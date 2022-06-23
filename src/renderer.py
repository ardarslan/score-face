import torch
import numpy as np
from pytorch3d.io import load_obj
from pytorch3d.renderer.mesh import Textures
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh.shader import ShaderBase
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.renderer.mesh.shading import phong_shading
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer
)


class HardPhongShader(ShaderBase):
    """
    Per pixel lighting - the lighting model is applied using the interpolated
    coordinates and normals for each pixel. The blending function hard assigns
    the color of the closest face for each pixel.
    To use the default values, simply initialize the shader with the desired
    device e.g.
    .. code-block::
        shader = HardPhongShader(device=torch.device("cuda:0"))
    """
    def __init__(self, backgrounds, *args, **kwargs):
        self.backgrounds = backgrounds
        super().__init__(*args, **kwargs)
    
    def _hard_rgb_blend(self, colors: torch.Tensor, backgrounds: torch.Tensor, fragments: torch.Tensor) -> torch.Tensor:
        """
        Naive blending of top K faces to return an RGBA image
        - **RGB** - choose color of the closest point i.e. K=0
        - **A** - 1.0

        Args:
            colors: (N, H, W, K, 3) RGB color for each of the top K faces per pixel.
            fragments: the outputs of rasterization. From this we use
                - pix_to_face: LongTensor of shape (N, H, W, K) specifying the indices
                of the faces (in the packed representation) which
                overlap each pixel in the image. This is used to
                determine the output shape.
            blend_params: BlendParams instance that contains a background_color
            field specifying the color for the background
        Returns:
            RGBA pixel_colors: (N, H, W, 4)
        """

        # Mask for the background.
        is_background = (fragments.pix_to_face[..., 0] < 0)
        mask = is_background.unsqueeze(-1).repeat(1, 1, 1, 3)  # (N, H, W, 3)
        pixel_colors = torch.where(mask, backgrounds, colors[..., 0, :])
        # Concat with the alpha channel.
        alpha = (~is_background).type_as(pixel_colors)[..., None]
        return torch.cat([pixel_colors, alpha], dim=-1)

    def forward(self, fragments: Fragments, meshes: Meshes, **kwargs) -> torch.Tensor:
        cameras = kwargs.get("cameras", self.cameras)
        texels = meshes.sample_textures(fragments)
        lights = kwargs.get("lights", self.lights)
        materials = kwargs.get("materials", self.materials)
        backgrounds = kwargs.get("backgrounds", self.backgrounds)
        colors = phong_shading(
            meshes=meshes,
            fragments=fragments,
            texels=texels,
            lights=lights,
            cameras=cameras,
            materials=materials,
        )
        images = self._hard_rgb_blend(colors, backgrounds, fragments)
        return images, fragments


class Renderer(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.raster_settings = RasterizationSettings(
            image_size=cfg["image_size"],
            blur_radius=0.0,
            bin_size=0
        )
        obj_path = self.cfg["obj_path"]
        self.verts, self.faces, self.aux = load_obj(obj_path, device=self.cfg["device"])
        self.verts_uvs = self.aux.verts_uvs[None, ...]  # (1, V, 2)
        self.faces_uvs = self.faces.textures_idx[None, ...]  # (1, F, 3)
    
    def get_random_elev_azimuth(self):
        if self.cfg["elev_azimuth_random"]:
            elev = np.random.random() * 30 - 60
            azimuth = np.random.random() * 30 - 60
        else:
            elev = 0.0
            azimuth = 0.0
        return elev, azimuth
    
    def render(self, texture, background, elev, azimuth):
        """
            Inputs:
                texture: torch.tensor with shape (N, C, H, W)
                background: torch.tensor with shape (N, C, H, W)
                elev: float
                azimuth: float
            Outputs:
                face: (N, C, H, W)
        """
        textures_uv = Textures(verts_uvs=self.verts_uvs, faces_uvs=self.faces_uvs, maps=texture.permute(0, 2, 3, 1))
        meshes = Meshes(verts=[self.verts], faces=[self.faces.verts_idx], textures=textures_uv)
        verts_packed = meshes.verts_packed()
        center = verts_packed.mean(0)
        scale = max((verts_packed - center).abs().max(0)[0])
        meshes.offset_verts_(-center)
        meshes.scale_verts_((1.0 / float(scale)))
        R, T = look_at_view_transform(1.25, elev, azimuth)
        cameras = FoVPerspectiveCameras(R=R, T=T, device=self.cfg["device"])
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=self.raster_settings
            ),
            shader=HardPhongShader(
                device=self.cfg["device"], 
                cameras=cameras,
                backgrounds=background.permute(0, 2, 3, 1)
            )
        )
        face, fragments = renderer(meshes)
        face = face[:, :, :, :3].permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        return face, fragments, textures_uv
