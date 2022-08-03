import torch
from pytorch3d.io import load_obj
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer.mesh import Textures
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRasterizer
)
from pytorch3d.transforms import quaternion_to_matrix
from typing import Dict, Any, Tuple


class Renderer(object):
    def __init__(self, cfg: Dict[str, Any]) -> None:
        image_size = cfg["image_size"]
        texture_size = cfg["large_texture_size"]
        camera_distance = cfg["camera_distance"]
        batch_size = cfg["batch_size"]
        num_channels = cfg["num_channels"]
        input_obj_path = cfg["input_obj_path"]
        self.device = cfg["device"]

        verts, faces, aux = load_obj(input_obj_path, device=self.device)
        verts_uvs = aux.verts_uvs[None, ...]  # (1, V, 2)
        faces_uvs = faces.textures_idx[None, ...]  # (1, F, 3)
        verts = verts[None, ...]
        faces = faces.verts_idx[None, ...]
        texture = torch.randn(size=(batch_size, texture_size, texture_size, num_channels), device=self.device)

        self.textures_uv = Textures(verts_uvs=verts_uvs, faces_uvs=faces_uvs, maps=texture)
        self.meshes = Meshes(verts=verts, faces=faces, textures=self.textures_uv)
        verts_packed = self.meshes.verts_packed()
        center = verts_packed.mean(0)
        scale = max((verts_packed - center).abs().max(0)[0])
        self.meshes.offset_verts_(-center)
        self.meshes.scale_verts_((1.0 / float(scale)))

        self.T = torch.tensor([[0.0, 0.0, camera_distance]], device=self.device)

        self.raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            bin_size=0
        )
    
    def render(self, texture: torch.Tensor, background: torch.Tensor, pixel_uvs: torch.Tensor, background_mask: torch.Tensor) -> torch.Tensor:
        texture_sampled = torch.nn.functional.grid_sample(torch.flip(texture, dims=[-2]), pixel_uvs, mode='nearest', padding_mode='border', align_corners=False)
        return torch.where(background_mask, background, texture_sampled)

    def prerender(self, quaternion: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            Inputs:
                quaternion: torch.Tensor with shape (N, 4)
            Outputs:
                pixel_uvs: torch.Tensor with shape (N, image_size, image_size, 2)
                background_mask: torch.Tensor with shape (N, num_channels, image_size, image_size)
        """

        R = quaternion_to_matrix(quaternion)

        cameras = FoVPerspectiveCameras(R=R, T=self.T, device=self.device)
        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=self.raster_settings
        )
        fragments = rasterizer(self.meshes)

        is_background = (fragments.pix_to_face[..., 0] < 0)
        background_mask = is_background.unsqueeze(-1).repeat(1, 1, 1, 3).permute(0, 3, 1, 2)

        packing_list = [
            i[j] for i, j in zip(self.textures_uv.verts_uvs_list(), self.textures_uv.faces_uvs_list())
        ]
        faces_verts_uvs = torch.cat(packing_list)
        pixel_uvs = interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, faces_verts_uvs
        )
        N, H_out, W_out, K = fragments.pix_to_face.shape
        pixel_uvs = pixel_uvs.permute(0, 3, 1, 2, 4).reshape(N * K, H_out, W_out, 2)
        pixel_uvs = pixel_uvs * 2.0 - 1.0

        return pixel_uvs, background_mask
