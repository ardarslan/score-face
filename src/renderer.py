import torch
import numpy as np
from pytorch3d.io import load_obj
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer.mesh import Textures
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRasterizer
)
from pytorch3d.transforms import quaternion_to_matrix, axis_angle_to_quaternion, matrix_to_quaternion, quaternion_multiply
from utils import load_image_axis_angle, axis_angle_rotation


class Renderer(object):
    def __init__(self, image_size, camera_distance, device):
        self.raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            bin_size=0
        )
        self.camera_distance = camera_distance
        self.device = device
    
    def render(self, texture, background, pixel_uvs, background_mask):
        texture_sampled = torch.nn.functional.grid_sample(torch.flip(texture, dims=[-2]), pixel_uvs, mode='nearest', padding_mode='border', align_corners=False)
        return torch.where(background_mask, background, texture_sampled)

    def prerender(self, obj_path, axis_angle_path, texture, elev, azimuth):
        """
            Inputs:
                obj_path: str
                texture: torch.tensor with shape (N, C, H, W)
                elev: float
                azimuth: float
        """
        verts, faces, aux = load_obj(obj_path, device=self.device)
        verts_uvs = aux.verts_uvs[None, ...]  # (1, V, 2)
        faces_uvs = faces.textures_idx[None, ...]  # (1, F, 3)
        verts = verts[None, ...]
        faces = faces.verts_idx[None, ...]
        textures_uv = Textures(verts_uvs=verts_uvs, faces_uvs=faces_uvs, maps=texture.permute(0, 2, 3, 1))
        meshes = Meshes(verts=verts, faces=faces, textures=textures_uv)
        
        # center the mesh
        verts_packed = meshes.verts_packed()
        center = verts_packed.mean(0)
        scale = max((verts_packed - center).abs().max(0)[0])
        meshes.offset_verts_(-center)
        meshes.scale_verts_((1.0 / float(scale)))

        # rotate the mesh
        axis_angle_image = load_image_axis_angle(axis_angle_path, self.device)
        quaternion_image = axis_angle_to_quaternion(axis_angle_image)
        quaternion_fix = matrix_to_quaternion(axis_angle_rotation(axis="Y", angle=torch.tensor([[np.pi]], device=self.device)))
        quaternion_frontal = quaternion_multiply(quaternion_image, quaternion_fix)
        quaternion_elev = matrix_to_quaternion(axis_angle_rotation(axis="X", angle=torch.tensor([[np.pi * elev / 180.0]], device=self.device)))
        quaternion_azimuth = matrix_to_quaternion(axis_angle_rotation(axis="Y", angle=torch.tensor([[np.pi * azimuth / 180.0]], device=self.device)))
        quaternion_elev_azimuth = quaternion_multiply(quaternion_azimuth, quaternion_elev)
        quaternion_final = quaternion_multiply(quaternion_frontal, quaternion_elev_azimuth)
        
        R = quaternion_to_matrix(quaternion_final)[0]
        T = torch.tensor([[0.0, 0.0, self.camera_distance]], device=self.device)

        cameras = FoVPerspectiveCameras(R=R, T=T, device=self.device)
        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=self.raster_settings
        )
        fragments = rasterizer(meshes)

        is_background = (fragments.pix_to_face[..., 0] < 0)
        background_mask = is_background.unsqueeze(-1).repeat(1, 1, 1, 3).permute(0, 3, 1, 2)

        packing_list = [
            i[j] for i, j in zip(textures_uv.verts_uvs_list(), textures_uv.faces_uvs_list())
        ]
        faces_verts_uvs = torch.cat(packing_list)
        pixel_uvs = interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, faces_verts_uvs
        )
        N, H_out, W_out, K = fragments.pix_to_face.shape
        pixel_uvs = pixel_uvs.permute(0, 3, 1, 2, 4).reshape(N * K, H_out, W_out, 2)
        pixel_uvs = pixel_uvs * 2.0 - 1.0

        return pixel_uvs, background_mask