import torch
from pytorch3d.io import load_obj
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer.mesh import Textures
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRasterizer
)


class Renderer(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.raster_settings = RasterizationSettings(
            image_size=cfg["image_size"],
            blur_radius=0.0,
            bin_size=0
        )
        self.verts, self.faces, self.aux = load_obj(cfg["obj_path"], device=self.cfg["device"])
        self.verts_uvs = self.aux.verts_uvs[None, ...]  # (1, V, 2)
        self.faces_uvs = self.faces.textures_idx[None, ...]  # (1, F, 3)
        self.verts = self.verts[None, ...]
        self.faces = self.faces.verts_idx[None, ...]
    
    def render_prerendering_results(self, texture, background, pixel_uvs, background_mask):
        texture_sampled = torch.nn.functional.grid_sample(torch.flip(texture, dims=[-2]), pixel_uvs, mode='nearest', padding_mode='border', align_corners=False)
        return torch.where(background_mask, background, texture_sampled)
    
    def is_any_neighbor_background(self, background_mask, sample_idx, y, x):
        def find_indices(index, max_index):
            if index == 0:
                indices = [index, index + 1]
            elif index == max_index:
                indices = [index - 1, index]
            else:
                indices = [index - 1, index, index + 1]
            return indices
    
        ys = find_indices(index=y, max_index=background_mask.shape[-2] - 1)
        xs = find_indices(index=x, max_index=background_mask.shape[-1] - 1)

        for current_y in ys:
            for current_x in xs:
                if current_y == y and current_x == x:
                    continue
                if background_mask[sample_idx, 0, current_y, current_x].sum() == 1:
                    return True
        return False

    def render(self, texture, background, elev, azimuth, result_keys):
        """
            Inputs:
                texture: torch.tensor with shape (N, C, H, W)
                background: torch.tensor with shape (N, C, H, W)
                elev: float
                azimuth: float
                result_keys: list
        """
        textures_uv = Textures(verts_uvs=self.verts_uvs, faces_uvs=self.faces_uvs, maps=texture.permute(0, 2, 3, 1))
        meshes = Meshes(verts=self.verts, faces=self.faces, textures=textures_uv)
        verts_packed = meshes.verts_packed()
        center = verts_packed.mean(0)
        scale = max((verts_packed - center).abs().max(0)[0])
        meshes.offset_verts_(-center)
        meshes.scale_verts_((1.0 / float(scale)))
        R, T = look_at_view_transform(1.2, elev, azimuth) # changed from 1.2
        cameras = FoVPerspectiveCameras(R=R, T=T, device=self.cfg["device"])
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

        face = self.render_prerendering_results(texture=texture, background=background, pixel_uvs=pixel_uvs, background_mask=background_mask)

        # if "foreground_edges" in result_keys:
        #     foreground_edges = torch.zeros_like(background_mask)
        #     for sample_idx in range(background_mask.shape[0]):
        #         for y in range(background_mask.shape[2]):
        #             for x in range(background_mask.shape[3]):
        #                 if background_mask[sample_idx, 0, y, x] == 0: # foreground pixel
        #                     if self.is_any_neighbor_background(background_mask, sample_idx, y, x): # has a background neighbor
        #                         foreground_edges[sample_idx, :, y, x] = True
        #     face = torch.where(foreground_edges, torch.ones_like(face) * 0.99, face)
        # else:
        #     foreground_edges = None

        unfilled_mask = (1 * (face.sum(axis=1, keepdim=True) == 3).repeat(repeats=[1, 3, 1, 1])) * (1 - 1 * background_mask)  # white pixels are unfilled

        filled_mask = 1 - unfilled_mask # other pixels are filled

        results = {
            "background_mask": background_mask,
            # "foreground_edges": foreground_edges,
            "pixel_uvs": pixel_uvs,
            "unfilled_mask": unfilled_mask,
            "filled_mask": filled_mask,
            "face": face
        }

        return (results[result_key] for result_key in result_keys)
