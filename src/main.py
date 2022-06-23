import cv2
import torch
from tqdm import tqdm
import torch.nn.functional as F
from pytorch3d.ops import interpolate_face_attributes

from utils import (
    get_argument_parser,
    set_seeds,
    get_experiment_name,
    save_cfg
)
from model_utils import (
    get_score_model,
    get_sde,
    get_score_fn,
)
from visualization import save_images
from renderer import Renderer


def main(cfg):
    cfg["experiment_name"] = get_experiment_name()
    if torch.cuda.is_available():
        cfg["device"] = torch.device(f"cuda:{cfg['cuda_device']}")
        torch.cuda.set_device(cfg["device"])
    else:
        cfg["device"] = torch.device("cpu")
    print(f"Using {cfg['device']}.")

    save_cfg(cfg)
    set_seeds(cfg)

    score_model = get_score_model(cfg)
    sde = get_sde()
    score_fn = get_score_fn(score_model=score_model, sde=sde)
    renderer = Renderer(cfg=cfg)

    with torch.no_grad():
        image_shape = (cfg["batch_size"], cfg["num_channels"], cfg["image_size"], cfg["image_size"])
        texture_shape = (cfg["batch_size"], cfg["num_channels"], cfg["texture_size"], cfg["texture_size"])

        background = torch.zeros(size=image_shape, device=cfg["device"])
        texture = torch.ones(size=texture_shape, device=cfg["device"])
        elev, azimuth = renderer.get_random_elev_azimuth()
        face, fragments, textures_uv = renderer.render(texture=torch.randn(size=texture_shape, device=cfg["device"]),
                                                       background=background,
                                                       elev=elev,
                                                       azimuth=azimuth)
        is_background = (fragments.pix_to_face[..., 0] < 0)
        face = torch.where(is_background, background, torch.randn(size=texture_shape, device=cfg["device"]) * cfg["initial_noise_coefficient"])

        timesteps = torch.linspace(cfg["sde_T"], cfg["sampling_eps"], cfg["sde_N"], device=cfg["device"])
        rsde = sde.reverse(score_fn, cfg["probability_flow"])
        alpha = torch.ones(cfg["batch_size"], device=cfg["device"])

        for i in tqdm(list(range(cfg["sde_N"]))):
            t = timesteps[i]
            vec_t = torch.ones(cfg["batch_size"], device=cfg["device"]) * t
            labels = sde.marginal_prob(torch.zeros(size=image_shape), vec_t)[1]

            # update face
            for j in range(cfg["num_corrector_steps"]):
                grad_face = score_model(face, labels)
                noise_face = torch.randn(size=image_shape, device=cfg["device"])
                grad_face_norm = torch.norm(grad_face.reshape(grad_face.shape[0], -1), dim=-1).mean()
                noise_face_norm = torch.norm(noise_face.reshape(noise_face.shape[0], -1), dim=-1).mean()
                step_size_face = (cfg["snr"] * noise_face_norm / grad_face_norm) ** 2 * 2 * alpha
                face_mean = face + step_size_face[:, None, None, None] * grad_face
                face = face_mean + torch.sqrt(step_size_face * 2)[:, None, None, None] * noise_face
            f, G = rsde.discretize(face, vec_t)
            z = torch.randn(size=image_shape, device=cfg["device"])
            face_mean = face - f
            face = face_mean + G[:, None, None, None] * z

            # face = torch.tensor(cv2.resize(cv2.imread("../asd/40044.png")[:, :, [2, 1, 0]], (cfg["image_size"], cfg["image_size"]), interpolation=cv2.INTER_CUBIC), dtype=torch.float32, device=cfg["device"]).permute(2, 0, 1).unsqueeze(0) / 255.0
            # print(face.shape)
            face = renderer.render(texture=torch.tensor(cv2.imread("assets/40044.png")[:, :, [2, 1, 0]], dtype=torch.float32, device=cfg["device"]).permute(2, 0, 1).unsqueeze(0) / 255.0,
                                   background=background,
                                   elev=elev,
                                   azimuth=azimuth)[0]
            save_images(cfg, face, "face_initial", i)

            # update texture
            packing_list = [
                i[j] for i, j in zip(textures_uv.verts_uvs_list(), textures_uv.faces_uvs_list())
            ]
            faces_verts_uvs = torch.cat(packing_list)

            pixel_uvs = interpolate_face_attributes(
                fragments.pix_to_face, fragments.bary_coords, faces_verts_uvs
            )

            for sample_idx in range(cfg["batch_size"]):
                for y in range(cfg["image_size"]):
                    for x in range(cfg["image_size"]):
                        v = int(cfg["texture_size"] - pixel_uvs[sample_idx, y, x, 0, 0] * 255 - 1)
                        u = int(cfg["texture_size"] - pixel_uvs[sample_idx, y, x, 0, 1] * 255 - 1)
                        texture[sample_idx, :, v, u] = face[sample_idx, :, y, x]

            save_images(cfg, texture, "texture_initial", i)

            N, H_out, W_out, K = fragments.pix_to_face.shape
            N, H_in, W_in, C = cfg["batch_size"], cfg["texture_size"], cfg["texture_size"], cfg["num_channels"]

            # pixel_uvs: (N, H, W, K, 2) -> (N, K, H, W, 2) -> (NK, H, W, 2)
            pixel_uvs = pixel_uvs.permute(0, 3, 1, 2, 4).reshape(N * K, H_out, W_out, 2)

            # textures.map:
            #   (N, H, W, C) -> (N, C, H, W) -> (1, N, C, H, W)
            #   -> expand (K, N, C, H, W) -> reshape (N*K, C, H, W)
            texture_maps = (
                texture[None, ...]
                .expand(K, -1, -1, -1, -1)
                .transpose(0, 1)
                .reshape(N * K, C, H_in, W_in)
            )

            # Textures: (N*K, C, H, W), pixel_uvs: (N*K, H, W, 2)
            # Now need to format the pixel uvs and the texture map correctly!
            # From pytorch docs, grid_sample takes `grid` and `input`:
            #   grid specifies the sampling pixel locations normalized by
            #   the input spatial dimensions It should have most
            #   values in the range of [-1, 1]. Values x = -1, y = -1
            #   is the left-top pixel of input, and values x = 1, y = 1 is the
            #   right-bottom pixel of input.

            pixel_uvs = pixel_uvs * 2.0 - 1.0

            texture_maps = torch.flip(texture_maps, [2])  # flip y axis of the texture map
            if texture_maps.device != pixel_uvs.device:
                texture_maps = texture_maps.to(pixel_uvs.device)
            texels = F.grid_sample(
                texture_maps,
                pixel_uvs,
                mode=textures_uv.sampling_mode,
                align_corners=textures_uv.align_corners,
                padding_mode=textures_uv.padding_mode,
            )
            # texels now has shape (NK, C, H_out, W_out)
            texels = texels.reshape(N, K, C, H_out, W_out).permute(0, 3, 4, 1, 2)

            save_images(cfg, texels[:, :, :, 0, :], "texels", i)
            
            elev, azimuth = renderer.get_random_elev_azimuth()
            face, fragments, textures_uv = renderer.render(texture=texture,
                                                           background=background,
                                                           elev=elev,
                                                           azimuth=azimuth)

            if (i + 1) % 1 == 0:
                save_images(cfg, face, "face", i)
                save_images(cfg, texture, "texture", i)
                save_images(cfg, background, "background", i)

if __name__ == "__main__":
    cfg = get_argument_parser().parse_args().__dict__
    main(cfg)
