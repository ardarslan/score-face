import functools
import torch
from score_sde.models import utils as mutils
from score_sde.models import ncsnpp  # this should be imported
from score_sde.sde_lib import VESDE
from score_sde.utils import restore_checkpoint
from score_sde.losses import get_optimizer
from score_sde.models.ema import ExponentialMovingAverage
from score_sde.controllable_generation import shared_predictor_update_fn, shared_corrector_update_fn
from score_sde.sampling import ReverseDiffusionPredictor, LangevinCorrector
from typing import Dict, Any, Callable


def get_sde(cfg: Dict[str, Any]) -> VESDE:
    if cfg["optimization_space"] == "image":
        from score_sde.configs.ve import ffhq_ncsnpp_continuous as configs
    elif cfg["optimization_space"] == "texture":
        from score_sde.configs.ve import ffhq_256_ncsnpp_continuous as configs
    else:
        raise Exception(f"Not a valid optimization_space {cfg['optimization_space']}.")

    sde_N = cfg["sde_N"]
    config = configs.get_config()
    sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=sde_N)
    return sde


def get_score_model(cfg: Dict[str, Any]) -> torch.nn.DataParallel:
    checkpoint_path = cfg["checkpoint_path"]
    batch_size = cfg["batch_size"]
    device = cfg["device"]

    if cfg["optimization_space"] == "image":
        from score_sde.configs.ve import ffhq_ncsnpp_continuous as configs
    elif cfg["optimization_space"] == "texture":
        from score_sde.configs.ve import ffhq_256_ncsnpp_continuous as configs
    else:
        raise Exception(f"Not a valid optimization_space {cfg['optimization_space']}.")

    config = configs.get_config()
    config.training.batch_size = batch_size
    config.eval.batch_size = batch_size
    config.device = device

    score_model = mutils.create_model(config)
    score_model.eval()

    optimizer = get_optimizer(config, score_model.parameters())
    ema = ExponentialMovingAverage(score_model.parameters(),
                                decay=config.model.ema_rate)
    state = dict(step=0, optimizer=optimizer, model=score_model, ema=ema)
    state = restore_checkpoint(checkpoint_path, state, device)
    ema.copy_to(score_model.parameters())
    return score_model


def get_score_fn(sde: VESDE, score_model: torch.nn.DataParallel) -> Callable:
    def score_fn(x, vec_t):
        labels = sde.marginal_prob(torch.zeros_like(x), vec_t)[1]
        score = score_model(x, labels)
        return score
    return score_fn


def get_inpaint_update_fn(update_fn: Callable, sde: VESDE) -> Callable:
    """Modify the update function of predictor & corrector to incorporate data information."""
    def inpaint_update_fn(model, data, mask, x, t):
        with torch.no_grad():
            vec_t = torch.ones(data.shape[0], device=data.device) * t
            x, x_mean = update_fn(x, vec_t, model=model)
            masked_data_mean, std = sde.marginal_prob(data, vec_t)
            masked_data = masked_data_mean + torch.randn_like(x) * std[:, None, None, None]
            x = x * (1. - mask) + masked_data * mask
            x_mean = x * (1. - mask) + masked_data_mean * mask
            return x, x_mean
    return inpaint_update_fn


def get_reverse_diffusion_predictor_inpaint_update_fn(sde: VESDE) -> Callable:
    predictor = ReverseDiffusionPredictor
    predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                            sde=sde,
                                            predictor=predictor,
                                            probability_flow=False,
                                            continuous=True)
    return get_inpaint_update_fn(predictor_update_fn, sde)


def get_langevin_corrector_inpaint_update_fn(cfg: Dict[str, Any], sde: VESDE) -> Callable:
    snr = cfg["snr"]
    num_corrector_steps = cfg["num_corrector_steps"]
    corrector = LangevinCorrector
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=True,
                                            snr=snr,
                                            n_steps=num_corrector_steps)
    return get_inpaint_update_fn(corrector_update_fn, sde)