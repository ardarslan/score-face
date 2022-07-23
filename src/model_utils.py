import functools

import torch
from score_sde.configs.ve import ffhq_256_ncsnpp_continuous as configs
from score_sde.models import utils as mutils
from score_sde.models import ncsnpp  # this should be imported
from score_sde.sde_lib import VESDE
from score_sde.utils import restore_checkpoint
from score_sde.losses import get_optimizer
from score_sde.models.ema import ExponentialMovingAverage
from score_sde.controllable_generation import shared_predictor_update_fn, shared_corrector_update_fn
from score_sde.sampling import ReverseDiffusionPredictor, LangevinCorrector


def get_sde(cfg):
    config = configs.get_config()
    sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=cfg["sde_N"])
    return sde


def get_score_model(cfg):
    config = configs.get_config()
    batch_size = cfg["batch_size"]
    config.training.batch_size = batch_size
    config.eval.batch_size = batch_size
    config.device = cfg["device"]

    score_model = mutils.create_model(config)
    score_model.eval()

    optimizer = get_optimizer(config, score_model.parameters())
    ema = ExponentialMovingAverage(score_model.parameters(),
                                decay=config.model.ema_rate)
    state = dict(step=0, optimizer=optimizer, model=score_model, ema=ema)
    state = restore_checkpoint("../assets/checkpoint_48.pth", state, config.device)
    ema.copy_to(score_model.parameters())
    return score_model


def get_inpaint_update_fn(update_fn, sde):
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


def get_reverse_diffusion_predictor_inpaint_update_fn(cfg, sde):
    predictor = ReverseDiffusionPredictor
    predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                            sde=sde,
                                            predictor=predictor,
                                            probability_flow=False,
                                            continuous=True)
    return get_inpaint_update_fn(predictor_update_fn, sde)


def get_langevin_corrector_inpaint_update_fn(cfg, sde):
    corrector = LangevinCorrector
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=True,
                                            snr=cfg["snr"],
                                            n_steps=cfg["num_corrector_steps"])
    return get_inpaint_update_fn(corrector_update_fn, sde)
