import torch
from score_sde.configs.ve import ffhq_256_ncsnpp_continuous as configs
from score_sde.models import utils as mutils
from score_sde.models import ncsnpp  # this should be imported
from score_sde.sde_lib import VESDE
from score_sde.utils import restore_checkpoint
from score_sde.losses import get_optimizer
from score_sde.models.ema import ExponentialMovingAverage


def get_sde(sde_N):
    config = configs.get_config()
    sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=sde_N)
    return sde


def get_score_model(checkpoint_path, batch_size, device):
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


def get_score_fn(sde, score_model):
    def score_fn(x, vec_t):
        labels = sde.marginal_prob(torch.zeros_like(x), vec_t)[1]
        score = score_model(x, labels)
        return score
    return score_fn
