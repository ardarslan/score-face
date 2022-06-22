import torch
from score_sde.configs.ve import ffhq_256_ncsnpp_continuous as configs
from score_sde.models import utils as mutils
from score_sde.models import ncsnpp  # this should be imported
from score_sde.sde_lib import VESDE
from score_sde.utils import restore_checkpoint
from score_sde.losses import get_optimizer
from score_sde.models.ema import ExponentialMovingAverage


def get_score_model(cfg):
    checkpoint_filepath = cfg["checkpoint_filepath"]
    config = configs.get_config()

    config.training.batch_size = cfg["batch_size"]
    config.eval.batch_size = cfg["batch_size"]

    score_model = mutils.create_model(config)
    score_model.eval()

    optimizer = get_optimizer(config, score_model.parameters())
    ema = ExponentialMovingAverage(score_model.parameters(),
                                   decay=config.model.ema_rate)
    state = dict(step=0, optimizer=optimizer,
                model=score_model, ema=ema)

    state = restore_checkpoint(checkpoint_filepath, state, config.device)
    ema.copy_to(score_model.parameters())
    
    return score_model


def get_sde():
    config = configs.get_config()
    return VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)


def get_score_fn(sde, score_model):
    def score_fn(x, vec_t):
        labels = sde.marginal_prob(torch.zeros_like(x), vec_t)[1]
        score = score_model(x, labels)
        return score
    return score_fn


def get_grad_texture_and_background(texture, background, grad_face, render_func):
    _, (grad_texture, grad_background) = torch.autograd.functional.vjp(func=render_func, inputs=(texture, background), v=grad_face, create_graph=False, strict=False)
    return grad_texture, grad_background

