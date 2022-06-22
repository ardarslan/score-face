from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import io
import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import importlib
import os
import functools
import itertools
import torch

import torch.nn as nn
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_gan as tfgan
import tqdm
import io
import inspect
sns.set(font_scale=2)
sns.set(style="whitegrid")

import score_sde.models
from score_sde.models import utils as mutils
from score_sde.models import ncsnv2
from score_sde.models import ncsnpp
from score_sde.models import ddpm as ddpm_model
from score_sde.models import layerspp
from score_sde.models import layers
from score_sde.models import normalization

#from configs.ncsnpp import cifar10_continuous_ve as configs
# from score_sde.configs.ddpm import cifar10_continuous_vp as configs
# config = configs.get_config()

# checkpoint = torch.load('exp/ddpm_continuous_vp.pth')

# #score_model = ncsnpp.NCSNpp(config)
# score_model = ddpm_model.DDPM(config)
# score_model.load_state_dict(checkpoint)
# score_model = score_model.eval()
# x = torch.ones(8, 3, 32, 32)
# y = torch.tensor([1] * 8)
# breakpoint()
# with torch.no_grad():
#   score = score_model(x, y)