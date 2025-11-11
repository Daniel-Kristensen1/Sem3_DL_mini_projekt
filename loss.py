
import torch
import config
from torch import nn as nn
from torch.nn import functional as F


class LossFunc(nn.Module):
    def __init__(self):
        super().__init__()
        self.coord = config.COORD
        self.noobj = config.NOOBJ