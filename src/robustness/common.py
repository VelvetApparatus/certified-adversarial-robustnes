import torch
from torch import nn
import random


class RobustnessRegularization(nn.Module):
    def __init__(
            self,
    ):
        super(RobustnessRegularization, self).__init__()

    def augment_on_batch(self, x, y, model):
        raise NotImplementedError
