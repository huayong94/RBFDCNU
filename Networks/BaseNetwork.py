import torch
import torch.nn as nn
from Modules.Interpolation import SpatialTransformer


class BaseRegistraionNetwork(nn.Module):
    def __init__(self, image_size: list):
        super(BaseRegistraionNetwork, self).__init__()
        self.transformer = SpatialTransformer(image_size)
        self.name = ''

    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        raise NotImplementedError('Please define the forward method')

    def test(self, src: torch.Tensor, tgt: torch.Tensor):
        raise NotImplementedError('Please define the test method')

    def objective(self, src: torch.Tensor, tgt: torch.Tensor):
        raise NotImplementedError('Please define the objective method')

    def setHyperparam(self, **kwargs):
        raise NotImplementedError


class GenerativeRegistrationNetwork(BaseRegistraionNetwork):
    def __init__(self, image_size: list):
        super(GenerativeRegistrationNetwork, self).__init__(image_size)

    def sample(self, z_param):
        raise NotImplementedError('Please define the sample method')

    def uncertainty(self, src, tgt, K):
        raise NotImplementedError('Please define the uncertainty method')
