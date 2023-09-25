import torch.nn as nn
from .Interpolate2D import BilinearInterpolate
from .Meshgrid import meshgrid2D


class ExponentiationLayer(nn.Module):
    def __init__(self, size, factor=4):
        super(ExponentiationLayer, self).__init__()
        self.factor = factor
        self.interpolate = BilinearInterpolate()
        grid = meshgrid2D(size[0], size[1])
        self.register_buffer('grid', grid)

    def forward(self, v, times):
        phi = self.grid + v * 2**(-self.factor)
        for i in range(times):
            phi = self.interpolate(phi, phi)
        phi = phi - self.grid
        return phi
