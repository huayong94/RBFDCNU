import math

import torch
from torch import nn
from torch.nn import functional as F

from .Meshgrid import meshgrid2D


class GaussianSmoothing2D(nn.Module):
    def __init__(self, kernel_size=15, sigma=3, channel=2):
        super(GaussianSmoothing2D, self).__init__()

        self.kernel_size = kernel_size
        self.channel = channel

        grid = meshgrid2D(kernel_size, kernel_size)
        grid = grid - kernel_size // 2
        grid_x, grid_y = grid[0], grid[1]

        distance = torch.sqrt(grid_x**2 + grid_y**2)
        kernel = 1 / (sigma * math.sqrt(2 * math.pi)) * torch.exp(
            -(distance / sigma)**2 / 2)
        kernel = kernel / torch.sum(kernel)
        kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(channel, 1, 1, 1)

        self.register_buffer('weight', kernel)

    def forward(self, flow):
        return F.conv2d(flow,
                        self.weight,
                        padding=self.kernel_size // 2,
                        groups=self.channel)
