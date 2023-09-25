"""
requirement:pytorch==1.10.1
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ) -> None:
        super(Conv2d, self).__init__()

        self.shape = {
            "weight": [out_channels, in_channels // groups, *kernel_size],
            "bias": [out_channels] if bias is True else None,
        }

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, input, parameters):
        return F.conv2d(
            input,
            parameters["weight"],
            parameters["bias"],
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class Conv2dRelu(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=1
    ) -> None:
        super(Conv2dRelu, self).__init__()

        self.conv2d = Conv2d(in_channels, out_channels, kernel_size, stride, padding)

        self.shape = self.conv2d.shape

    def forward(self, input, parameters):
        out = self.conv2d(input, parameters["weight"], parameters["bias"])
        out = F.leaky_relu(out, negative_slope=0.2)
        return out

