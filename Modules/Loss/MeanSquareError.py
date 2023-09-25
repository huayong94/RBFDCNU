import torch
import torch.nn as nn


class MeanSquareError(nn.Module):
    def __init__(self, weight=1, sum_up=False):
        super(MeanSquareError, self).__init__()
        self.sum_up = sum_up
        self.weight = weight

    def set(self, weight):
        self.weight = weight

    def forward(self, I: torch.Tensor, J: torch.Tensor) -> torch.Tensor:

        ndims = len(I.size()) - 2
        df = I - J
        if self.sum_up:
            return self.weight * torch.sum(df * df, dim=[*range(1, ndims + 2)])
        else:
            return self.weight * torch.mean(df * df,
                                            dim=[*range(1, ndims + 2)])
