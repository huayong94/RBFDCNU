import torch
import torch.nn as nn


class JacobianDeterminantMetric(nn.Module):
    def __init__(self, size=(128, 128)):
        super(JacobianDeterminantMetric, self).__init__()

    def forward(self, df):
        dx = df[:, :, :-1, 1:] - df[:, :, :-1, :-1]
        dy = df[:, :, 1:, :-1] - df[:, :, :-1, :-1]

        jacobian_map = (
            (dx[:, 0, :, :] + 1) *
            (dy[:, 1, :, :] + 1)) - (dx[:, 1, :, :] * dy[:, 0, :, :])

        return jacobian_map


class JacobianDeterminantLoss(nn.Module):
    def __init__(self, size=(128, 128)):
        super(JacobianDeterminantLoss, self).__init__()

    def forward(self, df):
        dx = df[:, :, :-1, 1:] - df[:, :, :-1, :-1]
        dy = df[:, :, 1:, :-1] - df[:, :, :-1, :-1]

        jacobian_map = (
            (dx[:, 0, :, :] + 1) *
            (dy[:, 1, :, :] + 1)) - (dx[:, 1, :, :] * dy[:, 0, :, :])

        # sigmoid = torch.max(-jacobian_map, torch.zeros_like(jacobian_map))

        sigmoid = (jacobian_map <= 0).float()
        jacobian_loss = torch.sum(sigmoid * (-jacobian_map + 1), dim=[1, 2])

        return jacobian_loss
