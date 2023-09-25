import torch
import torch.nn as nn
import numpy as np


class BendingEnergyLoss(nn.Module):
    def __init__(self):
        super(BendingEnergyLoss, self).__init__()

    def forward(self, flow):

        dx, dy = flow[:, 0, :, :], flow[:, 1, :, :]

        gx = dx[:, 1:-1, :-2] + dx[:, 1:-1, 2:] - 2 * dx[:, 1:-1, 1:-1]
        gy = dy[:, :-2, 1:-1] + dy[:, 2:, 1:-1] - 2 * dy[:, 1:-1, 1:-1]

        gx_y = dx[:, 1:-1, 1:-1] - dx[:, 1:-1, 2:] - dx[:, 2:,
                                                        1:-1] + dx[:, 2:, 2:]
        gy_x = dy[:, 1:-1, 1:-1] - dy[:, 2:, 1:-1] - dy[:, 1:-1,
                                                        2:] + dy[:, 2:, 2:]

        ex = gx * gx + gx_y * gx_y
        ey = gy * gy + gy_x * gy_x

        be = torch.mean(ex) + torch.mean(ey)

        return be / 2.0


class BendingEnergyMetric(nn.Module):
    def __init__(self):
        super(BendingEnergyMetric, self).__init__()

    def forward(self, flow):

        dx, dy = flow[:, 0, :, :], flow[:, 1, :, :]

        gx = dx[:, 1:-1, :-2] + dx[:, 1:-1, 2:] - 2 * dx[:, 1:-1, 1:-1]
        gy = dy[:, :-2, 1:-1] + dy[:, 2:, 1:-1] - 2 * dy[:, 1:-1, 1:-1]

        gx_y = dx[:, 1:-1, 1:-1] - dx[:, 1:-1, 2:] - dx[:, 2:,
                                                        1:-1] + dx[:, 2:, 2:]
        gy_x = dy[:, 1:-1, 1:-1] - dy[:, 2:, 1:-1] - dy[:, 1:-1,
                                                        2:] + dy[:, 2:, 2:]

        ex = gx * gx + gx_y * gx_y
        ey = gy * gy + gy_x * gy_x

        be = torch.mean(ex, [1, 2]) + torch.mean(ey, [1, 2])

        return be / 2.0


class RBFBendingEnergyLoss(nn.Module):
    def __init__(self, cpoint_pos, r):
        super(RBFBendingEnergyLoss, self).__init__()
        self.num_cp = cpoint_pos.size()[0]
        scppos = cpoint_pos.unsqueeze(1).repeat(1, self.num_cp, 1)
        despos = cpoint_pos.unsqueeze(0).repeat(self.num_cp, 1, 1)
        dis = torch.norm(scppos - despos, dim=2) / r
        filter_dis = dis < 1
        weight = torch.pow(1 - dis, 4) * (4 * dis + 1)
        weight = (weight * filter_dis.float()).unsqueeze(0)
        self.register_buffer('weight', weight)

    def be(self, alpha):
        flatted_alpha = torch.flatten(alpha, start_dim=1)
        tiled_alpha = flatted_alpha.unsqueeze(1).repeat(1, self.num_cp, 1)
        temp_res = torch.sum(tiled_alpha * self.weight, dim=2)
        be = torch.sum(flatted_alpha * temp_res, dim=1)
        return be

    def forward(self, alpha):
        be_x = self.be(alpha[:, :, 0])
        be_y = self.be(alpha[:, :, 1])
        return (be_x + be_y) / 2


class RBFBendingEnergyLossA(nn.Module):
    def __init__(self):
        super(RBFBendingEnergyLossA, self).__init__()

    def be(self, alpha, cpoint_pos, r):
        self.num_cp = cpoint_pos.size()[0]
        scppos = cpoint_pos.unsqueeze(1).repeat(1, self.num_cp, 1)
        despos = cpoint_pos.unsqueeze(0).repeat(self.num_cp, 1, 1)
        dis = torch.norm(scppos - despos, dim=2) / r
        filter_dis = dis < 1
        weight = torch.pow(1 - dis, 4) * (4 * dis + 1)
        weight = (weight * filter_dis.float()).unsqueeze(0)
        flatted_alpha = torch.flatten(alpha, start_dim=1)
        tiled_alpha = flatted_alpha.unsqueeze(1).repeat(1, self.num_cp, 1)
        temp_res = torch.sum(tiled_alpha * weight, dim=2)
        be = torch.sum(flatted_alpha * temp_res, dim=1)
        return be

    def forward(self, alpha, cpoint_pos, r):
        be_x = self.be(alpha[:, :, 0], cpoint_pos, r)
        be_y = self.be(alpha[:, :, 1], cpoint_pos, r)
        return (be_x + be_y) / 2