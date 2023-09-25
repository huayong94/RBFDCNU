import numpy as np
from numpy.lib.function_base import select
import torch
import torch.nn as nn
from ..Loss.Distance import MaxMinPointDist


class RadialBasisLayer(nn.Module):
    """RadialBasisFunction
    The layer is radial basis interpolation function.
    Given a group of control points (position and value) and a mesh,
    the interpolation result of the mesh is computed by the layer.
    """
    def __init__(self, cpoint_pos, img_size=(128, 128), c=2):
        """
        Args:
            cpoint_pos (tensor[n,2]): the position tensor of control points
            img_size (tuple, optional): the space size of tensor. Defaults to (128, 128).
            c (int, optional): the compact support. Defaults to 2.
        """
        super(RadialBasisLayer, self).__init__()
        cpoint_num = cpoint_pos.size()[0]
        cpoint_size = torch.max(cpoint_pos, 0)[0][[1, 0]]

        # a location mesh of output
        loc_vectors = [
            torch.linspace(0.0, c_s - 1, i_s)
            for (i_s, c_s) in zip(img_size, cpoint_size)
        ]
        loc = torch.meshgrid(loc_vectors)
        loc = torch.stack(loc, 2)
        loc = loc[:, :, [1, 0]].float().unsqueeze(2)
        # repeating for calculate the distance of contorl cpoints
        loc_tile = loc.repeat(1, 1, cpoint_num, 1)

        # a location mesh of control points
        cp_loc = cpoint_pos.unsqueeze(0).unsqueeze(0)
        cp_loc_tile = cp_loc.repeat(*img_size, 1, 1)

        # calculate r
        dist = torch.norm(loc_tile - cp_loc_tile, dim=3) / c
        # add mask for r < 1
        mask = dist < 1
        # weight if r<1 weight=(1-r)^4*(4r+1)
        #        else   weight=0
        # Todo: reduce weight size
        weight = torch.pow(1 - dist, 4) * (4 * dist + 1)
        weight = weight * mask.float()
        weight = weight.unsqueeze(0).unsqueeze(4)
        # print(weight[0, 64, 64, 64:])
        self.register_buffer('weight', weight)
        self.first = True

    def forward(self, alpha):
        alpha = alpha.unsqueeze(1).unsqueeze(1)
        phi = torch.sum(self.weight * alpha, 3)
        phi = phi.permute(0, 3, 1, 2)
        return phi


class RadialBasisArbitraryLayer(nn.Module):
    """RadialBasisFunction with Arbitrary Control point
    The layer is radial basis interpolation function.
    Given a group of control points (position and value) and a mesh,
    the interpolation result of the mesh is computed by the layer.
    """
    def __init__(self, i_size, c_factor=2, cpoint_num=160):
        super(RadialBasisArbitraryLayer, self).__init__()
        # a location mesh of output
        loc_vectors = [torch.linspace(0.0, i_s - 1, i_s) for i_s in i_size]
        loc = torch.meshgrid(loc_vectors)
        loc = torch.stack(loc, 2)
        loc = loc[:, :, [1, 0]].float().unsqueeze(2)
        # repeating for calculate the distance of contorl cpoints
        loc_tile = loc.repeat(1, 1, cpoint_num, 1)
        self.register_buffer('loc_tile', loc_tile)
        self.cpoint_maxmin = MaxMinPointDist(cpoint_num)
        self.c_factor = c_factor
        self.i_size = i_size

    def forward(self, cpoint_loc, alpha):
        # compute the compact support
        c = self.cpoint_maxmin(cpoint_loc) * self.c_factor
        c = c.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        # a location mesh of control points
        cp_loc = cpoint_loc.unsqueeze(1).unsqueeze(1)
        cp_loc_tile = cp_loc.repeat(1, *self.i_size, 1, 1)
        # calculate r
        # print(cp_loc.shape, cp_loc_tile.shape,c.shape)
        dist = torch.norm(self.loc_tile - cp_loc_tile, dim=4) / c
        # add mask for r < 1
        mask = dist < 1
        # weight if r<1 weight=(1-r)^4*(4r+1)
        #        else   weight=0
        # Todo: reduce weight size
        weight = torch.pow(1 - dist, 4) * (4 * dist + 1)
        weight = weight * mask.float()
        weight = weight.unsqueeze(4)

        alpha = alpha.unsqueeze(1).unsqueeze(1)
        phi = torch.sum(weight * alpha, 3)
        phi = phi.permute(0, 3, 1, 2)
        return phi


class RadialBasisArbitraryLayerG(nn.Module):
    def __init__(self, i_size, c_factor=2, cpoint_num=160):
        super(RadialBasisArbitraryLayerG, self).__init__()
        self.c_factor = c_factor
        self.i_size = i_size
        self.cpoint_maxmin = MaxMinPointDist(cpoint_num)
        self.cpoint_num = cpoint_num

    def forward(self, cpoint_loc, alpha):
        batch_size = alpha.size()[0]

        r = self.cpoint_maxmin(cpoint_loc) * self.c_factor
        r_max = torch.ceil(torch.max(r)).item()
        r_max = r_max if r_max <= 37 else 37
        r = r.unsqueeze(1).unsqueeze(1)
        win_loc = torch.linspace(-r_max,
                                 r_max - 1,
                                 int(2 * r_max),
                                 device=alpha.device,
                                 requires_grad=False)
        win_grid = torch.meshgrid([win_loc, win_loc])
        win_grid = torch.stack(win_grid, 2)[..., [1, 0]]
        win_grid = torch.flatten(win_grid.unsqueeze(0).unsqueeze(0),
                                 start_dim=2,
                                 end_dim=3)
        # cpoint_loc_tile = cpoint_loc.unsqueeze(2)
        cpoint_loc_tile = torch.zeros_like(cpoint_loc)
        cpoint_loc_tile[..., 0] = torch.clamp(cpoint_loc[..., 0], r_max,
                                              self.i_size[1] - r_max)
        cpoint_loc_tile[..., 1] = torch.clamp(cpoint_loc[..., 1], r_max,
                                              self.i_size[1] - r_max)
        cpoint_loc_tile = cpoint_loc_tile.unsqueeze(2)
        img_loc = torch.floor(win_grid + cpoint_loc_tile)
        dist = torch.norm(img_loc - cpoint_loc.unsqueeze(2), dim=3) / r

        # img_loc_x = img_loc[:, :, :, 0]
        # img_loc_y = img_loc[:, :, :, 1]

        # x_in = (img_loc_x >= 0).float() * (img_loc_x <=
        #                                    self.i_size[1] - 1).float()
        # y_in = (img_loc_y >= 0).float() * (img_loc_y <=
        #                                    self.i_size[0] - 1).float()
        # dist_in = (dist < 1).float()
        # mask = x_in * y_in * dist_in
        mask = (dist < 1).float()

        weight = torch.pow(1 - dist, 4) * (4 * dist + 1) * mask.float()
        val = weight.unsqueeze(3) * alpha.unsqueeze(2)
        val = torch.flatten(val, start_dim=0, end_dim=2)

        index_b = torch.arange(batch_size,
                               device=alpha.device,
                               requires_grad=False).unsqueeze(1).repeat(
                                   1, self.cpoint_num * 4 * int(r_max) *
                                   int(r_max)).view(-1).long()
        img_loc = torch.flatten(img_loc, start_dim=0, end_dim=2).long()

        phi_zero = torch.zeros(batch_size,
                               *self.i_size,
                               device=alpha.device,
                               requires_grad=False)
        phi_x = phi_zero.index_put((index_b, img_loc[:, 1], img_loc[:, 0]),
                                   val[:, 0],
                                   accumulate=True)
        phi_y = phi_zero.index_put((index_b, img_loc[:, 1], img_loc[:, 0]),
                                   val[:, 1],
                                   accumulate=True)
        phi = torch.stack([phi_x, phi_y], 1)
        return phi


class RadialBasisArbitraryLayerT(nn.Module):
    def __init__(self, i_size, c, cpoint_size):
        super(RadialBasisArbitraryLayerT, self).__init__()
        cpoint_yx = [
            torch.linspace(0, s - 1, c) for s, c in zip(i_size, cpoint_size)
        ]
        cpoint_grid = torch.flatten(torch.stack(torch.meshgrid(cpoint_yx),
                                                2)[..., [1, 0]],
                                    start_dim=0,
                                    end_dim=1).unsqueeze(0).unsqueeze(0)
        img_yx = [torch.linspace(0, s - 1, s) for s in i_size]
        img_grid = torch.stack(torch.meshgrid(img_yx), 2)[...,
                                                          [1, 0]].unsqueeze(2)

        dist = torch.norm(img_grid - cpoint_grid, dim=3) / c
        sorted_dist, index = torch.sort(dist, 2)

        mask = dist < 1
        cpoint_max = mask.sum(2).max()

        select_dist = sorted_dist[..., :cpoint_max]
        select_dist_index = index[..., :cpoint_max]
        select_mask = select_dist < 1

        select_cpoints_x = torch.take(cpoint_grid[..., 0], select_dist_index)
        select_cpoints_y = torch.take(cpoint_grid[..., 1], select_dist_index)
        select_cpoints = torch.stack([select_cpoints_x, select_cpoints_y], 3)

        phi_0 = torch.pow(1 - select_dist,
                          4) * (4 * select_dist + 1) * select_mask

        phi_r = -4 * torch.pow(1 - select_dist, 3) * (
            4 * select_dist + 1) + 4 * torch.pow(1 - select_dist, 4)

        r_x = (select_cpoints[..., 0] -
               img_grid[..., 0]) / (select_dist * c * c + 1e-5)
        r_y = (select_cpoints[..., 1] -
               img_grid[..., 1]) / (select_dist * c * c + 1e-5)

        phi_x = phi_r * r_x * select_mask
        phi_y = phi_r * r_y * select_mask

        self.register_buffer('select_index', select_dist_index)
        self.register_buffer('phi_0', phi_0)
        self.register_buffer('phi_x', phi_x)
        self.register_buffer('phi_y', phi_y)
        self.register_buffer('cpoints_0', select_cpoints)
        self.cpoint_num = np.prod(cpoint_size)

    def forward(self, cpoint_loc, alpha):
        batch_size = cpoint_loc.size()[0]
        base = torch.arange(batch_size,
                            device=cpoint_loc.device) * self.cpoint_num
        base = base.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        index = self.select_index.unsqueeze(0) + base
        select_cpoint_loc_x = torch.take(cpoint_loc[..., 0], index)
        select_cpoint_loc_y = torch.take(cpoint_loc[..., 1], index)
        select_alpha_x = torch.take(alpha[..., 0], index)
        select_alpha_y = torch.take(alpha[..., 1], index)
        select_alpha = torch.stack([select_alpha_x, select_alpha_y], 1)

        phi = self.phi_0 + (
            select_cpoint_loc_x - self.cpoints_0[..., 0]) * self.phi_x + (
                select_cpoint_loc_y - self.cpoints_0[..., 1]) * self.phi_y
        phi = phi.unsqueeze(1)

        flow = torch.sum(phi * select_alpha, dim=4)
        return flow


class RadialBasisArbitraryLayerTG(nn.Module):
    def __init__(self, cpoint_size):
        super(RadialBasisArbitraryLayerTG, self).__init__()
        self.cpoint_num = np.prod(cpoint_size)

    def forward(self, cpoint_loc, alpha, cpoints_0, select_index, phi_0, phi_x,
                phi_y):
        batch_size = cpoint_loc.size()[0]
        base = torch.arange(batch_size,
                            device=cpoint_loc.device) * self.cpoint_num
        base = base.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        index = select_index.unsqueeze(0) + base
        select_cpoint_loc_x = torch.take(cpoint_loc[..., 0], index)
        select_cpoint_loc_y = torch.take(cpoint_loc[..., 1], index)
        select_alpha_x = torch.take(alpha[..., 0], index)
        select_alpha_y = torch.take(alpha[..., 1], index)
        select_alpha = torch.stack([select_alpha_x, select_alpha_y], 1)

        phi = phi_0 + (select_cpoint_loc_x - cpoints_0[..., 0]) * phi_x + (
            select_cpoint_loc_y - cpoints_0[..., 1]) * phi_y
        phi = phi.unsqueeze(1)

        flow = torch.sum(phi * select_alpha, dim=4)
        return flow


class RadialBasisInitControlPoint(nn.Module):
    def __init__(self, i_size, c, cpoint_size):
        super(RadialBasisInitControlPoint, self).__init__()
        cpoint_yx = [
            torch.linspace(0, s - 1, c) for s, c in zip(i_size, cpoint_size)
        ]
        cpoint_grid = torch.flatten(torch.stack(torch.meshgrid(cpoint_yx),
                                                2)[..., [1, 0]],
                                    start_dim=0,
                                    end_dim=1)
        self.register_parameter('cpoint_grid', nn.Parameter(cpoint_grid))

        img_yx = [torch.linspace(0, s - 1, s) for s in i_size]
        img_grid = torch.stack(torch.meshgrid(img_yx), 2)[...,
                                                          [1, 0]].unsqueeze(2)
        self.register_buffer('img_grid', img_grid)
        self.c = c
        self.i_size = i_size

    def clipContorlPoints(self):
        for i in range(len(self.i_size)):
            self.cpoint_grid[..., i].data.clamp_(0, self.i_size[i] - 1)

    def forward(self):
        cpoint_grid = self.cpoint_grid.unsqueeze(0).unsqueeze(0)
        dist = torch.norm(self.img_grid - cpoint_grid, dim=3) / self.c
        sorted_dist, index = torch.sort(dist, 2)

        mask = dist < 1
        cpoint_max = mask.sum(2).max()

        select_dist = sorted_dist[..., :cpoint_max]
        select_dist_index = index[..., :cpoint_max]
        select_mask = select_dist < 1

        select_cpoints_x = torch.take(cpoint_grid[..., 0], select_dist_index)
        select_cpoints_y = torch.take(cpoint_grid[..., 1], select_dist_index)
        select_cpoints = torch.stack([select_cpoints_x, select_cpoints_y], 3)

        phi_0 = torch.pow(1 - select_dist,
                          4) * (4 * select_dist + 1) * select_mask

        phi_r = -4 * torch.pow(1 - select_dist, 3) * (
            4 * select_dist + 1) + 4 * torch.pow(1 - select_dist, 4)

        r_x = (select_cpoints[..., 0] -
               self.img_grid[..., 0]) / (select_dist * self.c * self.c + 1e-5)
        r_y = (select_cpoints[..., 1] -
               self.img_grid[..., 1]) / (select_dist * self.c * self.c + 1e-5)

        phi_x = phi_r * r_x * select_mask
        phi_y = phi_r * r_y * select_mask

        return select_cpoints, select_dist_index, phi_0, phi_x, phi_y
