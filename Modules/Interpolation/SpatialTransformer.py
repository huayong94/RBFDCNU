import torch
import torch.nn as nn
import torch.nn.functional as nnf


class SpatialTransformer(nn.Module):
    """
    [SpatialTransformer] represesents a spatial transformation block
    that uses the output from the UNet to preform an grid_sample
    https://pytorch.org/docs/stable/nn.functional.html#grid-sample
    """
    def __init__(self, size, need_grid=True):
        """
        Instiatiate the block
            :param size: size of input to the spatial transformer block
            :param mode: method of interpolation for grid_sampler
            :param need_grid: to determine whether the transformer create the sampling grid
        """
        super(SpatialTransformer, self).__init__()

        # Create sampling grid
        if need_grid:
            vectors = [torch.arange(0, s) for s in size]
            grids = torch.meshgrid(vectors)
            grid = torch.stack(grids)[[1, 0] if len(size) == 2 else
                                      [1, 0, 2]]  # y, x, z ==> x, y, z
            grid = torch.unsqueeze(grid, 0)  # add batch
            grid = grid.type(torch.FloatTensor)
            self.register_buffer('grid', grid)

        self.need_grid = need_grid

    def forward(self, src, flow, mode='bilinear', align_corners=True):
        """
        Push the src and flow through the spatial transform block
            :param src: the original moving image
            :param flow: the output from the U-Net
        """
        if self.need_grid:
            new_locs = self.grid + flow
        else:
            new_locs = flow * 1.0

        shape = flow.shape[2:]
        if len(shape) == 2:
            shape = [shape[1], shape[0]]
        elif len(shape) == 3:
            shape = [shape[1], shape[0], shape[2]]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i,
                     ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)

        return nnf.grid_sample(src,
                               new_locs,
                               mode=mode,
                               align_corners=align_corners)
