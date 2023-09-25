import torch
import torch.nn as nn
import torch.nn.functional as F


class BilinearInterpolate(nn.Module):
    def __init__(self):
        super(BilinearInterpolate, self).__init__()

    def repeat(self, x, n_repeats, device):
        rep = torch.ones(n_repeats, device=device).unsqueeze(0)
        x = torch.matmul(x.view(-1, 1).float(), rep)
        return x.view(-1).long()

    def forward(self, Im, G):
        Im = F.pad(Im, (1, 1, 1, 1), mode='replicate')
        x = G[:, 0, :, :].contiguous()
        y = G[:, 1, :, :].contiguous()
        device = Im.device

        size_Im = Im.size()
        num_batch, channels = size_Im[0], size_Im[1]
        height, width = size_Im[2], size_Im[3]

        size_G = G.size()
        out_height, out_width = size_G[2], size_G[3]

        x, y = x.view(-1) + 1, y.view(-1) + 1
        max_x, max_y = width - 1, height - 1

        dim1, dim2 = height * width, width
        base = self.repeat(torch.arange(num_batch, device=device) * dim1,
                           out_height * out_width,
                           device=device)
        x0, y0 = torch.floor(x).long(), torch.floor(y).long()
        x1, y1 = x0 + 1, y0 + 1

        x0, x1 = torch.clamp(x0, 0, max_x), torch.clamp(x1, 0, max_x)
        y0, y1 = torch.clamp(y0, 0, max_y), torch.clamp(y1, 0, max_y)

        base_y0, base_y1 = base + y0 * dim2, base + y1 * dim2

        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        Im_flat = Im.permute(0, 2, 3, 1).contiguous().view(-1,
                                                           channels).float()
        Ia = torch.index_select(Im_flat, 0, idx_a)
        Ib = torch.index_select(Im_flat, 0, idx_b)
        Ic = torch.index_select(Im_flat, 0, idx_c)
        Id = torch.index_select(Im_flat, 0, idx_d)

        x1_f, y1_f = x1.float(), y1.float()
        dx, dy = x1_f - x, y1_f - y

        wa = (dy * dx).unsqueeze(1)
        wb = ((1 - dy) * dx).unsqueeze(1)
        wc = (dy * (1 - dx)).unsqueeze(1)
        wd = ((1 - dy) * (1 - dx)).unsqueeze(1)

        output = wa * Ia + wb * Ib + wc * Ic + wd * Id

        return output.view(-1, out_height, out_width,
                           channels).permute(0, 3, 1, 2)


class NearestInterpolate(nn.Module):
    def __init__(self):
        super(NearestInterpolate, self).__init__()

    def repeat(self, x, n_repeats, device):
        rep = torch.ones(n_repeats, device=device).unsqueeze(0)
        x = torch.matmul(x.view(-1, 1).float(), rep)
        return x.view(-1).long()

    def forward(self, Im, G):
        Im = F.pad(Im, (1, 1, 1, 1, 1, 1), mode='replicate')
        x = G[:, 0, :, :].contiguous()
        y = G[:, 1, :, :].contiguous()
        device = Im.device

        size_Im = Im.size()
        num_batch, channels = size_Im[0], size_Im[1]
        height, width = size_Im[2], size_Im[3]

        size_G = G.size()
        out_height, out_width = size_G[2], size_G[3]

        x, y = x.view(-1) + 1, y.view(-1) + 1
        max_x, max_y = width - 1, height - 1

        dim1, dim2 = height * width, width
        base = self.repeat(torch.arange(num_batch, device=device) * dim1,
                           out_height * out_width,
                           device=device)
        x0, y0 = torch.floor(x + 0.5).long(), torch.floor(y + 0.5).long()
        x0 = torch.clamp(x0, 0, max_x)
        y0 = torch.clamp(y0, 0, max_y)

        idx = base + y0 * dim2 + x0

        Im_flat = Im.permute(0, 2, 3, 1).contiguous().view(-1,
                                                           channels).float()
        output = torch.index_select(Im_flat, 0, idx)

        return output.view(-1, out_height, out_width,
                           channels).permute(0, 3, 1, 2)
