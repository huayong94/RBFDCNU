import torch
from torch import nn


class Flow2RBG(nn.Module):
    def __init__(self, max_value=10.0):
        super(Flow2RBG, self).__init__()
        self.max_value = float(max_value)

    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        flow_size = flow.size()
        batch = flow_size[0]
        height = flow_size[2]
        width = flow_size[3]

        x, y = flow[:, 0, :, :], flow[:, 1, :, :]
        x_zero, y_zero = x == 0, y == 0
        both_zero = x_zero * y_zero
        x_np = x + both_zero * self.max_value
        y_np = y + both_zero * self.max_value

        rgb_map = torch.ones((batch, 3, height, width), device=flow.device)

        normalized_x_np = x_np / self.max_value
        normalized_y_np = y_np / self.max_value

        r = normalized_x_np.unsqueeze(1)
        g = (-0.5 * (normalized_x_np + normalized_y_np) *
             (both_zero == 0)).unsqueeze(1)
        b = normalized_y_np.unsqueeze(1)

        rgb = rgb_map + torch.cat([r, g, b], 1)

        rgb = torch.clamp(rgb, 0, 1)

        return rgb
