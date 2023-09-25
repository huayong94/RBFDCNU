import torch


def meshgrid2D(height, width):
    x_t = torch.matmul(torch.ones(height, 1),
                       torch.linspace(0.0, width - 1, width).unsqueeze(0))
    y_t = torch.matmul(
        torch.linspace(0.0, height - 1, height).unsqueeze(1),
        torch.ones(1, width))
    x_t, y_t = x_t.unsqueeze(0), y_t.unsqueeze(0)
    grid = torch.cat([x_t, y_t], 0)

    return grid
