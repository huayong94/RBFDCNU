import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class LocalCrossCorrelation2D(nn.Module):
    def __init__(self, win=[9, 9]):
        """Initialize the Local Cross Correlation (LCC) model for 2D images

        Args:
            win (list, optional): the size of the local windows. Defaults to [9, 9].
        """
        super(LocalCrossCorrelation2D, self).__init__()
        self.win = win

    def set(self, win):
        self.win = win

    def forward(self, I: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
        """Push two images I and J through LCC2D block

        Args:
            I (torch.Tensor): A batch of 2D images with the shape of [BxCxHxW]
            J (torch.Tensor): Another batch of 2D images with the shape of [BxCxHxW]

        Returns:
            torch.Tensor: The results of LCC with the shape of [Bx1]
        """
        I2 = I * I
        J2 = J * J
        IJ = I * J

        sum_filter = torch.ones([1, 1, self.win[0], self.win[1]],
                                device=I.device)

        I_sum = F.conv2d(I, sum_filter, padding=self.win[0] // 2)
        J_sum = F.conv2d(J, sum_filter, padding=self.win[0] // 2)
        I2_sum = F.conv2d(I2, sum_filter, padding=self.win[0] // 2)
        J2_sum = F.conv2d(J2, sum_filter, padding=self.win[0] // 2)
        IJ_sum = F.conv2d(IJ, sum_filter, padding=self.win[0] // 2)

        win_size = self.win[0] * self.win[1]

        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        # Here we filter the zero-value background to avoid NaN
        non_zero = I_var * J_var > np.power(np.e, -15)
        zero = I_var * J_var <= np.power(np.e, -15)
        cross = non_zero * cross + zero
        I_var = non_zero * I_var + zero
        J_var = non_zero * J_var + zero

        cc = cross * cross / (I_var * J_var + np.power(np.e, -15))

        return -1.0 * torch.mean(cc, dim=[1, 2, 3]) + 1


class WeightedLocalCrossCorrelation2D(nn.Module):
    def __init__(self, alpha=0.02, win=[9, 9]):
        """Initialize the WeightedL Local Cross Correlation (WLCC) model for 2D images

        Args:
            alpha (float, optional): The factor of the WLCC. Defaults to 0.02.
            win (list, optional): the size of the local windows. Defaults to [9, 9].
        """
        super(WeightedLocalCrossCorrelation2D, self).__init__()
        self.win = win
        self.normal = Normal(0, alpha, validate_args=None)

    def set(self, alpha, win):
        self.win = win
        self.normal = Normal(0, alpha, validate_args=None)

    def forward(self, I: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
        """Push two images I and J through WLCC2D block

        Args:
            I (torch.Tensor): A batch of 2D images with the shape of [BxCxHxW]
            J (torch.Tensor): Another batch of 2D images with the shape of [BxCxHxW]

        Returns:
            torch.Tensor: The results of LCC with the shape of [Bx1]
        """
        I2 = I * I
        J2 = J * J
        IJ = I * J

        sum_filter = torch.ones([1, 1, self.win[0], self.win[1]],
                                device=I.device)

        I_sum = F.conv2d(I, sum_filter, padding=self.win[0] // 2)
        J_sum = F.conv2d(J, sum_filter, padding=self.win[0] // 2)
        I2_sum = F.conv2d(I2, sum_filter, padding=self.win[0] // 2)
        J2_sum = F.conv2d(J2, sum_filter, padding=self.win[0] // 2)
        IJ_sum = F.conv2d(IJ, sum_filter, padding=self.win[0] // 2)

        win_size = self.win[0] * self.win[1]

        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        # Here we filter the zero-value background to avoid NaN
        non_zero = I_var * J_var > np.power(np.e, -15)
        zero = I_var * J_var <= np.power(np.e, -15)
        cross = non_zero * cross + zero
        I_var = non_zero * I_var + zero
        J_var = non_zero * J_var + zero

        cc = cross * cross / (I_var * J_var + np.power(np.e, -15))

        
        # calculating weight according the intensity difference
        
        P = self.normal.log_prob(torch.abs(I - J)).exp()
        weight = P / self.normal.log_prob(torch.tensor(0).cuda()).exp()

        dccp = weight + cc * (1 - weight)

        return -1.0 * torch.mean(dccp, dim=[1, 2, 3]) + 1
