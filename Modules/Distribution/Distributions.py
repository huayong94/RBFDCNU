import numpy as np
import torch
import torch.nn as nn


class Gaussian(nn.Module):
    def __init__(self, in_feature_dim, z_dim):
        super(Gaussian, self).__init__()

        self.mu = nn.Linear(in_feature_dim, z_dim)
        self.log_var = nn.Linear(in_feature_dim, z_dim)
        # torch.nn.init.normal_(self.log_var.weight, mean=0.0, std=1e-10)
        # torch.nn.init.constant_(self.log_var.bias, -10)

    def forward(self, x):
        mu = self.mu(x)
        log_var = self.log_var(x)

        return mu, log_var

    def sample_z(self, mu, log_var):
        eps = torch.randn(mu.size(), device=mu.device)

        std = torch.exp(0.5 * log_var)

        z = mu + std * eps

        return z

    def kld(self, q_mu, q_log_var, p_mu=0, p_log_var=0, dim=1):
        return torch.sum(
            -0.5 + (torch.exp(q_log_var) + torch.pow(p_mu - q_mu, 2)) /
            (2 * np.exp(p_log_var)) + 0.5 * (p_log_var - q_log_var),
            dim=dim)

    def bekld(self, q_mu, q_log_var, be, dim=1):
        return torch.sum(torch.exp(q_log_var) - q_log_var,
                         dim=dim) / 2 + be  # be has been divided by 2


class Gaussian2D(Gaussian):
    def __init__(self,
                 in_feature_dim,
                 out_feature_dim,
                 kernel_size,
                 stride=1,
                 padding=1):
        super(Gaussian2D, self).__init__(1, 1)
        self.mu = nn.Conv2d(in_feature_dim,
                            out_feature_dim,
                            kernel_size,
                            stride=stride,
                            padding=padding)
        self.log_var = nn.Conv2d(in_feature_dim,
                                 out_feature_dim,
                                 kernel_size,
                                 stride=stride,
                                 padding=padding)
        # torch.nn.init.normal_(self.log_var.weight, mean=0.0, std=1e-10)
        # torch.nn.init.constant_(self.log_var.bias, -10)

    def kld(self, q_mu, q_log_var, p_mu=0, p_log_var=0):
        return super(Gaussian2D, self).kld(q_mu,
                                           q_log_var,
                                           p_mu,
                                           p_log_var,
                                           dim=[1, 2, 3])

    def bekld(self, q_mu, q_log_var, be):
        return super(Gaussian2D, self).bekld(q_mu,
                                             q_log_var,
                                             be,
                                             dim=[1, 2, 3])
