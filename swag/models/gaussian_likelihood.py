"""Gaussian likelihood model"""

import numpy as np
import torch.nn as nn
import torch
from torch.distributions.multivariate_normal import MultivariateNormal


class GaussianLikelihood(nn.Module):

    def __init__(self, dim):
        super(GaussianLikelihood, self).__init__()
        self.mean = torch.nn.Parameter(torch.zeros(dim, dtype=torch.double),
                                       requires_grad=True)
        self.cov = torch.tensor(np.eye(dim))
        self.dist = MultivariateNormal(self.mean, self.cov)


    def forward(self, x):
        return - self.dist.log_prob(x)

    def true_posterior(self, cov_data):
        raise NotImplementedError
