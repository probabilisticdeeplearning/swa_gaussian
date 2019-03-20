"""Gaussian likelihood model"""

import numpy as np
import torch.nn as nn
import torch
from torch.distributions.multivariate_normal import MultivariateNormal


class GaussianLikelihood(nn.Module):
    """Gaussian likelihood model

    Quasi model with no response variable
    Finds parameters such that a given dataset is approximated
    by a multivariate gaussian.
    """

    def __init__(self, dim, device=None):
        super(GaussianLikelihood, self).__init__()

        self.mean = torch.nn.Parameter(
            torch.zeros(dim, dtype=torch.double, device=device))
        self.cov = torch.tensor(np.eye(dim), dtype=torch.double, device=device)
        self.device = device

        self.params = nn.ParameterDict({"mean": self.mean})
        print(self.params)
        self.dist = MultivariateNormal(self.mean, self.cov)
        self.criterion = torch.nn.L1Loss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=1e-1)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=20,
                                                         gamma=0.2)
        if self.device:
            self.to(self.device)

    def set_dist_device(self, device):
        """Need this fix to move the distributions to GPU"""
        self.dist.loc = self.dist.loc.to(device)
        self.dist.covariance_matrix = self.dist.covariance_matrix.to(device)

    def status(self, include_learning_rate=True):
        """Format parameter values into a tab separated string"""
        status = str()
        for key, param in self.params.items():
            val = param.data.cpu().numpy()
            status += "{}: {}\t".format(key, val)
        if include_learning_rate:
            status += "Learning rate: {}\t" .format(
                self.optimizer.param_groups[0]["lr"])
        return status

    def train_epoch(self, data_train_loader):
        """Train epoch"""
        for sample in data_train_loader:
            sample = sample.to(self.device)
            loss = self.neg_log_likelihood(sample)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def neg_log_likelihood(self, sample):
        """Custom loss

        Calc neg log lik by a forward pass
        Measure L1 norm from zero (conforming to pytorch)
        There is probably a better way.
        """

        neg_log_likelihood = self(sample)
        return self.criterion(neg_log_likelihood,
                              torch.zeros(1, dtype=torch.double,
                                          device=self.device))

    def update_learning_rate(self, epoch=None):
        """Update learning rate"""
        self.scheduler.step(epoch)

    def forward(self, sample):
        """Forward pass"""
        return - self.dist.log_prob(sample)

    def true_posterior(self, cov_data):
        """Calculate analytic posterior"""
        raise NotImplementedError
