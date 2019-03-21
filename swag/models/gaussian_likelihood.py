"""Gaussian likelihood model"""

import numpy as np
import torch.nn as nn
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import swag.utils as sw_utils


class GaussianLikelihood(nn.Module):
    """Gaussian likelihood model

    Quasi model with no response variable
    Finds parameters such that a given dataset is approximated
    by a multivariate gaussian.
    """

    def __init__(self, theta_0, cov_theta, cov_x, swag_settings, device=None):
        super(GaussianLikelihood, self).__init__()

        theta = torch.nn.Parameter(torch.zeros(theta_0.shape,
                                               dtype=torch.double,
                                               device=device))
        cov_x = torch.tensor(cov_x, dtype=torch.double,
                             device=device, requires_grad=False)
        cov_theta = torch.tensor(cov_theta, dtype=torch.double,
                                 device=device, requires_grad=False)
        theta_0 = torch.tensor(theta_0, dtype=torch.double,
                               device=device, requires_grad=False)
        self.device = device
        self.swag_settings = swag_settings

        self.params = nn.ParameterDict({"theta": theta})
        self.prior = MultivariateNormal(theta_0, cov_theta)
        self.likelihood = MultivariateNormal(theta, cov_x)
        self.posterior = Posterior(theta_0=theta_0, sigma_theta_0=cov_theta,
                                   sigma_x=cov_x)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=1e-1)
        if self.device:
            self.to(self.device)

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
        loss = torch.nn.L1Loss()
        return loss(neg_log_likelihood,
                    torch.zeros(1, dtype=torch.double,
                                device=self.device))

    def update_learning_rate(self, epoch):
        """Update learning rate"""
        initial_learning_rate = self.optimizer.defaults["lr"]
        new_learning_rate = sw_utils.schedule(epoch,
                                              initial_learning_rate,
                                              self.swag_settings.total_epochs,
                                              self.swag_settings.use_swag,
                                              self.swag_settings.swag_start_epoch,
                                              self.swag_settings.swag_lr)
        sw_utils.adjust_learning_rate(self.optimizer, new_learning_rate)

    def forward(self, sample):
        """Forward pass"""
        theta = self.params["theta"]
        prob = self.prior.log_prob(theta)
        prob += torch.sum(self.likelihood.log_prob(sample))
        return - prob

    def update_true_posterior(self, sample):
        """Calculate analytic posterior"""
        self.posterior.update(sample)


class Posterior:
    """True posterior"""

    def __init__(self, theta_0, sigma_theta_0, sigma_x):
        self.theta = theta_0
        # self.sigma_theta_inv = np.linalg.inv(sigma_theta_0)
        # self.sigma_x_inv = np.linalg.inv(sigma_x)

    def update(self, sample):
        self.theta = 0
