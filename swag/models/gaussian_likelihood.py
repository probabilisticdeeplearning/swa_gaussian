"""Gaussian likelihood model"""

import numpy as np
import torch.nn as nn
import torch
import torch.distributions.multivariate_normal as torch_mvn
import swag.utils as sw_utils
import decimal


class GaussianLikelihood(nn.Module):
    """Gaussian likelihood model

    Quasi model with no response variable
    Finds parameters such that a given dataset is approximated
    by a multivariate gaussian.
    """

    def __init__(self, theta_0, cov_theta, cov_x, swag_settings, device=None):
        super(GaussianLikelihood, self).__init__()

        theta = torch.nn.Parameter(
            torch.zeros(theta_0.shape, dtype=torch.double, device=device))
        cov_x = torch.tensor(cov_x,
                             dtype=torch.double,
                             device=device,
                             requires_grad=False)
        cov_theta = torch.tensor(cov_theta,
                                 dtype=torch.double,
                                 device=device,
                                 requires_grad=False)
        theta_0 = torch.tensor(theta_0,
                               dtype=torch.double,
                               device=device,
                               requires_grad=False)
        self.device = device
        self.swag_settings = swag_settings
        self.theta_store = list()

        self.params = nn.ParameterDict({"theta": theta})
        self.prior = torch_mvn.MultivariateNormal(loc=theta_0,
                                                  covariance_matrix=cov_theta)
        self.likelihood = torch_mvn.MultivariateNormal(loc=theta,
                                                       covariance_matrix=cov_x)
        self.posterior = Posterior(theta_0=theta_0,
                                   sigma_theta_0=cov_theta,
                                   sigma_x=cov_x,
                                   device=self.device)

        self.optimizer = torch.optim.SGD(self.parameters(), lr=1e-1)
        if self.device:
            self.to(self.device)

    def status(self, show_posterior=False, include_learning_rate=False):
        """Format parameter values into a tab separated string"""
        status = str()
        for key, param in self.params.items():
            val = param.data.cpu().numpy()
            status += "{}: {}\t".format(key, val)
        if show_posterior:
            status += "Posterior: {}\t".format(
                self.posterior.theta.data.cpu().numpy())
        if include_learning_rate:
            status += "Learning rate: {}\t".format(
                self.optimizer.param_groups[0]["lr"])
        return status

    def train_epoch(self, data_train_loader, store_swag=False):
        """Train epoch"""
        for sample in data_train_loader:
            sample = sample.to(self.device)
            loss = self.neg_log_likelihood(sample)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # self.posterior.update(sample)
            if store_swag:
                self.store_swag()

    def store_swag(self):
        self.theta_store.append(self.params["theta"].clone())

    def store_swag_to_numpy(self):
        self.theta_store = torch.stack(self.theta_store).data.cpu().numpy()

    def neg_log_likelihood(self, sample):
        """Custom loss

        Calc neg log lik by a forward pass
        Measure L1 norm from zero (conforming to pytorch)
        There is probably a better way.
        """

        neg_log_likelihood = self(sample)
        loss = torch.nn.L1Loss()
        return loss(neg_log_likelihood,
                    torch.zeros(1, dtype=torch.double, device=self.device))

    def update_learning_rate(self, epoch):
        """Update learning rate"""
        initial_learning_rate = self.optimizer.defaults["lr"]
        new_learning_rate = sw_utils.schedule(
            epoch, initial_learning_rate, self.swag_settings.total_epochs,
            self.swag_settings.use_swag, self.swag_settings.swag_start_epoch,
            self.swag_settings.swag_lr)
        sw_utils.adjust_learning_rate(self.optimizer, new_learning_rate)

    def forward(self, sample):
        """Forward pass

        Args:
          sample (torch.tensor): Size: (#samples, dimension)
        """
        theta = self.params["theta"]
        prob = self.prior.log_prob(theta)
        prob += torch.sum(self.likelihood.log_prob(sample))
        return -prob

    def update_true_posterior(self, sample):
        """Calculate analytic posterior"""
        self.posterior.update(sample)


class Posterior:
    """True posterior"""

    def __init__(self, theta_0, sigma_theta_0, sigma_x, device=None):
        self.device = device
        self.theta = theta_0.clone().detach()
        self.sigma_theta = sigma_theta_0.clone().detach()
        self.sigma_x_inv = torch.inverse(sigma_x).clone().detach()

    def __repr__(self):
        str_ = "Post mean:\n"
        for theta_coord in self.theta.data.cpu().numpy():
            str_ += "\t{}\n".format(theta_coord)

        str_ += "\n"
        str_ += "Post covariance:\n"
        for cov_part in self.sigma_theta.data.cpu().numpy():
            str_ += "\t{}\n".format(cov_part)

        return str_

    def update(self, sample):
        sample = sample.to(self.device)
        dim = self.theta.size()[0]
        self.theta = torch.reshape(self.theta, (dim, 1))
        if sample.nelement() == 1:
            batch_size = 1
        else:
            batch_size = sample.size()[0]
        old_sigma_theta_inv = torch.inverse(self.sigma_theta)
        new_sigma_theta = torch.inverse(old_sigma_theta_inv +
                                        batch_size * self.sigma_x_inv)

        sample_sum = torch.reshape(torch.sum(sample, 0), (dim, 1))
        mean_shift = torch.matmul(old_sigma_theta_inv, self.theta)\
            + torch.matmul(self.sigma_x_inv, sample_sum)

        self.theta = torch.matmul(new_sigma_theta, mean_shift)
        self.sigma_theta = new_sigma_theta


def kl_div_gaussian(mu_1, Sigma_1, mu_2, Sigma_2):
    """Calculates the KL div between two arb. gaussian distributions
    Represented by mu_1,2 as n x 1 torch tensors and
    Sigma_1,2 as n x n torch tensors
    """

    Sigma_2_inv = torch.inverse(Sigma_2)
    trace_term = torch.trace(torch.matmul(Sigma_2_inv, Sigma_1))

    mean_diff = mu_2 - mu_1
    quadratic_term = torch.matmul(mean_diff.t(),
                                  torch.matmul(Sigma_2_inv, mean_diff))

    determinant_term = np.log(np.linalg.det(Sigma_2) / np.linalg.det(Sigma_1))
    kl_div = (trace_term + quadratic_term - len(mu_1) + determinant_term) / 2
    return kl_div.item()
