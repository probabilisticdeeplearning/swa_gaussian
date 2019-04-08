"""Unittest: Swag for toy synthetic gaussian example"""
import unittest
import numpy as np
import torch

import swag.data as sw_data
import swag.models.gaussian_likelihood as sw_gauss
import swag.utils as sw_utils


class TestPosteriorCalculation(unittest.TestCase):
    def test_one_dim_single_sample(self):
        """One dim posterior
        Single sample
        """

        dim = 1
        theta_0 = 1 * np.ones(dim, dtype=np.double)
        cov_theta = np.eye(dim)
        cov_x = np.eye(dim)

        device = sw_utils.torch_settings()
        num_epochs = 50  # Not in use
        swag_settings = sw_utils.SwagSettings(use_swag=True,
                                              initial_learning_rate=0.1,
                                              swag_start_epoch=num_epochs - 25,
                                              swag_lr=0.01,
                                              total_epochs=num_epochs)

        model = sw_gauss.GaussianLikelihood(theta_0,
                                            cov_theta,
                                            cov_x,
                                            swag_settings=swag_settings,
                                            device=device)

        sample = torch.tensor([[2]], dtype=torch.double).to(device)
        model.update_true_posterior(sample)
        self.assertAlmostEqual(model.posterior.theta.item(), 3/2)
        self.assertAlmostEqual(model.posterior.sigma_theta.item(), 1/2)

    def test_one_dim_mult_sample(self):
        """One dim posterior
        Multiple samples
        """

        dim = 1
        theta_0 = 1 * np.ones(dim, dtype=np.double)
        cov_theta = np.eye(dim)
        cov_x = np.eye(dim)

        device = sw_utils.torch_settings()
        num_epochs = 50  # Not in use
        swag_settings = sw_utils.SwagSettings(use_swag=True,
                                              initial_learning_rate=0.1,
                                              swag_start_epoch=num_epochs - 25,
                                              swag_lr=0.01,
                                              total_epochs=num_epochs)

        model = sw_gauss.GaussianLikelihood(theta_0,
                                            cov_theta,
                                            cov_x,
                                            swag_settings=swag_settings,
                                            device=device)

        sample = torch.tensor([[1, 2, 3]], dtype=torch.double).to(device).t()
        model.update_true_posterior(sample)
        self.assertAlmostEqual(model.posterior.theta.item(), 7/4)
        self.assertAlmostEqual(model.posterior.sigma_theta.item(), 1/4)

    def test_two_dim_mult_sample(self):
        """two dim posterior
        Multiple samples
        """
        dim = 2
        theta_0 = torch.reshape(torch.tensor([1, 2]), (dim, 1))
        cov_theta = np.eye(dim)
        cov_x = np.eye(dim)

        device = sw_utils.torch_settings()
        num_epochs = 50  # Not in use
        swag_settings = sw_utils.SwagSettings(use_swag=True,
                                              initial_learning_rate=0.1,
                                              swag_start_epoch=num_epochs - 25,
                                              swag_lr=0.01,
                                              total_epochs=num_epochs)

        model = sw_gauss.GaussianLikelihood(theta_0,
                                            cov_theta,
                                            cov_x,
                                            swag_settings=swag_settings,
                                            device=device)

        sample = torch.tensor([[1, 2], [1, 2], [2, 3]],
                              dtype=torch.double).to(device)
        model.update_true_posterior(sample)
        self.assertTrue(torch.equal(model.posterior.theta,
                        torch.tensor([[5/4], [9/4]],
                                     device=device,
                                     dtype=torch.double)))
        self.assertTrue(torch.equal(model.posterior.sigma_theta,
                                    1/4 * torch.eye(dim,
                                                    dtype=torch.double,
                                                    device=device)))

    def test_2d_corr(self):
        """2D posterior
        Correlated variable
        """

        dim = 2
        theta_0 = torch.zeros(dim, 1)
        cov_theta = torch.tensor([[1, 1/2], [1/2, 1]])
        cov_x = cov_theta.clone()

        device = sw_utils.torch_settings()
        num_epochs = 50  # Not in use
        swag_settings = sw_utils.SwagSettings(use_swag=True,
                                              initial_learning_rate=0.1,
                                              swag_start_epoch=num_epochs - 25,
                                              swag_lr=0.01,
                                              total_epochs=num_epochs)

        model = sw_gauss.GaussianLikelihood(theta_0,
                                            cov_theta,
                                            cov_x,
                                            swag_settings=swag_settings,
                                            device=device)
        sample = torch.tensor([[0.1, 0.1], [-0.1, -0.1]],
                              dtype=torch.double).to(device)
        model.update_true_posterior(sample)
        true_theta = torch.zeros(dim, 1,
                                 device=device,
                                 dtype=torch.double)
        self.assertTrue(torch.equal(model.posterior.theta, true_theta))
        self.assertTrue(torch.equal(model.posterior.sigma_theta,
                                    torch.tensor([[1/3, 1/6], [1/6, 1/3]],
                                                    dtype=torch.double,
                                                    device=device)))


if __name__ == "__main__":
    unittest.main()
