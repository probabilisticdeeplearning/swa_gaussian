"""Unittest: Gaussian Likelihood model"""
import unittest
import numpy as np
import numpy.linalg as np_la
import torch
import math

import swag.models.gaussian_likelihood as sw_gauss
import swag.utils as sw_utils


class TestObjectiveFunction(unittest.TestCase):
    """Objective function"""

    def test_one_dim_single_sample(self):
        """One dim loss function
        Single sample
        """
        dim = 1
        theta_0 = 0 * np.ones(dim, dtype=np.double)
        cov_theta = 1 * np.eye(dim)
        cov_x = 1 * np.eye(dim)

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

        sample = torch.tensor([0], dtype=torch.double).to(device)
        sample_size = 1
        true_loss = (
            (sample_size + 1) * np.log(2 * math.pi) +
            np.log(np_la.det(cov_x)) + np.log(np_la.det(cov_theta))) / 2
        loss = model.forward(sample)
        self.assertAlmostEqual(loss.item(), true_loss)

    def test_one_dim_mult_sample(self):
        """One dim loss function
        Multiple samples
        """
        dim = 1
        theta_0 = 0 * np.ones(dim, dtype=np.double)
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

        sample = torch.tensor([[0], [0], [0]], dtype=torch.double).to(device)
        sample_size = 3
        true_loss = ((sample_size + 1) * dim * np.log(2 * math.pi) +
                     sample_size * np.log(np_la.det(cov_x)) +
                     np.log(np_la.det(cov_theta))) / 2
        loss = model.forward(sample)
        self.assertAlmostEqual(loss.item(), true_loss)

    def test_two_dim_single_sample(self):
        """Two dim loss function
        Single sample
        """
        dim = 2
        theta_0 = 0 * np.ones(dim, dtype=np.double)
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

        sample = torch.tensor([0, 0], dtype=torch.double).to(device)
        sample_size = 1
        true_loss = ((sample_size + 1) * dim * np.log(2 * math.pi) +
                     sample_size * np.log(np_la.det(cov_x)) +
                     np.log(np_la.det(cov_theta))) / 2
        loss = model.forward(sample)
        self.assertAlmostEqual(loss.item(), true_loss)

    def test_two_dim(self):
        """Two dim loss function
        Single sample
        """
        dim = 2
        theta_0 = 0 * np.ones(dim, dtype=np.double)
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

        sample = torch.tensor([[0, 0], [0, 0], [0, 0]],
                              dtype=torch.double).to(device)
        sample_size = sample.size()[0]
        true_loss = ((sample_size + 1) * dim * np.log(2 * math.pi) +
                     sample_size * np.log(np_la.det(cov_x)) +
                     np.log(np_la.det(cov_theta))) / 2
        loss = model.forward(sample)
        self.assertAlmostEqual(loss.item(), true_loss)


class TestPosteriorCalculation(unittest.TestCase):
    """Analytic posterior calculations"""

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
        self.assertAlmostEqual(model.posterior.theta.item(), 3 / 2)
        self.assertAlmostEqual(model.posterior.sigma_theta.item(), 1 / 2)

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
        self.assertAlmostEqual(model.posterior.theta.item(), 7 / 4)
        self.assertAlmostEqual(model.posterior.sigma_theta.item(), 1 / 4)

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
        self.assertTrue(
            torch.equal(
                model.posterior.theta,
                torch.tensor([[5 / 4], [9 / 4]],
                             device=device,
                             dtype=torch.double)))
        self.assertTrue(
            torch.equal(
                model.posterior.sigma_theta,
                1 / 4 * torch.eye(dim, dtype=torch.double, device=device)))

    def test_2d_corr(self):
        """2D posterior
        Correlated variable
        """

        dim = 2
        theta_0 = torch.zeros(dim, 1)
        cov_theta = torch.tensor([[1, 1 / 2], [1 / 2, 1]])
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
        true_theta = torch.zeros(dim, 1, device=device, dtype=torch.double)
        self.assertTrue(torch.equal(model.posterior.theta, true_theta))
        self.assertTrue(
            torch.equal(
                model.posterior.sigma_theta,
                torch.tensor([[1 / 3, 1 / 6], [1 / 6, 1 / 3]],
                             dtype=torch.double,
                             device=device)))


class TestKLDiv(unittest.TestCase):
    """Test KL Divergence between two gaussians"""

    def test_one_dim(self):
        """One dimensional KL"""
        dim = 1
        mu_1 = 1 * torch.ones(dim, 1)
        cov_1 = torch.eye(dim)

        mu_2 = mu_1.clone()
        cov_2 = cov_1.clone()

        self.assertEqual(sw_gauss.kl_div_gaussian(mu_1, cov_1, mu_2, cov_2), 0)

    def test_diag_std_norm_1d(self):
        """Test special case of KL
        1-dimensional
        Random mu and sigma_sq
        """

        dim = 1
        mu_1 = 1 * torch.ones(dim, 1)
        sigma_sq_1 = np.random.rand()
        cov_1 = sigma_sq_1 * torch.eye(dim)

        mu_2 = torch.zeros(dim, 1)
        cov_2 = torch.eye(dim)

        special_case_calc = calc_kl_special_case(mu_1, sigma_sq_1)
        general_calc = sw_gauss.kl_div_gaussian(mu_1, cov_1, mu_2, cov_2)
        self.assertAlmostEqual(general_calc, special_case_calc, places=6)

    def test_diag_std_norm_mvn_det(self):
        """Test special case of KL
        Multi dim
        Deterministic dim, mu and sigma_sq
        """

        dim = 10
        mu_1 = torch.ones(dim, 1)
        sigma_sq_1 = torch.ones(dim, 1)
        cov_1 = torch.diag(sigma_sq_1[:, 0])

        mu_2 = torch.zeros(dim, 1)
        cov_2 = torch.eye(dim)

        special_case_calc = calc_kl_special_case(mu_1, sigma_sq_1)
        general_calc = sw_gauss.kl_div_gaussian(mu_1, cov_1, mu_2, cov_2)
        self.assertAlmostEqual(general_calc, special_case_calc, places=6)

    def test_diag_std_norm_mvn_rand(self):
        """Test special case of KL
        Random dim, mu and sigma_sq
        """

        dim = np.random.randint(1, 10)
        mu_1 = torch.rand(dim, 1)
        sigma_sq_1 = torch.rand(dim, 1)
        cov_1 = torch.diag(sigma_sq_1[:, 0])

        mu_2 = torch.zeros(dim, 1)
        cov_2 = torch.eye(dim)

        mu_2 = torch.zeros(dim, 1)
        cov_2 = torch.eye(dim)

        special_case_calc = calc_kl_special_case(mu_1, sigma_sq_1)
        general_calc = sw_gauss.kl_div_gaussian(mu_1, cov_1, mu_2, cov_2)
        self.assertAlmostEqual(general_calc, special_case_calc, places=6)


def calc_kl_special_case(mu_1, sigma_sq_1):
    """Calc special case of KL between diagonal gaussion
    and a standard normal:
        KL(N(mu., diag(sigma^2)) || N(0, I)) =
        0.5 sum_k( sigma_k^2 + mu_k - ln(sigma_k^2) -1)
    """

    special_case_calc = (torch.sum(sigma_sq_1 + mu_1 * mu_1 -
                                   np.log(sigma_sq_1) - 1)) / 2
    return special_case_calc.item()


if __name__ == "__main__":
    unittest.main()
