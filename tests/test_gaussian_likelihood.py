"""Unittest: Gaussian Likelihood model"""
import unittest
import numpy as np
import torch

import swag.models.gaussian_likelihood as sw_gauss

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
        cov_1 =  sigma_sq_1 * torch.eye(dim)

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

    special_case_calc = (torch.sum(
        sigma_sq_1 + mu_1 * mu_1 - np.log(sigma_sq_1) - 1)) / 2
    return special_case_calc.item()


if __name__ == "__main__":
    unittest.main()
