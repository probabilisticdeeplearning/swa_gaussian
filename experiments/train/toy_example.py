"""Swag for toy example"""
import numpy as np
import torch

import swag.data as sw_data
import swag.models.gaussian_likelihood as sw_gauss
import swag.utils as sw_utils


def main():
    """Main entry point"""
    dim = 2
    batch_size = 5
    theta_0 = 0 * np.ones(dim, dtype=np.double)
    cov_theta = np.eye(dim)
    cov_x = np.eye(dim)
    dataset = sw_data.SyntheticGaussianData(theta_0=theta_0,
                                            cov_theta=cov_theta,
                                            cov_x=cov_x,
                                            n_samples=10)
    data_train_loader = torch.utils.data.DataLoader(dataset,
                                                    batch_size=batch_size,
                                                    shuffle=True)
    device = sw_utils.torch_settings()
    swag_settings = sw_utils.SwagSettings()
    model = sw_gauss.GaussianLikelihood(theta_0,
                                        cov_theta,
                                        cov_x,
                                        swag_settings=swag_settings,
                                        device=device)

    num_epochs = 100
    for epoch in range(num_epochs):
        print("Epoch: {}\t {}".format(epoch, model.status()))
        model.train_epoch(data_train_loader)
        model.update_learning_rate(epoch)


if __name__ == "__main__":
    main()
