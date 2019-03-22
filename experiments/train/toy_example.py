"""Swag for toy example"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib import cm
import torch

import swag.data as sw_data
import swag.models.gaussian_likelihood as sw_gauss
import swag.utils as sw_utils


def plot_bivariate(sample, true_post):

    mean = true_post.theta.data.cpu().numpy()
    cov = true_post.sigma_theta.data.cpu().numpy()
    x_vals = sample[:, 0]
    y_vals = sample[:, 1]
    plt.plot(x_vals, y_vals, "bo")
    plot_cov_ellipse(np.cov(sample.T), mean)

    plt.show()


def plot_cov_ellipse(cov, pos, nstd=1, ax=plt.gca(), **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellipse = patches.Ellipse(xy=pos, width=width, height=height,
                              angle=theta, **kwargs)

    ax.add_artist(ellipse)
    return ellipse


def main():
    """Main entry point"""
    dim = 2
    batch_size = 5
    num_epochs = 30
    theta_0 = 30 * np.ones(dim, dtype=np.double)
    cov_theta = np.eye(dim)
    cov_x = np.eye(dim)
    dataset = sw_data.SyntheticGaussianData(theta_0=theta_0,
                                            cov_theta=cov_theta,
                                            cov_x=cov_x,
                                            n_samples=100)
    data_train_loader = torch.utils.data.DataLoader(dataset,
                                                    batch_size=batch_size,
                                                    shuffle=True)
    device = sw_utils.torch_settings()
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

    for epoch in range(num_epochs):
        print("Epoch: {}\t {}".format(epoch, model.status()))
        model.train_epoch(data_train_loader, swag_settings.should_store(epoch))
        model.update_learning_rate(epoch)
    model.store_swag_to_numpy()

    print(np.cov(model.theta_store.T))
    print(model.posterior.sigma_theta.data.cpu().numpy())
    print("Post mean", model.posterior.theta.data.cpu().numpy())
    print("Sample mean", np.mean(model.theta_store))
    plot_bivariate(model.theta_store, model.posterior)


if __name__ == "__main__":
    main()
