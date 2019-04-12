"""Swag for toy example"""
import numpy as np
import numpy.linalg as np_la
from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib import cm
import matplotlib2tikz
import torch
import scipy.stats as sp_stats

import swag.data as sw_data
import swag.models.gaussian_likelihood as sw_gauss
import swag.utils as sw_utils


def plot_univariate(sample, true_post, to_tikz=False, num_points=50, bins=50):
    mean = true_post.theta.item()
    sigma = np.sqrt(true_post.sigma_theta.item())
    points = np.linspace(mean - 2 * sigma, mean + 2 * sigma, num_points)
    pdf = sp_stats.norm.pdf(points, loc=mean, scale=sigma)
    plt.hist(sample, density=True, bins=bins)
    plt.plot(points, pdf)
    if to_tikz:
        matplotlib2tikz.save("test.tex")
    plt.show()


def plot_quadratic_form(mean, cov, num_points=100):
    x_points = np.linspace(mean[0, 0] - 2 * cov[0, 0],
                           mean[0, 0] + 2 * cov[0, 0], num_points)
    y_points = np.linspace(mean[1, 0] - 2 * cov[1, 1],
                           mean[1, 0] + 2 * cov[1, 1], num_points)
    z_values = np.zeros((len(x_points), len(y_points)))
    for x_ind, x_coord in enumerate(x_points):
        for y_ind, y_coord in enumerate(y_points):
            diff = np.array([[x_coord], [y_coord]]) - mean
            tmp_val = np.matmul(np.matmul(diff.T, np_la.inv(cov)), diff)[0][0]
            z_values[x_ind][y_ind] = tmp_val
    plt.contour(x_points, y_points, z_values, 3)


def plot_bivariate(sample, true_post, to_tikz=False, show=True):
    mean = true_post.theta.data.cpu().numpy()
    cov = true_post.sigma_theta.data.cpu().numpy()
    x_vals = sample[:, 0]
    y_vals = sample[:, 1]
    plt.plot(x_vals, y_vals, "bo")
    plot_cov_ellipse(mean, cov)
    plot_quadratic_form(mean, cov)
    if to_tikz:
        matplotlib2tikz.save("test.tex")
    plt.show()


def plot_cov_ellipse(pos, cov, nstd=1, ax=plt.gca(), **kwargs):
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
    ellipse = patches.Ellipse(xy=pos, width=width, height=height, angle=theta)

    ax.add_artist(ellipse)
    return ellipse


def plot_dataset(data, posterior):
    plot_bivariate(data, posterior)


def main():
    """Main entry point"""
    dim = 2
    batch_size = 5
    num_epochs = 30
    theta_0 = 1 * np.array([1, 0], dtype=np.double).T
    cov_theta = np.array([[1, 0.5], [0.5, 1]])

    #theta_0 = np.zeros(1)
    #cov_theta = 0.25 * np.eye(dim)

    cov_x = cov_theta
    dataset_file = "data/gaussian/{}dim.csv".format(dim)
    dataset = sw_data.SyntheticGaussianData(theta_0=theta_0,
                                            cov_theta=cov_theta,
                                            cov_x=cov_x,
                                            store_file=dataset_file,
                                            n_samples=100)
    data_train_loader = torch.utils.data.DataLoader(dataset,
                                                    batch_size=batch_size,
                                                    shuffle=True)
    device = sw_utils.torch_settings()
    swag_settings = sw_utils.SwagSettings(use_swag=True,
                                          initial_learning_rate=0.1,
                                          swag_start_epoch=5,
                                          swag_lr=0.01,
                                          total_epochs=num_epochs)

    model = sw_gauss.GaussianLikelihood(theta_0,
                                        cov_theta,
                                        cov_x,
                                        swag_settings=swag_settings,
                                        device=device)

    model.posterior.update(
        torch.tensor(dataset.get_full_data(),
                     dtype=torch.double,
                     requires_grad=False))

    for epoch in range(num_epochs):
        print("Epoch: {}\t {}".format(epoch, model.status()))
        model.train_epoch(data_train_loader, swag_settings.should_store(epoch))
        model.update_learning_rate(epoch)
    model.store_swag_to_numpy()
    print(model.posterior)

    plot = True
    if dim == 2 and plot:
        plot_bivariate(model.theta_store, model.posterior, to_tikz=False)
    elif dim == 1 and plot:
        plot_univariate(model.theta_store, model.posterior, to_tikz=True)


if __name__ == "__main__":
    # mean = 1 * np.array([[1, 0]], dtype=np.double).T
    # cov = np.array([[1, 0.5], [0.5, 1]])
    # plot_quadratic_form(mean, cov)
    # plt.show()
    main()
