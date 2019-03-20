"""Swag for toy example"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import tabulate
import torch
import torch.nn.functional as F
import torchvision

import swag.data as sw_data
import swag.models as sw_models
import swag.utils as sw_utils
from swag.posteriors import SWAG


def main():
    """Main entry point"""
    mean = np.array([2, 2])
    cov = np.diag([1, 1])
    dataset = sw_data.SyntheticGaussianData(mean=mean, cov=cov, n_samples=1000)
    data_train_loader = torch.utils.data.DataLoader(dataset,
                                                    batch_size=5,
                                                    shuffle=True)
    device = sw_utils.torch_settings()
    model = sw_models.gaussian_likelihood.GaussianLikelihood(dim=2,
                                                             device=device)
    # model.set_dist_device(device)

    num_epochs = 100
    for epoch in range(num_epochs):
        print("Epoch: {}\t {}".format(epoch, model.status()))
        model.train_epoch(data_train_loader)
        model.update_learning_rate()


if __name__ == "__main__":
    main()
