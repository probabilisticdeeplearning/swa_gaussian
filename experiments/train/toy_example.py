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

# from swag import data, models, utils, losses
from swag import data
from swag import models
from swag.posteriors import SWAG


def main():
    """Main entry point"""
    mean = np.array([2, 2])
    cov = np.diag([1, 1])
    dataset = data.SyntheticGaussianData(mean=mean, cov=cov, n_samples=1000)
    data_train_loader = torch.utils.data.DataLoader(dataset,
                                                    batch_size=5,
                                                    shuffle=True)
    model = models.gaussian_likelihood.GaussianLikelihood(dim=2)

    num_epochs = 100
    for epoch in range(num_epochs):
        print("Epoch: {}\t {}".format(epoch, model.status()))
        model.train_epoch(data_train_loader)
        model.update_learning_rate()


if __name__ == "__main__":
    main()
