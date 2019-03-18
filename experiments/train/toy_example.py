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

from swag import data, models, utils, losses
from swag.posteriors import SWAG


def main():
    """Main entry point"""
    mean = np.array([2, 2])
    cov = np.diag([1, 1])
    dataset = data.SyntheticGaussianData(mean=mean, cov=cov, n_samples=100)
    data_train_loader = torch.utils.data.DataLoader(dataset,
                                                    batch_size=1,
                                                    shuffle=True)
    model = models.gaussian_likelihood.GaussianLikelihood(dim=2)
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=10,
                                                gamma=0.2)

    const_zero = torch.zeros(1, dtype=torch.double)
    num_epochs = 50
    for epoch in range(num_epochs):
        print("Epoch: {}\t Mean: {}".format(epoch,
                                            model.mean.data.cpu().numpy()))
        for sample in data_train_loader:
            loss = criterion(model(sample), const_zero)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()


if __name__ == "__main__":
    main()
