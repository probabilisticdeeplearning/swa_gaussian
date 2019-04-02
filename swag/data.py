import os
import csv
from pathlib import Path

import numpy as np
import torch
import torchvision

from .camvid import CamVid

c10_classes = np.array([
    [0, 1, 2, 8, 9],
    [3, 4, 5, 6, 7]
], dtype=np.int32)

class SyntheticGaussianData(torch.utils.data.Dataset):

    def __init__(self, theta_0, cov_theta, cov_x, store_file,
                 reuse_data=False, n_samples=100):
        super(SyntheticGaussianData).__init__()
        self.theta_0 = theta_0
        self.cov_x = cov_x
        self.cov_theta = cov_theta
        self.n_samples = n_samples
        self.file = Path(store_file)
        if self.file.exists() and reuse_data:
            with self.file.open(newline="") as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=",", quotechar="|")
                assert len(theta_0) == len(next(csv_reader))
                assert n_samples - 1 == sum(1 for row in csv_reader)
        else:
            self.file.parent.mkdir(parents=True, exist_ok=True)
            sampled_theta = np.random.multivariate_normal(mean=self.theta_0,
                                                          cov=self.cov_theta,
                                                          size=n_samples)
            sampled_x = np.array([
                np.random.multivariate_normal(mean=theta, cov=self.cov_x)
                for theta in sampled_theta
            ])
            np.savetxt(self.file, sampled_x, delimiter=",")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        sample = None
        with self.file.open(newline="") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",", quotechar="|")
            for count, row in enumerate(csv_reader):
                if count == index:
                    sample = row
                    break

        return np.array(sample, dtype=float)

    def calculate_sufficient_statistics(self):
        return self.n_samples, np.sum(self.get_full_data())

    def get_full_data(self):
        tmp_raw_data = list()
        with self.file.open(newline="") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",", quotechar="|")
            tmp_raw_data = [data for data in csv_reader]
        return np.array(tmp_raw_data, dtype=float)







def camvid_loaders(path, batch_size, num_workers, transform_train, transform_test,
                   use_validation, val_size, shuffle_train=True,
                   joint_transform=None, ft_joint_transform=None, ft_batch_size=1, **kwargs):

    #load training and finetuning datasets
    print(path)
    train_set = CamVid(root=path, split='train', joint_transform=joint_transform, transform=transform_train, **kwargs)
    ft_train_set = CamVid(root=path, split='train', joint_transform=ft_joint_transform, transform=transform_train, **kwargs)

    val_set = CamVid(root=path, split='val', joint_transform=None, transform=transform_test, **kwargs)
    test_set = CamVid(root=path, split='test', joint_transform=None, transform=transform_test, **kwargs)

    num_classes = 11 # hard coded labels ehre

    return {'train': torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True
    ),
            'fine_tune': torch.utils.data.DataLoader(
                ft_train_set,
                batch_size=ft_batch_size,
                shuffle=shuffle_train,
                num_workers=num_workers,
                pin_memory=True
            ),
            'val': torch.utils.data.DataLoader(
                val_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            ),
            'test': torch.utils.data.DataLoader(
                test_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )}, num_classes


def svhn_loaders(path, batch_size, num_workers, transform_train, transform_test, use_validation, val_size, shuffle_train=True):
    train_set = torchvision.datasets.SVHN(root=path, split='train', download = True, transform = transform_train)

    if use_validation:
        test_set = torchvision.datasets.SVHN(root=path, split='train', download = True, transform = transform_test)
        train_set.data = train_set.data[:-val_size]
        train_set.labels = train_set.labels[:-val_size]

        test_set.data = test_set.data[-val_size:]
        test_set.labels = test_set.labels[-val_size:]

    else:
        print('You are going to run models on the test set. Are you sure?')
        test_set = torchvision.datasets.SVHN(root=path, split='test', download = True, transform = transform_test)

    num_classes = 10

    return \
        {
            'train': torch.utils.data.DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=True and shuffle_train,
                num_workers=num_workers,
                pin_memory=True
            ),
            'test': torch.utils.data.DataLoader(
                test_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            ),
        }, \
        num_classes



def loaders(dataset, path, batch_size, num_workers, transform_train, transform_test,
            use_validation=True, val_size=5000, split_classes=None, shuffle_train=True,
            **kwargs):

    if dataset == 'CamVid':
        return camvid_loaders(path, batch_size=batch_size, num_workers=num_workers, transform_train=transform_train,
                              transform_test=transform_test, use_validation=use_validation, val_size=val_size, **kwargs)

    path = os.path.join(path, dataset.lower())

    ds = getattr(torchvision.datasets, dataset)

    if dataset == 'SVHN':
        return svhn_loaders(path, batch_size, num_workers, transform_train, transform_test, use_validation, val_size)
    else:
        ds = getattr(torchvision.datasets, dataset)

    if dataset == 'STL10':
        train_set = ds(root=path, split='train', download=True, transform=transform_train)
        num_classes = 10
        cls_mapping = np.array([0, 2, 1, 3, 4, 5, 7, 6, 8, 9])
        train_set.labels = cls_mapping[train_set.labels]
    else:
        train_set = ds(root=path, train=True, download=True, transform=transform_train)
        num_classes = max(train_set.targets) + 1

    if use_validation:
        print("Using train (" + str(len(train_set.train_data)-val_size) + ") + validation (" +str(val_size)+ ")")
        train_set.train_data = train_set.train_data[:-val_size]
        train_set.targets = train_set.targets[:-val_size]

        test_set = ds(root=path, train=True, download=True, transform=transform_test)
        test_set.train = False
        test_set.test_data = test_set.train_data[-val_size:]
        test_set.test_labels = test_set.targets[-val_size:]
        delattr(test_set, 'train_data')
        delattr(test_set, 'train_labels')
    else:
        print('You are going to run models on the test set. Are you sure?')
        if dataset == 'STL10':
            test_set = ds(root=path, split='test', download=True, transform=transform_test)
            test_set.labels = cls_mapping[test_set.labels]
        else:
            test_set = ds(root=path, train=False, download=True, transform=transform_test)

    if split_classes is not None:
        assert dataset == 'CIFAR10'
        assert split_classes in {0, 1}

        print('Using classes:', end='')
        print(c10_classes[split_classes])
        train_mask = np.isin(train_set.targets, c10_classes[split_classes])
        train_set.train_data = train_set.train_data[train_mask, :]
        train_set.targets = np.array(train_set.targets)[train_mask]
        train_set.targets = np.where(train_set.targets[:, None] == c10_classes[split_classes][None, :])[1].tolist()
        print('Train: %d/%d' % (train_set.train_data.shape[0], train_mask.size))

        test_mask = np.isin(test_set.test_labels, c10_classes[split_classes])
        test_set.test_data = test_set.test_data[test_mask, :]
        test_set.test_labels = np.array(test_set.test_labels)[test_mask]
        test_set.test_labels = np.where(test_set.test_labels[:, None] == c10_classes[split_classes][None, :])[1].tolist()
        print('Test: %d/%d' % (test_set.test_data.shape[0], test_mask.size))

    return \
        {
            'train': torch.utils.data.DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=True and shuffle_train,
                num_workers=num_workers,
                pin_memory=True
            ),
            'test': torch.utils.data.DataLoader(
                test_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            ),
        }, \
        num_classes

