
import os
import torch
import torchvision
from torchvision.transforms import RandomHorizontalFlip, RandomCrop, ToTensor, Normalize, Compose



_CIFAR10 = {
    'Normalize': {
        'mean': (0.4914, 0.4822, 0.4465),
        'std':  (0.2470, 0.2430, 0.2610)
    }
}


def get_transforms(augment):
    train_t = Compose([RandomCrop(32, padding=4),
                       RandomHorizontalFlip(),
                       ToTensor(),
                       Normalize(**_CIFAR10['Normalize'])])
    valid_t = Compose([ToTensor(),
                       Normalize(**_CIFAR10['Normalize'])])
    if not augment:
        train_t = valid_t
    transforms = {
        'training':   train_t,
        'validation': valid_t
    }
    return transforms


def load_data_sets(logbook):
    transforms  = get_transforms(logbook.config['data']['augment'])
    train_set   = torchvision.datasets.CIFAR10(logbook.dir_data, train=True, transform=transforms['training'], download=False)
    valid_set    = torchvision.datasets.CIFAR10(logbook.dir_data, train=False, transform=transforms['validation'], download=False)

    return train_set, valid_set