''' define two functions:
    get_transforms(augment)
        returns transforms for load_data_sets() to transform input data
    load_data_sets(logbook)
        returns train_set, valid_set for model training and validation
'''
import os
import torch
import torchvision
from torchvison import ToTensor, Normalize, RandomHorizontalFLip, Compose, RandomCrop



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


def load_data_sets(logboog):
    transforms  = get_transforms(logbook.config['data']['augment'])
    train_set   = torchvision.datasets.CIFAR10(logbook.dir_data, train=True, transform=transforms['training'], targettransform=None, download=False)
    valid_set    = torchvision.datasets.CIFAR10(logbook.dir_data, train=True, transform=transforms['validation'], targettransform=None, download=False)

    return train_set, valid_set