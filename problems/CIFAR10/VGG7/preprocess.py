import torchvision
from torchvision.transforms import Compose, RandomCrop, ToTensor, RandomHorizontalFlip, Normalize
from torch.utils.data import random_split

_CIFAR10 = {
    'Normalize': {
        'mean': (0.4914, 0.4822, 0.4465),
        'std': (0.247, 0.243, 0.261)
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
    data_config = logbook.config['data']
    dir_data = logbook.dir_data
    transforms           = get_transforms(data_config['augment'])
    train_set       = torchvision.datasets.CIFAR10(root=dir_data, train=True, transform=transforms['training'])

    valid_set             = torchvision.datasets.CIFAR10(root=dir_data, train=False, transform=transforms['validation'])

    return train_set, valid_set
