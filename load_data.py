import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import SubsetRandomSampler
from torchvision import transforms, datasets

def set_seed(seed: int):
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(mode=True)

def get_validation_set(dst_train, split: float = 0.1, seed: int = 42):

    set_seed(seed)

    indices = list(range(len(dst_train)))
    np.random.shuffle(indices)
    split = int(np.floor(split * len(dst_train)))
    train_indices, val_indices = indices[split:], indices[:split]
    print(len(train_indices), len(val_indices))
    train_sample = SubsetRandomSampler(train_indices)
    val_sample = SubsetRandomSampler(val_indices)

    return train_sample, val_sample

def get_test_val_set(dst_train, split_test=0.1, split_val=0.1, seed: int = 42):

    set_seed(seed)

    indices = list(range(len(dst_train)))
    np.random.shuffle(indices)
    split_test = int(np.floor(split_test * len(dst_train)))
    split_val = int(np.floor(split_val * len(dst_train)))
    train_indices, val_indices, test_indices = (
        indices[split_test + split_val :],
        indices[:split_val],
        indices[split_val : split_test + split_val],
    )
    train_sample = SubsetRandomSampler(train_indices)
    val_sample = SubsetRandomSampler(val_indices)
    test_sample = SubsetRandomSampler(test_indices)
    return train_sample, val_sample, test_sample


def get_dataset(dataset: str, data_path: str, batch_size: int, seed: int = 42):

    set_seed(seed)

    if dataset == "MNIST":
        channel = 1
        im_size = (32, 32)
        num_classes = 10
        transform = transforms.Compose([transforms.Resize(im_size), transforms.ToTensor()])
        dst_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
        dst_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
        train_sample, val_sample = get_validation_set(dst_train, split=0.1, seed=seed)
        class_names = [str(c) for c in range(num_classes)]

    elif dataset == "FMNIST":
        channel = 1
        im_size = (32, 32)
        num_classes = 10
        transform = transforms.Compose([transforms.Resize(im_size), transforms.ToTensor()])
        dst_train = datasets.FashionMNIST(data_path, train=True, download=True, transform=transform)
        dst_test = datasets.FashionMNIST(data_path, train=False, download=True, transform=transform)
        train_sample, val_sample = get_validation_set(dst_train, split=0.1, seed=seed)
        class_names = dst_train.classes

    elif dataset == "STL10":
        channel = 1
        im_size = (96, 96)
        num_classes = None
        class_names = None
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Grayscale(num_output_channels=channel)]
        )
        dst_train = datasets.STL10(
            data_path, split="unlabeled", download=True, folds=None, transform=transform
        )
        dst_test = None
        train_sample, val_sample, test_sample = get_test_val_set(
            dst_train, split_test=0.1, split_val=0.1, seed=seed
        )

    else:
        raise ValueError("unknown dataset: %s" % dataset)

    if dataset != "STL10":
        trainloader = torch.utils.data.DataLoader(
            dst_train, batch_size=batch_size, num_workers=0, sampler=train_sample
        )
        valoader = torch.utils.data.DataLoader(
            dst_train, batch_size=batch_size, num_workers=0, sampler=val_sample
        )
        testloader = torch.utils.data.DataLoader(
            dst_test, batch_size=batch_size, shuffle=False, num_workers=0
        )

    elif dataset == "STL10":
        trainloader = torch.utils.data.DataLoader(
            dst_train, batch_size=batch_size, num_workers=0, sampler=train_sample
        )
        valoader = torch.utils.data.DataLoader(
            dst_train, batch_size=batch_size, num_workers=0, sampler=val_sample
        )
        testloader = torch.utils.data.DataLoader(
            dst_train, batch_size=batch_size, num_workers=0, sampler=test_sample
        )

    return (
        channel,
        im_size,
        num_classes,
        class_names,
        dst_train,
        dst_test,
        testloader,
        trainloader,
        valoader,
    )