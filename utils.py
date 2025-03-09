import os
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import SubsetRandomSampler
from torchvision import transforms, datasets
from torch.autograd import Function
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import copy

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
class BinaryQuantize(Function):

    @staticmethod
    def forward(ctx, weights):
        # out = torch.sign(input) * 0.5 + 0.5
        ctx.save_for_backward(weights)
        return (weights >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        # weights, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # grad_input[weights.abs() > 1] = 0
        return grad_input


class BinaryQuantize_1(Function):

    @staticmethod
    def forward(ctx, input):
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input
    
def plot_H(H):
    H_np = H.detach().cpu().numpy()
    plt.figure(figsize=(8, 6))
    plt.imshow(H_np, cmap='gray')
    plt.show()

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

def save_coded_apertures(system_layer, row, pad, path, name, system):

    if system == "SPC":
        aperture_codes = (
            copy.deepcopy(system_layer.get_coded_aperture().detach()).squeeze(0).unsqueeze(1)
        )

    elif system == "CASSI":
        aperture_codes = copy.deepcopy(
            system_layer.get_coded_aperture().detach().permute(1, 0, 2, 3)
        )

    elif system == "CCASSI":
        cas = copy.deepcopy(system_layer.get_coded_aperture().detach())
        aperture_codes = cas.view(cas.shape[-4] * cas.shape[-3], 1, cas.shape[-2], cas.shape[-1])

    elif system == "MRI":
        aperture_codes = copy.deepcopy(system_layer.get_mask().detach()).permute(1, 0, 2, 3)

    elif system == "CT":
        aperture_codes = copy.deepcopy(system_layer.get_mask().detach())

    grid = vutils.make_grid(aperture_codes, nrow=row, padding=pad, normalize=True)
    vutils.save_image(grid, f"{path}/{name}.png")

    return grid


def save_reconstructed_images(imgs, recons, num_img, pad, path, name, PSNR, SSIM):

    grid = vutils.make_grid(
        torch.cat((imgs[:num_img], recons[:num_img])), nrow=num_img, padding=pad, normalize=True
    )
    vutils.save_image(grid, f"{path}/{name}.png")

    psnr_imgs = [
        np.round(PSNR(recons[i].unsqueeze(0), imgs[i].unsqueeze(0)).item(), 2)
        for i in range(num_img)
    ]
    ssim_imgs = [
        np.round(SSIM(recons[i].unsqueeze(0), imgs[i].unsqueeze(0)).item(), 3)
        for i in range(num_img)
    ]

    return grid, psnr_imgs, ssim_imgs


def hadamard(n):

    if n == 1:
        return np.array([[1]])

    else:
        h = hadamard(n // 2)
        return np.block([[h, h], [h, -h]])


def save_metrics(save_path):

    images_path = save_path + "/images"
    model_path = save_path + "/model"
    metrics_path = save_path + "/metrics"

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(metrics_path, exist_ok=True)

    return images_path, model_path, metrics_path

def print_dict(dictionary):
    for key, value in dictionary.items():
        print(f"{key}: {value}")