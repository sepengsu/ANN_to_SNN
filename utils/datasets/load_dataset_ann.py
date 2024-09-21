import torch
import torchvision
import torchvision.datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import os

def normalize(X):
    return 2 * X - 1.
    
def load_mnist(data_path, batch_size):
    SetRange = transforms.Lambda(normalize)
    transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        SetRange])

    trainset = torchvision.datasets.MNIST(data_path,
                                            train=True,
                                            transform=transform,
                                            download=True)

    testset = torchvision.datasets.MNIST(data_path,
                                            train=False,
                                            transform=transform,
                                            download=True)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return trainloader, testloader

def load_MIAD_metal_welding(data_path, batch_size):
    train_path = os.path.join(data_path, "MIAD/train/")
    test_path = os.path.join(data_path, "MIAD/test/")
    SetRange = transforms.Lambda(normalize)  # Normalize 적용
    
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        SetRange])

    trainset = torchvision.datasets.ImageFolder(root=train_path, transform=transform)
    testset = torchvision.datasets.ImageFolder(root=test_path, transform=transform)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False)
    
    return trainloader, testloader

