import os
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import random_split
import torch
import utils.global_v as glv
import torchvision
from torch.utils.data import DataLoader, random_split

def load_mnist(data_path, batch_size=None, input_size=None, small=False):
    print("loading mnist")
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    if batch_size is None:
        batch_size = glv.network_config['batch_size']
    if input_size is None:
        input_size = glv.network_config['input_size']

    SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    transform_train = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        SetRange
    ])

    transform_test = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        SetRange
    ])
    trainset = torchvision.datasets.MNIST(data_path, train=True, transform=transform_train, download=True)
    testset = torchvision.datasets.MNIST(data_path, train=False, transform=transform_test, download=True)

    if small:
        trainset, _ = random_split(dataset=trainset, lengths=[int(0.1 * len(trainset)), int(0.9 * len(trainset))],
                                   generator=torch.Generator().manual_seed(0))
        testset, _ = random_split(dataset=testset, lengths=[int(0.1 * len(testset)), int(0.9 * len(testset))],
                                   generator=torch.Generator().manual_seed(0))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size*2, shuffle=False, num_workers=2, pin_memory=True)
    return trainloader, testloader

def load_MIAD_metal_welding(data_path, batch_size=None, input_size=None, small=False):
    """
    MIAD Metal Welding Dataset Loader Function

    Args:
    - data_path (str): Path to the dataset folder
    - batch_size (int): Size of the batches
    - input_size (int): Input image size (for resizing)
    - small (bool): If True, loads only a small subset of the dataset for testing purposes

    Returns:
    - trainloader (DataLoader): DataLoader for the training set
    - testloader (DataLoader): DataLoader for the test set
    """
    print("loading MIAD_metal_welding")

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    if batch_size is None:
        batch_size = 32  # Default batch size
    if input_size is None:
        input_size = 32  # Default input size

    # 이미지 정규화 범위를 [-1, 1]로 맞추는 람다 함수 적용
    SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    
    # Train과 Test에 사용할 변환 정의
    transform_train = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        SetRange
    ])

    transform_test = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        SetRange
    ])

    # 데이터 경로 설정
    train_path = os.path.join(data_path, "train/")
    test_path = os.path.join(data_path, "test/")
    
    # ImageFolder로 데이터셋 불러오기
    trainset = torchvision.datasets.ImageFolder(root=train_path, transform=transform_train)
    testset = torchvision.datasets.ImageFolder(root=test_path, transform=transform_test)

    # small 옵션이 활성화된 경우 데이터셋의 일부만 로드
    if small:
        trainset, _ = random_split(dataset=trainset, lengths=[int(0.1 * len(trainset)), int(0.9 * len(trainset))],
                                   generator=torch.Generator().manual_seed(0))
        testset, _ = random_split(dataset=testset, lengths=[int(0.1 * len(testset)), int(0.9 * len(testset))],
                                   generator=torch.Generator().manual_seed(0))

    # DataLoader 설정
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size * 2, shuffle=False, num_workers=2, pin_memory=True)

    return trainloader, testloader


