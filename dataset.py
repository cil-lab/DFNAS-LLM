import torchvision.transforms as transforms
import torch
import torchvision
import torch.nn as nn
import numpy as np



def build_cifar10(batch_size):
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./input/cifar', train=True, download=True, transform=transform_train) 
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=5)

    testset = torchvision.datasets.CIFAR10(
        root='./input/cifar', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=5)
    return trainloader, testloader




class Cutout(nn.Module):
    def __init__(self, length=16, prob=1.0):
        super().__init__()
        self.length = length  # 遮挡区域边长
        self.prob = prob      # 应用Cutout的概率

    def forward(self, img):
        if torch.rand(1) > self.prob:
            return img  # 按概率跳过Cutout
        
        h, w = img.shape[1], img.shape[2]
        mask = torch.ones((h, w), dtype=torch.float32)
        
        # 随机生成遮挡中心坐标
        y = torch.randint(0, h, (1,)).item()
        x = torch.randint(0, w, (1,)).item()
        
        # 计算遮挡区域边界（防止越界）
        y1 = max(0, y - self.length // 2)
        y2 = min(h, y + self.length // 2)
        x1 = max(0, x - self.length // 2)
        x2 = min(w, x + self.length // 2)
        
        # 应用遮挡
        mask[y1:y2, x1:x2] = 0
        mask = mask.expand_as(img)  # 扩展到所有通道
        img = img * mask
        
        return img

def build_cifar10_cut(batch_size):
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    Cutout(length=16, prob=0.5)  # 遮挡16x16区域，50%概率应用
])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./input/cifar', train=True, download=True, transform=transform_train) 
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=5)

    testset = torchvision.datasets.CIFAR10(
        root='./input/cifar', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=5)
    return trainloader, testloader



class Cutout(object):
    def __init__(self, length=16, n_cut=1):
        self.length = length  # 遮挡边长
        self.n_cut = n_cut    # 遮挡数量

    def __call__(self, img):
        h, w = img.size(1), img.size(2)  # 图像高度和宽度（CIFAR-10为32×32）
        mask = np.ones((h, w), np.float32)
        
        for _ in range(self.n_cut):
            y = np.random.randint(h)     # 随机生成遮挡中心y坐标
            x = np.random.randint(w)     # 随机生成遮挡中心x坐标
            
            # 计算遮挡区域边界（防止越界）
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            
            # 将遮挡区域置为0
            mask[y1:y2, x1:x2] = 0.
        
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)  # 扩展到与图像相同的通道数（如RGB）
        img = img * mask            # 应用遮挡
        
        return img
    





def build_cifar10_final(batch_size):


    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    Cutout(length=16, n_cut=1)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./input/cifar', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=5)

    testset = torchvision.datasets.CIFAR10(
        root='./input/cifar', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=256, shuffle=False, num_workers=5)

    return trainloader, testloader





class Lighting(object):
    """Lighting noise (AlexNet-style PCA-based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec
        self.alpha = torch.Tensor(3).normal_(0, self.alphastd)

    def __call__(self, img):
        if self.alphastd == 0:
            return img
        rgb = torch.mm(self.eigvec, torch.mul(self.alpha, self.eigval).view(3, 1))

        return img + rgb.view(3, 1, 1)




def build_imagenet(batch_size):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    eigval = torch.Tensor([0.2175, 0.0188, 0.0045])
    eigvec = torch.Tensor([
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203]
    ])
    transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                Lighting(0.1, eigval, eigvec),
                transforms.Normalize(mean, std),
            ])

    transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])


    trainset = torchvision.datasets.ImageNet(
        root='./input/imagenet', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.ImageNet(
        root='./input/imagenet', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return trainloader, testloader




def build_cifar100(batch_size):
    mean = [129.3 / 255, 124.1 / 255, 112.4 / 255]
    std = [68.2 / 255, 65.4 / 255, 70.4 / 255]
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Pad(padding=4),
        transforms.RandomCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    trainset = torchvision.datasets.CIFAR100(
        root='./input/cifar', train=True, download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(
        root='./input/cifar', train=False, download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return trainloader, testloader





def build_svhn(batch_size=128):
    mean = [129.3 / 255, 124.1 / 255, 112.4 / 255]
    std = [68.2 / 255, 65.4 / 255, 70.4 / 255]
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Pad(padding=4),
        transforms.RandomCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    trainset = torchvision.datasets.SVHN(
        root='./input/svhn', split='train', download=True)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.SVHN(
        root='./input/svhn', split='test', download=True)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return trainloader, testloader
