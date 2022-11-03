import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

def build_cifar10(download=False):
    aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    aug.append(transforms.ToTensor())

    aug.append(
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), )
    transform_train = transforms.Compose(aug)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_dataset = CIFAR10(root='./dataset/',
                            train=True, download=download, transform=transform_train)
    val_dataset = CIFAR10(root='./dataset/',
                            train=False, download=download, transform=transform_test)

    return train_dataset, val_dataset