import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_mnist_loaders(batch_size=128, train_subset=10000, val_subset=2000):
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root="./data", train=True, download=True,
                                   transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True,
                                  transform=transform)

    train_indices = list(range(train_subset))
    val_indices = list(range(train_subset, train_subset + val_subset))

    train_subset_ds = Subset(train_dataset, train_indices)
    val_subset_ds = Subset(train_dataset, val_indices)

    train_loader = DataLoader(train_subset_ds, batch_size=batch_size,
                              shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset_ds, batch_size=batch_size,
                            shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=2)
    return train_loader, val_loader, test_loader
