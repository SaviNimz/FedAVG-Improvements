import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def load_data(dataset_name, batch_size):
    """
    Loads datasets and returns DataLoader instances.
    
    Parameters:
        dataset_name (str): The name of the dataset to load ('CIFAR-10', 'FEMNIST', etc.).
        batch_size (int): The batch size for loading data.
    
    Returns:
        DataLoader: A PyTorch DataLoader instance for the dataset.
    """
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    if dataset_name == 'CIFAR-10':
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    elif dataset_name == 'FEMNIST':
        train_dataset = datasets.FEMNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.FEMNIST(root='./data', train=False, download=True, transform=transform)
    
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
