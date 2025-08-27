import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def _get_transform(dataset_name: str):
    """Return dataset-specific transforms."""
    name = dataset_name.lower()
    if name == 'cifar-10':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    if name in ('emnist', 'femnist'):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
    raise ValueError(f"Unsupported dataset for transforms: {dataset_name}")


def _split_dataset(dataset, num_clients: int, non_iid: bool, shards_per_client: int = 2):
    """Split a dataset into subsets for each client."""
    if num_clients <= 1:
        return [dataset]

    indices = np.arange(len(dataset))
    targets = np.array(dataset.targets if hasattr(dataset, 'targets') else dataset.labels)

    if non_iid:
        # Sort by labels and divide into shards
        sorted_indices = indices[np.argsort(targets)]
        shards = np.array_split(sorted_indices, num_clients * shards_per_client)
        np.random.shuffle(shards)
        client_indices = [
            np.concatenate(shards[i * shards_per_client:(i + 1) * shards_per_client])
            for i in range(num_clients)
        ]
    else:
        np.random.shuffle(indices)
        client_indices = np.array_split(indices, num_clients)

    return [Subset(dataset, idx) for idx in client_indices]


def load_data(dataset_name: str, batch_size: int, num_clients: int = 1, non_iid: bool = False):
    """Load dataset and return DataLoaders for clients and test set."""

    transform = _get_transform(dataset_name)
    name = dataset_name.upper()

    if name == 'CIFAR-10':
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif name in ('EMNIST', 'FEMNIST'):
        train_dataset = datasets.EMNIST(root='./data', split='byclass', train=True, download=True, transform=transform)
        test_dataset = datasets.EMNIST(root='./data', split='byclass', train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    subsets = _split_dataset(train_dataset, num_clients, non_iid)
    train_loaders = [
        DataLoader(subset, batch_size=batch_size, shuffle=True) for subset in subsets
    ]
    if num_clients == 1:
        train_loaders = train_loaders[0]

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loaders, test_loader
