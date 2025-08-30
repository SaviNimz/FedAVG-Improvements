from __future__ import annotations

from typing import Any, Dict
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.data_loader import load_data
from utils.evaluation import evaluate_model
from utils.loss_function import cross_entropy_loss
from training.client_update_baseline import client_update_baseline
from training.client_update import client_update
from training.server_aggregation import server_aggregation
from models.architectures import (
    CIFARCNN,
    FEMNISTCNN,
    MNISTCNN,
    ShakespeareLSTM,
)


def _create_model(config: Dict[str, Any]) -> torch.nn.Module:
    """Return an architecture appropriate for the selected dataset."""
    dataset = config.get("dataset_name", "").lower()
    model_name = config.get("model_name", "").lower()

    if dataset == "cifar-10" or model_name == "cifar_cnn":
        return CIFARCNN()
    if dataset in ("femnist", "emnist") or model_name in ("femnist_cnn", "emnist_cnn"):
        num_classes = config.get("num_classes", 10)
        return FEMNISTCNN(num_classes=num_classes)
    if dataset == "mnist" or model_name == "mnist_cnn":
        return MNISTCNN()
    if dataset == "shakespeare" or model_name in ("shakespeare", "shakespeare_lstm"):
        vocab_size = config.get("vocab_size", 80)
        return ShakespeareLSTM(vocab_size=vocab_size)
    raise ValueError(f"Unsupported dataset/model combination: {dataset}, {model_name}")


def _seed_everything(seed: int | None) -> None:
    """Seed all random number generators for reproducibility."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _train_algorithm(
    algorithm: str,
    base_model: torch.nn.Module,
    train_loaders: list[DataLoader],
    test_loader: DataLoader,
    config: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    """Train a model using the specified algorithm and collect metrics."""
    epochs = config["epochs"]
    lr = config["learning_rate"]
    local_epochs = config.get("local_epochs", 1)
    lambda_ = config.get("lambda_", 0.5)
    T = config.get("T", 2.0)
    tau = config.get("tau", 0.9)

    model = base_model.to(device)
    metrics = {"accuracy": [], "loss": []}

    num_clients = len(train_loaders)
    for _ in range(epochs):
        client_weights = []
        client_sizes = []
        for loader in train_loaders:
            if algorithm == "fedavg":
                updated = client_update_baseline(
                    model, loader, lr, device, local_epochs
                )
            else:
                updated = client_update(
                    model, loader, lambda_, T, tau, lr, device, local_epochs
                )
            client_weights.append(updated)
            client_sizes.append(len(loader.dataset))

        aggregated = server_aggregation(client_weights, client_sizes)
        model.load_state_dict(aggregated)

        acc, loss = evaluate_model(model, test_loader, cross_entropy_loss, device)
        metrics["accuracy"].append(acc)
        metrics["loss"].append(loss)

    return {"metrics": metrics, "model_state": model.state_dict()}


def run_comparison(config: Dict[str, Any], tag: str) -> Dict[str, Any]:
    """Compare baseline FedAvg with knowledge distillation variant.

    The function trains two separate models using the provided configuration:
    the standard Federated Averaging baseline (``fedavg``) and the knowledge
    distillation enhanced version (``fedavg_kd``). Per-epoch accuracy and loss
    are collected for each algorithm along with the final model weights.

    Parameters
    ----------
    config: dict
        Experiment configuration containing dataset and hyperparameters.
    tag: str
        Identifier for the comparison run.

    Returns
    -------
    dict
        Dictionary holding metrics and final model states for both algorithms.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loaders, test_loader = load_data(
        config["dataset_name"],
        config["batch_size"],
        num_clients=config.get("num_clients", 1),
        non_iid=config.get("non_iid", True),
        shards_per_client=config.get("shards_per_client", 2),
        seed=config.get("seed"),
    )

    if isinstance(train_loaders, DataLoader):
        train_loaders = [train_loaders]

    results: Dict[str, Any] = {"tag": tag}

    for algorithm in ["fedavg", "fedavg_kd"]:
        _seed_everything(config.get("seed"))
        base_model = _create_model(config)
        results[algorithm] = _train_algorithm(
            algorithm, base_model, train_loaders, test_loader, config, device
        )

    return results
