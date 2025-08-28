import argparse
from torch.utils.data import DataLoader
import yaml
import mlflow
import mlflow.pytorch
from training.client_update import client_update
from training.server_aggregation import server_aggregation
from utils.data_loader import load_data
from utils.evaluation import evaluate_model
from models.architectures import CIFARCNN, FEMNISTCNN, ShakespeareLSTM
from utils.loss_function import total_loss


def create_model(config):
    """Return an architecture appropriate for the selected dataset."""
    dataset = config['dataset_name'].lower()
    model_name = config.get('model_name', '').lower()

    if dataset == 'cifar-10' or model_name == 'cifar_cnn':
        return CIFARCNN()
    if dataset in ('femnist', 'emnist') or model_name in ('femnist_cnn', 'emnist_cnn'):
        num_classes = config.get('num_classes', 62)
        return FEMNISTCNN(num_classes=num_classes)
    if dataset == 'shakespeare' or model_name in ('shakespeare', 'shakespeare_lstm'):
        vocab_size = config.get('vocab_size', 80)
        return ShakespeareLSTM(vocab_size=vocab_size)
    raise ValueError(f"Unsupported dataset/model combination: {dataset}, {model_name}")

def load_config(config_file):
    """
    Loads the configuration from a YAML file.
    
    Parameters:
        config_file (str): Path to the configuration file.
    
    Returns:
        dict: Dictionary containing configuration values.
    """
    with open(config_file, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config

def run_experiment(config):
    """
    Runs the experiment, trains the model, and evaluates it.

    Parameters:
        config (dict): Configuration containing hyperparameters and dataset details.
    """
    with mlflow.start_run():
        mlflow.log_params(config)

        # Load datasets
        train_loaders, test_loader = load_data(
            config['dataset_name'],
            config['batch_size'],
            num_clients=config.get('num_clients', 1),
            non_iid=config.get('non_iid', False),
            seed=config.get('seed'),
        )

        # Initialize global model based on dataset/model selection
        global_model = create_model(config)

        # Hyperparameters
        lambda_ = config['lambda_']
        T = config['T']
        tau = config['tau']
        epochs = config['epochs']
        learning_rate = config['learning_rate']

        # Ensure train_loaders is iterable
        if isinstance(train_loaders, DataLoader):
            train_loaders = [train_loaders]

        # Training loop
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")

            client_weights = []
            client_sizes = []
            # Iterate over each client-specific DataLoader
            for client_loader in train_loaders:
                updated_weights = client_update(
                    global_model, client_loader, lambda_, T, tau, learning_rate
                )
                client_weights.append(updated_weights)
                client_sizes.append(len(client_loader.dataset))

            aggregated_weights = server_aggregation(client_weights, client_sizes)
            global_model.load_state_dict(aggregated_weights)

            accuracy, avg_loss = evaluate_model(global_model, test_loader, total_loss)
            print(f"Validation Accuracy: {accuracy:.2f}% | Validation Loss: {avg_loss:.4f}")
            mlflow.log_metric("accuracy", accuracy, step=epoch)
            mlflow.log_metric("loss", avg_loss, step=epoch)

        mlflow.pytorch.log_model(global_model, "model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Federated Learning experiments")
    parser.add_argument('--config', default='config/config.yaml', help='Path to config file')
    parser.add_argument('--dataset', help='Override dataset name from config')
    parser.add_argument('--model', help='Override model architecture')
    args = parser.parse_args()

    config = load_config(args.config)
    if args.dataset:
        config['dataset_name'] = args.dataset
    if args.model:
        config['model_name'] = args.model

    run_experiment(config)
