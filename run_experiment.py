import torch
import torch.optim as optim
import yaml
from training.client_update import client_update
from training.server_aggregation import server_aggregation
from utils.data_loader import load_data
from utils.evaluation import evaluate_model
from models.student_model import StudentModel
from models.teacher_model import TeacherModel
from utils.loss_function import total_loss

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
    # Load datasets
    train_loader, test_loader = load_data(config['dataset_name'], config['batch_size'])

    # Initialize global model (use any base architecture for experimentation)
    global_model = StudentModel(global_model=None)  # Replace with a base model (e.g., CNN)

    # Hyperparameters
    lambda_ = config['lambda_']
    T = config['T']
    tau = config['tau']
    epochs = config['epochs']
    learning_rate = config['learning_rate']

    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # Simulate the communication with clients (just a single client here)
        client_weights = []
        for i, (inputs, labels) in enumerate(train_loader):
            # Create teacher-student models
            teacher = TeacherModel(global_model)
            student = StudentModel(global_model)

            # Train the student model
            updated_weights = client_update(global_model, train_loader, lambda_, T, tau)
            client_weights.append(updated_weights)

        # Aggregate the client weights on the server
        aggregated_weights = server_aggregation(client_weights)
        global_model.load_state_dict(aggregated_weights)  # Update global model with aggregated weights

        # Evaluate the model after each epoch
        accuracy, avg_loss = evaluate_model(global_model, test_loader, total_loss)
        print(f"Validation Accuracy: {accuracy:.2f}% | Validation Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    # Load configuration file (you can specify the path to your YAML file)
    config = load_config('config/config.yaml')

    # Run the experiment
    run_experiment(config)
