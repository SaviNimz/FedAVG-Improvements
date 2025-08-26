import torch

def server_aggregation(client_weights):
    """
    Aggregates the client weights using FedAvg method (simple averaging).
    
    Parameters:
        client_weights (list of dicts): List of model weights received from clients.
    
    Returns:
        dict: Aggregated global model weights.
    """
    global_model_weights = {}
    num_clients = len(client_weights)
    
    # Initialize global model weights to zero
    for key in client_weights[0].keys():
        global_model_weights[key] = torch.zeros_like(client_weights[0][key])

    # Perform weighted averaging of client weights
    for client_weight in client_weights:
        for key in client_weight.keys():
            global_model_weights[key] += client_weight[key] / num_clients

    return global_model_weights
