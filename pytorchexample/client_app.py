# pytorchexample/client_app.py
import flwr as fl
from flwr.common import Context
import torch
import torch.optim as optim
from collections import OrderedDict

from .task import load_data, get_model, train, NUM_CLASSES


class FlowerClient(fl.client.NumPyClient):
    """Flower client representing a UAV in the swarm"""
    
    def __init__(self, model, trainloader, valloader, device):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
    
    def get_parameters(self, config):
        """Get model parameters for sending to server"""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        """Set model parameters received from server"""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        """Train model on local data"""
        self.set_parameters(parameters)
        
        # Training hyperparameters
        epochs = config.get("local_epochs", 1)
        lr = config.get("learning_rate", 0.001)
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Train
        train(self.model, self.trainloader, optimizer, epochs, self.device)
        
        # Return updated parameters and number of training samples
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}


def client_fn(context: Context):
    """Create a Flower client for a UAV in the swarm"""
    
    # Get client configuration
    partition_id = context.node_config.get("partition-id", 0)
    num_partitions = context.node_config.get("num-partitions", 3)
    
    # Load data for this client (simulates UAV's local data)
    trainloader = load_data(partition_id, num_partitions, batch_size=8)
    
    # Create model
    model = get_model()
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Client {partition_id} using device: {device}")
    
    # Return Flower client
    return FlowerClient(
        model=model,
        trainloader=trainloader,
        valloader=None,
        device=device,
    ).to_client()