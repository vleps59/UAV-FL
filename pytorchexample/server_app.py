# pytorchexample/server_app.py
import flwr as fl
from flwr.common import Context
from flwr.server.strategy import FedAvg
from flwr.server import ServerConfig


def server_fn(context: Context):
    """Create Flower server with FedAvg strategy"""
    
    # FedAvg strategy configuration
    strategy = FedAvg(
        fraction_fit=1.0,           # Use all available clients each round
        fraction_evaluate=0.5,      # Use 50% for evaluation
        min_fit_clients=2,          # Minimum clients for training
        min_evaluate_clients=2,     # Minimum clients for evaluation
        min_available_clients=3,    # Minimum clients needed to start
    )
    
    # Server configuration
    config = ServerConfig(
        num_rounds=5,               # Number of FL communication rounds
        round_timeout=600,          # Timeout in seconds (10 minutes)
    )
    
    print("\n🚁 UAV-Swarm Federated Learning Server Starting...")
    print(f"   Strategy: FedAvg")
    print(f"   Rounds: {config.num_rounds}")
    
    return fl.server.ServerApp(
        config=config,
        strategy=strategy,
    )