# run_simulation.py
"""Direct simulation script for Flower FL"""

import flwr as fl
from flwr.simulation import run_simulation
from flwr.server.strategy import FedAvg
from flwr.server import ServerConfig

from pytorchexample.client_app import client_fn
from pytorchexample.task import get_model, NUM_CLASSES

def main():
    """Run the FL simulation directly"""
    
    print("=" * 60)
    print("🚁 UAV-Swarm Federated Learning Simulation")
    print("=" * 60)
    print(f"   Number of classes: {NUM_CLASSES}")
    print(f"   Model: FlyingObjectDetector")
    print(f"   Strategy: FedAvg")
    print("=" * 60)
    
    # Server configuration
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.5,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=3,
    )
    
    config = ServerConfig(
        num_rounds=5,
        round_timeout=600,
    )
    
    # Client resources
    backend_config = {
        "client_resources": {
            "num_cpus": 1,
            "num_gpus": 0.0,
        }
    }
    
    # Run simulation
    print("\n🌟 Starting simulation...\n")
    
    run_simulation(
        server_app=fl.server.ServerApp(config=config, strategy=strategy),
        client_app=fl.client.ClientApp(client_fn=client_fn),
        num_supernodes=3,  # Number of clients
        backend_config=backend_config,
    )
    
    print("\n✅ Simulation complete!")

if __name__ == "__main__":
    main()
    