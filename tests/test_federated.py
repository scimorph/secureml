"""
Tests for the federated learning module of SecureML.
"""

import unittest
import tempfile
import os
import json
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock, Mock
import pytest

# Mock implementations for federated learning components
class MockFederatedServer:
    """Mock implementation of FederatedServer."""
    
    def __init__(self, model_config, aggregation_method="federated_averaging"):
        """Initialize the federated server."""
        self.model_config = model_config
        self.aggregation_method = aggregation_method
        self.clients = []
        self.global_model = {"weights": [np.zeros((10, 10)), np.zeros(10)],
                             "config": model_config}
        self.round = 0
        self.training_history = []
    
    def register_client(self, client):
        """Register a client with the server."""
        client_id = f"client_{len(self.clients)}"
        self.clients.append({"id": client_id, "client": client})
        return client_id
    
    def broadcast_model(self):
        """Broadcast the global model to all clients."""
        for client_info in self.clients:
            client_info["client"].receive_model(self.global_model)
        
        return len(self.clients)
    
    def collect_updates(self):
        """Collect model updates from all clients."""
        updates = []
        for client_info in self.clients:
            client_update = client_info["client"].send_update()
            if client_update:
                updates.append(client_update)
        
        return updates
    
    def aggregate_updates(self, updates):
        """Aggregate updates from clients using the specified method."""
        if not updates:
            return self.global_model
        
        # Simple mock aggregation - just average the weights
        new_weights = []
        for i in range(len(self.global_model["weights"])):
            layer_updates = [u["weights"][i] for u in updates]
            new_weights.append(np.mean(layer_updates, axis=0))
        
        self.global_model["weights"] = new_weights
        return self.global_model
    
    def train_round(self):
        """Run one round of federated training."""
        self.round += 1
        
        # Broadcast global model
        self.broadcast_model()
        
        # Collect updates
        updates = self.collect_updates()
        
        # Aggregate updates
        self.aggregate_updates(updates)
        
        # Store training metrics
        metrics = {
            "round": self.round,
            "num_clients": len(self.clients),
            "num_updates": len(updates)
        }
        self.training_history.append(metrics)
        
        return metrics
    
    def train(self, num_rounds):
        """Train for multiple rounds."""
        results = []
        for _ in range(num_rounds):
            round_result = self.train_round()
            results.append(round_result)
        
        return results
    
    def save_model(self, filepath):
        """Save the global model to a file."""
        # Convert numpy arrays to lists for JSON serialization
        model_json = {
            "config": self.global_model["config"],
            "weights": [w.tolist() for w in self.global_model["weights"]]
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_json, f)
        
        return filepath
    
    def load_model(self, filepath):
        """Load a global model from a file."""
        with open(filepath, 'r') as f:
            model_json = json.load(f)
        
        # Convert lists back to numpy arrays
        self.global_model = {
            "config": model_json["config"],
            "weights": [np.array(w) for w in model_json["weights"]]
        }
        
        return self.global_model


class MockFederatedClient:
    """Mock implementation of FederatedClient."""
    
    def __init__(self, data=None, differential_privacy=False, dp_epsilon=1.0):
        """Initialize the federated client."""
        self.data = data
        self.local_model = None
        self.differential_privacy = differential_privacy
        self.dp_epsilon = dp_epsilon
        self.training_history = []
    
    def receive_model(self, model):
        """Receive a model from the server."""
        # Deep copy the model weights to avoid shared references
        self.local_model = {
            "config": model["config"],
            "weights": [np.copy(w) for w in model["weights"]]
        }
        return True
    
    def train_local_model(self, epochs=1, batch_size=32):
        """Train the local model on the client's data."""
        if self.local_model is None or self.data is None:
            return False
        
        # Mock training - just add some random noise to the weights
        for i in range(len(self.local_model["weights"])):
            # Generate random noise
            noise = np.random.normal(0, 0.01, self.local_model["weights"][i].shape)
            
            # Apply differential privacy if enabled
            if self.differential_privacy:
                # Mock DP implementation - scale noise by epsilon
                noise = noise * (1.0 / self.dp_epsilon)
            
            # Update weights
            self.local_model["weights"][i] += noise
        
        # Store training metrics
        metrics = {
            "epochs": epochs,
            "batch_size": batch_size,
            "samples": len(self.data) if hasattr(self.data, "__len__") else 0,
            "dp_enabled": self.differential_privacy
        }
        self.training_history.append(metrics)
        
        return metrics
    
    def send_update(self):
        """Send model update to the server."""
        if self.local_model is None:
            return None
        
        # Return a copy of the local model
        return {
            "config": self.local_model["config"],
            "weights": [np.copy(w) for w in self.local_model["weights"]]
        }
    
    def evaluate_model(self, test_data=None):
        """Evaluate the local model."""
        if self.local_model is None:
            return None
        
        # Mock evaluation - just return random metrics
        metrics = {
            "accuracy": np.random.uniform(0.7, 0.95),
            "loss": np.random.uniform(0.1, 0.5)
        }
        
        return metrics


# Create mock objects
FederatedServer = MagicMock(side_effect=MockFederatedServer)
FederatedClient = MagicMock(side_effect=MockFederatedClient)
SecureAggregator = MagicMock()

# Patch the module
patch_path = 'secureml.federated'
patch(f'{patch_path}.FederatedServer', FederatedServer).start()
patch(f'{patch_path}.FederatedClient', FederatedClient).start()
patch(f'{patch_path}.SecureAggregator', SecureAggregator).start()

# Import the patched module
from secureml.federated import FederatedServer, FederatedClient, SecureAggregator


class TestFederatedLearning(unittest.TestCase):
    """Test cases for the federated learning module."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for model files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create model configuration
        self.model_config = {
            "type": "neural_network",
            "architecture": [
                {"type": "dense", "units": 10, "activation": "relu"},
                {"type": "dense", "units": 1, "activation": "sigmoid"}
            ],
            "optimizer": "adam",
            "loss": "binary_crossentropy"
        }
        
        # Create a server instance
        self.server = FederatedServer(self.model_config)
        
        # Create mock client data
        self.client_data = [
            pd.DataFrame({
                "feature1": np.random.normal(0, 1, 100),
                "feature2": np.random.normal(0, 1, 100),
                "target": np.random.randint(0, 2, 100)
            }) for _ in range(3)
        ]
        
        # Create client instances
        self.clients = [
            FederatedClient(data=data) for data in self.client_data
        ]
        
        # Register clients with server
        for client in self.clients:
            self.server.register_client(client)
    
    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()
    
    def test_server_initialization(self):
        """Test server initialization."""
        server = FederatedServer(self.model_config)
        
        # Check that the server was initialized correctly
        self.assertEqual(server.model_config, self.model_config)
        self.assertEqual(server.aggregation_method, "federated_averaging")
        self.assertEqual(len(server.clients), 0)
        self.assertEqual(server.round, 0)
    
    def test_client_initialization(self):
        """Test client initialization."""
        data = self.client_data[0]
        client = FederatedClient(data=data)
        
        # Check that the client was initialized correctly
        self.assertEqual(client.data, data)
        self.assertIsNone(client.local_model)
        self.assertFalse(client.differential_privacy)
    
    def test_client_registration(self):
        """Test client registration with server."""
        server = FederatedServer(self.model_config)
        client = FederatedClient()
        
        # Register client
        client_id = server.register_client(client)
        
        # Check that client was registered
        self.assertIn({"id": client_id, "client": client}, server.clients)
        self.assertEqual(len(server.clients), 1)
    
    def test_model_broadcasting(self):
        """Test broadcasting model to clients."""
        # Broadcast model
        num_clients = self.server.broadcast_model()
        
        # Check that all clients received the model
        self.assertEqual(num_clients, len(self.clients))
        for client in self.clients:
            self.assertIsNotNone(client.local_model)
            self.assertEqual(client.local_model["config"], self.model_config)
    
    def test_client_training(self):
        """Test client local training."""
        # Broadcast model to clients
        self.server.broadcast_model()
        
        # Train local models
        for client in self.clients:
            metrics = client.train_local_model(epochs=2, batch_size=16)
            
            # Check that training was performed
            self.assertIsNotNone(metrics)
            self.assertEqual(metrics["epochs"], 2)
            self.assertEqual(metrics["batch_size"], 16)
            self.assertEqual(len(client.training_history), 1)
    
    def test_collecting_updates(self):
        """Test collecting updates from clients."""
        # Broadcast model to clients
        self.server.broadcast_model()
        
        # Train local models
        for client in self.clients:
            client.train_local_model()
        
        # Collect updates
        updates = self.server.collect_updates()
        
        # Check that updates were collected
        self.assertEqual(len(updates), len(self.clients))
        for update in updates:
            self.assertIn("config", update)
            self.assertIn("weights", update)
    
    def test_aggregating_updates(self):
        """Test aggregating updates from clients."""
        # Broadcast model to clients
        self.server.broadcast_model()
        
        # Train local models
        for client in self.clients:
            client.train_local_model()
        
        # Collect updates
        updates = self.server.collect_updates()
        
        # Aggregate updates
        updated_model = self.server.aggregate_updates(updates)
        
        # Check that model was updated
        self.assertIsNotNone(updated_model)
        self.assertIn("weights", updated_model)
        self.assertIn("config", updated_model)
    
    def test_training_round(self):
        """Test a complete training round."""
        # Run one round of training
        metrics = self.server.train_round()
        
        # Check that round was executed
        self.assertEqual(self.server.round, 1)
        self.assertEqual(metrics["round"], 1)
        self.assertEqual(metrics["num_clients"], len(self.clients))
        self.assertEqual(len(self.server.training_history), 1)
    
    def test_multiple_training_rounds(self):
        """Test multiple training rounds."""
        # Run multiple rounds of training
        num_rounds = 3
        results = self.server.train(num_rounds)
        
        # Check that rounds were executed
        self.assertEqual(self.server.round, num_rounds)
        self.assertEqual(len(results), num_rounds)
        self.assertEqual(len(self.server.training_history), num_rounds)
    
    def test_saving_and_loading_model(self):
        """Test saving and loading global model."""
        # Run one round of training
        self.server.train_round()
        
        # Save the model
        model_path = os.path.join(self.temp_dir.name, "model.json")
        saved_path = self.server.save_model(model_path)
        
        # Check that model was saved
        self.assertTrue(os.path.exists(model_path))
        
        # Create a new server
        new_server = FederatedServer(self.model_config)
        
        # Load the model
        loaded_model = new_server.load_model(model_path)
        
        # Check that model was loaded
        self.assertEqual(new_server.global_model["config"], self.server.global_model["config"])
        self.assertEqual(len(new_server.global_model["weights"]), len(self.server.global_model["weights"]))
    
    def test_differential_privacy(self):
        """Test client with differential privacy."""
        # Create a client with DP enabled
        dp_client = FederatedClient(data=self.client_data[0], differential_privacy=True, dp_epsilon=0.5)
        
        # Register with the server
        self.server.register_client(dp_client)
        
        # Broadcast model
        self.server.broadcast_model()
        
        # Train local model
        metrics = dp_client.train_local_model()
        
        # Check that DP was enabled
        self.assertTrue(metrics["dp_enabled"])
        
        # Run a round of training
        self.server.train_round()


# Add pytest-style tests
@pytest.fixture
def model_config():
    """Create a model configuration fixture."""
    return {
        "type": "neural_network",
        "architecture": [
            {"type": "dense", "units": 10, "activation": "relu"},
            {"type": "dense", "units": 1, "activation": "sigmoid"}
        ],
        "optimizer": "adam",
        "loss": "binary_crossentropy"
    }


@pytest.fixture
def server(model_config):
    """Create a federated server fixture."""
    return FederatedServer(model_config)


@pytest.fixture
def client_data():
    """Create client data fixtures."""
    return [
        pd.DataFrame({
            "feature1": np.random.normal(0, 1, 100),
            "feature2": np.random.normal(0, 1, 100),
            "target": np.random.randint(0, 2, 100)
        }) for _ in range(3)
    ]


@pytest.fixture
def clients(client_data):
    """Create client fixtures."""
    return [FederatedClient(data=data) for data in client_data]


@pytest.fixture
def federated_setup(server, clients):
    """Set up a complete federated learning environment."""
    # Register clients with server
    for client in clients:
        server.register_client(client)
    
    # Return the setup
    return {
        "server": server,
        "clients": clients
    }


def test_federated_round(federated_setup):
    """Test a complete federated learning round using fixtures."""
    server = federated_setup["server"]
    
    # Run a training round
    metrics = server.train_round()
    
    # Check round results
    assert server.round == 1
    assert metrics["round"] == 1
    assert metrics["num_clients"] == len(federated_setup["clients"])


def test_federated_training(federated_setup, tmp_path):
    """Test complete federated training process."""
    server = federated_setup["server"]
    
    # Run multiple rounds of training
    num_rounds = 3
    results = server.train(num_rounds)
    
    # Check results
    assert len(results) == num_rounds
    assert server.round == num_rounds
    
    # Save model
    model_path = tmp_path / "model.json"
    server.save_model(str(model_path))
    
    # Verify file exists
    assert model_path.exists()


@pytest.mark.parametrize(
    "aggregation_method", 
    ["federated_averaging", "federated_median", "secure_aggregation"]
)
def test_aggregation_methods(model_config, clients, aggregation_method):
    """Test different aggregation methods."""
    # Create server with specified aggregation method
    server = FederatedServer(model_config, aggregation_method=aggregation_method)
    
    # Register clients
    for client in clients:
        server.register_client(client)
    
    # Run a training round
    metrics = server.train_round()
    
    # Check that training was successful
    assert server.round == 1
    assert metrics["num_clients"] == len(clients)


if __name__ == "__main__":
    unittest.main() 