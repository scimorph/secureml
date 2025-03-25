"""
Federated learning functionality for SecureML.

This module provides tools for training machine learning models in a federated 
setting using Flower, where data remains distributed across multiple clients
without being centralized.
"""

from typing import Any, Callable, Dict, Optional, Tuple, Union
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Placeholder for optional imports
_HAS_FLOWER = False
_HAS_PYTORCH = False
_HAS_TENSORFLOW = False

# Try importing Flower
try:
    import flwr as fl
    from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, Parameters
    from flwr.server.client_proxy import ClientProxy
    print("Flower imported")
    _HAS_FLOWER = True
except ImportError:
    pass

# Try importing PyTorch
try:
    import torch
    _HAS_PYTORCH = True
except ImportError:
    pass

# Try importing TensorFlow
try:
    import tensorflow as tf
    _HAS_TENSORFLOW = True
except ImportError:
    pass


class FederatedConfig:
    """
    Configuration options for federated learning.
    """

    def __init__(
        self,
        num_rounds: int = 3,
        fraction_fit: float = 1.0,
        min_fit_clients: int = 2,
        min_available_clients: int = 2,
        server_address: str = "0.0.0.0:8080",
        use_secure_aggregation: bool = False,
        apply_differential_privacy: bool = False,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        **kwargs: Any,
    ):
        """
        Initialize federated learning configuration.

        Args:
            num_rounds: Number of federated training rounds
            fraction_fit: Fraction of clients used for training in each round
            min_fit_clients: Minimum number of clients for training
            min_available_clients: Minimum number of available clients to start round
            server_address: Server address in the format 'host:port'
            use_secure_aggregation: Whether to use secure aggregation protocol
            apply_differential_privacy: Whether to apply differential privacy
            epsilon: Privacy budget for differential privacy (if enabled)
            delta: Privacy failure probability for differential privacy (if enabled)
            **kwargs: Additional parameters for specific federated learning setups
        """
        self.num_rounds = num_rounds
        self.fraction_fit = fraction_fit
        self.min_fit_clients = min_fit_clients
        self.min_available_clients = min_available_clients
        self.server_address = server_address
        self.use_secure_aggregation = use_secure_aggregation
        self.apply_differential_privacy = apply_differential_privacy
        self.epsilon = epsilon
        self.delta = delta
        self.extra_kwargs = kwargs


def train_federated(
    model: Any,
    client_data_fn: Callable[[], Dict[str, Union[pd.DataFrame, np.ndarray]]],
    config: Optional[FederatedConfig] = None,
    framework: str = "auto",
    model_save_path: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """
    Train a model using federated learning with Flower.

    This function sets up a federated learning simulation where the model
    is trained across multiple clients without centralizing the data.

    Args:
        model: The model architecture to train (must be compatible with chosen framework)
        client_data_fn: A function that returns a dictionary mapping client IDs to 
                       their local datasets
        config: Configuration for federated learning
        framework: ML framework to use ('pytorch', 'tensorflow', or 'auto' to detect)
        model_save_path: Path to save the final federated model
        **kwargs: Additional parameters passed to client and server setup functions

    Returns:
        The trained federated model

    Raises:
        ImportError: If Flower or required ML framework is not installed
        ValueError: If the framework is not supported or cannot be detected
    """
    if not _HAS_FLOWER:
        raise ImportError(
            "Flower is not installed. Please install it with 'pip install flwr'."
        )

    # Set default configuration if not provided
    if config is None:
        config = FederatedConfig()

    # Determine which framework to use
    if framework == "auto":
        framework = _detect_framework(model)

    # Get the client data
    client_datasets = client_data_fn()

    # Client and server builder functions
    if framework == "pytorch":
        if not _HAS_PYTORCH:
            raise ImportError(
                "PyTorch is required for PyTorch models. "
                "Please install it with 'pip install torch'."
            )
        client_fn = _create_pytorch_client_fn(model, client_datasets, config, **kwargs)
        server = _create_pytorch_server(model, config, **kwargs)

    elif framework == "tensorflow":
        if not _HAS_TENSORFLOW:
            raise ImportError(
                "TensorFlow is required for TensorFlow models. "
                "Please install it with 'pip install tensorflow'."
            )
        client_fn = _create_tensorflow_client_fn(model, client_datasets, config, **kwargs)
        server = _create_tensorflow_server(model, config, **kwargs)

    else:
        raise ValueError(
            f"Unsupported framework: {framework}. "
            f"Supported frameworks are 'pytorch' and 'tensorflow'."
        )

    # Start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(client_datasets),
        server=server,
        config=fl.server.ServerConfig(num_rounds=config.num_rounds),
    )

    # Save the trained model if path is provided
    if model_save_path is not None:
        _save_model(model, model_save_path, framework)

    return model


def start_federated_server(
    model: Any,
    config: Optional[FederatedConfig] = None,
    framework: str = "auto",
    **kwargs: Any,
) -> None:
    """
    Start a Flower federated learning server.

    This function starts a server that coordinates the federated learning process
    among connected clients.

    Args:
        model: The initial model architecture to distribute
        config: Configuration for federated learning
        framework: ML framework to use ('pytorch', 'tensorflow', or 'auto' to detect)
        **kwargs: Additional parameters for specific server configurations

    Raises:
        ImportError: If Flower or required ML framework is not installed
        ValueError: If the framework is not supported or cannot be detected
    """
    if not _HAS_FLOWER:
        raise ImportError(
            "Flower is not installed. Please install it with 'pip install flwr'."
        )

    # Set default configuration if not provided
    if config is None:
        config = FederatedConfig()

    # Determine which framework to use
    if framework == "auto":
        framework = _detect_framework(model)

    # Create the appropriate server
    if framework == "pytorch":
        if not _HAS_PYTORCH:
            raise ImportError(
                "PyTorch is required for PyTorch models. "
                "Please install it with 'pip install torch'."
            )
        server = _create_pytorch_server(model, config, **kwargs)

    elif framework == "tensorflow":
        if not _HAS_TENSORFLOW:
            raise ImportError(
                "TensorFlow is required for TensorFlow models. "
                "Please install it with 'pip install tensorflow'."
            )
        server = _create_tensorflow_server(model, config, **kwargs)

    else:
        raise ValueError(
            f"Unsupported framework: {framework}. "
            f"Supported frameworks are 'pytorch' and 'tensorflow'."
        )

    # Start the server
    fl.server.start_server(
        server_address=config.server_address,
        server=server,
        config=fl.server.ServerConfig(
            num_rounds=config.num_rounds
        )
    )


def start_federated_client(
    model: Any,
    data: Union[pd.DataFrame, np.ndarray],
    server_address: str,
    framework: str = "auto",
    apply_differential_privacy: bool = False,
    epsilon: float = 1.0,
    delta: float = 1e-5,
    **kwargs: Any,
) -> None:
    """
    Start a Flower federated learning client.

    This function starts a client that participates in the federated learning process
    by training the model on local data and sending the updates to the server.

    Args:
        model: The model architecture to train locally
        data: The local training data
        server_address: Address of the federated learning server (host:port)
        framework: ML framework to use ('pytorch', 'tensorflow', or 'auto' to detect)
        apply_differential_privacy: Whether to apply differential privacy to local updates
        epsilon: Privacy budget for differential privacy (if enabled)
        delta: Privacy failure probability for differential privacy (if enabled)
        **kwargs: Additional parameters for specific client configurations

    Raises:
        ImportError: If Flower or required ML framework is not installed
        ValueError: If the framework is not supported or cannot be detected
    """
    if not _HAS_FLOWER:
        raise ImportError(
            "Flower is not installed. Please install it with 'pip install flwr'."
        )

    # Determine which framework to use
    if framework == "auto":
        framework = _detect_framework(model)

    # Create and start the appropriate client
    if framework == "pytorch":
        if not _HAS_PYTORCH:
            raise ImportError(
                "PyTorch is required for PyTorch models. "
                "Please install it with 'pip install torch'."
            )
        client = _create_pytorch_client(
            model, 
            data, 
            apply_differential_privacy=apply_differential_privacy,
            epsilon=epsilon,
            delta=delta,
            **kwargs
        )

    elif framework == "tensorflow":
        if not _HAS_TENSORFLOW:
            raise ImportError(
                "TensorFlow is required for TensorFlow models. "
                "Please install it with 'pip install tensorflow'."
            )
        client = _create_tensorflow_client(
            model, 
            data, 
            apply_differential_privacy=apply_differential_privacy,
            epsilon=epsilon,
            delta=delta,
            **kwargs
        )
    else:
        raise ValueError(
            f"Unsupported framework: {framework}. "
            f"Supported frameworks are 'pytorch' and 'tensorflow'."
        )

    # Start the client
    fl.client.start_client(server_address=server_address, client=client)


def _detect_framework(model: Any) -> str:
    """
    Detect which ML framework the model is built with.

    Args:
        model: The model to detect the framework for

    Returns:
        The detected framework ('pytorch' or 'tensorflow')

    Raises:
        ValueError: If the framework cannot be detected
    """
    # Check for PyTorch
    if _HAS_PYTORCH and isinstance(model, torch.nn.Module):
        return "pytorch"

    # Check for TensorFlow
    if _HAS_TENSORFLOW and (
        isinstance(model, tf.keras.Model) or isinstance(model, tf.Module)
    ):
        return "tensorflow"

    # If we got here, we couldn't detect the framework
    raise ValueError(
        "Could not detect the model framework. "
        "Please specify the framework manually."
    )


def _create_pytorch_client(
    model: Any,
    data: Union[pd.DataFrame, np.ndarray],
    apply_differential_privacy: bool = False,
    epsilon: float = 1.0,
    delta: float = 1e-5,
    **kwargs: Any,
) -> fl.client.Client:
    """
    Create a Flower client for PyTorch models.
    
    Args:
        model: PyTorch model
        data: Local training data
        apply_differential_privacy: Whether to apply DP to local training
        epsilon: Privacy budget
        delta: Privacy delta
        **kwargs: Additional parameters
        
    Returns:
        A Flower client instance
    """
    if not _HAS_PYTORCH:
        raise ImportError("PyTorch is required but not installed.")

    try:
        from flwr.client import NumPyClient
    except ImportError:
        raise ImportError("Flower is required but not installed.")

    # Split data into train/test if needed
    train_data = data
    test_data = None
    if "test_data" in kwargs:
        test_data = kwargs["test_data"]
    elif kwargs.get("test_split", 0.0) > 0:
        # Simple split - in practice you'd want to use proper stratification
        split_idx = int(len(data) * (1 - kwargs.get("test_split", 0.0)))
        if isinstance(data, pd.DataFrame):
            train_data = data.iloc[:split_idx]
            test_data = data.iloc[split_idx:]
        else:  # numpy array
            train_data = data[:split_idx]
            test_data = data[split_idx:]

    # Convert data to PyTorch format if needed
    x_train, y_train = _prepare_data_for_pytorch(train_data, **kwargs)
    x_test, y_test = (None, None) if test_data is None else _prepare_data_for_pytorch(test_data, **kwargs)

    # Setup for differential privacy if requested
    if apply_differential_privacy:
        try:
            from opacus import PrivacyEngine
            from opacus.validators import ModuleValidator
        except ImportError:
            raise ImportError(
                "Opacus is required for differential privacy with PyTorch. "
                "Please install it with 'pip install opacus'."
            )

        if not ModuleValidator.is_valid(model):
            model = ModuleValidator.fix(model)

    # Create a PyTorch DataLoader
    batch_size = kwargs.get("batch_size", 32)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        ),
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = None
    if x_test is not None and y_test is not None:
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.tensor(x_test, dtype=torch.float32),
                torch.tensor(y_test, dtype=torch.long)
            ),
            batch_size=batch_size,
            shuffle=False
        )

    # Create optimizer
    optimizer_class = kwargs.get("optimizer", torch.optim.Adam)
    optimizer = optimizer_class(model.parameters(), lr=kwargs.get("learning_rate", 0.001))
    
    # Setup loss function
    loss_fn = kwargs.get("loss_fn", torch.nn.CrossEntropyLoss())

    # Apply DP if requested
    if apply_differential_privacy:
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            epochs=kwargs.get("epochs", 1),
            target_epsilon=epsilon,
            target_delta=delta,
            max_grad_norm=kwargs.get("max_grad_norm", 1.0),
        )

    # Create the NumPyClient
    class PytorchClient(NumPyClient):
        def get_parameters(self, config):
            """Get model parameters as a list of NumPy arrays."""
            return [val.cpu().numpy() for _, val in model.state_dict().items()]

        def set_parameters(self, parameters):
            """Set model parameters from a list of NumPy arrays."""
            params_dict = zip(model.state_dict().keys(), parameters)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            model.load_state_dict(state_dict, strict=True)

        def fit(self, parameters, config):
            """Train the model on the local dataset."""
            self.set_parameters(parameters)
            
            # Training
            model.train()
            local_epochs = config.get("epochs", 1)
            
            for _ in range(local_epochs):
                for inputs, targets in train_loader:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = loss_fn(outputs, targets)
                    loss.backward()
                    optimizer.step()

            return self.get_parameters(config={}), len(train_loader.dataset), {}

        def evaluate(self, parameters, config):
            """Evaluate the model on the local test dataset."""
            self.set_parameters(parameters)
            
            if test_loader is None:
                # If no test data, return minimal metrics
                return 0.0, len(train_loader.dataset), {"accuracy": 0.0}
            
            # Evaluation
            model.eval()
            loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, targets in test_loader:
                    outputs = model(inputs)
                    loss += loss_fn(outputs, targets).item()
                    _, predicted = torch.max(outputs, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()

            accuracy = correct / total if total > 0 else 0.0
            avg_loss = loss / len(test_loader) if len(test_loader) > 0 else 0.0
            
            return avg_loss, total, {"accuracy": accuracy}

    # Create and return Flower client
    return PytorchClient().to_client()


def _create_tensorflow_client(
    model: Any,
    data: Union[pd.DataFrame, np.ndarray],
    apply_differential_privacy: bool = False,
    epsilon: float = 1.0,
    delta: float = 1e-5,
    **kwargs: Any,
) -> fl.client.Client:
    """
    Create a Flower client for TensorFlow models.
    
    Args:
        model: TensorFlow model
        data: Local training data
        apply_differential_privacy: Whether to apply DP to local training
        epsilon: Privacy budget
        delta: Privacy delta
        **kwargs: Additional parameters
        
    Returns:
        A Flower client instance
    """
    if not _HAS_TENSORFLOW:
        raise ImportError("TensorFlow is required but not installed.")

    try:
        from flwr.client import NumPyClient
    except ImportError:
        raise ImportError("Flower is required but not installed.")

    # Split data into train/test if needed
    train_data = data
    test_data = None
    if "test_data" in kwargs:
        test_data = kwargs["test_data"]
    elif kwargs.get("test_split", 0.0) > 0:
        # Simple split - in practice you'd want to use proper stratification
        split_idx = int(len(data) * (1 - kwargs.get("test_split", 0.0)))
        if isinstance(data, pd.DataFrame):
            train_data = data.iloc[:split_idx]
            test_data = data.iloc[split_idx:]
        else:  # numpy array
            train_data = data[:split_idx]
            test_data = data[split_idx:]

    # Prepare data
    x_train, y_train = _prepare_data_for_tensorflow(train_data, **kwargs)
    x_test, y_test = (None, None) if test_data is None else _prepare_data_for_tensorflow(test_data, **kwargs)

    # For TensorFlow, we can convert to tf.data.Dataset
    batch_size = kwargs.get("batch_size", 32)
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
    
    test_dataset = None
    if x_test is not None and y_test is not None:
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    # Setup for differential privacy if requested
    if apply_differential_privacy:
        try:
            import tensorflow_privacy
            from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPKerasAdamOptimizer
        except ImportError:
            raise ImportError(
                "TensorFlow Privacy is required for differential privacy with TensorFlow. "
                "Please install it with 'pip install tensorflow-privacy'."
            )

        # DP optimizer
        optimizer = DPKerasAdamOptimizer(
            l2_norm_clip=kwargs.get("max_grad_norm", 1.0),
            noise_multiplier=kwargs.get("noise_multiplier", 1.0),
            num_microbatches=kwargs.get("microbatches", 1),
            learning_rate=kwargs.get("learning_rate", 0.001),
        )
    else:
        # Regular optimizer
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=kwargs.get("learning_rate", 0.001)
        )

    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss=kwargs.get("loss", "sparse_categorical_crossentropy"),
        metrics=kwargs.get("metrics", ["accuracy"]),
    )

    # Create the NumPyClient
    class TensorFlowClient(NumPyClient):
        def get_parameters(self, config):
            """Get model parameters as a list of NumPy arrays."""
            return [v.numpy() for v in model.weights]

        def set_parameters(self, parameters):
            """Set model parameters from a list of NumPy arrays."""
            # This is a simplified approach - in real code you might need more
            # sophisticated weight updating
            for i, w in enumerate(model.weights):
                w.assign(parameters[i])

        def fit(self, parameters, config):
            """Train the model on the local dataset."""
            self.set_parameters(parameters)
            
            # Training
            local_epochs = config.get("epochs", 1)
            history = model.fit(
                train_dataset,
                epochs=local_epochs,
                verbose=kwargs.get("verbose", 0),
            )
            
            # Return updated weights, sample size, and metrics
            return self.get_parameters(config={}), len(x_train), {
                "loss": history.history.get("loss", [0])[-1]
            }

        def evaluate(self, parameters, config):
            """Evaluate the model on the local test dataset."""
            self.set_parameters(parameters)
            
            if test_dataset is None:
                # If no test data, return minimal metrics
                return 0.0, len(x_train), {"accuracy": 0.0}
            
            # Evaluation
            loss, accuracy = model.evaluate(test_dataset, verbose=kwargs.get("verbose", 0))
            
            return loss, len(x_test), {"accuracy": accuracy}

    # Create and return Flower client
    return TensorFlowClient().to_client()


def _create_pytorch_client_fn(
    model: Any,
    client_datasets: Dict[str, Union[pd.DataFrame, np.ndarray]],
    config: FederatedConfig,
    **kwargs: Any,
) -> Callable[[str], fl.client.Client]:
    """
    Create a function that returns PyTorch clients for simulation.
    
    Args:
        model: The PyTorch model architecture (will be copied for each client)
        client_datasets: Dictionary mapping client IDs to their datasets
        config: Federated learning configuration
        **kwargs: Additional parameters
        
    Returns:
        A function that creates clients based on client ID
    """
    import copy

    def client_fn(cid: str) -> fl.client.Client:
        # Create a copy of the model for this client
        client_model = copy.deepcopy(model)
        
        # Get this client's dataset
        data = client_datasets.get(cid, next(iter(client_datasets.values())))
        
        # Create the client
        return _create_pytorch_client(
            model=client_model,
            data=data,
            apply_differential_privacy=config.apply_differential_privacy,
            epsilon=config.epsilon,
            delta=config.delta,
            **kwargs
        )
    
    return client_fn


def _create_tensorflow_client_fn(
    model: Any,
    client_datasets: Dict[str, Union[pd.DataFrame, np.ndarray]],
    config: FederatedConfig,
    **kwargs: Any,
) -> Callable[[str], fl.client.Client]:
    """
    Create a function that returns TensorFlow clients for simulation.
    
    Args:
        model: The TensorFlow model architecture (will be copied for each client)
        client_datasets: Dictionary mapping client IDs to their datasets
        config: Federated learning configuration
        **kwargs: Additional parameters
        
    Returns:
        A function that creates clients based on client ID
    """
    import copy

    def client_fn(cid: str) -> fl.client.Client:
        # Create a copy of the model for this client
        client_model = tf.keras.models.clone_model(model)
        client_model.set_weights(model.get_weights())
        
        # Get this client's dataset
        data = client_datasets.get(cid, next(iter(client_datasets.values())))
        
        # Create the client
        return _create_tensorflow_client(
            model=client_model,
            data=data,
            apply_differential_privacy=config.apply_differential_privacy,
            epsilon=config.epsilon,
            delta=config.delta,
            **kwargs
        )
    
    return client_fn


def _create_pytorch_server(
    model: Any,
    config: FederatedConfig,
    **kwargs: Any,
) -> fl.server.Server:
    """
    Create a Flower server for coordinating federated learning with PyTorch models.
    
    Args:
        model: Initial PyTorch model
        config: Federated learning configuration
        **kwargs: Additional parameters
        
    Returns:
        A configured Flower server
    """

    if not _HAS_FLOWER:  # Add missing Flower check
        raise ImportError("Flower is required but not installed.")

    # Make sure PyTorch is available
    if not _HAS_PYTORCH:
        raise ImportError("PyTorch is required but not installed.")
    
    # Define model weights as strategy parameters conversion function
    def get_weights(model):
        return [val.cpu().numpy() for _, val in model.state_dict().items()]
        
    def set_weights(model, weights):
        params_dict = zip(model.state_dict().keys(), weights)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)
        return model
    
    # Define strategy for server
    if config.use_secure_aggregation:
        from flwr.server.strategy import FedAvgWithSecAgg
        
        strategy = FedAvgWithSecAgg(
            initial_parameters=fl.common.ndarrays_to_parameters(get_weights(model)),
            fraction_fit=config.fraction_fit,
            min_fit_clients=config.min_fit_clients,
            min_available_clients=config.min_available_clients,
            eval_fn=kwargs.get("eval_fn", None),
            **kwargs.get("strategy_kwargs", {})
        )
    else:
        from flwr.server.strategy import FedAvg
        
        strategy = FedAvg(
            initial_parameters=fl.common.ndarrays_to_parameters(get_weights(model)),
            fraction_fit=config.fraction_fit,
            min_fit_clients=config.min_fit_clients,
            min_available_clients=config.min_available_clients,
            eval_fn=kwargs.get("eval_fn", None),
            **kwargs.get("strategy_kwargs", {})
        )
    from flwr.server.client_manager import SimpleClientManager
    
    # Create and return server
    return fl.server.Server(strategy=strategy, client_manager=SimpleClientManager())


def _create_tensorflow_server(
    model: Any,
    config: FederatedConfig,
    **kwargs: Any,
) -> fl.server.Server:
    """
    Create a Flower server for coordinating federated learning with TensorFlow models.
    
    Args:
        model: Initial TensorFlow model
        config: Federated learning configuration
        **kwargs: Additional parameters
        
    Returns:
        A configured Flower server
    """
    # Make sure TensorFlow is available
    if not _HAS_TENSORFLOW:
        raise ImportError("TensorFlow is required but not installed.")
    
    # Define model weights as strategy parameters conversion function
    def get_weights(model):
        return [w.numpy() for w in model.weights]
    
    # Define strategy for server
    if config.use_secure_aggregation:
        from flwr.server.strategy import FedAvgWithSecAgg
        
        strategy = FedAvgWithSecAgg(
            initial_parameters=fl.common.ndarrays_to_parameters(get_weights(model)),
            fraction_fit=config.fraction_fit,
            min_fit_clients=config.min_fit_clients,
            min_available_clients=config.min_available_clients,
            eval_fn=kwargs.get("eval_fn", None),
            **kwargs.get("strategy_kwargs", {})
        )
    else:
        from flwr.server.strategy import FedAvg
        
        strategy = FedAvg(
            initial_parameters=fl.common.ndarrays_to_parameters(get_weights(model)),
            fraction_fit=config.fraction_fit,
            min_fit_clients=config.min_fit_clients,
            min_available_clients=config.min_available_clients,
            eval_fn=kwargs.get("eval_fn", None),
            **kwargs.get("strategy_kwargs", {})
        )
    
    # Create and return server
    from flwr.server.client_manager import SimpleClientManager
    return fl.server.Server(strategy=strategy, client_manager=SimpleClientManager())


def _prepare_data_for_pytorch(
    data: Union[pd.DataFrame, np.ndarray], **kwargs: Any
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data for use with PyTorch models.
    
    Args:
        data: Input data (DataFrame or numpy array)
        **kwargs: Additional parameters for data preparation
        
    Returns:
        Tuple of (features, labels) as numpy arrays
    """
    # If data is a DataFrame
    if isinstance(data, pd.DataFrame):
        # Check if target column is specified
        target_col = kwargs.get("target_column")
        
        if target_col is not None and target_col in data.columns:
            # Extract features and target
            x = data.drop(columns=[target_col]).values
            y = data[target_col].values
        else:
            # Assume last column is the target
            x = data.iloc[:, :-1].values
            y = data.iloc[:, -1].values
    else:
        # If data is a numpy array, assume last column is the target
        x = data[:, :-1]
        y = data[:, -1]
    
    return x, y


def _prepare_data_for_tensorflow(
    data: Union[pd.DataFrame, np.ndarray], **kwargs: Any
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data for use with TensorFlow models.
    
    Args:
        data: Input data (DataFrame or numpy array)
        **kwargs: Additional parameters for data preparation
        
    Returns:
        Tuple of (features, labels) as numpy arrays
    """
    # For TensorFlow, use the same preparation as for PyTorch
    return _prepare_data_for_pytorch(data, **kwargs)


def _save_model(model: Any, path: str, framework: str) -> None:
    """
    Save a trained model to disk.
    
    Args:
        model: The model to save
        path: Path where to save the model
        framework: Framework of the model ('pytorch' or 'tensorflow')
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    
    if framework == "pytorch":
        if not _HAS_PYTORCH:
            warnings.warn("PyTorch is not installed, cannot save model.")
            return
        torch.save(model.state_dict(), path)
    
    elif framework == "tensorflow":
        if not _HAS_TENSORFLOW:
            warnings.warn("TensorFlow is not installed, cannot save model.")
            return
        model.save(path)
    
    else:
        warnings.warn(f"Unknown framework {framework}, model not saved.") 