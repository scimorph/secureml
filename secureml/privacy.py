"""
Privacy-preserving training methods for SecureML.

This module provides tools for training machine learning models
with privacy guarantees like differential privacy.
"""

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from secureml.isolated_environments.tf_privacy import run_tf_privacy_function


def differentially_private_train(
    model: Any,
    data: Union[pd.DataFrame, np.ndarray],
    epsilon: float = 1.0,
    delta: float = 1e-5,
    noise_multiplier: Optional[float] = None,
    max_grad_norm: float = 1.0,
    framework: str = "auto",
    **kwargs: Any,
) -> Any:
    """
    Train a model using differential privacy.

    This function wraps various differential privacy implementations to provide
    privacy-preserving training for machine learning models.

    Args:
        model: The model architecture to train (compatible with the chosen framework)
        data: The training data (DataFrame or numpy array)
        epsilon: The privacy budget (smaller values provide stronger privacy guarantees)
        delta: The privacy delta parameter (smaller values provide stronger privacy)
        noise_multiplier: Manually set the noise multiplier instead of epsilon/delta
        max_grad_norm: Maximum norm of gradients for clipping
        framework: ML framework to use ('pytorch', 'tensorflow', or 'auto' to detect)
        **kwargs: Additional parameters passed to the underlying training function

    Returns:
        The trained model

    Raises:
        ValueError: If the framework is not supported or cannot be detected
        ImportError: If the required dependencies are not installed
    """
    # Determine which framework to use
    if framework == "auto":
        framework = _detect_framework(model)

    # Apply differential privacy based on the framework
    if framework == "pytorch":
        return _train_with_opacus(
            model, data, epsilon, delta, noise_multiplier, max_grad_norm, **kwargs
        )
    elif framework == "tensorflow":
        return _train_with_tf_privacy(
            model, data, epsilon, delta, noise_multiplier, max_grad_norm, **kwargs
        )
    else:
        raise ValueError(
            f"Unsupported framework: {framework}. "
            f"Supported frameworks are 'pytorch' and 'tensorflow'."
        )


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
    try:
        import torch
        if isinstance(model, torch.nn.Module):
            return "pytorch"
    except ImportError:
        pass

    # Check for TensorFlow
    try:
        import tensorflow as tf
        if isinstance(model, tf.keras.Model) or isinstance(model, tf.Module):
            return "tensorflow"
    except ImportError:
        pass

    # If we got here, we couldn't detect the framework
    raise ValueError(
        "Could not detect the model framework. "
        "Please specify the framework manually."
    )


def _train_with_opacus(
    model: Any,
    data: Union[pd.DataFrame, np.ndarray],
    epsilon: float,
    delta: float,
    noise_multiplier: Optional[float],
    max_grad_norm: float,
    **kwargs: Any,
) -> Any:
    """
    Train a PyTorch model with differential privacy using Opacus.

    Args:
        model: PyTorch model to train
        data: Training data
        epsilon: Privacy budget
        delta: Privacy delta
        noise_multiplier: Noise multiplier for privacy
        max_grad_norm: Maximum gradient norm for clipping
        **kwargs: Additional training parameters

    Returns:
        The trained model

    Raises:
        ImportError: If Opacus or PyTorch is not installed
    """
    try:
        import torch
        from opacus import PrivacyEngine
        from opacus.validators import ModuleValidator
    except ImportError:
        raise ImportError(
            "Opacus or PyTorch is not installed. "
            "Please install it with 'pip install opacus torch'."
        )

    # Extract training parameters from kwargs
    batch_size = kwargs.get("batch_size", 64)
    epochs = kwargs.get("epochs", 10)
    learning_rate = kwargs.get("learning_rate", 0.001)
    validation_split = kwargs.get("validation_split", 0.2)
    shuffle = kwargs.get("shuffle", True)
    verbose = kwargs.get("verbose", True)
    criterion = kwargs.get("criterion", torch.nn.CrossEntropyLoss())
    device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    # Move model to the appropriate device
    model = model.to(device)

    # Convert data to PyTorch format if needed
    x_data, y_data = _prepare_data_for_torch(data, **kwargs)
    
    # Create validation split if requested
    if validation_split > 0 and len(x_data) > 1:
        val_size = int(len(x_data) * validation_split)
        indices = torch.randperm(len(x_data))
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]
        
        x_train, y_train = x_data[train_indices], y_data[train_indices]
        x_val, y_val = x_data[val_indices], y_data[val_indices]
    else:
        x_train, y_train = x_data, y_data
        x_val, y_val = None, None

    # Create a DataLoader for training
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle
    )
    
    # Create validation DataLoader if validation data exists
    val_loader = None
    if x_val is not None and y_val is not None:
        val_dataset = torch.utils.data.TensorDataset(
            torch.tensor(x_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long)
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False
        )

    # Validate the model for DP training
    if not ModuleValidator.is_valid(model):
        model = ModuleValidator.fix(model)

    # Setup optimizer
    optimizer_class = kwargs.get("optimizer", torch.optim.Adam)
    optimizer = optimizer_class(
        model.parameters(), 
        lr=learning_rate,
        **kwargs.get("optimizer_kwargs", {})
    )

    # Setup PrivacyEngine
    privacy_engine = PrivacyEngine()
    
    # Make the model private with either epsilon/delta or noise_multiplier
    if noise_multiplier is not None:
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
        )
    else:
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            epochs=epochs,
            target_epsilon=epsilon,
            target_delta=delta,
            max_grad_norm=max_grad_norm,
        )

    # Training loop
    best_val_loss = float('inf')
    early_stopping_patience = kwargs.get("early_stopping_patience", 0)
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
            # Move tensors to the appropriate device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
        
        # Calculate average training metrics
        train_loss = train_loss / train_total
        train_accuracy = train_correct / train_total
        
        # Validation phase if validation data exists
        val_loss = 0.0
        val_accuracy = 0.0
        
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    # Move tensors to the appropriate device
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    # Forward pass
                    outputs = model(inputs)
                    
                    # Calculate loss
                    loss = criterion(outputs, targets)
                    
                    # Update metrics
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += targets.size(0)
                    val_correct += (predicted == targets).sum().item()
            
            # Calculate average validation metrics
            val_loss = val_loss / val_total
            val_accuracy = val_correct / val_total
            
            # Check for early stopping
            if early_stopping_patience > 0:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save the best model state
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch+1}")
                        # Load the best model state
                        if best_model_state is not None:
                            model.load_state_dict(best_model_state)
                        break
        
        # Print epoch results
        if verbose:
            if val_loader is not None:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Loss: {train_loss:.4f} - Accuracy: {train_accuracy:.4f} - "
                      f"Val Loss: {val_loss:.4f} - Val Accuracy: {val_accuracy:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Loss: {train_loss:.4f} - Accuracy: {train_accuracy:.4f}")
    
    # Get the privacy budget spent
    if hasattr(privacy_engine, "get_epsilon"):
        spent_epsilon = privacy_engine.get_epsilon(delta)
        if verbose:
            print(f"Privacy budget spent (ε = {spent_epsilon:.4f}, δ = {delta})")
    
    return model


def _prepare_data_for_torch(data: Union[pd.DataFrame, np.ndarray], **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data for PyTorch model training.
    
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


def _train_with_tf_privacy(
    model: Any,
    data: Union[pd.DataFrame, np.ndarray],
    epsilon: float,
    delta: float,
    noise_multiplier: Optional[float],
    max_grad_norm: float,
    **kwargs: Any,
) -> Any:
    """
    Train a TensorFlow model with differential privacy using TensorFlow Privacy in an isolated environment.

    This function uses a separate virtual environment with tensorflow-privacy installed
    to avoid dependency conflicts with the main project.

    Args:
        model: TensorFlow model to train
        data: Training data
        epsilon: Privacy budget
        delta: Privacy delta
        noise_multiplier: Noise multiplier for privacy
        max_grad_norm: Maximum gradient norm for clipping
        **kwargs: Additional training parameters

    Returns:
        The trained model

    Raises:
        RuntimeError: If there's an error when running the function in the isolated environment
        ImportError: If TensorFlow is not installed in the main environment
    """
    try:
        import tensorflow as tf
        import json
        import tempfile
        import os
    except ImportError:
        raise ImportError(
            "TensorFlow is required but not installed. "
            "Please install it with 'pip install tensorflow'."
        )
    
    # Extract training parameters from kwargs to match the PyTorch implementation
    batch_size = kwargs.get("batch_size", 64)
    epochs = kwargs.get("epochs", 10)
    learning_rate = kwargs.get("learning_rate", 0.001)
    validation_split = kwargs.get("validation_split", 0.2)
    shuffle = kwargs.get("shuffle", True)
    verbose = kwargs.get("verbose", True)
    early_stopping_patience = kwargs.get("early_stopping_patience", 0)
    
    # Create a temporary directory for model and data exchange
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save the model architecture and weights
        model_json_path = os.path.join(temp_dir, "model_architecture.json")
        weights_path = os.path.join(temp_dir, "model_weights.h5")
        final_weights_path = os.path.join(temp_dir, "final_weights.h5")
        
        # Serialize model architecture to JSON
        model_json = model.to_json()
        with open(model_json_path, "w") as json_file:
            json_file.write(model_json)
        
        # Save model weights
        model.save_weights(weights_path)
        
        # Prepare data
        if isinstance(data, pd.DataFrame):
            data_path = os.path.join(temp_dir, "data.csv")
            data.to_csv(data_path, index=False)
            data_type = "dataframe"
        else:  # numpy array
            data_path = os.path.join(temp_dir, "data.npy")
            np.save(data_path, data)
            data_type = "numpy"
        
        # Prepare metadata for target column
        target_column = kwargs.get("target_column")
        metadata = {
            "target_column": target_column,
            "data_type": data_type,
            "model_architecture_path": model_json_path,
            "weights_path": weights_path,
            "final_weights_path": final_weights_path,
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "validation_split": validation_split,
            "shuffle": shuffle,
            "verbose": verbose,
            "early_stopping_patience": early_stopping_patience,
            "optimizer": kwargs.get("optimizer", "adam"),
            "loss": kwargs.get("loss", "sparse_categorical_crossentropy"),
            "metrics": kwargs.get("metrics", ["accuracy"]),
            "epsilon": epsilon,
            "delta": delta,
            "noise_multiplier": noise_multiplier,
            "max_grad_norm": max_grad_norm
        }
        
        # Save metadata
        metadata_path = os.path.join(temp_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)
        
        # Run the training function in the isolated environment
        if verbose:
            print("Starting TensorFlow Privacy training in isolated environment...")
        
        result = run_tf_privacy_function(
            "secureml.isolated_environments.tf_privacy_trainer.train_model",
            {
                "metadata_path": metadata_path,
                "data_path": data_path
            }
        )
        
        # Check if training was successful
        if not result.get("success", False):
            error_message = result.get("error", "Unknown error in isolated environment")
            raise RuntimeError(f"TensorFlow Privacy training failed: {error_message}")
        
        # Load the trained weights back into the original model
        if os.path.exists(final_weights_path):
            model.load_weights(final_weights_path)
            
            # Print training metrics if available and verbose mode is on
            if verbose and "metrics" in result:
                metrics = result["metrics"]
                print("\nTraining Results:")
                for key, value in metrics.items():
                    print(f"  {key}: {value}")
                
                # Print privacy budget spent
                if "privacy_spent" in result:
                    privacy = result["privacy_spent"]
                    print(f"Privacy budget spent (ε = {privacy['epsilon']:.4f}, δ = {privacy['delta']})")
        else:
            raise RuntimeError("Training completed but model weights file not found")
            
    return model 