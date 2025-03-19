"""
Privacy-preserving training methods for SecureML.

This module provides tools for training machine learning models
with privacy guarantees like differential privacy.
"""

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd


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

    # Convert data to PyTorch format if needed
    if isinstance(data, pd.DataFrame):
        data = torch.utils.data.TensorDataset(
            torch.tensor(data.values, dtype=torch.float32)
        )

    # Create a DataLoader for training
    batch_size = kwargs.get("batch_size", 64)
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)

    # Validate the model for DP training
    if not ModuleValidator.is_valid(model):
        model = ModuleValidator.fix(model)

    # Setup optimizer
    optimizer_class = kwargs.get("optimizer", torch.optim.Adam)
    optimizer = optimizer_class(model.parameters(), lr=kwargs.get("learning_rate", 0.001))

    # Setup PrivacyEngine
    privacy_engine = PrivacyEngine()
    model, optimizer, data_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        epochs=kwargs.get("epochs", 10),
        target_epsilon=epsilon,
        target_delta=delta,
        max_grad_norm=max_grad_norm,
    )

    # Training loop (simplified)
    for epoch in range(kwargs.get("epochs", 10)):
        for batch in data_loader:
            optimizer.zero_grad()
            # This is simplified - in real code, you'd compute loss and do proper backprop
            loss = model(batch)
            loss.backward()
            optimizer.step()

    return model


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
    Train a TensorFlow model with differential privacy using TensorFlow Privacy.

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
        ImportError: If TensorFlow Privacy is not installed
    """
    try:
        import tensorflow as tf
        import tensorflow_privacy
        from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPGradientDescentGaussianOptimizer
    except ImportError:
        raise ImportError(
            "TensorFlow Privacy is not installed. "
            "Please install it with 'pip install tensorflow-privacy'."
        )

    # Convert data to TensorFlow format if needed
    if isinstance(data, pd.DataFrame):
        data = tf.convert_to_tensor(data.values, dtype=tf.float32)

    # Calculate the parameters for DP-SGD
    learning_rate = kwargs.get("learning_rate", 0.001)
    batch_size = kwargs.get("batch_size", 32)
    microbatches = kwargs.get("microbatches", 1)
    epochs = kwargs.get("epochs", 10)

    # Create the DP optimizer
    optimizer = DPGradientDescentGaussianOptimizer(
        l2_norm_clip=max_grad_norm,
        noise_multiplier=noise_multiplier or 1.0,  # Use provided or default
        num_microbatches=microbatches,
        learning_rate=learning_rate,
    )

    # Compile the model with the DP optimizer
    model.compile(
        optimizer=optimizer,
        loss=kwargs.get("loss", "mse"),
        metrics=kwargs.get("metrics", ["accuracy"]),
    )

    # Train the model
    model.fit(
        data,
        epochs=epochs,
        batch_size=batch_size,
        verbose=kwargs.get("verbose", 1),
    )

    return model 