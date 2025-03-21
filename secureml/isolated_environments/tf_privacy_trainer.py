"""
TensorFlow Privacy trainer functionality to be run in the isolated environment.

This module contains functions that will run in the isolated tensorflow-privacy environment.
"""

from typing import Any, Dict, Union

import numpy as np
import pandas as pd


def train(
    model_config: Dict[str, Any],
    data: Dict[str, Any],
    epsilon: float,
    delta: float,
    noise_multiplier: Union[float, None],
    max_grad_norm: float,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Train a TensorFlow model with differential privacy using TensorFlow Privacy.
    
    This function runs in the isolated environment with tensorflow-privacy installed.
    
    Args:
        model_config: The serialized TensorFlow model configuration
        data: The serialized training data
        epsilon: Privacy budget
        delta: Privacy delta
        noise_multiplier: Noise multiplier for privacy
        max_grad_norm: Maximum gradient norm for clipping
        **kwargs: Additional training parameters
        
    Returns:
        A dictionary with the trained model weights and training metrics
    """
    import tensorflow as tf
    from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPGradientDescentGaussianOptimizer
    
    # Deserialize the data
    if "array" in data:
        # It was a numpy array
        train_data = np.array(data["array"])
    else:
        # It was a pandas DataFrame
        train_data = pd.DataFrame(data)
        train_data = tf.convert_to_tensor(train_data.values, dtype=tf.float32)
    
    # Recreate the model from its config
    # For a simple model:
    try:
        model = tf.keras.models.model_from_config(model_config)
    except:
        # For a more complex model, you might need additional logic:
        # This is just a placeholder for demonstration
        input_shape = train_data.shape[1]
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
    
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
    
    # Prepare the target data if provided
    target_data = kwargs.get("target_data")
    if target_data is not None:
        if isinstance(target_data, dict) and "array" in target_data:
            target_data = np.array(target_data["array"])
        
        # Train the model with separate features and targets
        history = model.fit(
            train_data,
            target_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=kwargs.get("verbose", 1),
        )
    else:
        # Train the model with the data as is (for simplicity)
        history = model.fit(
            train_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=kwargs.get("verbose", 1),
        )
    
    # Return the model weights and training history
    return {
        "weights": [w.tolist() for w in model.get_weights()],
        "history": {k: (v if isinstance(v, (int, float)) else v.tolist()) 
                    for k, v in history.history.items()}
    } 