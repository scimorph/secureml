"""
TensorFlow Privacy training implementation for isolated environment.

This module contains functions for training TensorFlow models with differential privacy
in an isolated environment to avoid dependency conflicts.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Tuple, Union


def train_model(metadata_path: str, data_path: str) -> Dict[str, Any]:
    """
    Train a TensorFlow model with differential privacy.
    
    Args:
        metadata_path: Path to the JSON file containing training metadata
        data_path: Path to the data file (CSV for DataFrame, NPY for NumPy array)
        
    Returns:
        Dictionary containing training results and status
    """
    try:
        # Import TensorFlow and TensorFlow Privacy
        import tensorflow as tf
        from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer
        from tensorflow_privacy.privacy.analysis.compute_dp_sgd_privacy import compute_dp_sgd_privacy
        
        # Load metadata
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        # Extract parameters from metadata
        model_architecture_path = metadata["model_architecture_path"]
        weights_path = metadata["weights_path"]
        final_weights_path = metadata["final_weights_path"]
        data_type = metadata["data_type"]
        target_column = metadata["target_column"]
        batch_size = metadata["batch_size"]
        epochs = metadata["epochs"]
        learning_rate = metadata["learning_rate"]
        validation_split = metadata["validation_split"]
        shuffle = metadata["shuffle"]
        verbose = metadata["verbose"]
        early_stopping_patience = metadata["early_stopping_patience"]
        optimizer_name = metadata["optimizer"]
        loss = metadata["loss"]
        metrics = metadata["metrics"]
        epsilon = metadata["epsilon"]
        delta = metadata["delta"]
        noise_multiplier = metadata["noise_multiplier"]
        max_grad_norm = metadata["max_grad_norm"]
        
        # Load data
        if data_type == "dataframe":
            data = pd.read_csv(data_path)
            # Extract features and target
            if target_column is not None and target_column in data.columns:
                x = data.drop(columns=[target_column]).values
                y = data[target_column].values
            else:
                # Assume last column is the target
                x = data.iloc[:, :-1].values
                y = data.iloc[:, -1].values
        else:  # numpy array
            data = np.load(data_path)
            # Assume last column is the target
            x = data[:, :-1]
            y = data[:, -1]
        
        # Load model architecture
        with open(model_architecture_path, "r") as json_file:
            model_json = json_file.read()
        
        model = tf.keras.models.model_from_json(model_json)
        
        # Load initial weights
        model.load_weights(weights_path)
        
        # Determine the number of training examples
        n_train_samples = x.shape[0]
        
        # Calculate steps per epoch
        steps_per_epoch = n_train_samples // batch_size
        
        # Create validation split if requested
        x_train = x
        y_train = y
        x_val = None
        y_val = None
        validation_data = None
        
        if validation_split > 0 and len(x) > 1:
            # Create validation indices
            indices = np.random.permutation(len(x))
            val_size = int(len(x) * validation_split)
            train_indices = indices[val_size:]
            val_indices = indices[:val_size]
            
            x_train, y_train = x[train_indices], y[train_indices]
            x_val, y_val = x[val_indices], y[val_indices]
            validation_data = (x_val, y_val)
        
        # Setup optimizer with differential privacy
        if noise_multiplier is None:
            # Calculate noise multiplier from epsilon and delta
            from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
            from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent
            
            # Use binary search to find noise multiplier for target epsilon
            def compute_epsilon(n_multiplier):
                """Compute epsilon for given noise multiplier and parameters."""
                orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
                rdp = compute_rdp(
                    q=batch_size / n_train_samples,
                    noise_multiplier=n_multiplier,
                    steps=epochs * steps_per_epoch,
                    orders=orders
                )
                return get_privacy_spent(orders, rdp, target_delta=delta)[0]
            
            # Binary search for noise multiplier
            low, high = 0.1, 10.0
            while high - low > 0.01:
                mid = (low + high) / 2
                current_eps = compute_epsilon(mid)
                if current_eps > epsilon:
                    low = mid
                else:
                    high = mid
            
            noise_multiplier = high
        
        # Create DP optimizer
        if optimizer_name.lower() == "sgd":
            optimizer = DPKerasSGDOptimizer(
                l2_norm_clip=max_grad_norm,
                noise_multiplier=noise_multiplier,
                learning_rate=learning_rate
            )
        else:
            # For other optimizers, use SGD with DP as the default
            optimizer = DPKerasSGDOptimizer(
                l2_norm_clip=max_grad_norm,
                noise_multiplier=noise_multiplier,
                learning_rate=learning_rate
            )
        
        # Compile the model
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        # Setup callbacks
        callbacks = []
        
        # Add early stopping if requested
        if early_stopping_patience > 0 and validation_data is not None:
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True
            )
            callbacks.append(early_stopping)
        
        # Train the model
        history = model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            validation_data=validation_data,
            callbacks=callbacks,
            shuffle=shuffle
        )
        
        # Save the trained weights
        model.save_weights(final_weights_path)
        
        # Calculate final privacy spent
        eps, _ = compute_dp_sgd_privacy(
            n=n_train_samples,
            batch_size=batch_size,
            noise_multiplier=noise_multiplier,
            epochs=epochs,
            delta=delta
        )
        
        # Prepare training metrics to return
        train_metrics = {}
        for key, values in history.history.items():
            train_metrics[key] = float(values[-1])  # Get the last value
        
        return {
            "success": True,
            "metrics": train_metrics,
            "privacy_spent": {
                "epsilon": float(eps),
                "delta": float(delta),
                "noise_multiplier": float(noise_multiplier)
            }
        }
        
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        } 