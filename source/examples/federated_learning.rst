Federated Learning Examples
========================

This section demonstrates how to use SecureML's federated learning capabilities to train models collaboratively across multiple data sources without sharing the raw data.

Basic Federated Learning with PyTorch
------------------------------------

The following example shows how to train a PyTorch model using federated learning with simulated clients:

.. code-block:: python

    import torch
    import torch.nn as nn
    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.preprocessing import StandardScaler
    from secureml.federated import train_federated, FederatedConfig
    
    # Create synthetic dataset
    X, y = make_classification(
        n_samples=1000, 
        n_features=20, 
        n_informative=10, 
        n_classes=2, 
        random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Define a simple PyTorch model
    class BinaryClassifier(nn.Module):
        def __init__(self, input_dim):
            super(BinaryClassifier, self).__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
            
        def forward(self, x):
            return self.layers(x)
    
    # Create a model instance
    input_dim = X_scaled.shape[1]
    model = BinaryClassifier(input_dim)
    
    # Simulate clients by splitting the data
    def get_client_data():
        # Split data into 3 parts to simulate 3 clients
        splits = np.array_split(np.arange(len(X_scaled)), 3)
        
        # Create client datasets
        client_data = {}
        for i, indices in enumerate(splits):
            # Combine features and target into one array for each client
            client_X = X_scaled[indices]
            client_y = y[indices].reshape(-1, 1)
            client_data[f"client-{i+1}"] = np.hstack((client_X, client_y))
            
        return client_data
    
    # Configure federated learning
    config = FederatedConfig(
        num_rounds=5,
        fraction_fit=1.0,
        min_fit_clients=2,
        min_available_clients=2
    )
    
    # Train the model using federated learning
    trained_model = train_federated(
        model=model,
        client_data_fn=get_client_data,
        config=config,
        framework="pytorch",
        batch_size=32,
        learning_rate=0.01,
        epochs=2  # Local epochs per round
    )

This example demonstrates:

* How to define a PyTorch model for federated learning
* How to split data across multiple simulated clients
* How to configure basic federated learning parameters
* How to train a model without centralizing the data

Federated Learning with TensorFlow
--------------------------------

SecureML also supports TensorFlow models for federated learning:

.. code-block:: python

    import tensorflow as tf
    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.preprocessing import StandardScaler
    from secureml.federated import train_federated, FederatedConfig
    
    # Create synthetic dataset for multi-class classification
    X, y = make_classification(
        n_samples=1000, 
        n_features=20, 
        n_informative=10,
        n_classes=3,  # Multi-class classification
        n_clusters_per_class=2,
        random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Define a TensorFlow model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_scaled.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')  # 3 classes
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Simulate clients by splitting the data
    def get_client_data():
        # Split data into 4 parts to simulate 4 clients
        splits = np.array_split(np.arange(len(X_scaled)), 4)
        
        # Create client datasets
        client_data = {}
        for i, indices in enumerate(splits):
            # Get client data subset
            client_X = X_scaled[indices]
            client_labels = y[indices].reshape(-1, 1)
            client_data[f"client-{i+1}"] = np.hstack((client_X, client_labels))
            
        return client_data
    
    # Configure federated learning
    config = FederatedConfig(
        num_rounds=5,
        fraction_fit=0.75,  # Use 75% of clients per round
        min_fit_clients=2,
        min_available_clients=3
    )
    
    # Train the model using federated learning
    trained_model = train_federated(
        model=model,
        client_data_fn=get_client_data,
        config=config,
        framework="tensorflow",
        batch_size=32,
        epochs=3  # Local epochs per round
    )

Privacy-Preserving Federated Learning
-----------------------------------

For enhanced privacy protection, you can combine federated learning with differential privacy and secure aggregation:

.. code-block:: python

    from secureml.federated import train_federated, FederatedConfig
    
    # Configure federated learning with privacy features
    config = FederatedConfig(
        num_rounds=8,
        fraction_fit=0.8,
        min_fit_clients=3,
        min_available_clients=4,
        use_secure_aggregation=True,  # Enable secure aggregation
        apply_differential_privacy=True,  # Enable differential privacy
        epsilon=2.0,  # Privacy budget
        delta=1e-5,  # Privacy failure probability
        # Advanced weight update configuration
        weight_update_strategy="ema",  # Use exponential moving average
        weight_mixing_rate=0.5,  # Mix 50% of new and old weights
        warmup_rounds=2  # Gradually increase mixing rate over first 2 rounds
    )
    
    # Train the model with privacy features
    trained_model = train_federated(
        model=model,
        client_data_fn=get_client_data,
        config=config,
        framework="pytorch",
        batch_size=64,
        learning_rate=0.005,
        epochs=1,  # Use fewer local epochs when applying DP
        max_grad_norm=1.0  # Clipping parameter for differential privacy
    )

This approach:

* Prevents the server from inspecting individual updates using secure aggregation
* Adds carefully calibrated noise to model updates for differential privacy guarantees
* Uses a sophisticated weight update strategy to handle the additional noise
* Provides formal privacy guarantees via the epsilon and delta parameters

Comparing Weight Update Strategies
--------------------------------

SecureML offers different weight update strategies to improve federated learning with heterogeneous data. This example compares these strategies on a regression task:

.. code-block:: python

    from secureml.federated import train_federated, FederatedConfig
    import matplotlib.pyplot as plt
    
    # Function to train and evaluate model with a specific strategy
    def train_with_strategy(strategy, mixing_rate=0.5, momentum=0.9, constraints=False):
        # Create a new model instance
        model = RegressionModel(input_dim)
        
        # Configure with the specific strategy
        config = FederatedConfig(
            num_rounds=10,
            weight_update_strategy=strategy,
            weight_mixing_rate=mixing_rate,
            weight_momentum=momentum,
            apply_weight_constraints=constraints,
            max_weight_change=0.2 if constraints else None
        )
        
        # Train the model
        trained_model = train_federated(
            model=model,
            client_data_fn=get_client_data_non_iid,  # Non-IID data distribution
            config=config,
            framework="pytorch"
        )
        
        # Evaluate and return results
        # ...
        
        return mse, trained_model
    
    # Compare different strategies
    strategies = {
        "direct": {},  # Default direct update
        "ema": {"mixing_rate": 0.3},  # Exponential moving average
        "momentum": {"mixing_rate": 0.1, "momentum": 0.9, "constraints": False},  # Momentum
        "momentum_constrained": {"mixing_rate": 0.1, "momentum": 0.9, "constraints": True}  # With constraints
    }
    
    results = {}
    for name, params in strategies.items():
        mse, model = train_with_strategy(
            strategy=name.split("_")[0],  # Extract base strategy name
            **params
        )
        results[name] = mse

Available strategies include:

1. **Direct Updates**: The simplest approach where clients directly apply weight updates.

2. **Exponential Moving Average (EMA)**: Creates a weighted average between old and new weights:

   .. code-block:: text

      updated_weight = (1 - mixing_rate) * old_weight + mixing_rate * new_weight

3. **Momentum-Based Updates**: Uses momentum for smoother, more effective updates:

   .. code-block:: text

      momentum_update = momentum * previous_update + mixing_rate * (new_weight - old_weight)
      updated_weight = old_weight + momentum_update

4. **Constrained Updates**: Limits the maximum change in any weight to prevent instability.

Client-Server Deployment
----------------------

For real-world deployment, you can set up a federated learning server and clients:

**Server Side:**

.. code-block:: python

    import torch.nn as nn
    from secureml.federated import start_federated_server, FederatedConfig
    
    # Define your model architecture
    class MedicalDiagnosisModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(20, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 5)  # 5 diagnostic categories
            )
            
        def forward(self, x):
            return self.layers(x)
    
    # Create a model instance
    model = MedicalDiagnosisModel()
    
    # Configure the federated learning
    config = FederatedConfig(
        num_rounds=30,
        fraction_fit=0.8,
        min_fit_clients=5,
        min_available_clients=8,
        server_address="0.0.0.0:8080",  # Listen on all interfaces
        use_secure_aggregation=True,
        apply_differential_privacy=True,
        epsilon=1.0,
        delta=1e-5,
        weight_update_strategy="momentum",
        weight_mixing_rate=0.1,
        weight_momentum=0.9,
        apply_weight_constraints=True,
        max_weight_change=0.15
    )
    
    # Start the server
    start_federated_server(
        model=model,
        config=config,
        framework="pytorch"
    )

**Client Side:**

.. code-block:: python

    import torch.nn as nn
    import pandas as pd
    from secureml.federated import start_federated_client
    
    # Define the same model architecture as the server
    class MedicalDiagnosisModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(20, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 5)  # 5 diagnostic categories
            )
            
        def forward(self, x):
            return self.layers(x)
    
    # Create a model instance
    model = MedicalDiagnosisModel()
    
    # Load local data
    local_data = pd.read_csv("hospital_data.csv")
    
    # Start the client
    start_federated_client(
        model=model,
        data=local_data,
        server_address="fl-server.example.com:8080",
        framework="pytorch",
        apply_differential_privacy=True,
        epsilon=1.0,
        delta=1e-5,
        test_split=0.2,  # Use 20% of data for local evaluation
        batch_size=32,
        learning_rate=0.001,
        target_column="diagnosis"  # Specify the target column in the DataFrame
    )

In a real deployment:

1. The server runs on a dedicated machine or cloud instance
2. Each client is a separate organization with its own private data
3. Clients connect to the server to participate in training rounds
4. The raw data never leaves the client's environment

Best Practices
------------

When using federated learning, consider these best practices:

1. **Match model architectures**: Ensure server and client models have identical architectures
2. **Start with simulations**: Test your setup using simulated clients before deployment
3. **Privacy protection**: Combine federated learning with differential privacy for maximum privacy
4. **Non-IID data handling**: Use momentum or EMA updates for data that is not identically distributed
5. **Batch size selection**: Balance between compute efficiency and update quality
6. **Communication efficiency**: Keep model sizes reasonable to reduce communication overhead
7. **Client selection**: Configure `fraction_fit` appropriately to manage client participation
8. **Privacy budget**: Start with higher epsilon values and reduce them to find the right privacy-utility tradeoff 