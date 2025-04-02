===================
Federated Learning
===================

Federated Learning (FL) is a machine learning technique that trains models across multiple decentralized devices or servers holding local data samples, without exchanging the actual data. SecureML provides a robust framework for implementing secure and privacy-preserving federated learning systems.

Core Concepts
------------

**Federated Learning Types**:

* **Cross-device FL**: Learning across many (thousands to millions) mobile or IoT devices
* **Cross-silo FL**: Learning across a small number of organizations or data silos
* **Vertical FL**: Learning when different organizations have different features for the same entities
* **Horizontal FL**: Learning when different organizations have the same features for different entities

**Key Components**:

* **Federated Clients**: Devices or servers that hold local data
* **Federated Server**: Central server that orchestrates the learning process
* **Aggregation Algorithms**: Methods to combine model updates from multiple clients

Basic Usage
----------

Training with Federated Learning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The main way to use federated learning in SecureML is with the `train_federated` function:

.. code-block:: python

    from secureml.federated import train_federated, FederatedConfig
    import torch.nn as nn
    
    # Define your model (PyTorch example)
    class SimpleNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(10, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
            
        def forward(self, x):
            return self.layers(x)
    
    # Create a model
    model = SimpleNN()
    
    # Define a function that returns client data
    def get_client_data():
        # Return a dictionary mapping client IDs to their datasets
        return {
            "client-001": client_1_data,
            "client-002": client_2_data,
            "client-003": client_3_data
        }
    
    # Configure federated learning
    config = FederatedConfig(
        num_rounds=10,
        fraction_fit=1.0,
        min_fit_clients=2,
        min_available_clients=2,
        use_secure_aggregation=True,
        apply_differential_privacy=True,
        epsilon=1.0,
        delta=1e-5,
        weight_update_strategy="ema",
        weight_mixing_rate=0.5
    )
    
    # Train the model with federated learning
    trained_model = train_federated(
        model=model,
        client_data_fn=get_client_data,
        config=config,
        framework="pytorch",  # or "tensorflow", or "auto"
        model_save_path="federated_model.pt"
    )

Setting Up a Federated Server
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For real-world deployments, you can set up a federated learning server:

.. code-block:: python

    from secureml.federated import start_federated_server, FederatedConfig
    import torch.nn as nn
    
    # Define your model (PyTorch example)
    class SimpleNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(10, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
            
        def forward(self, x):
            return self.layers(x)
    
    # Create a model
    model = SimpleNN()
    
    # Configure federated learning
    config = FederatedConfig(
        num_rounds=10,
        fraction_fit=0.8,
        min_fit_clients=3,
        min_available_clients=5,
        server_address="0.0.0.0:8080",
        use_secure_aggregation=True
    )
    
    # Start the federated server
    start_federated_server(
        model=model,
        config=config,
        framework="pytorch"  # or "tensorflow", or "auto"
    )

Setting Up a Federated Client
^^^^^^^^^^^^^^^^^^^^^^^^^^^

On each client device or server:

.. code-block:: python

    from secureml.federated import start_federated_client
    import torch.nn as nn
    import pandas as pd
    
    # Define your model (must match the server's architecture)
    class SimpleNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(10, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
            
        def forward(self, x):
            return self.layers(x)
    
    # Create a model
    model = SimpleNN()
    
    # Load local data (pandas DataFrame or NumPy array)
    local_data = pd.read_csv("client_data.csv")
    
    # Start the federated client
    start_federated_client(
        model=model,
        data=local_data,
        server_address="fl-server.example.com:8080",
        framework="pytorch",  # or "tensorflow", or "auto"
        apply_differential_privacy=True,
        epsilon=1.0,
        delta=1e-5,
        test_split=0.2,  # Optional: Use 20% of data for local evaluation
        batch_size=64,
        learning_rate=0.001
    )

Advanced Techniques
------------------

Secure Aggregation
^^^^^^^^^^^^^^^^

SecureML supports secure aggregation to protect client updates:

.. code-block:: python

    from secureml.federated import FederatedConfig, train_federated
    
    # Configure federated learning with secure aggregation
    config = FederatedConfig(
        num_rounds=10,
        fraction_fit=1.0,
        min_fit_clients=2,
        min_available_clients=2,
        use_secure_aggregation=True  # Enable secure aggregation
    )
    
    # Train with secure aggregation
    trained_model = train_federated(
        model=model,
        client_data_fn=get_client_data,
        config=config
    )

Differential Privacy in Federated Learning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Add differential privacy to client updates:

.. code-block:: python

    from secureml.federated import start_federated_client
    
    # Start a client with differential privacy
    start_federated_client(
        model=model,
        data=local_data,
        server_address="fl-server.example.com:8080",
        apply_differential_privacy=True,
        epsilon=1.0,
        delta=1e-5,
        max_grad_norm=1.0,  # Clipping parameter
        noise_multiplier=1.1  # Noise level (optional)
    )
    
    # Or configure it system-wide
    from secureml.federated import FederatedConfig, train_federated
    
    config = FederatedConfig(
        num_rounds=10,
        apply_differential_privacy=True,
        epsilon=1.0,
        delta=1e-5
    )
    
    trained_model = train_federated(
        model=model,
        client_data_fn=get_client_data,
        config=config
    )

Advanced Weight Update Strategies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

SecureML provides sophisticated weight update mechanisms for federated learning to improve convergence and stability:

.. code-block:: python

    from secureml.federated import FederatedConfig, train_federated
    
    # Configure federated learning with Exponential Moving Average (EMA) weight updates
    ema_config = FederatedConfig(
        num_rounds=10,
        weight_update_strategy="ema",       # Use exponential moving average
        weight_mixing_rate=0.5,             # 50% mix of new weights, 50% of old weights
        warmup_rounds=2                     # Gradually increase mixing rate over first 2 rounds
    )
    
    # Train with EMA updates
    trained_model = train_federated(
        model=model,
        client_data_fn=get_client_data,
        config=ema_config
    )
    
    # Use momentum-based weight updates
    momentum_config = FederatedConfig(
        num_rounds=10,
        weight_update_strategy="momentum",  # Use momentum-based updates
        weight_mixing_rate=0.1,             # Small update step size
        weight_momentum=0.9,                # High momentum coefficient
        apply_weight_constraints=True,      # Constrain updates to prevent instability
        max_weight_change=0.3               # Maximum 30% change in any weight
    )
    
    # Train with momentum updates
    trained_model = train_federated(
        model=model,
        client_data_fn=get_client_data,
        config=momentum_config
    )

Weight Update Strategy Types
''''''''''''''''''''''''''''

SecureML supports three different strategies for updating model weights in federated learning:

1. **Direct Updates** (``strategy="direct"``): The simplest strategy, where client models directly adopt the weights received from the server. This is the classic federated learning approach.

2. **Exponential Moving Average (EMA)** (``strategy="ema"``): A weighted average between old and new weights. This creates smoother updates and can improve training stability:

   .. code-block:: text

      updated_weight = (1 - mixing_rate) * old_weight + mixing_rate * new_weight

3. **Momentum-Based Updates** (``strategy="momentum"``): Uses a momentum term to accelerate training and avoid local minima:

   .. code-block:: text

      momentum_update = momentum * previous_update + mixing_rate * (new_weight - old_weight)
      updated_weight = old_weight + momentum_update

Key Configuration Parameters
''''''''''''''''''''''''''''

- **weight_mixing_rate**: Controls how much of the new weights to incorporate (0.0 to 1.0). Lower values make smaller, more conservative updates.

- **weight_momentum**: For momentum strategy, determines how much previous updates influence current ones (typically 0.9 to 0.99).

- **warmup_rounds**: Number of initial rounds with gradually increasing mixing rates. Useful for stabilizing early training.

- **apply_weight_constraints**: When ``True``, prevents any weight from changing too dramatically in a single update.

- **max_weight_change**: Maximum relative change allowed in any weight when constraints are enabled (e.g., 0.2 = 20% maximum change).

Choosing a Strategy
'''''''''''''''''

- Use **Direct** for simpler models and homogeneous data distributions.
- Use **EMA** for improved stability and when working with sensitive data that might create noisy updates.
- Use **Momentum** for faster convergence on complex problems and when clients have heterogeneous data distributions.

For maximum stability, especially with differential privacy enabled, combine momentum with weight constraints:

.. code-block:: python

    from secureml.federated import FederatedConfig, train_federated
    
    # Configuration for stable training with differential privacy
    config = FederatedConfig(
        num_rounds=20,
        weight_update_strategy="momentum", 
        weight_momentum=0.95,
        apply_weight_constraints=True,
        max_weight_change=0.25,
        apply_differential_privacy=True,
        epsilon=1.0
    )
    
    trained_model = train_federated(
        model=model,
        client_data_fn=get_client_data,
        config=config
    )

Supported Frameworks
-----------------------------

SecureML supports multiple frameworks for federated learning:

**PyTorch Models**

.. code-block:: python

    import torch.nn as nn
    from secureml.federated import train_federated
    
    # Define a PyTorch model
    class SimpleNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(10, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
            
        def forward(self, x):
            return self.layers(x)
    
    # Create and train the model
    model = SimpleNN()
    
    trained_model = train_federated(
        model=model,
        client_data_fn=get_client_data,
        framework="pytorch"
    )

**TensorFlow Models**

.. code-block:: python

    import tensorflow as tf
    from secureml.federated import train_federated
    
    # Define a TensorFlow model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile the model (this is optional, will be done internally if needed)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Train the model
    trained_model = train_federated(
        model=model,
        client_data_fn=get_client_data,
        framework="tensorflow"
    )

Best Practices
-------------

1. **Start with simulation**: Test your federated learning setup in a simulated environment using `train_federated` before deploying to real clients with `start_federated_server` and `start_federated_client`.

2. **Handle heterogeneous data**: Use advanced weight update strategies like momentum or EMA to handle non-IID data distributions.

3. **Consider communication costs**: Keep model sizes reasonable and choose appropriate batch sizes to manage communication overhead.

4. **Apply privacy protections**: Combine federated learning with differential privacy and secure aggregation for maximum privacy protection.

5. **Monitor convergence**: Carefully monitor convergence rates and model performance, as federated learning may converge differently than centralized training.

6. **Framework detection**: You can set `framework="auto"` to let SecureML automatically detect whether you're using PyTorch or TensorFlow, but it's best to explicitly specify the framework when possible.

7. **Data preparation**: Ensure your data is properly formatted before training. SecureML expects a pandas DataFrame or numpy array, with the target variable either specified via the `target_column` parameter or assumed to be the last column.

Further Reading
-------------

* :doc:`/api/federated_learning` - Complete API reference for federated learning functions
* :doc:`/examples/federated_learning` - More examples of federated learning techniques
* `Communication-Efficient Learning of Deep Networks from Decentralized Data <https://arxiv.org/abs/1602.05629>`_ - Original FedAvg paper by McMahan et al. 