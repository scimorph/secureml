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

Setting Up a Federated Learning System
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The simplest way to use federated learning is with the high-level API:

.. code-block:: python

    from secureml.federated_learning import FederatedLearningSystem
    
    # Create a federated learning system
    fl_system = FederatedLearningSystem(
        model_architecture='logistic_regression',  # or 'random_forest', 'neural_network', etc.
        aggregation_method='fedavg',  # FederatedAveraging algorithm
        num_rounds=10,
        min_clients_per_round=5,
        encryption_enabled=True  # Enable secure aggregation
    )
    
    # Define the data architecture
    fl_system.configure_data(
        features=['age', 'gender', 'income', 'location'],
        target='purchase_decision',
        task_type='classification'
    )
    
    # Start the system
    fl_system.start_server(port=8080)

Setting Up a Federated Client
^^^^^^^^^^^^^^^^^^^^^^^^^^^

On each client device or server:

.. code-block:: python

    from secureml.federated_learning import FederatedClient
    
    # Create a federated client
    client = FederatedClient(
        server_address='https://fl-server.example.com:8080',
        client_id='client-001',
        data=client_data,  # Local pandas DataFrame
        features=['age', 'gender', 'income', 'location'],
        target='purchase_decision',
        privacy_budget=1.0  # Optional: Apply differential privacy
    )
    
    # Start the client
    client.start()
    
    # Stop the client when done
    client.stop()

Running a Federated Learning Simulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For experimentation and testing:

.. code-block:: python

    from secureml.federated_learning import FederatedSimulation
    
    # Create a simulation with synthetic clients
    simulation = FederatedSimulation(
        num_clients=100,
        data_distribution='non_iid',  # Options: 'iid', 'non_iid', 'label_skew', 'feature_skew'
        model_architecture='cnn',
        aggregation_method='fedavg'
    )
    
    # Load data and partition it to simulate clients
    simulation.load_and_partition_data(
        X=X_full,
        y=y_full,
        partition_strategy='dirichlet',  # Other options: 'random', 'shard', 'pathological'
        alpha=0.5  # Concentration parameter for Dirichlet distribution
    )
    
    # Run the simulation
    results = simulation.run(
        num_rounds=50,
        local_epochs=5,
        batch_size=64,
        learning_rate=0.01
    )
    
    # Evaluate the final global model
    global_model_accuracy = simulation.evaluate(X_test, y_test)
    print(f"Global model accuracy: {global_model_accuracy:.4f}")
    
    # Plot the training progress
    simulation.plot_metrics()

Advanced Techniques
------------------

Secure Aggregation
^^^^^^^^^^^^^^^^

Protect client updates from being inspected:

.. code-block:: python

    from secureml.federated_learning import SecureAggregator
    
    # On the server side
    secure_aggregator = SecureAggregator(
        encryption_type='homomorphic',  # Options: 'homomorphic', 'secure_multiparty', 'paillier'
        security_bits=128,
        threshold=0.7  # Minimum fraction of clients required
    )
    
    # Configure the FL system to use secure aggregation
    fl_system = FederatedLearningSystem(
        model_architecture='logistic_regression',
        aggregator=secure_aggregator,
        num_rounds=10
    )

Differential Privacy in Federated Learning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Add differential privacy to client updates:

.. code-block:: python

    from secureml.federated_learning import DPFederatedClient
    
    # Create a differentially private federated client
    dp_client = DPFederatedClient(
        server_address='https://fl-server.example.com:8080',
        client_id='client-002',
        data=client_data,
        epsilon=1.0,
        delta=1e-5,
        clip_norm=1.0,
        noise_multiplier=1.1
    )
    
    # Start the client with DP protection
    dp_client.start()

Custom Aggregation Algorithms
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Implement custom aggregation strategies:

.. code-block:: python

    from secureml.federated_learning import FederatedAggregator, FederatedLearningSystem
    
    # Create a custom aggregator
    class WeightedAggregator(FederatedAggregator):
        def __init__(self, client_weights=None):
            super().__init__()
            self.client_weights = client_weights or {}
            
        def aggregate(self, client_updates):
            """Aggregate updates with custom weights for each client"""
            weighted_updates = {}
            for client_id, update in client_updates.items():
                weight = self.client_weights.get(client_id, 1.0)
                weighted_updates[client_id] = {k: v * weight for k, v in update.items()}
            
            # Perform weighted average
            aggregated = {}
            for param_name in next(iter(client_updates.values())).keys():
                sum_weights = sum(self.client_weights.get(client_id, 1.0) for client_id in client_updates.keys())
                aggregated[param_name] = sum(update[param_name] for update in weighted_updates.values()) / sum_weights
                
            return aggregated
    
    # Use the custom aggregator
    custom_aggregator = WeightedAggregator(
        client_weights={'client-001': 1.5, 'client-002': 0.8, 'client-003': 1.2}
    )
    
    fl_system = FederatedLearningSystem(
        model_architecture='neural_network',
        aggregator=custom_aggregator,
        num_rounds=20
    )

Client Selection Strategies
^^^^^^^^^^^^^^^^^^^^^^^^^

Control how clients are selected for each round:

.. code-block:: python

    from secureml.federated_learning import ClientSelector, FederatedLearningSystem
    
    # Define a custom client selection strategy
    class PerformanceBasedSelector(ClientSelector):
        def __init__(self, historical_performance=None):
            self.historical_performance = historical_performance or {}
            self.min_performance = 0.7  # Minimum acceptable performance
            
        def select_clients(self, available_clients, num_to_select):
            """Select clients based on their historical performance"""
            # Filter clients with acceptable performance
            qualified_clients = [
                c for c in available_clients 
                if self.historical_performance.get(c, 1.0) >= self.min_performance
            ]
            
            # Sort by performance (higher is better)
            qualified_clients.sort(key=lambda c: self.historical_performance.get(c, 0), reverse=True)
            
            # Select the top performers
            return qualified_clients[:num_to_select]
    
    # Use the custom selector
    custom_selector = PerformanceBasedSelector(
        historical_performance={'client-001': 0.95, 'client-002': 0.85, 'client-003': 0.65}
    )
    
    fl_system = FederatedLearningSystem(
        model_architecture='neural_network',
        client_selector=custom_selector,
        num_rounds=20,
        min_clients_per_round=5
    )

Handling Non-IID Data
^^^^^^^^^^^^^^^^^^^

Deal with non-independent and identically distributed data:

.. code-block:: python

    from secureml.federated_learning import FederatedLearningSystem
    
    # Configure a system to handle non-IID data
    fl_system = FederatedLearningSystem(
        model_architecture='neural_network',
        aggregation_method='fedprox',  # Use FedProx instead of FedAvg for non-IID data
        proximal_term_strength=0.01,   # Regularization parameter to handle heterogeneous data
        num_rounds=30,
        min_clients_per_round=5
    )

Supported Models and Frameworks
-----------------------------

SecureML supports multiple frameworks for federated learning:

**Scikit-learn Models**

.. code-block:: python

    from secureml.federated_learning.sklearn import FederatedLogisticRegression, FederatedRandomForest
    
    # Create federated scikit-learn models
    fed_logreg = FederatedLogisticRegression(
        server_address='https://fl-server.example.com:8080',
        aggregation_method='fedavg',
        num_rounds=10
    )
    
    fed_rf = FederatedRandomForest(
        server_address='https://fl-server.example.com:8080',
        aggregation_method='fedavg',
        num_rounds=10,
        n_estimators=100
    )

**PyTorch Models**

.. code-block:: python

    from secureml.federated_learning.torch import FederatedPyTorchModel
    import torch.nn as nn
    
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
    
    # Create the federated PyTorch model
    model = SimpleNN()
    fed_torch_model = FederatedPyTorchModel(
        model=model,
        server_address='https://fl-server.example.com:8080',
        aggregation_method='fedavg',
        num_rounds=20,
        optimizer='adam',
        learning_rate=0.001
    )

**TensorFlow Models**

.. code-block:: python

    from secureml.federated_learning.tensorflow import FederatedTensorFlowModel
    import tensorflow as tf
    
    # Define a TensorFlow model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Create the federated TensorFlow model
    fed_tf_model = FederatedTensorFlowModel(
        model=model,
        server_address='https://fl-server.example.com:8080',
        aggregation_method='fedavg',
        num_rounds=20,
        optimizer='adam',
        learning_rate=0.001
    )

Deployment Scenarios
------------------

Cross-silo Federated Learning
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For federated learning across organizations:

.. code-block:: python

    from secureml.federated_learning import CrossSiloFederatedSystem
    
    # Create a cross-silo federated system
    cross_silo_system = CrossSiloFederatedSystem(
        model_architecture='neural_network',
        aggregation_method='fedavg',
        num_rounds=20,
        authentication_required=True,
        secure_connection_type='tls',
        secure_aggregation=True
    )
    
    # Configure the system
    cross_silo_system.configure_server(
        server_address='fl-server.example.com',
        server_port=8080,
        tls_cert='server_cert.pem',
        tls_key='server_key.pem'
    )
    
    # Start the server
    cross_silo_system.start_server()

Cross-device Federated Learning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For federated learning across mobile devices:

.. code-block:: python

    from secureml.federated_learning import CrossDeviceFederatedSystem
    
    # Create a cross-device federated system
    cross_device_system = CrossDeviceFederatedSystem(
        model_architecture='mobile_net',
        aggregation_method='fedavg',
        num_rounds=50,
        min_clients_per_round=100,
        client_sampling_rate=0.1,  # Sample 10% of available clients each round
        communication_efficient=True  # Enable communication-efficient updates
    )
    
    # Configure the system for mobile deployment
    cross_device_system.configure_server(
        server_address='fl-server.example.com',
        server_port=8080,
        model_compression=True,
        model_update_size_limit_mb=5  # Limit update size for mobile devices
    )
    
    # Start the server
    cross_device_system.start_server()

Monitoring and Evaluation
-----------------------

Tracking Federated Learning Progress
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Monitor the federated learning process:

.. code-block:: python

    from secureml.federated_learning import FederatedMonitor
    
    # Create a monitor for the federated system
    monitor = FederatedMonitor(
        fl_system=fl_system,
        metrics=['accuracy', 'loss', 'communication_cost', 'convergence_rate'],
        log_dir='fl_logs',
        save_model_checkpoints=True
    )
    
    # Start monitoring
    monitor.start()
    
    # Access current metrics
    current_metrics = monitor.get_current_metrics()
    print(f"Current global model accuracy: {current_metrics['accuracy']:.4f}")
    
    # Generate a visualization dashboard
    monitor.generate_dashboard('fl_dashboard.html')

Evaluating Model Performance
^^^^^^^^^^^^^^^^^^^^^^^^^

Assess the performance of federated models:

.. code-block:: python

    from secureml.federated_learning.evaluation import evaluate_federated_model
    
    # Evaluate the global model
    global_performance = evaluate_federated_model(
        model=fl_system.get_global_model(),
        test_data=X_test,
        test_labels=y_test,
        metrics=['accuracy', 'precision', 'recall', 'f1']
    )
    
    print(f"Global model performance: {global_performance}")
    
    # Evaluate models on individual clients
    client_performances = []
    for client_id in ['client-001', 'client-002', 'client-003']:
        client_perf = evaluate_federated_model(
            model=fl_system.get_global_model(),
            test_data=client_test_data[client_id],
            test_labels=client_test_labels[client_id],
            metrics=['accuracy']
        )
        client_performances.append((client_id, client_perf['accuracy']))
    
    # Print performance across clients
    for client_id, accuracy in client_performances:
        print(f"Client {client_id} accuracy: {accuracy:.4f}")

Best Practices
-------------

1. **Start with simulation**: Test your federated learning setup in a simulated environment before deploying to real clients

2. **Handle heterogeneous data**: Use techniques like FedProx or client weighting to handle non-IID data distributions

3. **Consider communication costs**: Implement model compression and efficient communication protocols, especially for cross-device FL

4. **Apply privacy protections**: Combine federated learning with differential privacy and secure aggregation for maximum privacy

5. **Monitor convergence**: Carefully monitor convergence rates and model performance, as federated learning may converge differently than centralized training

6. **Client selection strategies**: Develop thoughtful client selection strategies that balance data quality, client reliability, and fairness

7. **Security considerations**: Implement proper authentication, encryption, and access controls for all components

Further Reading
-------------

* :doc:`/api/federated_learning` - Complete API reference for federated learning functions
* :doc:`/examples/federated_learning` - More examples of federated learning techniques
* `Communication-Efficient Learning of Deep Networks from Decentralized Data <https://arxiv.org/abs/1602.05629>`_ - Original FedAvg paper by McMahan et al. 