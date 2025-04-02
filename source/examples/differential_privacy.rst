Differential Privacy Examples
============================

This section demonstrates how to use SecureML's differential privacy features to train machine learning models with formal privacy guarantees.

PyTorch Model with Differential Privacy
--------------------------------------

In this example, we'll train a PyTorch neural network with differential privacy using Opacus under the hood:

.. code-block:: python

    import numpy as np
    import torch
    import torch.nn as nn
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from secureml.privacy import differentially_private_train
    
    # Create a synthetic dataset
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    # Generate features and binary classification target
    X = np.random.randn(n_samples, n_features)
    w = np.random.randn(n_features)
    y = (np.dot(X, w) + 0.1 * np.random.randn(n_samples) > 0).astype(int)
    
    # Split and scale the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Prepare data for SecureML (expects DataFrame or numpy array with target as last column)
    train_data = np.column_stack((X_train, y_train))
    
    # Define a PyTorch model
    class BinaryClassifier(nn.Module):
        def __init__(self, input_dim):
            super(BinaryClassifier, self).__init__()
            self.layer1 = nn.Linear(input_dim, 32)
            self.layer2 = nn.Linear(32, 16)
            self.layer3 = nn.Linear(16, 1)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
            
        def forward(self, x):
            x = self.relu(self.layer1(x))
            x = self.relu(self.layer2(x))
            x = self.sigmoid(self.layer3(x))
            return x
    
    # Create the model
    model = BinaryClassifier(X_train.shape[1])
    
    # Train with differential privacy
    dp_model = differentially_private_train(
        model=model,
        data=train_data,
        epsilon=1.0,  # Privacy budget
        delta=1e-5,   # Privacy delta
        batch_size=64,
        epochs=5,
        learning_rate=0.001,
        validation_split=0.1,
        framework="pytorch",
        verbose=True
    )
    
    # Evaluate the model
    dp_model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_pred = dp_model(X_test_tensor).numpy().flatten() > 0.5
        accuracy = np.mean(y_pred == y_test)
        print(f"Test accuracy with differential privacy: {accuracy:.4f}")

TensorFlow Model with Differential Privacy
---------------------------------------

SecureML also supports differential privacy for TensorFlow models using TensorFlow Privacy in an isolated environment:

.. code-block:: python

    import numpy as np
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from secureml.privacy import differentially_private_train
    
    # Create a synthetic multi-class dataset
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    n_classes = 3
    
    # Generate features and multi-class classification target
    X = np.random.randn(n_samples, n_features)
    w = np.random.randn(n_features, n_classes)
    logits = np.dot(X, w)
    y = np.argmax(logits, axis=1)
    
    # Split and scale the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Prepare data for SecureML
    train_data = np.column_stack((X_train, y_train))
    
    # Define a TensorFlow model
    tf_model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(n_features,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(n_classes, activation='softmax')
    ])
    
    # Compile the model
    tf_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train with differential privacy
    dp_tf_model = differentially_private_train(
        model=tf_model,
        data=train_data,
        epsilon=1.0,
        delta=1e-5,
        max_grad_norm=1.0,
        batch_size=64,
        epochs=5,
        framework="tensorflow",
        verbose=True
    )
    
    # Evaluate the model
    test_loss, test_accuracy = dp_tf_model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy with differential privacy: {test_accuracy:.4f}")

Privacy-Utility Tradeoff
----------------------

One important aspect of differential privacy is understanding the tradeoff between privacy and utility. Here's an example that evaluates model performance across different privacy budgets (epsilon values):

.. code-block:: python

    import numpy as np
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from secureml.privacy import differentially_private_train
    
    # Create dataset and prepare data as in previous examples
    # ...
    
    # Define different privacy budgets to test
    epsilons = [0.1, 0.5, 1.0, 5.0, 10.0]
    accuracies = []
    
    # Function to create a model with the same architecture
    def create_model():
        return nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    # Train and evaluate for each epsilon
    for epsilon in epsilons:
        model = create_model()
        
        # Train with differential privacy
        dp_model = differentially_private_train(
            model=model,
            data=train_data,
            epsilon=epsilon,
            delta=1e-5,
            batch_size=64,
            epochs=5,
            framework="pytorch",
            verbose=False
        )
        
        # Evaluate
        dp_model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            y_pred = dp_model(X_test_tensor).numpy().flatten() > 0.5
            accuracy = np.mean(y_pred == y_test)
            accuracies.append(accuracy)
    
    # Train a non-private model for comparison
    non_private_model = create_model()
    # Train the non-private model...
    
    # Plot the privacy-utility tradeoff
    plt.figure(figsize=(10, 6))
    plt.plot(epsilons, accuracies, 'o-', label='DP Model')
    plt.axhline(y=non_private_accuracy, color='r', linestyle='--', label='Non-private Model')
    plt.xscale('log')
    plt.xlabel('Privacy Budget (ε)')
    plt.ylabel('Test Accuracy')
    plt.title('Privacy-Utility Tradeoff')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('privacy_utility_tradeoff.png')

This will produce a graph showing how model accuracy changes as the privacy budget (epsilon) increases. Typically, as epsilon increases (less privacy), accuracy improves and approaches the non-private model's performance.

Federated Learning with Differential Privacy
-----------------------------------------

SecureML allows combining federated learning with differential privacy for enhanced privacy guarantees:

.. code-block:: python

    import torch
    import torch.nn as nn
    from secureml.federated import start_federated_client
    
    # Define a model
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
        nn.Sigmoid()
    )
    
    # Start a federated learning client with differential privacy
    start_federated_client(
        model=model,
        data=client_data,  # Client's local dataset
        server_address="localhost:8080",
        apply_differential_privacy=True,
        epsilon=1.0,
        delta=1e-5,
        max_grad_norm=1.0
    )

Advanced Options and Best Practices
--------------------------------

Setting Hyperparameters
^^^^^^^^^^^^^^^^^^^^^

When training with differential privacy, several hyperparameters can significantly affect both privacy and utility:

.. code-block:: python

    dp_model = differentially_private_train(
        model=model,
        data=train_data,
        epsilon=1.0,
        delta=1e-5,
        
        # Noise and clipping parameters
        noise_multiplier=None,  # Auto-calculated from epsilon if None
        max_grad_norm=1.0,      # Clipping threshold for gradients
        
        # Training parameters
        batch_size=64,          # Larger batch sizes need less noise
        learning_rate=0.001,    # May need adjustment compared to non-DP training
        epochs=10,
        
        # Validation and early stopping
        validation_split=0.2,
        early_stopping_patience=3,
        
        # Other parameters
        verbose=True,
        framework="pytorch"     # or "tensorflow"
    )

Data Preparation Tips
^^^^^^^^^^^^^^^^^^^

For better performance with differential privacy:

1. **Normalize your data**: Normalized features perform better with gradient clipping
2. **Balance classes**: Imbalanced datasets can make private training more challenging
3. **Remove outliers**: Extreme values can have a disproportionate effect on gradients

.. code-block:: python

    # Example of proper data preparation
    from sklearn.preprocessing import StandardScaler
    
    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Create a balanced subsample if needed
    from sklearn.utils import resample
    X_balanced, y_balanced = resample(X_train, y_train, stratify=y_train, random_state=42)

Monitoring Privacy Budget
^^^^^^^^^^^^^^^^^^^^^

Both PyTorch and TensorFlow implementations allow you to monitor the privacy budget spent:

.. code-block:: python

    # For PyTorch (Opacus)
    from opacus import PrivacyEngine
    
    # After training with Opacus, the privacy engine has a get_epsilon method
    privacy_engine = PrivacyEngine()
    # Training code...
    
    # Get the privacy budget spent
    spent_epsilon = privacy_engine.get_epsilon(delta=1e-5)
    print(f"Privacy budget spent (ε = {spent_epsilon:.4f})")
    
    # For TensorFlow Privacy, the spent budget is returned as part of the result
    # when verbose=True is set in differentially_private_train 