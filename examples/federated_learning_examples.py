import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# For PyTorch examples
import torch
import torch.nn as nn
import torch.optim as optim

# For TensorFlow examples
import tensorflow as tf

# Import SecureML federated learning components
from secureml.federated import (
    train_federated,
    start_federated_server,
    start_federated_client,
    FederatedConfig
)

# Create a colorful print header for examples
def print_header(title):
    """Print a section header with formatting."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


# Example 1: Basic Simulation with PyTorch
def example_federated_pytorch():
    """
    Basic federated learning simulation with PyTorch.
    This example demonstrates how to use SecureML's federated learning 
    with a simple PyTorch model on synthetic data.
    """
    print_header("Example 1: Basic Federated Learning with PyTorch")
    
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
    
    # Simulate 3 clients by splitting the data
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
    print("Starting federated training with PyTorch model...")
    trained_model = train_federated(
        model=model,
        client_data_fn=get_client_data,
        config=config,
        framework="pytorch",
        batch_size=32,
        learning_rate=0.01,
        epochs=2,  # Local epochs per round
        verbose=True
    )
    
    print("Federated training completed successfully.")
    
    # Test the trained model
    X_test, y_test = make_classification(
        n_samples=200, 
        n_features=20, 
        n_informative=10,
        n_classes=2, 
        random_state=43
    )
    X_test_scaled = scaler.transform(X_test)
    
    # Evaluate the model
    trained_model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
        outputs = trained_model(X_tensor)
        predicted = (outputs > 0.5).float().numpy().flatten()
        
    accuracy = np.mean(predicted == y_test)
    print(f"Test accuracy: {accuracy:.4f}")
    
    return trained_model


# Example 2: Federated Learning with TensorFlow
def example_federated_tensorflow():
    """
    Federated learning simulation with TensorFlow.
    This example shows how to use SecureML's federated learning with TensorFlow.
    """
    print_header("Example 2: Federated Learning with TensorFlow")
    
    # Create synthetic dataset
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
    
    # One-hot encode the labels for multi-class classification
    y_onehot = tf.keras.utils.to_categorical(y)
    
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
    
    # Simulate 4 clients by splitting the data
    def get_client_data():
        # Split data into 4 parts to simulate 4 clients
        splits = np.array_split(np.arange(len(X_scaled)), 4)
        
        # Create client datasets
        client_data = {}
        for i, indices in enumerate(splits):
            # For TensorFlow, we need to concatenate features and one-hot encoded labels
            client_X = X_scaled[indices]
            client_y = y_onehot[indices]
            
            # For the federated learning function, we'll convert back to regular labels
            # because the function handles one-hot encoding internally
            client_labels = np.argmax(client_y, axis=1).reshape(-1, 1)
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
    print("Starting federated training with TensorFlow model...")
    trained_model = train_federated(
        model=model,
        client_data_fn=get_client_data,
        config=config,
        framework="tensorflow",
        batch_size=32,
        epochs=3,  # Local epochs per round
        verbose=True
    )
    
    print("Federated training completed successfully.")
    
    # Test the trained model
    X_test, y_test = make_classification(
        n_samples=200, 
        n_features=20, 
        n_informative=10,
        n_classes=3,
        n_clusters_per_class=2,
        random_state=43
    )
    X_test_scaled = scaler.transform(X_test)
    
    # Evaluate the model
    loss, accuracy = trained_model.evaluate(
        X_test_scaled, 
        tf.keras.utils.to_categorical(y_test),
        verbose=0
    )
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    
    return trained_model


# Example 3: Federated Learning with Privacy Features
def example_federated_with_privacy():
    """
    Federated learning with differential privacy and secure aggregation.
    This example demonstrates how to enable privacy-preserving features in federated learning.
    """
    print_header("Example 3: Federated Learning with Privacy Features")
    
    # Create synthetic dataset
    X, y = make_classification(
        n_samples=1000, 
        n_features=10, 
        n_informative=5,
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
                nn.Linear(input_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
            
        def forward(self, x):
            return self.layers(x)
    
    # Create a model instance
    input_dim = X_scaled.shape[1]
    model = BinaryClassifier(input_dim)
    
    # Simulate 5 clients by splitting the data
    def get_client_data():
        # Split data into 5 parts to simulate 5 clients
        splits = np.array_split(np.arange(len(X_scaled)), 5)
        
        # Create client datasets
        client_data = {}
        for i, indices in enumerate(splits):
            # Combine features and target into one array for each client
            client_X = X_scaled[indices]
            client_y = y[indices].reshape(-1, 1)
            client_data[f"client-{i+1}"] = np.hstack((client_X, client_y))
            
        return client_data
    
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
    
    # Train the model using federated learning with privacy
    print("Starting privacy-preserving federated training...")
    trained_model = train_federated(
        model=model,
        client_data_fn=get_client_data,
        config=config,
        framework="pytorch",
        batch_size=64,
        learning_rate=0.005,
        epochs=1,  # Use fewer local epochs when applying DP
        verbose=True,
        max_grad_norm=1.0  # Clipping parameter for differential privacy
    )
    
    print("Privacy-preserving federated training completed successfully.")
    
    # Test the trained model
    X_test, y_test = make_classification(
        n_samples=200, 
        n_features=10, 
        n_informative=5,
        n_classes=2, 
        random_state=43
    )
    X_test_scaled = scaler.transform(X_test)
    
    # Evaluate the model
    trained_model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
        outputs = trained_model(X_tensor)
        predicted = (outputs > 0.5).float().numpy().flatten()
        
    accuracy = np.mean(predicted == y_test)
    print(f"Test accuracy: {accuracy:.4f}")
    
    return trained_model


# Example 4: Advanced Weight Update Strategies
def example_federated_weight_strategies():
    """
    Comparing different weight update strategies in federated learning.
    This example demonstrates the different weight update strategies available.
    """
    print_header("Example 4: Comparing Weight Update Strategies")
    
    # Create synthetic dataset
    X, y = make_regression(
        n_samples=1000, 
        n_features=20, 
        n_informative=10,
        noise=0.1,
        random_state=42
    )
    
    # Scale features and target
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_scaled = X_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()
    
    # Define a simple regression model with PyTorch
    class RegressionModel(nn.Module):
        def __init__(self, input_dim):
            super(RegressionModel, self).__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
            
        def forward(self, x):
            return self.layers(x)
    
    # Simulate non-IID data by creating skewed distributions for clients
    def get_client_data_non_iid():
        """Create non-IID data distribution across clients."""
        # Split samples into 4 clients
        client_data = {}
        
        # Client 1: mostly lower values of feature 0
        c1_indices = np.where(X_scaled[:, 0] < -0.5)[0][:200]
        client_data["client-1"] = np.hstack((
            X_scaled[c1_indices], 
            y_scaled[c1_indices].reshape(-1, 1)
        ))
        
        # Client 2: mostly higher values of feature 0
        c2_indices = np.where(X_scaled[:, 0] > 0.5)[0][:200]
        client_data["client-2"] = np.hstack((
            X_scaled[c2_indices], 
            y_scaled[c2_indices].reshape(-1, 1)
        ))
        
        # Client 3: mostly mid-range values of feature 0
        c3_indices = np.where((X_scaled[:, 0] >= -0.2) & (X_scaled[:, 0] <= 0.2))[0][:200]
        client_data["client-3"] = np.hstack((
            X_scaled[c3_indices], 
            y_scaled[c3_indices].reshape(-1, 1)
        ))
        
        # Client 4: random mix
        remaining = list(set(range(len(X_scaled))) - 
                        set(c1_indices) - 
                        set(c2_indices) - 
                        set(c3_indices))
        c4_indices = np.random.choice(remaining, 200, replace=False)
        client_data["client-4"] = np.hstack((
            X_scaled[c4_indices], 
            y_scaled[c4_indices].reshape(-1, 1)
        ))
        
        return client_data
    
    # Create test data
    X_test, y_test = make_regression(
        n_samples=200, 
        n_features=20, 
        n_informative=10,
        noise=0.1,
        random_state=43
    )
    X_test_scaled = X_scaler.transform(X_test)
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).flatten()
    
    # Function to train and evaluate model with a specific strategy
    def train_with_strategy(strategy, mixing_rate=0.5, momentum=0.9, constraints=False):
        # Create a new model instance
        input_dim = X_scaled.shape[1]
        model = RegressionModel(input_dim)
        
        # Configure with the specific strategy
        config = FederatedConfig(
            num_rounds=10,
            fraction_fit=1.0,
            min_fit_clients=3,
            min_available_clients=3,
            weight_update_strategy=strategy,
            weight_mixing_rate=mixing_rate,
            weight_momentum=momentum,
            apply_weight_constraints=constraints,
            max_weight_change=0.2 if constraints else None,
            warmup_rounds=2
        )
        
        # Train the model
        print(f"Training with {strategy} strategy...")
        trained_model = train_federated(
            model=model,
            client_data_fn=get_client_data_non_iid,
            config=config,
            framework="pytorch",
            batch_size=32,
            learning_rate=0.01,
            epochs=2,
            verbose=False  # Reduce output verbosity
        )
        
        # Evaluate the model
        trained_model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
            predictions = trained_model(X_tensor).numpy().flatten()
            
        # Calculate MSE
        mse = np.mean((predictions - y_test_scaled) ** 2)
        
        print(f"  {strategy} strategy - Test MSE: {mse:.5f}")
        return mse, trained_model
    
    # Compare different strategies
    results = {}
    
    # Direct strategy
    direct_mse, direct_model = train_with_strategy("direct")
    results["direct"] = direct_mse
    
    # EMA strategy
    ema_mse, ema_model = train_with_strategy("ema", mixing_rate=0.3)
    results["ema"] = ema_mse
    
    # Momentum strategy
    momentum_mse, momentum_model = train_with_strategy(
        "momentum", 
        mixing_rate=0.1, 
        momentum=0.9, 
        constraints=False
    )
    results["momentum"] = momentum_mse
    
    # Momentum with constraints
    momentum_constrained_mse, momentum_constrained_model = train_with_strategy(
        "momentum", 
        mixing_rate=0.1, 
        momentum=0.9, 
        constraints=True
    )
    results["momentum_constrained"] = momentum_constrained_mse
    
    # Display results
    print("\nStrategy Comparison Results:")
    for strategy, mse in results.items():
        print(f"  {strategy}: MSE = {mse:.5f}")
        
    # Plot the results
    plt.figure(figsize=(10, 6))
    strategies = list(results.keys())
    mses = [results[s] for s in strategies]
    
    plt.bar(strategies, mses)
    plt.title('Comparison of Weight Update Strategies')
    plt.xlabel('Strategy')
    plt.ylabel('Mean Squared Error (lower is better)')
    plt.ylim(0, max(mses) * 1.2)
    
    # Add values on top of the bars
    for i, mse in enumerate(mses):
        plt.text(i, mse + max(mses) * 0.05, f"{mse:.5f}", ha='center')
    
    plt.savefig('federated_strategies_comparison.png')
    plt.close()
    
    print("\nResults have been saved to 'federated_strategies_comparison.png'")
    
    return results, direct_model, ema_model, momentum_model, momentum_constrained_model


# Example 5: Client/Server Setup (this is for demonstration - not actually run)
def example_client_server_setup():
    """
    Demonstrate how to set up a federated learning server and client.
    Note: This code is not actually run, but shows the API usage.
    """
    print_header("Example 5: Client/Server Setup (Demonstration Only)")
    
    print("Note: This example demonstrates the API but doesn't actually execute.")
    print("In a real deployment, these would be run on separate machines.")
    
    # === Server Setup ===
    print("\n--- Server Side Code ---")
    code = """
    # Import required libraries
    import torch.nn as nn
    from secureml.federated import start_federated_server, FederatedConfig
    
    # Define your model architecture (must match the client's)
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
    """
    print(code)
    
    # === Client Setup ===
    print("\n--- Client Side Code ---")
    code = """
    # Import required libraries
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
        apply_differential_privacy=True,  # Enable DP locally as well
        epsilon=1.0,
        delta=1e-5,
        test_split=0.2,  # Use 20% of data for local evaluation
        batch_size=32,
        learning_rate=0.001,
        optimizer="adam",
        loss_fn="cross_entropy",
        target_column="diagnosis"  # Specify the target column in the DataFrame
    )
    """
    print(code)
    
    print("\nIn a real-world deployment:")
    print("1. The server would run on a dedicated machine/cloud instance")
    print("2. Each client would be a separate organization/device/hospital")
    print("3. Clients connect to the server when they're ready to participate")
    print("4. The training happens without sharing the raw data")
    
    return None


# Run the examples
if __name__ == "__main__":
    print("SecureML Federated Learning Examples")
    print("-----------------------------------")
    
    # Basic PyTorch example
    pytorch_model = example_federated_pytorch()
    
    # TensorFlow example
    tensorflow_model = example_federated_tensorflow()
    
    # Privacy example
    private_model = example_federated_with_privacy()
    
    # Strategy comparison example
    strategy_results = example_federated_weight_strategies()
    
    # Client/server setup example (demonstration only)
    example_client_server_setup()
    
    print("\nAll examples completed successfully.") 