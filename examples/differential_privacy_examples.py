import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import the privacy module
from secureml.privacy import differentially_private_train

# Example 1: PyTorch Model with Differential Privacy
def example_pytorch_dp():
    print("\nExample 1: PyTorch Model with Differential Privacy")
    print("=" * 50)
    
    # Import PyTorch
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("PyTorch is not installed. Please install with: pip install torch")
        return
    
    # Create a synthetic dataset
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    # Generate features and target
    X = np.random.randn(n_samples, n_features)
    # Generate binary classification target
    w = np.random.randn(n_features)
    y = (np.dot(X, w) + 0.1 * np.random.randn(n_samples) > 0).astype(int)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Create a DataFrame for training (SecureML expects a DataFrame or numpy array)
    train_data = np.column_stack((X_train, y_train))
    test_data = np.column_stack((X_test, y_test))
    
    # Print dataset information
    print(f"Training data shape: {train_data.shape}")
    print(f"Testing data shape: {test_data.shape}")
    
    # Define a simple PyTorch model
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
    input_dim = X_train.shape[1]
    model = BinaryClassifier(input_dim)
    print("\nModel architecture:")
    print(model)
    
    # Train the model with differential privacy
    print("\nTraining with differential privacy (PyTorch/Opacus)...")
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
        verbose=True,
        criterion=nn.BCELoss()
    )
    
    # Evaluate the model
    dp_model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_pred = dp_model(X_test_tensor).numpy().flatten() > 0.5
        accuracy = np.mean(y_pred == y_test)
        print(f"\nTest accuracy with differential privacy: {accuracy:.4f}")
    
    # Compare with non-private training
    print("\nComparing with non-private training...")
    
    # Create a new model for non-private training
    model_non_private = BinaryClassifier(input_dim)
    
    # Manual training without DP
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model_non_private.parameters(), lr=0.001)
    
    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    
    # Training loop
    model_non_private.train()
    for epoch in range(5):
        optimizer.zero_grad()
        outputs = model_non_private(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 1 == 0:
            print(f"Epoch {epoch+1}/5, Loss: {loss.item():.4f}")
    
    # Evaluate the non-private model
    model_non_private.eval()
    with torch.no_grad():
        y_pred_non_private = model_non_private(X_test_tensor).numpy().flatten() > 0.5
        accuracy_non_private = np.mean(y_pred_non_private == y_test)
        print(f"\nTest accuracy without differential privacy: {accuracy_non_private:.4f}")
    
    print("\nComparing the effect of differential privacy on model accuracy:")
    print(f"DP model accuracy: {accuracy:.4f}")
    print(f"Non-private model accuracy: {accuracy_non_private:.4f}")
    print(f"Accuracy difference: {accuracy_non_private - accuracy:.4f}")

# Example 2: TensorFlow Model with Differential Privacy
def example_tensorflow_dp():
    print("\nExample 2: TensorFlow Model with Differential Privacy")
    print("=" * 50)
    
    # Import TensorFlow
    try:
        import tensorflow as tf
        print("TensorFlow version:", tf.__version__)
    except ImportError:
        print("TensorFlow is not installed. Please install with: pip install tensorflow")
        return
    
    # Create a synthetic dataset
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    n_classes = 3
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate multi-class classification target
    w = np.random.randn(n_features, n_classes)
    logits = np.dot(X, w)
    y = np.argmax(logits, axis=1)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Create a DataFrame for training
    train_data = np.column_stack((X_train, y_train))
    
    # Print dataset information
    print(f"Training data shape: {train_data.shape}")
    print(f"Number of classes: {n_classes}")
    
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
    
    print("\nModel summary:")
    tf_model.summary()
    
    # Train with differential privacy
    print("\nTraining with differential privacy (TensorFlow Privacy)...")
    print("This will use the isolated environment for TensorFlow Privacy.")
    
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
    print("\nEvaluating the differentially private model...")
    test_loss, test_accuracy = dp_tf_model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy with differential privacy: {test_accuracy:.4f}")
    
    # Compare with non-private training
    print("\nComparing with non-private training...")
    
    # Create a new model for non-private training
    tf_model_non_private = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(n_features,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(n_classes, activation='softmax')
    ])
    
    # Compile the model
    tf_model_non_private.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train without differential privacy
    tf_model_non_private.fit(
        X_train, y_train,
        batch_size=64,
        epochs=5,
        verbose=1
    )
    
    # Evaluate the non-private model
    _, accuracy_non_private = tf_model_non_private.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy without differential privacy: {accuracy_non_private:.4f}")
    
    print("\nComparing the effect of differential privacy on model accuracy:")
    print(f"DP model accuracy: {test_accuracy:.4f}")
    print(f"Non-private model accuracy: {accuracy_non_private:.4f}")
    print(f"Accuracy difference: {accuracy_non_private - test_accuracy:.4f}")

# Example 3: Privacy Budget Experiment
def example_privacy_budget_comparison():
    print("\nExample 3: Privacy Budget Experiment")
    print("=" * 50)
    
    # Import PyTorch for this example
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("PyTorch is not installed. Please install with: pip install torch")
        return
    
    # Create a synthetic dataset
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    # Generate features and binary classification target
    X = np.random.randn(n_samples, n_features)
    w = np.random.randn(n_features)
    y = (np.dot(X, w) + 0.1 * np.random.randn(n_samples) > 0).astype(int)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Create a DataFrame for training
    train_data = np.column_stack((X_train, y_train))
    
    # Define different privacy budgets to test
    epsilons = [0.1, 0.5, 1.0, 5.0, 10.0]
    accuracies = []
    
    print("Testing different privacy budgets (epsilon values)...")
    
    # Create a simple model architecture
    def create_model():
        return nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    # Create test data tensor
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    
    # Train and evaluate for each epsilon
    for epsilon in epsilons:
        print(f"\nTraining with epsilon = {epsilon}")
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
            y_pred = dp_model(X_test_tensor).numpy().flatten() > 0.5
            accuracy = np.mean(y_pred == y_test)
            accuracies.append(accuracy)
            print(f"Test accuracy: {accuracy:.4f}")
    
    # Train a non-private model for comparison
    print("\nTraining without differential privacy...")
    non_private_model = create_model()
    
    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    
    # Training loop
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(non_private_model.parameters(), lr=0.001)
    
    non_private_model.train()
    for epoch in range(5):
        optimizer.zero_grad()
        outputs = non_private_model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
    
    # Evaluate
    non_private_model.eval()
    with torch.no_grad():
        y_pred = non_private_model(X_test_tensor).numpy().flatten() > 0.5
        non_private_accuracy = np.mean(y_pred == y_test)
        print(f"Test accuracy without DP: {non_private_accuracy:.4f}")
    
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
    print("\nPrivacy-utility tradeoff plot saved as 'privacy_utility_tradeoff.png'")
    
    # Print summary
    print("\nSummary of Privacy-Utility Tradeoff:")
    print("Epsilon (ε) | Accuracy | Accuracy Difference")
    print("-" * 45)
    for i, epsilon in enumerate(epsilons):
        diff = non_private_accuracy - accuracies[i]
        print(f"{epsilon:10.1f} | {accuracies[i]:8.4f} | {diff:19.4f}")

# Example 4: Federated Learning with Differential Privacy
def example_federated_learning_with_dp():
    print("\nExample 4: Federated Learning with Differential Privacy")
    print("=" * 50)
    
    try:
        from secureml.federated import start_federated_client
    except ImportError:
        print("The federated module could not be imported.")
        return
    
    print("This example demonstrates how to set up a federated learning client with differential privacy.")
    print("Note: Running an actual federated learning setup requires a federated server.")
    print("      This example shows the client setup code only.\n")
    
    # Create a synthetic client dataset
    np.random.seed(42)
    n_samples = 500
    n_features = 10
    
    # Generate features and binary classification target
    X = np.random.randn(n_samples, n_features)
    w = np.random.randn(n_features)
    y = (np.dot(X, w) + 0.1 * np.random.randn(n_samples) > 0).astype(int)
    
    # Create a client dataset
    client_data = np.column_stack((X, y))
    
    # Import PyTorch
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("PyTorch is not installed. Please install with: pip install torch")
        return
    
    # Define a PyTorch model
    model = nn.Sequential(
        nn.Linear(n_features, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
        nn.Sigmoid()
    )
    
    print("Client model architecture:")
    print(model)
    
    print("\nFederated Learning Client Configuration:")
    print("- Server Address: localhost:8080")
    print("- Differential Privacy: Enabled")
    print("- Epsilon: 1.0")
    print("- Delta: 1e-5")
    print("- Max Gradient Norm: 1.0")
    
    # The code to start a federated client (not actually executed)
    print("\nClient setup code:")
    print("""
    # Start a federated learning client with differential privacy
    start_federated_client(
        model=model,
        data=client_data,
        server_address="localhost:8080",
        apply_differential_privacy=True,
        epsilon=1.0,
        delta=1e-5,
        max_grad_norm=1.0
    )
    """)
    
    print("\nNote: This code is not executed as it requires a running federated server.")

if __name__ == "__main__":
    print("=" * 70)
    print("SecureML: Differential Privacy Examples")
    print("=" * 70)
    
    # Run PyTorch example
    example_pytorch_dp()
    
    # Run TensorFlow example
    example_tensorflow_dp()
    
    # Run privacy budget experiment
    example_privacy_budget_comparison()
    
    # Run federated learning example
    example_federated_learning_with_dp()
    
    print("\nAll examples completed.") 