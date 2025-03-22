"""
Tests for the federated learning module of SecureML.
"""

import unittest
import tempfile
import os
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import pytest

# Import the module to be tested
from secureml.federated import (
    FederatedConfig,
    train_federated,
    start_federated_server,
    start_federated_client,
    _detect_framework,
    _prepare_data_for_pytorch,
    _prepare_data_for_tensorflow,
    _create_pytorch_client,
    _create_tensorflow_client,
    _save_model
)

# Setup mock objects to avoid actual network connections and ML framework dependencies
# Create a mock Flower module for patching
class MockFlower:
    """Mock class for the flwr (Flower) module"""
    class simulation:
        @staticmethod
        def start_simulation(*args, **kwargs):
            pass
    
    class server:
        class ServerConfig:
            def __init__(self, num_rounds=3):
                self.num_rounds = num_rounds
        
        @staticmethod
        def start_server(*args, **kwargs):
            pass
        
        @staticmethod
        def Server(*args, **kwargs):
            return MagicMock()
    
    class client:
        @staticmethod
        def start_client(*args, **kwargs):
            pass
        
        class NumPyClient:
            def to_client(self):
                return MagicMock()
            
        Client = MagicMock()    
    
    class common:
        @staticmethod
        def ndarrays_to_parameters(weights):
            return weights

# Patch modules
fl_mock = MockFlower()
torch_mock = MagicMock()
tf_mock = MagicMock()

@pytest.fixture(autouse=True)
def setup_mocks(monkeypatch):
    monkeypatch.setattr("secureml.federated.fl", fl_mock)
    monkeypatch.setattr("secureml.federated.torch", MagicMock())
    monkeypatch.setattr("secureml.federated._HAS_PYTORCH", True)
    monkeypatch.setattr("secureml.federated.tf", MagicMock())
    monkeypatch.setattr("secureml.federated._HAS_TENSORFLOW", True)
    monkeypatch.setattr("secureml.federated._HAS_FLOWER", True)


class TestFederatedConfig(unittest.TestCase):
    """Test cases for the FederatedConfig class."""
    
    def test_config_initialization(self):
        """Test that the FederatedConfig class initializes with correct defaults."""
        config = FederatedConfig()
        
        # Check default values
        self.assertEqual(config.num_rounds, 3)
        self.assertEqual(config.fraction_fit, 1.0)
        self.assertEqual(config.min_fit_clients, 2)
        self.assertEqual(config.min_available_clients, 2)
        self.assertEqual(config.server_address, "0.0.0.0:8080")
        self.assertFalse(config.use_secure_aggregation)
        self.assertFalse(config.apply_differential_privacy)
        self.assertEqual(config.epsilon, 1.0)
        self.assertEqual(config.delta, 1e-5)
    
    def test_config_custom_values(self):
        """Test that FederatedConfig accepts custom values."""
        config = FederatedConfig(
            num_rounds=5,
            fraction_fit=0.8,
            min_fit_clients=3,
            min_available_clients=4,
            server_address="localhost:8081",
            use_secure_aggregation=True,
            apply_differential_privacy=True,
            epsilon=0.5,
            delta=1e-6,
            custom_param="custom_value"
        )
        
        # Check custom values
        self.assertEqual(config.num_rounds, 5)
        self.assertEqual(config.fraction_fit, 0.8)
        self.assertEqual(config.min_fit_clients, 3)
        self.assertEqual(config.min_available_clients, 4)
        self.assertEqual(config.server_address, "localhost:8081")
        self.assertTrue(config.use_secure_aggregation)
        self.assertTrue(config.apply_differential_privacy)
        self.assertEqual(config.epsilon, 0.5)
        self.assertEqual(config.delta, 1e-6)
        self.assertEqual(config.extra_kwargs["custom_param"], "custom_value")


class TestFrameworkDetection(unittest.TestCase):
    """Test cases for framework detection."""
    
    @patch("secureml.federated._HAS_PYTORCH", True)
    def test_detect_pytorch(self):
        """Test that PyTorch models are correctly detected."""
        # Create a mock PyTorch model
        model = MagicMock()
        torch_mock.nn.Module = MagicMock
        
        # Ensure the model is detected as a PyTorch model
        with patch("secureml.federated.isinstance", return_value=True):
            framework = _detect_framework(model)
            self.assertEqual(framework, "pytorch")
    
    @patch("secureml.federated._HAS_TENSORFLOW", True)
    def test_detect_tensorflow(self):
        """Test that TensorFlow models are correctly detected."""
        # Create a mock TensorFlow model
        model = MagicMock()
        tf_mock.keras.Model = MagicMock
        tf_mock.Module = MagicMock
        
        # Mock the isinstance check to return True for TensorFlow models
        def mock_isinstance(obj, class_or_tuple):
            if class_or_tuple == torch_mock.nn.Module:
                return False
            return True
        
        # Ensure the model is detected as a TensorFlow model
        with patch("secureml.federated.isinstance", mock_isinstance):
            framework = _detect_framework(model)
            self.assertEqual(framework, "tensorflow")
    
    @patch("secureml.federated._HAS_PYTORCH", True)
    @patch("secureml.federated._HAS_TENSORFLOW", True)
    def test_detect_unknown_framework(self):
        """Test that ValueError is raised for unknown models."""
        # Create a model that is neither PyTorch nor TensorFlow
        model = MagicMock()
        
        # Mock the isinstance check to always return False
        with patch("secureml.federated.isinstance", return_value=False):
            with self.assertRaises(ValueError):
                _detect_framework(model)


class TestDataPreparation(unittest.TestCase):
    """Test cases for data preparation functions."""
    
    def test_prepare_data_for_pytorch_dataframe(self):
        """Test preparing pandas DataFrame for PyTorch."""
        # Create a test DataFrame
        df = pd.DataFrame({
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "target": [0, 1, 0]
        })
        
        # Prepare data with target column specified
        x, y = _prepare_data_for_pytorch(df, target_column="target")
        
        # Check that features and target are correctly separated
        self.assertEqual(x.shape, (3, 2))
        self.assertEqual(y.shape, (3,))
        self.assertTrue(np.array_equal(y, np.array([0, 1, 0])))
        
        # Prepare data without target column specified (using last column)
        x, y = _prepare_data_for_pytorch(df)
        
        # Check that features and target are correctly separated
        self.assertEqual(x.shape, (3, 2))
        self.assertEqual(y.shape, (3,))
        self.assertTrue(np.array_equal(y, np.array([0, 1, 0])))
    
    def test_prepare_data_for_pytorch_numpy(self):
        """Test preparing numpy array for PyTorch."""
        # Create a test numpy array
        array = np.array([
            [1, 4, 0],
            [2, 5, 1],
            [3, 6, 0]
        ])
        
        # Prepare data (using last column as target)
        x, y = _prepare_data_for_pytorch(array)
        
        # Check that features and target are correctly separated
        self.assertEqual(x.shape, (3, 2))
        self.assertEqual(y.shape, (3,))
        self.assertTrue(np.array_equal(y, np.array([0, 1, 0])))
    
    def test_prepare_data_for_tensorflow(self):
        """Test that tensorflow data preparation uses pytorch preparation."""
        # Create a test DataFrame
        df = pd.DataFrame({
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "target": [0, 1, 0]
        })
        
        # Verify that the tensorflow preparation function calls the pytorch one
        with patch("secureml.federated._prepare_data_for_pytorch") as mock_prepare:
            mock_prepare.return_value = (np.array([[1, 4], [2, 5], [3, 6]]), np.array([0, 1, 0]))
            x, y = _prepare_data_for_tensorflow(df, target_column="target")
            mock_prepare.assert_called_once_with(df, target_column="target")
            
            # Check that the output matches what we expect from _prepare_data_for_pytorch
            self.assertEqual(x.shape, (3, 2))
            self.assertEqual(y.shape, (3,))


@patch("secureml.federated._HAS_FLOWER", True)
class TestTrainFederated(unittest.TestCase):
    """Test cases for the train_federated function."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for model files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create a mock model
        self.model = MagicMock()
        
        # Create mock client data
        self.client_data = {
            "client1": pd.DataFrame({
                "feature1": np.random.normal(0, 1, 10),
                "feature2": np.random.normal(0, 1, 10),
                "target": np.random.randint(0, 2, 10)
            }),
            "client2": pd.DataFrame({
                "feature1": np.random.normal(0, 1, 10),
                "feature2": np.random.normal(0, 1, 10),
                "target": np.random.randint(0, 2, 10)
            })
        }
        
        # Create a client data function
        self.client_data_fn = lambda: self.client_data
    
    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()
    
    def test_train_federated_pytorch(self):
        """Test the train_federated function with PyTorch."""
        # Setup required mocks
        simulation_mock = MagicMock()
        with patch.object(fl_mock.simulation, 'start_simulation', simulation_mock):
            
            # Set up framework detection to return pytorch
            with patch("secureml.federated._detect_framework", return_value="pytorch"):
                with patch("secureml.federated._create_pytorch_client_fn") as mock_client_fn:
                    with patch("secureml.federated._create_pytorch_server") as mock_server:
                        # Run function
                        result = train_federated(
                            model=self.model,
                            client_data_fn=self.client_data_fn,
                            framework="pytorch"
                        )
                        
                        # Assert correct functions were called
                        mock_client_fn.assert_called_once()
                        mock_server.assert_called_once()
                        simulation_mock.assert_called_once()
                        
                        # Check that the model was returned
                        self.assertEqual(result, self.model)
    
    def test_train_federated_tensorflow(self):
        """Test the train_federated function with TensorFlow."""
        # Setup required mocks
        simulation_mock = MagicMock()
        with patch.object(fl_mock.simulation, 'start_simulation', simulation_mock):
            
            # Set up framework detection to return tensorflow
            with patch("secureml.federated._detect_framework", return_value="tensorflow"):
                with patch("secureml.federated._create_tensorflow_client_fn") as mock_client_fn:
                    with patch("secureml.federated._create_tensorflow_server") as mock_server:
                        # Run function
                        result = train_federated(
                            model=self.model,
                            client_data_fn=self.client_data_fn,
                            framework="tensorflow"
                        )
                        
                        # Assert correct functions were called
                        mock_client_fn.assert_called_once()
                        mock_server.assert_called_once()
                        simulation_mock.assert_called_once()
                        
                        # Check that the model was returned
                        self.assertEqual(result, self.model)
    
    def test_train_federated_auto_detection(self):
        """Test train_federated with auto framework detection."""
        # Setup required mocks
        simulation_mock = MagicMock()
        with patch.object(fl_mock.simulation, 'start_simulation', simulation_mock):
            
            # Set up framework detection to return pytorch
            with patch("secureml.federated._detect_framework", return_value="pytorch") as mock_detect:
                with patch("secureml.federated._create_pytorch_client_fn"):
                    with patch("secureml.federated._create_pytorch_server"):
                        # Run function with auto framework detection
                        train_federated(
                            model=self.model,
                            client_data_fn=self.client_data_fn,
                            framework="auto"
                        )
                        
                        # Check that framework detection was called
                        mock_detect.assert_called_once_with(self.model)
    
    def test_train_federated_save_model(self):
        """Test that train_federated saves the model when a path is provided."""
        # Setup required mocks
        simulation_mock = MagicMock()
        with patch.object(fl_mock.simulation, 'start_simulation', simulation_mock):
            
            # Create a file path for saving the model
            model_path = os.path.join(self.temp_dir.name, "model.pt")
            
            with patch("secureml.federated._detect_framework", return_value="pytorch"):
                with patch("secureml.federated._create_pytorch_client_fn"):
                    with patch("secureml.federated._create_pytorch_server"):
                        with patch("secureml.federated._save_model") as mock_save:
                            # Run function with model save path
                            train_federated(
                                model=self.model,
                                client_data_fn=self.client_data_fn,
                                framework="pytorch",
                                model_save_path=model_path
                            )
                            
                            # Check that save_model was called
                            mock_save.assert_called_once_with(
                                self.model, model_path, "pytorch"
                            )
    
    def test_train_federated_custom_config(self):
        """Test train_federated with a custom config."""
        # Setup required mocks
        simulation_mock = MagicMock()
        with patch.object(fl_mock.simulation, 'start_simulation', simulation_mock):
            
            # Create a custom config
            config = FederatedConfig(
                num_rounds=5,
                fraction_fit=0.8,
                min_fit_clients=3,
                apply_differential_privacy=True
            )
            
            with patch("secureml.federated._detect_framework", return_value="pytorch"):
                with patch("secureml.federated._create_pytorch_client_fn") as mock_client_fn:
                    with patch("secureml.federated._create_pytorch_server") as mock_server:
                        # Run function with custom config
                        train_federated(
                            model=self.model,
                            client_data_fn=self.client_data_fn,
                            config=config,
                            framework="pytorch"
                        )
                        
                        # Check that the config was passed to the client_fn and server functions
                        args, kwargs = mock_client_fn.call_args
                        self.assertEqual(args[2], config)
                        
                        args, kwargs = mock_server.call_args
                        self.assertEqual(args[1], config)
                        
                        # Check that the simulation was started with the correct number of rounds
                        args, kwargs = simulation_mock.call_args
                        self.assertEqual(kwargs["config"].num_rounds, 5)


@patch("secureml.federated._HAS_FLOWER", True)
class TestStartFederatedServer(unittest.TestCase):
    """Test cases for the start_federated_server function."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a mock model
        self.model = MagicMock()
    
    def test_start_federated_server_pytorch(self):
        """Test start_federated_server with PyTorch."""
        # Setup required mocks
        server_mock = MagicMock()
        with patch.object(fl_mock.server, 'start_server', server_mock):
            
            with patch("secureml.federated._detect_framework", return_value="pytorch"):
                with patch("secureml.federated._create_pytorch_server") as mock_server:
                    # Run function
                    start_federated_server(
                        model=self.model,
                        framework="pytorch"
                    )
                    
                    # Check that the server was created and started
                    mock_server.assert_called_once()
                    server_mock.assert_called_once()
    
    def test_start_federated_server_tensorflow(self):
        """Test start_federated_server with TensorFlow."""
        # Setup required mocks
        server_mock = MagicMock()
        with patch.object(fl_mock.server, 'start_server', server_mock):
            
            with patch("secureml.federated._detect_framework", return_value="tensorflow"):
                with patch("secureml.federated._create_tensorflow_server") as mock_server:
                    # Run function
                    start_federated_server(
                        model=self.model,
                        framework="tensorflow"
                    )
                    
                    # Check that the server was created and started
                    mock_server.assert_called_once()
                    server_mock.assert_called_once()
    
    def test_start_federated_server_custom_config(self):
        """Test start_federated_server with custom config."""
        # Setup required mocks
        server_mock = MagicMock()
        with patch.object(fl_mock.server, 'start_server', server_mock):
            
            # Create a custom config
            config = FederatedConfig(
                num_rounds=5,
                server_address="localhost:8081",
                use_secure_aggregation=True
            )
            
            with patch("secureml.federated._detect_framework", return_value="pytorch"):
                with patch("secureml.federated._create_pytorch_server") as mock_server:
                    # Run function with custom config
                    start_federated_server(
                        model=self.model,
                        config=config,
                        framework="pytorch"
                    )
                    
                    # Check that the server was created with the config
                    mock_server.assert_called_once_with(self.model, config)
                    
                    # Check that the server was started with the correct address and config
                    args, kwargs = server_mock.call_args
                    self.assertEqual(kwargs["server_address"], "localhost:8081")
                    self.assertEqual(kwargs["config"].num_rounds, 5)


@patch("secureml.federated._HAS_FLOWER", True)
class TestStartFederatedClient(unittest.TestCase):
    """Test cases for the start_federated_client function."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a mock model
        self.model = MagicMock()
        
        # Create mock data
        self.data = pd.DataFrame({
            "feature1": np.random.normal(0, 1, 10),
            "feature2": np.random.normal(0, 1, 10),
            "target": np.random.randint(0, 2, 10)
        })
    
    def test_start_federated_client_pytorch(self):
        """Test start_federated_client with PyTorch."""
        # Setup required mocks
        client_mock = MagicMock()
        with patch.object(fl_mock.client, 'start_client', client_mock):
            
            with patch("secureml.federated._detect_framework", return_value="pytorch"):
                with patch("secureml.federated._create_pytorch_client") as mock_client:
                    # Run function
                    start_federated_client(
                        model=self.model,
                        data=self.data,
                        server_address="localhost:8080",
                        framework="pytorch"
                    )
                    
                    # Check that the client was created and started
                    mock_client.assert_called_once()
                    client_mock.assert_called_once_with(
                        server_address="localhost:8080",
                        client=mock_client.return_value
                    )
    
    def test_start_federated_client_tensorflow(self):
        """Test start_federated_client with TensorFlow."""
        # Setup required mocks
        client_mock = MagicMock()
        with patch.object(fl_mock.client, 'start_client', client_mock):
            
            with patch("secureml.federated._detect_framework", return_value="tensorflow"):
                with patch("secureml.federated._create_tensorflow_client") as mock_client:
                    # Run function
                    start_federated_client(
                        model=self.model,
                        data=self.data,
                        server_address="localhost:8080",
                        framework="tensorflow"
                    )
                    
                    # Check that the client was created and started
                    mock_client.assert_called_once()
                    client_mock.assert_called_once_with(
                        server_address="localhost:8080",
                        client=mock_client.return_value
                    )
    
    def test_start_federated_client_with_differential_privacy(self):
        """Test start_federated_client with differential privacy enabled."""
        # Setup required mocks
        client_mock = MagicMock()
        with patch.object(fl_mock.client, 'start_client', client_mock):
            
            with patch("secureml.federated._detect_framework", return_value="pytorch"):
                with patch("secureml.federated._create_pytorch_client") as mock_client:
                    # Run function with differential privacy
                    start_federated_client(
                        model=self.model,
                        data=self.data,
                        server_address="localhost:8080",
                        framework="pytorch",
                        apply_differential_privacy=True,
                        epsilon=0.5,
                        delta=1e-6
                    )
                    
                    # Check that the client was created with DP parameters
                    mock_client.assert_called_once_with(
                        self.model,
                        self.data,
                        apply_differential_privacy=True,
                        epsilon=0.5,
                        delta=1e-6
                    )


@patch("secureml.federated._HAS_PYTORCH", True)
class TestSaveModel(unittest.TestCase):
    """Test cases for the _save_model function."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for model files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create a mock model
        self.model = MagicMock()
    
    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()
    
    def test_save_pytorch_model(self):
        """Test saving a PyTorch model."""
        # Create a file path for saving the model
        model_path = os.path.join(self.temp_dir.name, "model.pt")
        
        # Save the model
        with patch("secureml.federated.torch.save") as mock_save:
            _save_model(self.model, model_path, "pytorch")
            
            # Check that torch.save was called
            mock_save.assert_called_once()
    
    def test_save_tensorflow_model(self):
        """Test saving a TensorFlow model."""
        # Create a file path for saving the model
        model_path = os.path.join(self.temp_dir.name, "model")
        
        # Save the model
        with patch("secureml.federated._HAS_TENSORFLOW", True):
            _save_model(self.model, model_path, "tensorflow")
            
            # Check that model.save was called
            self.model.save.assert_called_once_with(model_path)
    
    def test_save_unknown_framework_model(self):
        """Test that warning is issued for unknown frameworks."""
        # Create a file path for saving the model
        model_path = os.path.join(self.temp_dir.name, "model")
        
        # Save the model with an unknown framework
        with patch("secureml.federated.warnings.warn") as mock_warn:
            _save_model(self.model, model_path, "unknown")
            
            # Check that a warning was issued
            mock_warn.assert_called_once()


# Add pytest-style tests
@pytest.fixture
def mock_model():
    """Create a mock model fixture."""
    return MagicMock()


@pytest.fixture
def mock_data():
    """Create mock data fixture."""
    return pd.DataFrame({
        "feature1": np.random.normal(0, 1, 10),
        "feature2": np.random.normal(0, 1, 10),
        "target": np.random.randint(0, 2, 10)
    })


@pytest.fixture
def mock_client_data_fn():
    """Create a mock client data function fixture."""
    data = {
        "client1": pd.DataFrame({
            "feature1": np.random.normal(0, 1, 10),
            "feature2": np.random.normal(0, 1, 10),
            "target": np.random.randint(0, 2, 10)
        }),
        "client2": pd.DataFrame({
            "feature1": np.random.normal(0, 1, 10),
            "feature2": np.random.normal(0, 1, 10),
            "target": np.random.randint(0, 2, 10)
        })
    }
    return lambda: data


@pytest.mark.parametrize(
    "framework", 
    ["pytorch", "tensorflow", "auto"]
)
def test_train_federated_frameworks(mock_model, mock_client_data_fn, framework, monkeypatch):
    """Test train_federated with different frameworks."""
    # Setup mocks
    simulation_mock = MagicMock()
    monkeypatch.setattr("secureml.federated.fl.simulation.start_simulation", simulation_mock)
    monkeypatch.setattr("secureml.federated._HAS_FLOWER", True)
    
    # Mock framework detection
    detect_mock = MagicMock(return_value="pytorch")
    monkeypatch.setattr("secureml.federated._detect_framework", detect_mock)
    
    # Mock client and server creation
    client_fn_mock = MagicMock()
    server_mock = MagicMock()
    monkeypatch.setattr("secureml.federated._create_pytorch_client_fn", client_fn_mock)
    monkeypatch.setattr("secureml.federated._create_pytorch_server", server_mock)
    
    # Run function
    train_federated(
        model=mock_model,
        client_data_fn=mock_client_data_fn,
        framework=framework
    )
    
    # Check that simulation was started
    assert simulation_mock.call_count == 1
    
    # If auto framework detection was used, check it was called
    if framework == "auto":
        assert detect_mock.call_count == 1
        detect_mock.assert_called_with(mock_model)


@pytest.mark.parametrize(
    "differential_privacy,epsilon,delta", 
    [
        (False, 1.0, 1e-5),
        (True, 0.5, 1e-6)
    ]
)
def test_client_differential_privacy(mock_model, mock_data, differential_privacy, epsilon, delta, monkeypatch):
    """Test client creation with different differential privacy settings."""
    # Setup mocks
    client_mock = MagicMock()
    monkeypatch.setattr("secureml.federated.fl.client.start_client", client_mock)
    monkeypatch.setattr("secureml.federated._HAS_FLOWER", True)
    
    # Mock client creation
    client_instance_mock = MagicMock()
    create_client_mock = MagicMock(return_value=client_instance_mock)
    monkeypatch.setattr("secureml.federated._create_pytorch_client", create_client_mock)
    monkeypatch.setattr("secureml.federated._detect_framework", MagicMock(return_value="pytorch"))
    
    # Run function
    start_federated_client(
        model=mock_model,
        data=mock_data,
        server_address="localhost:8080",
        framework="pytorch",
        apply_differential_privacy=differential_privacy,
        epsilon=epsilon,
        delta=delta
    )
    
    # Check client was created with correct DP settings
    create_client_mock.assert_called_once_with(
        mock_model,
        mock_data,
        apply_differential_privacy=differential_privacy,
        epsilon=epsilon,
        delta=delta
    )
    
    # Check client was started
    client_mock.assert_called_once_with(
        server_address="localhost:8080",
        client=client_instance_mock
    )


if __name__ == "__main__":
    unittest.main() 