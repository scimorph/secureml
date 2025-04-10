"""
Isolated environment functionality for TensorFlow Privacy.

This module provides functionality to run TensorFlow Privacy code in an isolated
environment to avoid dependency conflicts with the main project.
"""

import os
import sys
import json
import subprocess
import tempfile
import importlib
from typing import Any, Dict


def run_tf_privacy_function(function_path: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a function in an isolated environment with TensorFlow Privacy installed.
    
    Args:
        function_path: The import path to the function (e.g., 'module.submodule.function')
        args: Arguments to pass to the function
        
    Returns:
        The function result (must be JSON serializable)
        
    Raises:
        RuntimeError: If there's an error setting up the environment or running the function
    """
    # Get the path to the virtual environment
    venv_path = _get_or_create_venv()
    
    # Create temporary files for input and output
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as input_file:
        input_path = input_file.name
        # Write arguments to input file
        json.dump(args, input_file, indent=2)
    
    output_path = input_path + '.output.json'
    
    # Replace backslashes with forward slashes for cross-platform compatibility
    input_path_normalized = input_path.replace("\\", "/")
    output_path_normalized = output_path.replace("\\", "/")
    
    # Create the runner script content with normalized paths
    runner_script = f"""
import json
import sys
import traceback

# Load input arguments
with open("{input_path_normalized}") as f:
    args = json.load(f)

try:
    # Import the function dynamically
    module_path, function_name = "{function_path}".rsplit('.', 1)
    module = __import__(module_path, fromlist=[function_name])
    function = getattr(module, function_name)
    
    # Call the function with the arguments
    result = function(**args)
    
    # Save the result
    with open("{output_path_normalized}", 'w') as f:
        json.dump(result, f, indent=2)
    
    sys.exit(0)
except Exception as e:
    # Save error information
    with open("{output_path_normalized}", 'w') as f:
        error_info = {{
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }}
        json.dump(error_info, f, indent=2)
    
    sys.exit(1)
"""
    
    # Write the runner script to a temporary file with explicit UTF-8 encoding
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as script_file:
        script_path = script_file.name
        script_file.write(runner_script)
    
    try:
        # Build the command to run the script in the virtual environment
        if sys.platform.startswith('win'):
            python_executable = os.path.join(venv_path, 'Scripts', 'python.exe')
        else:
            python_executable = os.path.join(venv_path, 'bin', 'python')
        
        # Run the script in the virtual environment
        result = subprocess.run(
            [python_executable, script_path],
            capture_output=True,
            text=True,
        )
        
        # Check if the script execution was successful
        if result.returncode != 0:
            raise RuntimeError(
                f"Error executing function in isolated environment:\n"
                f"STDOUT: {result.stdout}\n"
                f"STDERR: {result.stderr}"
            )
        
        # Read the output file
        with open(output_path, 'r') as f:
            function_result = json.load(f)
        
        return function_result
    
    finally:
        # Clean up temporary files
        for path in [input_path, output_path, script_path]:
            try:
                if os.path.exists(path):
                    os.unlink(path)
            except Exception:
                pass


def _get_or_create_venv() -> str:
    """
    Get the path to the TensorFlow Privacy virtual environment or create it if it doesn't exist.
    
    Returns:
        Path to the virtual environment
        
    Raises:
        RuntimeError: If there's an error creating the virtual environment
    """
    # Get the path to the virtual environment
    secureml_dir = os.path.expanduser("~/.secureml")
    venv_path = os.path.join(secureml_dir, "tf_privacy_venv")
    
    # Check if the virtual environment already exists
    if _is_venv_valid(venv_path):
        return venv_path
    
    # Create the directory if it doesn't exist
    os.makedirs(secureml_dir, exist_ok=True)
    
    # Create the virtual environment
    print("Creating isolated virtual environment for TensorFlow Privacy...")

    # Determine the Python executable to use
    tf_privacy_python = os.environ.get("TF_PRIVACY_PYTHON")
    if tf_privacy_python:
        python_executable = tf_privacy_python
    elif sys.version_info[:2] == (3, 11):
        python_executable = sys.executable
    else:
        raise RuntimeError(
            "The isolated environment for TensorFlow Privacy requires Python 3.11.\n"
            "Please set the TF_PRIVACY_PYTHON environment variable to the path of a Python 3.11 executable.\n"
            "For example, on Unix-like systems: export TF_PRIVACY_PYTHON=/usr/bin/python3.11\n"
            "On Windows: set TF_PRIVACY_PYTHON=C:\\Python311\\python.exe"
        )
    
    try:
        # Create virtual environment with the chosen executable
        subprocess.run(
            [python_executable, "-m", "venv", venv_path],
            check=True,
            capture_output=True,
        )
        
        # Determine Python executable path
        if sys.platform.startswith('win'):
            python_executable = os.path.join(venv_path, 'Scripts', 'python.exe')
        else:
            python_executable = os.path.join(venv_path, 'bin', 'python')
        
        # Upgrade pip using python -m pip
        subprocess.run(
            [python_executable, "-m", "pip", "install", "--upgrade", "pip"],
            check=True,
            capture_output=True,
        )
        
        # Install tensorflow and tensorflow-privacy
        print("Installing TensorFlow Privacy in the isolated environment...")
        subprocess.run(
            [python_executable, "-m", "pip", "install", "tensorflow==2.15.0", "tensorflow-privacy"],
            check=True,
            capture_output=True,
        )
        
        # Install other dependencies
        subprocess.run(
            [python_executable, "-m", "pip", "install", "numpy", "pandas"],
            check=True,
            capture_output=True,
        )
        
        # Install SecureML in development mode if applicable
        secureml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        if os.path.exists(os.path.join(secureml_path, "setup.py")):
            subprocess.run(
                [python_executable, "-m", "pip", "install", "-e", secureml_path],
                check=True,
                capture_output=True,
            )
        
        print("TensorFlow Privacy environment setup complete.")
        
        return venv_path
    
    except subprocess.CalledProcessError as e:
        # Clean up the virtual environment if creation failed
        import shutil
        if os.path.exists(venv_path):
            shutil.rmtree(venv_path, ignore_errors=True)
        
        raise RuntimeError(
            f"Error creating virtual environment for TensorFlow Privacy:\n"
            f"STDOUT: {e.stdout.decode() if e.stdout else ''}\n"
            f"STDERR: {e.stderr.decode() if e.stderr else ''}"
        )


def _is_venv_valid(venv_path: str) -> bool:
    """
    Check if the TensorFlow Privacy virtual environment is valid and has all required packages.
    
    Args:
        venv_path: Path to the virtual environment
        
    Returns:
        True if the virtual environment is valid, False otherwise
    """
    if not os.path.exists(venv_path):
        return False
    
    # Check if python executable exists
    if sys.platform.startswith('win'):
        python_executable = os.path.join(venv_path, 'Scripts', 'python.exe')
    else:
        python_executable = os.path.join(venv_path, 'bin', 'python')
    
    if not os.path.exists(python_executable):
        return False
    
    # Check Python version
    try:
        result = subprocess.run(
            [python_executable, "-c", "import sys; print('.'.join(map(str, sys.version_info[:2])))"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or result.stdout.strip() != "3.11":
            return False
    except Exception:
        return False
    
    # Check if required packages are installed
    try:
        result = subprocess.run(
            [python_executable, "-c", "import tensorflow, tensorflow_privacy, numpy, pandas"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except Exception:
        return False


def get_env_path() -> str:
    """
    Get the path to the TensorFlow Privacy virtual environment.
    
    Returns:
        Path to the virtual environment
    """
    secureml_dir = os.path.expanduser("~/.secureml")
    return os.path.join(secureml_dir, "tf_privacy_venv")


def is_env_valid() -> bool:
    """
    Check if the TensorFlow Privacy virtual environment is valid and ready to use.
    
    Returns:
        True if the virtual environment is valid, False otherwise
    """
    venv_path = get_env_path()
    return _is_venv_valid(venv_path)


def setup_tf_privacy_environment() -> None:
    """
    Set up the TensorFlow Privacy environment.
    
    This function can be called explicitly to set up the environment in advance.
    
    Raises:
        RuntimeError: If there's an error setting up the environment
    """
    _get_or_create_venv()
    print("TensorFlow Privacy environment is ready.") 