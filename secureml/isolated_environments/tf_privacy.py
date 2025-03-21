"""
TensorFlow Privacy isolated environment manager.

This module provides functionality to create and interact with an isolated
environment containing tensorflow-privacy with its specific dependencies.
"""

import json
import os
import subprocess
import sys
import tempfile
import venv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

ENV_NAME = "tf_privacy_env"
REQUIREMENTS = [
    "tensorflow-privacy==0.9.0",
    "packaging==22.0",
    "numpy>=1.24.0",
    "pandas>=1.5.0",
    "tensorflow>=2.11.0,<3.0.0",
]


def get_env_path() -> Path:
    """Get the path to the isolated environment."""
    # Store the env in the user's home directory to avoid permission issues
    base_dir = Path.home() / ".secureml" / "envs"
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / ENV_NAME


def create_environment(force: bool = False) -> Path:
    """
    Create a virtual environment for tensorflow-privacy.
    
    Args:
        force: If True, recreate the environment even if it exists
        
    Returns:
        Path to the virtual environment
    """
    env_path = get_env_path()
    
    # Check if environment already exists
    if env_path.exists() and not force:
        return env_path
        
    # Remove existing environment if forced
    if env_path.exists() and force:
        import shutil
        shutil.rmtree(env_path)
    
    # Create the virtual environment
    venv.create(env_path, with_pip=True)
    
    # Get pip path
    if sys.platform == "win32":
        pip_path = env_path / "Scripts" / "pip.exe"
    else:
        pip_path = env_path / "bin" / "pip"
    
    # Install requirements
    for req in REQUIREMENTS:
        subprocess.check_call([str(pip_path), "install", req])
    
    return env_path


def get_python_executable() -> str:
    """Get the path to the Python executable in the isolated environment."""
    env_path = get_env_path()
    
    if not env_path.exists():
        env_path = create_environment()
    
    if sys.platform == "win32":
        python_path = env_path / "Scripts" / "python.exe"
    else:
        python_path = env_path / "bin" / "python"
    
    return str(python_path)


def run_tf_privacy_function(
    function_name: str, 
    args: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Run a tensorflow-privacy function in the isolated environment.
    
    Args:
        function_name: The name of the function to run 
                      (in the format 'module.submodule.function')
        args: Dictionary of arguments to pass to the function
        
    Returns:
        The return value of the function as a dictionary
    """
    # Ensure the environment exists
    if not get_env_path().exists():
        create_environment()
    
    # Create a temporary file to pass the arguments
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as f:
        json.dump(args, f)
        args_file = f.name
    
    # Create a temporary file to store the result
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        result_file = f.name
    
    # Create a Python script to run in the isolated environment
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.py', delete=False) as f:
        f.write(f"""
import json
import sys

# Load the arguments
with open("{args_file}", "r") as f:
    args = json.load(f)

# Import the function
function_path = "{function_name}"
module_path, function_name = function_path.rsplit(".", 1)
module = __import__(module_path, fromlist=[function_name])
function = getattr(module, function_name)

# Call the function
result = function(**args)

# Convert result to a serializable format if needed
if hasattr(result, "to_dict"):
    result = result.to_dict()
elif hasattr(result, "__dict__"):
    result = result.__dict__

# Save the result
with open("{result_file}", "w") as f:
    json.dump(result, f)
""")
        script_file = f.name
    
    # Run the script in the isolated environment
    python_executable = get_python_executable()
    subprocess.check_call([python_executable, script_file])
    
    # Load the result
    with open(result_file, "r") as f:
        result = json.load(f)
    
    # Clean up temporary files
    os.unlink(args_file)
    os.unlink(result_file)
    os.unlink(script_file)
    
    return result 