"""
Isolated environments for handling dependency conflicts.

This package contains modules for running code in isolated environments
to avoid dependency conflicts with the main SecureML package.
"""

from secureml.isolated_environments.tf_privacy import (
    run_tf_privacy_function, 
    setup_tf_privacy_environment,
    get_env_path,
    is_env_valid
)

__all__ = [
    "run_tf_privacy_function",
    "setup_tf_privacy_environment",
    "get_env_path",
    "is_env_valid",
] 