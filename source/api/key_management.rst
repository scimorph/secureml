=================
Key Management API
=================

.. module:: secureml.key_management

This module provides tools for securely managing encryption keys and other secrets using HashiCorp Vault, implementing best practices for secure key storage, rotation, and access.

KeyManager Class
---------------

.. autoclass:: KeyManager
   :members:
   :special-members: __init__

The ``KeyManager`` class provides methods for retrieving and managing secrets from various backends, with a primary focus on HashiCorp Vault for production environments and fallback to environment variables or default values for development.

Basic Usage Example:

.. code-block:: python

    from secureml.key_management import KeyManager
    
    # Initialize a key manager
    key_manager = KeyManager(
        vault_url="https://vault.example.com:8200",
        vault_token="s.your-vault-token",
        vault_path="secureml",
        use_env_fallback=True,
        use_default_fallback=False
    )
    
    # Retrieve a secret
    api_key = key_manager.get_secret("api_key")
    
    # Get an encryption key
    encryption_key = key_manager.get_encryption_key(
        key_name="master_key", 
        key_bytes=32, 
        encoding="bytes"
    )

Global Configuration
-------------------

.. autofunction:: configure_default_key_manager

The module provides a global configuration function to set up a default key manager for the entire application:

.. code-block:: python

    from secureml.key_management import configure_default_key_manager
    
    # Configure the default key manager (do this once at startup)
    configure_default_key_manager(
        vault_url="https://vault.example.com:8200",
        vault_token="s.your-vault-token",
        vault_path="secureml",
        use_env_fallback=True
    )

Utility Functions
---------------

.. autofunction:: get_encryption_key

This utility function uses the default key manager to retrieve encryption keys:

.. code-block:: python

    from secureml.key_management import get_encryption_key
    
    # Get an encryption key using the default key manager
    key = get_encryption_key(
        key_name="data_encryption_key",
        encoding="base64"
    )
    
    # Use the key for encryption operations
    # ...

HashiCorp Vault Integration
------------------------

The module seamlessly integrates with HashiCorp Vault for secure key storage and management. To use Vault:

1. Ensure the `hvac` package is installed: `pip install secureml[vault]`
2. Configure Vault connection details when initializing KeyManager
3. Store your secrets in Vault using the `set_secret` method

Fallback Mechanisms
----------------

For ease of development and handling different environments, the module provides fallback mechanisms:

1. **Environment Variables**: When Vault is unavailable, the manager can fall back to environment variables
   prefixed with "SECUREML_" (e.g., `SECUREML_API_KEY`)

2. **Default Values**: For development environments only, the manager can generate deterministic default values

Key Derivation
------------

The `derive_key` method allows deriving purpose-specific keys from a base key, implementing key separation:

.. code-block:: python

    # Get the master key
    master_key = key_manager.get_encryption_key(key_name="master_key")
    
    # Derive purpose-specific keys
    encryption_key = key_manager.derive_key(master_key, purpose="data_encryption")
    hmac_key = key_manager.derive_key(master_key, purpose="data_integrity")

Best Practices
------------

1. **Use Vault in Production**: Always use HashiCorp Vault or another secure key management solution in production
2. **Disable Default Fallbacks**: Set `use_default_fallback=False` in production to prevent insecure defaults
3. **Key Rotation**: Implement regular key rotation practices
4. **Separate Keys by Purpose**: Use key derivation to create separate keys for different purposes
5. **Environment Variables**: For simple deployments, use environment variables with appropriate access controls
6. **Never Hardcode Secrets**: Never include secrets or keys directly in your code
