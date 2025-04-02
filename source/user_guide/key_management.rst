=============
Key Management
=============

Secure key management is critical for protecting sensitive data in machine learning workflows. SecureML provides robust key management capabilities, with special focus on integration with HashiCorp Vault for enterprise-grade key storage and management.

Core Concepts
------------

**Encryption Keys**: Cryptographic keys used to encrypt and decrypt sensitive data.

**Key Rotation**: The practice of periodically changing encryption keys to limit the impact of key compromise.

**Key Hierarchy**: A structured approach to organizing keys, typically with master keys protecting data encryption keys.

**Secret Storage**: Secure storage for API keys, credentials, and other sensitive configuration information.

Basic Usage
----------

Setting Up Key Management
^^^^^^^^^^^^^^^^^^^^^^^

Initialize the key management system:

.. code-block:: python

    from secureml.key_management import KeyManager
    
    # Initialize the key manager
    key_manager = KeyManager(
        vault_url='https://vault.example.com:8200',
        vault_token=os.environ.get('VAULT_TOKEN'),
        vault_path='secureml',
        use_env_fallback=True,
        use_default_fallback=False
    )
    
    # Check if the connection was successful
    if key_manager.vault_client:
        print("Successfully connected to HashiCorp Vault")
    else:
        print("Failed to connect to Vault, using fallback mechanisms")

Storing and Retrieving Secrets
^^^^^^^^^^^^^^^^^^^^^^^^

Store and retrieve secrets:

.. code-block:: python

    # Store a secret in Vault
    success = key_manager.set_secret(
        secret_name='api_key',
        value='your-api-key-value'
    )
    
    if success:
        print("Secret stored successfully")
    
    # Retrieve a secret
    api_key = key_manager.get_secret(
        secret_name='api_key',
        default='fallback-api-key'  # Optional default value
    )
    
    # Use the secret in your application
    print(f"Using API key: {api_key[:5]}...")

Getting Encryption Keys
^^^^^^^^^^^^^^^^^^^^^

Get encryption keys for secure operations:

.. code-block:: python

    # Get an encryption key in bytes format
    key_bytes = key_manager.get_encryption_key(
        key_name='master_key',
        key_bytes=32,
        encoding='bytes'
    )
    
    # Get a key in hexadecimal format
    key_hex = key_manager.get_encryption_key(
        key_name='master_key',
        key_bytes=32,
        encoding='hex'
    )
    
    # Get a key in base64 format
    key_base64 = key_manager.get_encryption_key(
        key_name='master_key',
        key_bytes=32,
        encoding='base64'
    )

Deriving Purpose-Specific Keys
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Derive keys for specific purposes from a master key:

.. code-block:: python

    # Get the master key
    master_key = key_manager.get_encryption_key(key_name='master_key')
    
    # Derive a key for data encryption
    data_key = key_manager.derive_key(
        base_key=master_key,
        purpose='data_encryption',
        key_bytes=32
    )
    
    # Derive a key for model encryption
    model_key = key_manager.derive_key(
        base_key=master_key,
        purpose='model_encryption',
        key_bytes=32
    )

Global Configuration
----------------

To make key management available throughout your application, use the global configuration:

.. code-block:: python

    from secureml.key_management import configure_default_key_manager, get_encryption_key
    
    # Configure the default key manager once at application startup
    configure_default_key_manager(
        vault_url='https://vault.example.com:8200',
        vault_token=os.environ.get('VAULT_TOKEN'),
        vault_path='secureml',
        use_env_fallback=True,
        use_default_fallback=False  # Only use for development!
    )
    
    # Later, use the global function to get encryption keys anywhere in your code
    key = get_encryption_key(
        key_name='master_key',
        key_bytes=32,
        encoding='bytes'
    )

HashiCorp Vault Integration
-------------------------

SecureML integrates with HashiCorp Vault for enterprise-grade key management.

Authentication
^^^^^^^^^^^

SecureML uses token-based authentication with Vault by default. You can provide the token directly or via environment variables:

.. code-block:: python

    # Direct token authentication
    key_manager = KeyManager(
        vault_url='https://vault.example.com:8200',
        vault_token='hvs.example'
    )
    
    # Environment-based authentication
    # First set: export SECUREML_VAULT_URL=https://vault.example.com:8200
    # First set: export SECUREML_VAULT_TOKEN=hvs.example
    key_manager = KeyManager()  # Will use environment variables

Vault KV Secret Engine
^^^^^^^^^^^^^^^^^^

SecureML uses Vault's KV (Key-Value) v2 secret engine to store secrets:

.. code-block:: python

    # Store a secret
    key_manager.set_secret(
        secret_name='database_credentials',
        value={
            'username': 'db_user',
            'password': 'db_password',
            'host': 'db.example.com'
        }
    )
    
    # Retrieve a secret
    credentials = key_manager.get_secret('database_credentials')

Fallback Mechanisms
^^^^^^^^^^^^^^^^

SecureML provides fallback mechanisms when Vault is unavailable:

.. code-block:: python

    # Configure with fallbacks
    key_manager = KeyManager(
        vault_url='https://vault.example.com:8200',
        vault_token=os.environ.get('VAULT_TOKEN'),
        use_env_fallback=True,  # Try environment variables if Vault fails
        use_default_fallback=False  # Don't use default values in production!
    )
    
    # When retrieving a secret, environment variables like SECUREML_SECRET_NAME will be checked
    # if Vault is unavailable
    api_key = key_manager.get_secret('api_key')

Development Mode
^^^^^^^^^^^^

For development environments, you can enable default fallback keys:

.. code-block:: python

    # DEVELOPMENT ONLY - NOT FOR PRODUCTION!
    key_manager = KeyManager(
        use_env_fallback=True,
        use_default_fallback=True  # Automatically generates deterministic development keys
    )
    
    # This will provide a development key if not found in Vault or environment
    dev_key = key_manager.get_encryption_key('dev_key')

Integration with ML Workflows
--------------------------

Using Key Management in ML Pipelines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Integrate key management with ML workflows:

.. code-block:: python

    from secureml.key_management import get_encryption_key
    from secureml.anonymization import anonymize
    
    # Get an encryption key for anonymization
    key = get_encryption_key(key_name='anonymization_key')
    
    # Use the key for anonymizing data
    anonymized_df = anonymize(
        data=sensitive_df,
        method='pseudonymization',
        strategy='fpe',  # Format-preserving encryption
        sensitive_columns=['name', 'email', 'ssn'],
        master_key=key  # Pass the key directly
    )

Best Practices
-------------

1. **Never store keys in code or config files**: Always use a dedicated key management solution

2. **Implement least privilege access**: Only grant access to keys that are necessary for specific operations

3. **Rotate keys regularly**: Establish and follow a key rotation schedule

4. **Use key derivation**: Derive purpose-specific keys from master keys for better security

5. **Monitor key usage**: Set up alerts for unusual key access patterns

6. **Backup keys securely**: Ensure you have secure backups of critical keys

7. **Audit key operations**: Keep detailed logs of all key management operations

8. **Disable default fallbacks in production**: Only use default fallback keys for development

Further Reading
-------------

* :doc:`/api/key_management` - Complete API reference for key management functions
* :doc:`/examples/key_management` - More examples of key management techniques
* `HashiCorp Vault Documentation <https://www.vaultproject.io/docs>`_ - Official documentation for Vault 