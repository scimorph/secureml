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
        backend='vault',  # Options: 'vault', 'aws_kms', 'azure_key_vault', 'gcp_kms', 'file'
        config={
            'vault_url': 'https://vault.example.com:8200',
            'vault_token': os.environ.get('VAULT_TOKEN'),
            'mount_point': 'secureml'
        }
    )
    
    # Test the connection
    if key_manager.test_connection():
        print("Successfully connected to key management backend")
    else:
        print("Failed to connect to key management backend")

Generating and Storing Keys
^^^^^^^^^^^^^^^^^^^^^^^^

Generate and store encryption keys:

.. code-block:: python

    # Generate a new encryption key
    key_id = key_manager.generate_key(
        key_name='customer_data_encryption_key',
        key_type='aes',
        key_size=256,
        description='Key for encrypting customer financial data',
        metadata={
            'department': 'data_science',
            'project': 'credit_risk_modeling',
            'data_classification': 'sensitive'
        }
    )
    
    print(f"Generated key with ID: {key_id}")
    
    # Store an existing key
    existing_key = b'...'  # Your existing key material
    key_id = key_manager.store_key(
        key_name='existing_model_key',
        key_material=existing_key,
        description='Imported key for model protection',
        metadata={'source': 'legacy_system'}
    )

Retrieving and Using Keys
^^^^^^^^^^^^^^^^^^^^^

Retrieve and use encryption keys:

.. code-block:: python

    # Retrieve a key by ID
    key = key_manager.get_key(key_id)
    
    # Or retrieve by name
    key = key_manager.get_key_by_name('customer_data_encryption_key')
    
    # Use the key for encryption
    from secureml.encryption import encrypt_data
    
    encrypted_data = encrypt_data(
        data=sensitive_data,
        key=key,
        algorithm='AES-GCM'
    )
    
    # Later, decrypt the data
    from secureml.encryption import decrypt_data
    
    decrypted_data = decrypt_data(
        encrypted_data=encrypted_data,
        key=key,
        algorithm='AES-GCM'
    )

Key Lifecycle Management
^^^^^^^^^^^^^^^^^^^^^

Manage the lifecycle of encryption keys:

.. code-block:: python

    # Rotate a key
    new_key_id = key_manager.rotate_key(
        key_id=key_id,
        reason='quarterly_rotation'
    )
    
    # Archive a key (make it unavailable for encryption but available for decryption)
    key_manager.archive_key(key_id)
    
    # Delete a key (use with extreme caution - may result in data loss)
    key_manager.delete_key(key_id, force=True)
    
    # Check key status
    status = key_manager.get_key_status(key_id)
    print(f"Key status: {status}")  # 'active', 'archived', 'expired', etc.
    
    # List all keys
    keys = key_manager.list_keys()
    for key_id, key_info in keys.items():
        print(f"Key {key_id}: {key_info['name']} ({key_info['status']})")

HashiCorp Vault Integration
-------------------------

SecureML provides deep integration with HashiCorp Vault for enterprise-grade key management.

Setting Up Vault Integration
^^^^^^^^^^^^^^^^^^^^^^^^^

Configure HashiCorp Vault integration:

.. code-block:: python

    from secureml.key_management.vault import VaultKeyManager
    
    # Initialize the Vault key manager
    vault_manager = VaultKeyManager(
        vault_url='https://vault.example.com:8200',
        vault_token=os.environ.get('VAULT_TOKEN'),  # Or use other authentication methods
        mount_point='secureml',
        namespace='ml-team'  # Optional, for Vault Enterprise
    )
    
    # Alternatively, use environment variables
    # export VAULT_ADDR='https://vault.example.com:8200'
    # export VAULT_TOKEN='hvs.example'
    vault_manager = VaultKeyManager()

Authentication Methods
^^^^^^^^^^^^^^^^^^^

Authenticate to Vault using different methods:

.. code-block:: python

    # Token authentication (most common)
    vault_manager = VaultKeyManager(
        vault_url='https://vault.example.com:8200',
        vault_token='hvs.example'
    )
    
    # AppRole authentication
    vault_manager = VaultKeyManager(
        vault_url='https://vault.example.com:8200',
        auth_method='approle',
        auth_params={
            'role_id': 'role-id-example',
            'secret_id': 'secret-id-example'
        }
    )
    
    # Kubernetes authentication
    vault_manager = VaultKeyManager(
        vault_url='https://vault.example.com:8200',
        auth_method='kubernetes',
        auth_params={
            'role': 'secureml-role',
            'jwt_path': '/var/run/secrets/kubernetes.io/serviceaccount/token'
        }
    )
    
    # AWS IAM authentication
    vault_manager = VaultKeyManager(
        vault_url='https://vault.example.com:8200',
        auth_method='aws',
        auth_params={
            'role': 'secureml-role',
            'aws_region': 'us-west-2'
        }
    )

Working with Transit Engine
^^^^^^^^^^^^^^^^^^^^^^^^

Use Vault's Transit Engine for encryption operations:

.. code-block:: python

    # Enable and configure the transit engine
    vault_manager.setup_transit_engine(
        transit_mount='transit',
        key_name='secureml-encryption-key',
        key_type='aes256-gcm96',
        exportable=False,
        allow_plaintext_backup=False
    )
    
    # Encrypt data using the transit engine
    ciphertext = vault_manager.transit_encrypt(
        key_name='secureml-encryption-key',
        plaintext=sensitive_data
    )
    
    # Decrypt data using the transit engine
    plaintext = vault_manager.transit_decrypt(
        key_name='secureml-encryption-key',
        ciphertext=ciphertext
    )
    
    # Rotate the transit key
    vault_manager.transit_rotate_key('secureml-encryption-key')
    
    # Reencrypt data with the latest key version
    updated_ciphertext = vault_manager.transit_reencrypt(
        key_name='secureml-encryption-key',
        ciphertext=ciphertext
    )

Secrets Management
^^^^^^^^^^^^^^^

Store and retrieve sensitive configuration:

.. code-block:: python

    # Store a secret
    vault_manager.store_secret(
        path='api_credentials/data_provider',
        data={
            'api_key': 'api-key-example',
            'api_secret': 'api-secret-example',
            'endpoint': 'https://api.provider.com/v1'
        }
    )
    
    # Retrieve a secret
    credentials = vault_manager.get_secret('api_credentials/data_provider')
    api_key = credentials['api_key']
    
    # Update a secret
    vault_manager.update_secret(
        path='api_credentials/data_provider',
        data={
            'api_key': 'new-api-key',
            'api_secret': 'new-api-secret',
            'endpoint': 'https://api.provider.com/v2'
        }
    )
    
    # Delete a secret
    vault_manager.delete_secret('api_credentials/old_provider')

Advanced Techniques
------------------

Key Hierarchies and Envelope Encryption
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Implement key hierarchies for better security:

.. code-block:: python

    from secureml.key_management import KeyHierarchy
    
    # Create a key hierarchy
    hierarchy = KeyHierarchy(key_manager)
    
    # Generate a master key
    master_key_id = hierarchy.create_master_key(
        name='master_key',
        description='Root key for the key hierarchy'
    )
    
    # Generate data encryption keys protected by the master key
    dek_id = hierarchy.create_data_encryption_key(
        name='customer_data_dek',
        master_key_id=master_key_id,
        description='DEK for customer data'
    )
    
    # Use envelope encryption
    encrypted_data = hierarchy.encrypt(
        data=sensitive_data,
        dek_id=dek_id
    )
    
    # Decrypt with envelope encryption
    decrypted_data = hierarchy.decrypt(
        encrypted_data=encrypted_data,
        dek_id=dek_id
    )

Automatic Key Rotation
^^^^^^^^^^^^^^^^^^^

Set up automatic key rotation:

.. code-block:: python

    from secureml.key_management import KeyRotationScheduler
    
    # Create a rotation scheduler
    scheduler = KeyRotationScheduler(key_manager)
    
    # Schedule key rotation
    scheduler.schedule_rotation(
        key_id=key_id,
        interval='90d',  # Rotate every 90 days
        auto_reencrypt=True,  # Automatically re-encrypt data with new key
        notification_email='security@example.com'
    )
    
    # View rotation schedule
    schedule = scheduler.get_rotation_schedule()
    for key_id, rotation_info in schedule.items():
        print(f"Key {key_id} next rotation: {rotation_info['next_rotation']}")
    
    # Execute pending rotations manually
    rotated_keys = scheduler.execute_pending_rotations()
    print(f"Rotated {len(rotated_keys)} keys")

Key Access Policies
^^^^^^^^^^^^^^^

Define and enforce key access policies:

.. code-block:: python

    from secureml.key_management import KeyAccessPolicy
    
    # Create an access policy
    policy = KeyAccessPolicy(
        key_id=key_id,
        allowed_users=['data_scientist_role', 'model_training_service'],
        allowed_operations=['encrypt', 'decrypt'],
        time_restrictions={
            'valid_from': '2023-01-01T00:00:00Z',
            'valid_until': '2023-12-31T23:59:59Z'
        },
        ip_restrictions=['10.0.0.0/24', '192.168.1.0/24']
    )
    
    # Apply the policy
    key_manager.apply_access_policy(policy)
    
    # Check access permission
    has_access = key_manager.check_access(
        key_id=key_id,
        operation='decrypt',
        user='data_scientist_role',
        context={'ip_address': '10.0.0.5'}
    )
    
    if has_access:
        print("Access granted")
    else:
        print("Access denied")

Command-Line Interface
-------------------

SecureML provides a command-line interface for key management operations:

.. code-block:: bash

    # Initialize key management
    secureml keys init --backend vault --vault-url https://vault.example.com:8200
    
    # Generate a new key
    secureml keys generate --name customer_data_key --type aes --size 256
    
    # List all keys
    secureml keys list
    
    # Get key information
    secureml keys info KEY_ID
    
    # Rotate a key
    secureml keys rotate KEY_ID
    
    # Export a key (if allowed)
    secureml keys export KEY_ID --output encrypted_key.bin
    
    # Import a key
    secureml keys import --name imported_key --file key_material.bin
    
    # Delete a key (with confirmation)
    secureml keys delete KEY_ID

Integration with ML Workflows
--------------------------

Using Key Management in ML Pipelines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Integrate key management with ML workflows:

.. code-block:: python

    from secureml.key_management import KeyManager
    from secureml.data import EncryptedDataLoader
    from secureml.model import EncryptedModelStorage
    
    # Initialize key management
    key_manager = KeyManager(backend='vault')
    
    # Load encrypted data
    data_key = key_manager.get_key_by_name('customer_data_key')
    data_loader = EncryptedDataLoader(
        file_path='encrypted_customer_data.csv',
        encryption_key=data_key
    )
    df = data_loader.load()
    
    # Train a model
    model = train_model(df)
    
    # Store the encrypted model
    model_key = key_manager.get_key_by_name('model_encryption_key')
    model_storage = EncryptedModelStorage(encryption_key=model_key)
    model_id = model_storage.save(model, 'credit_risk_model_v1')
    
    # Later, load the encrypted model
    loaded_model = model_storage.load(model_id)

Best Practices
-------------

1. **Never store keys in code or config files**: Always use a dedicated key management solution

2. **Implement least privilege access**: Only grant access to keys that are necessary for specific operations

3. **Rotate keys regularly**: Establish and follow a key rotation schedule

4. **Use key hierarchies**: Implement master keys and data encryption keys for better security

5. **Monitor key usage**: Set up alerts for unusual key access patterns

6. **Backup keys securely**: Ensure you have secure backups of critical keys

7. **Audit key operations**: Keep detailed logs of all key management operations

8. **Automate key management**: Use automation to reduce human error in key management

9. **Plan for key compromise**: Have a clear process for responding to key compromise incidents

10. **Test key recovery procedures**: Regularly test your ability to recover from key loss

Further Reading
-------------

* :doc:`/api/key_management` - Complete API reference for key management functions
* :doc:`/examples/key_management` - More examples of key management techniques
* `HashiCorp Vault Documentation <https://www.vaultproject.io/docs>`_ - Official documentation for Vault 