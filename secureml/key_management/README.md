# Key Management for SecureML

This module provides secure key management capabilities for SecureML using HashiCorp Vault, an industry-standard solution for managing secrets, tokens, keys, and certificates.

## Installation

To use HashiCorp Vault with SecureML, you need to install the `hvac` package:

```bash
pip install secureml[vault]
```

Or if you're using Poetry:

```bash
poetry add secureml -E vault
```

## Setting Up HashiCorp Vault

### Option 1: Using Vault in Development (Docker)

For local development, you can quickly set up a Vault instance using Docker:

```bash
docker run --cap-add=IPC_LOCK -p 8200:8200 -e 'VAULT_DEV_ROOT_TOKEN_ID=myroot' hashicorp/vault:latest

# This gives you:
# - Vault running on http://127.0.0.1:8200
# - Root token: myroot
# - Dev mode (no persistence, automatic unsealing)
```

### Option 2: Production Vault Installation

For production environments, follow the [official HashiCorp Vault installation guide](https://developer.hashicorp.com/vault/tutorials/getting-started/getting-started-install).

## Configuring SecureML to Use Vault

### Via Environment Variables

Set these environment variables to configure SecureML to use your Vault instance:

```bash
export SECUREML_VAULT_URL=http://127.0.0.1:8200
export SECUREML_VAULT_TOKEN=myroot
```

### Via CLI

You can use the SecureML CLI to configure and test your Vault connection:

```bash
# Test Vault connection
secureml keys configure-vault --vault-url http://127.0.0.1:8200 --vault-token myroot --test-connection

# Generate and store a new encryption key
secureml keys generate-key --key-name master_key

# Retrieve a stored key
secureml keys get-key --key-name master_key
```

### Via Python Code

```python
from secureml import configure_default_key_manager, get_encryption_key

# Configure the key manager
configure_default_key_manager(
    vault_url="http://127.0.0.1:8200",
    vault_token="myroot",
    vault_path="secureml",  # Base path for storing SecureML secrets
    use_env_fallback=True,  # Fall back to environment variables if Vault is unavailable
    use_default_fallback=False  # Whether to use default keys (unsafe for production)
)

# Get an encryption key for use in your code
master_key = get_encryption_key(key_name="master_key", encoding="bytes")
```

## Usage in SecureML Operations

Once configured, SecureML will automatically use Vault for key management without any changes to your code:

```python
import pandas as pd
from secureml import anonymize

# SecureML will automatically use the key stored in Vault
df = pd.DataFrame({
    "name": ["John Doe", "Jane Smith"],
    "ssn": ["123-45-6789", "987-65-4321"]
})

anonymized_df = anonymize(
    df,
    method="pseudonymization",
    strategy="fpe",  # Format-preserving encryption
    sensitive_columns=["name", "ssn"]
)
```

## Key Rotation and Management

For best security practices, it's recommended to rotate encryption keys periodically:

```bash
# Generate a new key
secureml keys generate-key --key-name master_key_new

# In your next deployment, update to use the new key
export SECUREML_MASTER_KEY_NAME=master_key_new
```

## Security Considerations

1. **Vault Authentication**: In production, use more secure authentication methods like AppRole or LDAP instead of token auth.

2. **Key Isolation**: Use different keys for different environments (dev, staging, prod).

3. **Access Control**: Configure Vault policies to limit which applications can access which secrets.

4. **Network Security**: Ensure Vault is properly secured behind a firewall and uses TLS for all communications.

5. **Auditing**: Enable audit logging in Vault to track all access to secrets.

## Fallback Options

SecureML's key management provides flexible fallback options:

1. **Vault**: Primary storage for production environments
2. **Environment Variables**: Secondary option if Vault is unavailable
3. **Default Values**: Last resort for development (should never be used in production)

By configuring `use_env_fallback` and `use_default_fallback` appropriately, you can control this behavior. 