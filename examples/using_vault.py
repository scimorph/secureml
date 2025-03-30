#!/usr/bin/env python
"""
Example script demonstrating HashiCorp Vault integration with SecureML.

This script shows how to:
1. Configure SecureML to use HashiCorp Vault for key management
2. Generate and store encryption keys
3. Use them in anonymization operations
"""

import os
import pandas as pd
from secureml import (
    anonymize,
    configure_default_key_manager,
    get_encryption_key,
    KeyManager
)

def main():
    """Run the example."""
    print("SecureML - HashiCorp Vault Integration Example")
    print("==============================================")
    
    # Check if Vault is available
    vault_url = os.environ.get("SECUREML_VAULT_URL")
    vault_token = os.environ.get("SECUREML_VAULT_TOKEN")
    
    if not vault_url or not vault_token:
        print("Warning: Vault environment variables not set. Using example values.")
        print("For production use, set SECUREML_VAULT_URL and SECUREML_VAULT_TOKEN")
        
        # For demo purposes only - in production, these would be environment variables
        vault_url = "http://127.0.0.1:8200"
        vault_token = "myroot"  # Example dev token
    
    # Step 1: Configure the key manager
    print("\n1. Configuring key manager...")
    configure_default_key_manager(
        vault_url=vault_url,
        vault_token=vault_token,
        vault_path="secureml",
        use_env_fallback=True,
        use_default_fallback=True  # Allow default keys for this example
    )
    
    # Step 2: Create a sample dataset with sensitive information
    print("\n2. Creating sample dataset...")
    df = pd.DataFrame({
        "name": ["John Doe", "Jane Smith", "Bob Johnson", "Alice Williams"],
        "email": [
            "john.doe@example.com",
            "jane.smith@example.com",
            "bob.johnson@example.com",
            "alice.williams@example.com"
        ],
        "ssn": ["123-45-6789", "987-65-4321", "456-78-9012", "321-54-9876"],
        "credit_card": [
            "4111-1111-1111-1111",
            "5555-5555-5555-4444",
            "3782-822463-10005",
            "6011-1111-1111-1117"
        ],
        "age": [32, 45, 28, 36],
        "income": [75000, 82000, 65000, 95000]
    })
    
    print(df.head())
    
    # Step 3: Try to store a key in Vault (this will only work if Vault is running)
    print("\n3. Attempting to store a key in Vault...")
    key_manager = KeyManager(
        vault_url=vault_url,
        vault_token=vault_token
    )
    
    try:
        # Generate and store a new key
        if key_manager.vault_client:
            success = key_manager.set_secret("fpe_master_key", os.urandom(32).hex())
            if success:
                print("✓ Successfully stored key in Vault")
            else:
                print("✗ Failed to store key in Vault")
                print("  Using fallback mechanisms instead")
        else:
            print("✗ Vault connection failed")
            print("  Using fallback mechanisms instead")
    except Exception as e:
        print(f"✗ Error storing key: {str(e)}")
        print("  Using fallback mechanisms instead")
    
    # Step 4: Use the key for anonymization
    print("\n4. Anonymizing data using Format-Preserving Encryption...")
    anonymized_df = anonymize(
        df,
        method="pseudonymization",
        strategy="fpe",  # Format-preserving encryption
        sensitive_columns=["name", "email", "ssn", "credit_card"],
        # The key name should match what we stored in Vault
        master_key_name="fpe_master_key"
    )
    
    print("\nAnonymized data:")
    print(anonymized_df.head())
    
    print("\nNote: Even if Vault is not available, this example works because:")
    print("1. SecureML falls back to environment variables if configured")
    print("2. For this example, we also allowed fallback to default keys")
    print("3. In production, you should disable default fallback keys")
    
    # Step 5: Demonstrate that we can get the key directly
    print("\n5. Demonstrating direct key access...")
    try:
        key_hex = get_encryption_key(key_name="fpe_master_key", encoding="hex")
        print(f"Successfully retrieved key: {key_hex[:6]}...{key_hex[-6:]}")
    except Exception as e:
        print(f"Error retrieving key: {str(e)}")


if __name__ == "__main__":
    main() 