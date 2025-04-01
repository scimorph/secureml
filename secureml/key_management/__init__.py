"""
Key management functionality for SecureML.

This module provides tools for securely managing encryption keys and other secrets
using HashiCorp Vault. It implements best practices for secure key storage,
rotation, and access.
"""

import os
import logging
import hashlib
import hmac
import base64
from typing import Optional, Dict, Any, Union, List, Tuple
import warnings

# Configure logging
logger = logging.getLogger("secureml.key_management")

# Optional Vault dependency - will be imported only if vault_client is used
try:
    import hvac
    _HAS_VAULT = True
except ImportError:
    _HAS_VAULT = False
    warnings.warn(
        "HashiCorp Vault client (hvac) not installed. "
        "Install with: pip install hvac"
    )


class KeyManager:
    """
    Manage encryption keys and other secrets for SecureML operations.
    
    This class provides methods for retrieving and managing secrets from various
    backends, with a primary focus on HashiCorp Vault for production environments
    and fallback to environment variables or default values for development.
    """
    
    def __init__(
        self,
        vault_url: Optional[str] = None,
        vault_token: Optional[str] = None,
        vault_path: str = "secureml",
        use_env_fallback: bool = True,
        use_default_fallback: bool = False,
    ):
        """
        Initialize the key manager.
        
        Args:
            vault_url: URL for the HashiCorp Vault server
            vault_token: Authentication token for Vault
            vault_path: Base path in Vault where SecureML secrets are stored
            use_env_fallback: Whether to fall back to environment variables if Vault is unavailable
            use_default_fallback: Whether to fall back to default values (DEVELOPMENT ONLY)
        """
        self.vault_url = vault_url or os.environ.get("SECUREML_VAULT_URL")
        self.vault_token = vault_token or os.environ.get("SECUREML_VAULT_TOKEN")
        self.vault_path = vault_path
        self.use_env_fallback = use_env_fallback
        self.use_default_fallback = use_default_fallback
        self.vault_client = None
        
        # Only try to initialize Vault client if hvac is installed and URL is provided
        if _HAS_VAULT and self.vault_url and self.vault_token:
            try:
                self.vault_client = hvac.Client(
                    url=self.vault_url,
                    token=self.vault_token
                )
                # Verify connection and authentication
                if not self.vault_client.is_authenticated():
                    logger.warning("Failed to authenticate with Vault")
                    self.vault_client = None
                else:
                    logger.info(f"Successfully connected to Vault at {self.vault_url}")
            except Exception as e:
                logger.error(f"Error connecting to Vault: {str(e)}")
                self.vault_client = None
        
        if not self.vault_client and not self.use_env_fallback and not self.use_default_fallback:
            logger.warning(
                "No valid secret backend configured. "
                "Encryption operations may fail unless keys are provided explicitly."
            )
    
    def get_secret(
        self, 
        secret_name: str, 
        default: Optional[Any] = None
    ) -> Any:
        """
        Retrieve a secret from the configured backend.
        
        This method attempts to retrieve the secret from Vault first, then falls back
        to environment variables, and finally to the provided default value if allowed.
        
        Args:
            secret_name: Name of the secret to retrieve
            default: Default value to return if the secret is not found
            
        Returns:
            The secret value if found, otherwise the default value
            
        Raises:
            ValueError: If no secret is found and no default is provided or allowed
        """
        # Try to get from Vault
        if self.vault_client is not None:
            try:
                secret_path = f"{self.vault_path}/data/{secret_name}"
                response = self.vault_client.secrets.kv.v2.read_secret_version(
                    path=secret_path
                )
                if response and "data" in response and "data" in response["data"]:
                    return response["data"]["data"].get("value")
            except Exception as e:
                logger.warning(f"Error retrieving secret from Vault: {str(e)}")
        
        # Try environment variable
        if self.use_env_fallback:
            env_var = f"SECUREML_{secret_name.upper()}"
            if env_var in os.environ:
                return os.environ[env_var]
        
        # Use default if allowed
        if default is not None or self.use_default_fallback:
            if default is not None:
                return default
            
            # Generate a consistent default for development only
            logger.warning(
                f"Using generated default for {secret_name}. "
                "This is UNSAFE for production use!"
            )
            return self._generate_development_default(secret_name)
        
        raise ValueError(f"Secret {secret_name} not found and no default provided")
    
    def set_secret(self, secret_name: str, value: Any) -> bool:
        """
        Store a secret in Vault.
        
        Args:
            secret_name: Name of the secret to store
            value: Value of the secret
            
        Returns:
            True if the secret was stored successfully, False otherwise
        """
        if not self.vault_client:
            logger.error("Cannot store secret: Vault client not configured")
            return False
        
        try:
            secret_path = f"{self.vault_path}/data/{secret_name}"
            self.vault_client.secrets.kv.v2.create_or_update_secret(
                path=secret_path,
                secret={"value": value}
            )
            logger.info(f"Secret {secret_name} stored successfully")
            return True
        except Exception as e:
            logger.error(f"Error storing secret in Vault: {str(e)}")
            return False
    
    def get_encryption_key(
        self, 
        key_name: str = "master_key",
        key_bytes: int = 32,
        encoding: str = "bytes"
    ) -> Union[bytes, str]:
        """
        Get an encryption key from the secret backend.
        
        Args:
            key_name: Name of the key to retrieve
            key_bytes: Number of bytes for the key (if generating a default)
            encoding: Output encoding ('bytes', 'hex', or 'base64')
            
        Returns:
            The encryption key in the requested encoding
        """
        # Get raw key material
        key_material = self.get_secret(key_name)
        
        # If it's a string, convert to bytes
        if isinstance(key_material, str):
            # Check if it's hex encoded
            if all(c in "0123456789abcdefABCDEF" for c in key_material):
                key_bytes = bytes.fromhex(key_material)
            else:
                # Assume it's utf-8 encoded
                key_bytes = key_material.encode('utf-8')
        elif not isinstance(key_material, bytes):
            # If it's neither string nor bytes, hash it to get bytes
            key_bytes = hashlib.sha256(str(key_material).encode()).digest()
        else:
            key_bytes = key_material
            
        # Ensure key is the right length by hashing if necessary
        if len(key_bytes) != key_bytes:
            key_bytes = hashlib.sha256(key_bytes).digest()[:key_bytes]
            
        # Return in the requested encoding
        if encoding == "bytes":
            return key_bytes
        elif encoding == "hex":
            return key_bytes.hex()
        elif encoding == "base64":
            return base64.b64encode(key_bytes).decode('ascii')
        else:
            raise ValueError(f"Unsupported encoding: {encoding}")
    
    def _generate_development_default(self, secret_name: str) -> bytes:
        """
        Generate a default value for development purposes only.
        
        This should NEVER be used in production environments.
        
        Args:
            secret_name: Name of the secret to generate a default for
            
        Returns:
            A deterministic but development-only default value
        """
        # Use a fixed development seed combined with the secret name
        dev_seed = b"SECUREML_DEVELOPMENT_ONLY_SEED"
        return hmac.new(
            dev_seed, 
            secret_name.encode(), 
            hashlib.sha256
        ).digest()
    
    def derive_key(
        self, 
        base_key: Union[str, bytes],
        purpose: str,
        key_bytes: int = 32
    ) -> bytes:
        """
        Derive a purpose-specific key from a base key.
        
        This implements key separation, allowing different keys
        to be used for different purposes while all being derived
        from a single master key.
        
        Args:
            base_key: The base key to derive from
            purpose: A string describing the key's purpose
            key_bytes: Length of the derived key in bytes
            
        Returns:
            The derived key as bytes
        """
        if isinstance(base_key, str):
            base_key = base_key.encode('utf-8')
            
        return hmac.new(
            base_key,
            purpose.encode(),
            hashlib.sha256
        ).digest()[:key_bytes]


# Singleton instance for use throughout the application
default_key_manager = KeyManager(
    use_env_fallback=True,
    use_default_fallback=False
)


def configure_default_key_manager(
    vault_url: Optional[str] = None,
    vault_token: Optional[str] = None,
    vault_path: str = "secureml",
    use_env_fallback: bool = True,
    use_default_fallback: bool = False,
) -> None:
    """
    Configure the default key manager used by SecureML.
    
    This function should be called early in your application's lifecycle
    to ensure secure key storage is properly configured.
    
    Args:
        vault_url: URL for the HashiCorp Vault server
        vault_token: Authentication token for Vault
        vault_path: Base path in Vault where SecureML secrets are stored
        use_env_fallback: Whether to fall back to environment variables
        use_default_fallback: Whether to fall back to default values (DEVELOPMENT ONLY)
    """
    global default_key_manager
    default_key_manager = KeyManager(
        vault_url=vault_url,
        vault_token=vault_token,
        vault_path=vault_path,
        use_env_fallback=use_env_fallback,
        use_default_fallback=use_default_fallback
    )


def get_encryption_key(
    key_name: str = "master_key",
    key_bytes: int = 32,
    encoding: str = "bytes"
) -> Union[bytes, str]:
    """
    Get an encryption key using the default key manager.
    
    Args:
        key_name: Name of the key to retrieve
        key_bytes: Number of bytes for the key (if generating a default)
        encoding: Output encoding ('bytes', 'hex', or 'base64')
        
    Returns:
        The encryption key in the requested encoding
    """
    return default_key_manager.get_encryption_key(
        key_name=key_name,
        key_bytes=key_bytes,
        encoding=encoding
    )

# Export these symbols
__all__ = ['KeyManager', 'configure_default_key_manager', 'get_encryption_key']