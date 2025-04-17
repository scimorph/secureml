"""
Command Line Interface for SecureML.

This module provides a CLI to interact with the key functionalities of the 
SecureML library, making it easier to perform common tasks from the command line.
"""

import click
import json
import pandas as pd
import yaml
from pathlib import Path
import sys
import os
import shutil

from secureml import (
    anonymize,
    check_compliance,
    generate_synthetic_data,
    list_available_presets,
    load_preset,
)

from secureml.isolated_environments.tf_privacy import (
    setup_tf_privacy_environment,
    get_env_path,
    is_env_valid
)

# For sensitive column detection in the CLI
from secureml.synthetic import _identify_sensitive_columns


@click.group()
@click.version_option()
def cli():
    """SecureML: Privacy-preserving machine learning tools.
    
    This CLI provides access to the core features of SecureML, 
    making it easy to perform common privacy and compliance tasks
    directly from the command line.
    """
    pass


@cli.group()
def anonymization():
    """Data anonymization commands."""
    pass


@anonymization.command("k-anonymize")
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path())
@click.option(
    "--sensitive", "-s", multiple=True, help="Sensitive columns to preserve"
)
@click.option(
    "--quasi-id", "-q", multiple=True, help="Quasi-identifier columns"
)
@click.option(
    "--k", type=int, default=5, help="K value for k-anonymity (default: 5)"
)
@click.option(
    "--format",
    type=click.Choice(["csv", "json", "parquet"]),
    default="csv",
    help="Output file format (default: csv)",
)
def k_anonymize(input_file, output_file, sensitive, quasi_id, k, format):
    """Apply k-anonymity to a dataset.
    
    This command reads a dataset, applies k-anonymity to protect privacy while
    preserving utility, and saves the anonymized version to a new file.
    
    Example:
        secureml anonymization k-anonymize data.csv anonymized.csv \\
        --quasi-id age --quasi-id zipcode --sensitive medical_condition --k 5
    """
    click.echo(f"Reading data from {input_file}...")
    
    # Determine input format
    input_path = Path(input_file)
    if input_path.suffix == '.csv':
        data = pd.read_csv(input_file)
    elif input_path.suffix == '.json':
        data = pd.read_json(input_file)
    elif input_path.suffix in ['.parquet', '.pq']:
        data = pd.read_parquet(input_file)
    else:
        click.echo(
            f"Unsupported input format: {input_path.suffix}. Trying csv format..."
        )
        data = pd.read_csv(input_file)
    
    click.echo(f"Applying k-anonymity with k={k}...")
    
    # Apply k-anonymity
    anonymized_data = anonymize.k_anonymize(
        data,
        sensitive_columns=list(sensitive),
        quasi_identifiers=list(quasi_id),
        k=k
    )
    
    # Save output in the requested format
    if format == 'csv':
        anonymized_data.to_csv(output_file, index=False)
    elif format == 'json':
        anonymized_data.to_json(output_file, orient='records')
    elif format == 'parquet':
        anonymized_data.to_parquet(output_file, index=False)
    
    click.echo(f"Anonymized data saved to {output_file}")


@cli.group()
def compliance():
    """Compliance checking commands."""
    pass


@compliance.command("check")
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--regulation",
    "-r",
    type=click.Choice(["GDPR", "CCPA", "HIPAA", "LGPD"]),
    required=True,
    help="Regulation to check compliance against",
)
@click.option(
    "--metadata",
    type=click.Path(exists=True),
    help="JSON or YAML file with dataset metadata",
)
@click.option(
    "--model-config",
    type=click.Path(exists=True),
    help="JSON or YAML file with model configuration",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Path to save the compliance report",
)
@click.option(
    "--format",
    type=click.Choice(["json", "html", "pdf"]),
    default="json",
    help="Output format for compliance report (default: json)",
)
def check_compliance_cmd(input_file, regulation, metadata, model_config, output, format):
    """Check dataset and model compliance with privacy regulations.
    
    This command checks a dataset and optional model configuration against
    privacy regulations like GDPR, CCPA, HIPAA, or LGPD.
    
    Examples:
        secureml compliance check data.csv --regulation GDPR
        secureml compliance check data.csv --regulation HIPAA \\
        --metadata metadata.json --model-config model.json \\
        --output report.html --format html
    """
    click.echo(f"Reading data from {input_file}...")
    
    # Load dataset
    input_path = Path(input_file)
    if input_path.suffix == '.csv':
        data = pd.read_csv(input_file)
    elif input_path.suffix == '.json':
        data = pd.read_json(input_file)
    elif input_path.suffix in ['.parquet', '.pq']:
        data = pd.read_parquet(input_file)
    else:
        click.echo(
            f"Unsupported input format: {input_path.suffix}. Trying csv format..."
        )
        data = pd.read_csv(input_file)
    
    # Load metadata if provided
    metadata_dict = {}
    if metadata:
        metadata_path = Path(metadata)
        if metadata_path.suffix in ['.json']:
            with open(metadata, 'r') as f:
                metadata_dict = json.load(f)
        elif metadata_path.suffix in ['.yaml', '.yml']:
            with open(metadata, 'r') as f:
                metadata_dict = yaml.safe_load(f)
        else:
            click.echo(f"Unsupported metadata format: {metadata_path.suffix}")
            return
    
    # Load model config if provided
    model_config_dict = {}
    if model_config:
        config_path = Path(model_config)
        if config_path.suffix in ['.json']:
            with open(model_config, 'r') as f:
                model_config_dict = json.load(f)
        elif config_path.suffix in ['.yaml', '.yml']:
            with open(model_config, 'r') as f:
                model_config_dict = yaml.safe_load(f)
        else:
            click.echo(
                f"Unsupported model config format: {config_path.suffix}"
            )
            return
    
    click.echo(f"Checking compliance with {regulation}...")
    
    # Create a combined dataset dictionary with metadata
    dataset_dict = {'data': data, **metadata_dict}
    
    # Check compliance
    report = check_compliance(
        dataset_dict,
        model_config=model_config_dict if model_config else None,
        regulation=regulation
    )
    
    # Output report
    if output:
        if format == 'json':
            with open(output, 'w') as f:
                json.dump(report.to_dict(), f, indent=2)
        elif format in ['html', 'pdf']:
            report.generate_report(output, format=format)
        click.echo(f"Compliance report saved to {output}")
    else:
        click.echo("Compliance Summary:")
        click.echo(f"Regulation: {regulation}")
        if report.has_issues():
            click.echo(f"Issues found: {len(report.issues)}")
            for i, issue in enumerate(report.issues, 1):
                click.echo(
                    f"{i}. [{issue['severity'].upper()}] {issue['issue']}"
                )
                click.echo(f"   Recommendation: {issue['recommendation']}")
        else:
            click.echo("No compliance issues found.")


@cli.group()
def synthetic():
    """Synthetic data generation commands."""
    pass


@synthetic.command("generate")
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path())
@click.option(
    "--method",
    type=click.Choice(["simple", "statistical", "sdv-copula", "sdv-ctgan", "gan", "copula"]),
    default="statistical",
    help="Generation method (default: statistical)",
)
@click.option(
    "--samples",
    "-n",
    type=int,
    default=1000,
    help="Number of samples to generate (default: 1000)",
)
@click.option(
    "--sensitive",
    "-s",
    multiple=True,
    help="Sensitive columns to handle specially",
)
@click.option(
    "--auto-detect-sensitive",
    is_flag=True,
    help="Automatically detect sensitive columns if none specified",
)
@click.option(
    "--sensitivity-confidence",
    type=float,
    default=0.5,
    help="Confidence threshold for sensitive column detection (0.0-1.0)",
)
@click.option(
    "--sensitivity-sample-size",
    type=int,
    default=100,
    help="Number of sample rows to examine for sensitive data detection",
)
@click.option(
    "--preserve-outliers",
    is_flag=True,
    help="Preserve outlier patterns from original data (statistical method)",
)
@click.option(
    "--handle-skewness",
    is_flag=True,
    default=True,
    help="Handle skewed numerical distributions (statistical/copula methods)",
)
@click.option(
    "--categorical-threshold",
    type=int,
    default=20,
    help="Max unique values to treat as categorical (statistical/copula methods)",
)
@click.option(
    "--epochs",
    type=int,
    default=300,
    help="Number of training epochs for GAN method",
)
@click.option(
    "--batch-size",
    type=int,
    default=32,
    help="Batch size for GAN training",
)
@click.option(
    "--learning-rate",
    type=float,
    default=0.001,
    help="Learning rate for GAN optimizer",
)
@click.option(
    "--noise-dim",
    type=int,
    default=100,
    help="Dimension of noise input for GAN generator",
)
@click.option(
    "--copula-type",
    type=click.Choice(["gaussian", "t"]),
    default="gaussian",
    help="Type of copula to use (copula method)",
)
@click.option(
    "--fit-method",
    type=click.Choice(["ml", "rank"]),
    default="ml",
    help="Method for fitting copula (copula method)",
)
@click.option(
    "--handle-missing",
    type=click.Choice(["mean", "median", "zero"]),
    default="mean",
    help="How to handle missing values (copula method)",
)
@click.option(
    "--format",
    type=click.Choice(["csv", "json", "parquet"]),
    default="csv",
    help="Output file format (default: csv)",
)
def generate_data(
    input_file, 
    output_file, 
    method, 
    samples, 
    sensitive, 
    auto_detect_sensitive, 
    sensitivity_confidence, 
    sensitivity_sample_size,
    preserve_outliers,
    handle_skewness,
    categorical_threshold,
    epochs,
    batch_size,
    learning_rate,
    noise_dim,
    copula_type,
    fit_method,
    handle_missing,
    format
):
    """Generate synthetic data based on a real dataset.
    
    This command creates synthetic data that has similar statistical properties
    to the input dataset but doesn't contain any actual sensitive information.
    
    Example:
        secureml synthetic generate real_data.csv synthetic_data.csv \\
        --method sdv-copula --samples 5000
        
    Advanced sensitive column detection:
        secureml synthetic generate real_data.csv synthetic_data.csv \\
        --auto-detect-sensitive --sensitivity-confidence 0.7 \\
        --sensitivity-sample-size 200
        
    Advanced statistical modeling:
        secureml synthetic generate real_data.csv synthetic_data.csv \\
        --method statistical --preserve-outliers \\
        --handle-skewness --categorical-threshold 15
        
    GAN-based generation:
        secureml synthetic generate real_data.csv synthetic_data.csv \\
        --method gan --epochs 500 --batch-size 64 \\
        --learning-rate 0.0002 --noise-dim 128
        
    Copula-based generation:
        secureml synthetic generate real_data.csv synthetic_data.csv \\
        --method copula --copula-type gaussian --fit-method ml \\
        --handle-missing mean --categorical-threshold 15
    """
    click.echo(f"Reading template data from {input_file}...")
    
    # Load the template dataset
    input_path = Path(input_file)
    if input_path.suffix == '.csv':
        data = pd.read_csv(input_file)
    elif input_path.suffix == '.json':
        data = pd.read_json(input_file)
    elif input_path.suffix in ['.parquet', '.pq']:
        data = pd.read_parquet(input_file)
    else:
        click.echo(
            f"Unsupported input format: {input_path.suffix}. Trying csv format..."
        )
        data = pd.read_csv(input_file)
    
    # Configure sensitivity detection
    sensitivity_detection = {
        "auto_detect": sensitive == () or auto_detect_sensitive,
        "confidence_threshold": sensitivity_confidence,
        "sample_size": sensitivity_sample_size
    }
    
    # Method-specific parameters
    method_params = {}
    
    # Statistical modeling parameters
    if method == "statistical":
        method_params = {
            "preserve_outliers": preserve_outliers,
            "handle_skewness": handle_skewness,
            "categorical_threshold": categorical_threshold,
        }
    # GAN parameters
    elif method == "gan":
        method_params = {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "noise_dim": noise_dim,
        }
    # Copula parameters
    elif method == "copula":
        method_params = {
            "copula_type": copula_type,
            "fit_method": fit_method,
            "handle_missing": handle_missing,
            "handle_skewness": handle_skewness,
            "categorical_threshold": categorical_threshold,
        }
    
    click.echo(
        f"Generating {samples} synthetic samples using {method} method..."
    )
    
    # Generate synthetic data
    synthetic_data = generate_synthetic_data(
        template=data,
        num_samples=samples,
        method=method,
        sensitive_columns=list(sensitive) if sensitive else None,
        sensitivity_detection=sensitivity_detection,
        **method_params
    )
    
    # Save output in the requested format
    if format == 'csv':
        synthetic_data.to_csv(output_file, index=False)
    elif format == 'json':
        synthetic_data.to_json(output_file, orient='records')
    elif format == 'parquet':
        synthetic_data.to_parquet(output_file, index=False)
    
    click.echo(f"Synthetic data saved to {output_file}")
    
    # If auto-detection was used, show which columns were detected as sensitive
    if sensitivity_detection["auto_detect"] and not sensitive:
        detected_cols = _identify_sensitive_columns(
            data, 
            sample_size=sensitivity_sample_size,
            confidence_threshold=sensitivity_confidence
        )
        if detected_cols:
            click.echo("\nDetected sensitive columns:")
            for col in detected_cols:
                click.echo(f"- {col}")
        else:
            click.echo("\nNo sensitive columns were detected with the current confidence threshold.")
            click.echo(f"Try lowering the confidence threshold (current: {sensitivity_confidence}).")


@cli.group()
def presets():
    """Regulation presets commands."""
    pass


@presets.command("list")
def list_presets():
    """List available regulation presets.
    
    This command shows all the built-in regulation presets that can be used
    for compliance checking and other privacy operations.
    
    Example:
        secureml presets list
    """
    available_presets = list_available_presets()
    click.echo("Available regulation presets:")
    for preset in available_presets:
        click.echo(f"- {preset}")


@presets.command("show")
@click.argument("preset_name")
@click.option(
    "--field", "-f", help="Show a specific field from the preset"
)
@click.option(
    "--output", "-o", type=click.Path(), help="Save preset to a file"
)
def show_preset(preset_name, field, output):
    """Show details of a regulation preset.
    
    This command displays the content of a regulation preset, or a specific
    field from the preset if requested.
    
    Examples:
        secureml presets show gdpr
        secureml presets show gdpr --field personal_data_identifiers
    """
    try:
        if field:
            # Show a specific field
            from secureml.presets import get_preset_field
            value = get_preset_field(preset_name, field)
            if isinstance(value, (list, dict)):
                if output:
                    with open(output, 'w') as f:
                        json.dump(value, f, indent=2)
                    click.echo(f"Field '{field}' saved to {output}")
                else:
                    click.echo(json.dumps(value, indent=2))
            else:
                if output:
                    with open(output, 'w') as f:
                        f.write(str(value))
                    click.echo(f"Field '{field}' saved to {output}")
                else:
                    click.echo(value)
        else:
            # Show the entire preset
            preset = load_preset(preset_name)
            if output:
                with open(output, 'w') as f:
                    json.dump(preset, f, indent=2)
                click.echo(f"Preset '{preset_name}' saved to {output}")
            else:
                click.echo(json.dumps(preset, indent=2))
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)


@cli.group()
def environments():
    """Manage isolated environments for dependency-conflicting packages."""
    pass


@environments.command("setup-tf-privacy")
@click.option(
    "--force", is_flag=True, help="Force recreation of the environment even if it exists"
)
def setup_tf_privacy(force):
    """Set up the isolated environment for TensorFlow Privacy."""
    click.echo("Setting up TensorFlow Privacy isolated environment...")
    try:
        if force:
            # If force flag is set, we need to delete the existing environment first
            venv_path = get_env_path()
            if os.path.exists(venv_path):
                click.echo(f"Removing existing environment at {venv_path}...")
                shutil.rmtree(venv_path, ignore_errors=True)
        
        # Set up the environment
        setup_tf_privacy_environment()
        venv_path = get_env_path()
        click.echo(f"Environment created at: {venv_path}")
    except RuntimeError as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@environments.command("info")
def environment_info():
    """Show information about isolated environments."""
    # Get the path to the TensorFlow Privacy environment
    venv_path = get_env_path()
    # Check if it's valid
    valid = is_env_valid()
    
    if valid:
        click.echo(f"TensorFlow Privacy environment: {venv_path} (Installed and valid)")
    else:
        click.echo(f"TensorFlow Privacy environment: {venv_path} (Not properly installed)")
        click.echo("Run 'secureml environments setup-tf-privacy' to set up the environment")


@cli.group()
def keys():
    """Manage encryption keys and secrets for SecureML."""
    pass


@keys.command()
@click.option(
    "--vault-url",
    "-u",
    help="HashiCorp Vault server URL",
    envvar="SECUREML_VAULT_URL",
)
@click.option(
    "--vault-token",
    "-t",
    help="HashiCorp Vault authentication token",
    envvar="SECUREML_VAULT_TOKEN",
)
@click.option(
    "--vault-path",
    "-p",
    help="Base path in Vault for SecureML secrets",
    default="secureml",
)
@click.option(
    "--test-connection",
    "-c",
    is_flag=True,
    help="Test connection to Vault server",
)
def configure_vault(vault_url, vault_token, vault_path, test_connection):
    """
    Configure HashiCorp Vault for secure key storage.
    
    This command will configure the connection to HashiCorp Vault
    for secure key storage and management. If you don't specify
    --vault-url or --vault-token, it will check for SECUREML_VAULT_URL
    and SECUREML_VAULT_TOKEN environment variables.
    """
    from secureml.key_management import KeyManager
    
    # If no parameters provided, print help and exit
    if not vault_url and not vault_token and not test_connection:
        click.echo("No parameters provided. Use --help for usage information.")
        return
    
    # Initialize key manager
    try:
        key_manager = KeyManager(
            vault_url=vault_url,
            vault_token=vault_token,
            vault_path=vault_path,
            use_env_fallback=True,
            use_default_fallback=False,
        )
        
        if test_connection:
            if key_manager.vault_client and key_manager.vault_client.is_authenticated():
                click.secho("✓ Vault connection successful", fg="green")
                click.echo(f"  URL: {vault_url}")
                click.echo(f"  Path: {vault_path}")
            else:
                click.secho("✗ Vault connection failed", fg="red")
                if not vault_url:
                    click.echo("  Vault URL not provided")
                if not vault_token:
                    click.echo("  Vault token not provided")
        else:
            click.echo("Vault configuration saved.")
            click.echo("Use environment variables or the configure_default_key_manager() function to use this configuration in your code.")
    except Exception as e:
        click.secho(f"Error configuring Vault: {str(e)}", fg="red")


@keys.command()
@click.option(
    "--vault-url",
    "-u",
    help="HashiCorp Vault server URL",
    envvar="SECUREML_VAULT_URL",
)
@click.option(
    "--vault-token",
    "-t",
    help="HashiCorp Vault authentication token",
    envvar="SECUREML_VAULT_TOKEN",
)
@click.option(
    "--vault-path",
    "-p",
    help="Base path in Vault for SecureML secrets",
    default="secureml",
)
@click.option(
    "--key-name",
    "-k",
    help="Name of the key to store",
    required=True,
)
@click.option(
    "--length",
    "-l",
    help="Length of the generated key in bytes",
    default=32,
    type=int,
)
@click.option(
    "--encoding",
    "-e",
    help="Encoding format for the generated key",
    type=click.Choice(["base64", "hex"]),
    default="hex",
)
def generate_key(vault_url, vault_token, vault_path, key_name, length, encoding):
    """
    Generate and store a new encryption key in Vault.
    
    This command generates a cryptographically secure random key
    and stores it in HashiCorp Vault for later use. The key can
    be used for encryption, pseudonymization, or other security
    operations in SecureML.
    """
    import os
    import base64
    from secureml.key_management import KeyManager
    
    try:
        # Generate a secure random key
        random_key = os.urandom(length)
        
        # Format the key according to the requested encoding
        if encoding == "hex":
            key_value = random_key.hex()
        else:  # base64
            key_value = base64.b64encode(random_key).decode('ascii')
        
        # Initialize key manager
        key_manager = KeyManager(
            vault_url=vault_url,
            vault_token=vault_token,
            vault_path=vault_path,
            use_env_fallback=True,
            use_default_fallback=False,
        )
        
        # Store the key in Vault
        if key_manager.vault_client:
            success = key_manager.set_secret(key_name, key_value)
            if success:
                click.secho(f"✓ Key '{key_name}' generated and stored successfully", fg="green")
                click.echo(f"  Path: {vault_path}/data/{key_name}")
                click.echo(f"  Length: {length} bytes")
                click.echo(f"  Encoding: {encoding}")
            else:
                click.secho(f"✗ Failed to store key in Vault", fg="red")
        else:
            click.secho("✗ Vault connection not configured", fg="red")
            click.echo("  Use --vault-url and --vault-token options or set")
            click.echo("  SECUREML_VAULT_URL and SECUREML_VAULT_TOKEN environment variables")
    except Exception as e:
        click.secho(f"Error generating key: {str(e)}", fg="red")


@keys.command()
@click.option(
    "--vault-url",
    "-u",
    help="HashiCorp Vault server URL",
    envvar="SECUREML_VAULT_URL",
)
@click.option(
    "--vault-token",
    "-t",
    help="HashiCorp Vault authentication token",
    envvar="SECUREML_VAULT_TOKEN",
)
@click.option(
    "--vault-path",
    "-p",
    help="Base path in Vault for SecureML secrets",
    default="secureml",
)
@click.option(
    "--key-name",
    "-k",
    help="Name of the key to retrieve",
    required=True,
)
@click.option(
    "--encoding",
    "-e",
    help="Desired output encoding format for the key",
    type=click.Choice(["base64", "hex"]),
    default="hex",
)
def get_key(vault_url, vault_token, vault_path, key_name, encoding):
    """
    Retrieve an encryption key from Vault.
    
    This command retrieves a key from HashiCorp Vault for use
    in encryption, pseudonymization, or other security operations.
    """
    from secureml.key_management import KeyManager
    
    try:
        # Initialize key manager
        key_manager = KeyManager(
            vault_url=vault_url,
            vault_token=vault_token,
            vault_path=vault_path,
            use_env_fallback=True,
            use_default_fallback=False,
        )
        
        # Retrieve the key
        try:
            key = key_manager.get_encryption_key(key_name=key_name, encoding=encoding)
            if key:
                click.secho(f"✓ Key '{key_name}' retrieved successfully", fg="green")
                click.echo(f"  Key: {key}")
                click.echo(f"  Encoding: {encoding}")
            else:
                click.secho(f"✗ Key '{key_name}' not found", fg="red")
        except ValueError as e:
            click.secho(f"✗ {str(e)}", fg="red")
    except Exception as e:
        click.secho(f"Error retrieving key: {str(e)}", fg="red")


if __name__ == "__main__":
    cli() 