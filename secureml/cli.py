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
    type=click.Choice(["GDPR", "CCPA", "HIPAA"]),
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
    privacy regulations like GDPR, CCPA, or HIPAA.
    
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
    type=click.Choice(["simple", "statistical", "sdv-copula", "sdv-ctgan"]),
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
    
    click.echo(
        f"Generating {samples} synthetic samples using {method} method..."
    )
    
    # Generate synthetic data
    synthetic_data = generate_synthetic_data(
        template=data,
        num_samples=samples,
        method=method,
        sensitive_columns=list(sensitive) if sensitive else None,
        sensitivity_detection=sensitivity_detection
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


if __name__ == "__main__":
    cli() 