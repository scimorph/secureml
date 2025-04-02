===================
Command-Line Interface
===================

SecureML provides a command-line interface (CLI) that allows you to perform common privacy-enhancing operations directly from the command line. The CLI is designed to be easy to use and follows standard command patterns.

Installation
----------

The SecureML CLI is installed automatically when you install the SecureML package:

.. code-block:: bash

    pip install secureml

After installation, the ``secureml`` command will be available in your terminal.

Basic Usage
----------

The SecureML CLI follows a command structure with subcommands and options:

.. code-block:: bash

    secureml [command] [subcommand] [options]

General options:

.. code-block:: bash

    secureml --help                 # Show general help
    secureml [command] --help       # Show help for a specific command
    secureml --version              # Show SecureML version

Available Commands
-------------

Anonymization Commands
^^^^^^^^^^^^^^^^^^^^

Anonymize datasets from the command line:

.. code-block:: bash

    # Apply k-anonymity to a CSV file
    secureml anonymization k-anonymize data.csv anonymized_data.csv \
        --quasi-id age --quasi-id zipcode \
        --sensitive diagnosis --sensitive income \
        --k 5
    
    # Specify the output format
    secureml anonymization k-anonymize data.csv anonymized_data.json \
        --quasi-id age --quasi-id zipcode \
        --format json

Compliance Commands
^^^^^^^^^^^^^^^^^

Check compliance with privacy regulations:

.. code-block:: bash

    # Check dataset compliance with GDPR
    secureml compliance check data.csv \
        --regulation GDPR
    
    # Include metadata and save the report as HTML
    secureml compliance check data.csv \
        --regulation HIPAA \
        --metadata metadata.json \
        --output compliance_report.html \
        --format html
    
    # Check both dataset and model compliance
    secureml compliance check data.csv \
        --regulation GDPR \
        --metadata metadata.json \
        --model-config model_config.json \
        --output compliance_report.pdf \
        --format pdf

Synthetic Data Commands
^^^^^^^^^^^^^^^^^^^^

Generate synthetic data based on real datasets:

.. code-block:: bash

    # Generate synthetic data using the statistical method
    secureml synthetic generate real_data.csv synthetic_data.csv \
        --method statistical \
        --samples 1000
    
    # Automatically detect sensitive columns
    secureml synthetic generate real_data.csv synthetic_data.csv \
        --auto-detect-sensitive \
        --sensitivity-confidence 0.7 \
        --sensitivity-sample-size 200
    
    # Specify sensitive columns and generation method
    secureml synthetic generate real_data.csv synthetic_data.csv \
        --method gan \
        --sensitive name --sensitive email \
        --epochs 300 --batch-size 32 \
        --format parquet

Regulation Presets Commands
^^^^^^^^^^^^^^^^^^^^^^^^^

Work with regulation presets:

.. code-block:: bash

    # List available presets
    secureml presets list
    
    # View a specific preset
    secureml presets show gdpr
    
    # Extract a specific field from a preset
    secureml presets show gdpr --field personal_data_identifiers
    
    # Save a preset to a file
    secureml presets show hipaa --output hipaa_preset.json

Isolated Environment Commands
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Manage isolated environments for libraries with conflicting dependencies:

.. code-block:: bash

    # Set up the TensorFlow Privacy environment
    secureml environments setup-tf-privacy
    
    # Get information about the environments
    secureml environments info
    
    # Force recreation of an environment
    secureml environments setup-tf-privacy --force

Key Management Commands
^^^^^^^^^^^^^^^^^^^^

Manage encryption keys using HashiCorp Vault:

.. code-block:: bash

    # Configure Vault connection
    secureml keys configure-vault \
        --vault-url https://vault.example.com:8200 \
        --vault-token hvs.example \
        --vault-path secureml
    
    # Test the Vault connection
    secureml keys configure-vault --test-connection
    
    # Generate a new encryption key
    secureml keys generate-key \
        --key-name customer_data_key \
        --length 32 \
        --encoding hex
    
    # Retrieve a key from Vault
    secureml keys get-key \
        --key-name customer_data_key \
        --encoding base64

Environment Variables
^^^^^^^^^^^^^^^^^^

Configure the CLI behavior using environment variables:

.. code-block:: bash

    # Set environment variables for Vault access
    export SECUREML_VAULT_URL=https://vault.example.com:8200
    export SECUREML_VAULT_TOKEN=hvs.example
    
    # Run commands (will use environment variables)
    secureml keys get-key --key-name my_encryption_key

Best Practices
-------------

1. **Use environment variables**: Store sensitive values like Vault tokens in environment variables

2. **Script automation**: Create shell scripts for common workflows

3. **Input format detection**: The CLI will attempt to detect input formats based on file extensions

4. **Sensitive data handling**: Use the synthetic data generator for sharing datasets that contain sensitive information

5. **Pipeline approach**: Chain commands together in scripts to create end-to-end privacy workflows

Further Reading
-------------

* ``secureml --help`` - Comprehensive help documentation for all commands
* :doc:`/examples/cli` - More examples of CLI usage
* :doc:`/api/cli` - Reference for programmatically extending the CLI 