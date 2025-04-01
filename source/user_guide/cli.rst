===================
Command-Line Interface
===================

SecureML provides a powerful command-line interface (CLI) that allows you to perform most common operations without writing Python code. The CLI is designed to be easy to use, scriptable, and follows standard Unix/Linux command-line conventions.

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
    secureml --verbose              # Enable verbose output
    secureml --config CONFIG_FILE   # Use a specific configuration file

Common Commands
-------------

Anonymization Commands
^^^^^^^^^^^^^^^^^^^^

Anonymize datasets from the command line:

.. code-block:: bash

    # Anonymize a CSV file
    secureml anonymize data.csv --output anonymized_data.csv \
        --quasi-identifiers age,zipcode,gender \
        --sensitive-attributes income,disease \
        --k 5 --l 3 --t 0.2
    
    # Anonymize with differential privacy
    secureml anonymize data.csv --output anonymized_data.csv \
        --quasi-identifiers age,zipcode,gender \
        --sensitive-attributes income,disease \
        --apply-dp --epsilon 1.0 --delta 1e-5
    
    # Save the anonymization configuration for future use
    secureml anonymize data.csv --output anonymized_data.csv \
        --quasi-identifiers age,zipcode,gender \
        --sensitive-attributes income,disease \
        --k 5 --save-config anonymization_config.json
    
    # Reuse a saved configuration
    secureml anonymize new_data.csv --output new_anonymized_data.csv \
        --config anonymization_config.json

Differential Privacy Commands
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Train models with differential privacy:

.. code-block:: bash

    # Train a differentially private model
    secureml dp-train --data training_data.csv --output dp_model.pkl \
        --target-column target --model-type logistic_regression \
        --epsilon 1.0 --delta 1e-5 --batch-size 64
    
    # Evaluate a differentially private model
    secureml dp-evaluate dp_model.pkl --data test_data.csv \
        --target-column target --metrics accuracy,precision,recall,f1
    
    # Generate a privacy-utility curve
    secureml dp-privacy-utility-curve --data training_data.csv \
        --test-data test_data.csv --target-column target \
        --model-type logistic_regression \
        --epsilons 0.1,0.5,1.0,2.0,5.0,10.0 \
        --output privacy_utility_curve.png

Synthetic Data Commands
^^^^^^^^^^^^^^^^^^^^

Generate and evaluate synthetic data:

.. code-block:: bash

    # Generate synthetic data
    secureml synthetic-generate --data original_data.csv \
        --output synthetic_data.csv --method ctgan \
        --categorical-columns gender,occupation,education \
        --continuous-columns age,income,height,weight \
        --epochs 300 --batch-size 500
    
    # Evaluate synthetic data quality
    secureml synthetic-evaluate --real-data original_data.csv \
        --synthetic-data synthetic_data.csv \
        --categorical-columns gender,occupation,education \
        --continuous-columns age,income,height,weight \
        --output evaluation_report.html
    
    # Generate synthetic data with privacy guarantees
    secureml synthetic-generate --data original_data.csv \
        --output synthetic_data.csv --method dp_ctgan \
        --categorical-columns gender,occupation,education \
        --continuous-columns age,income,height,weight \
        --epsilon 3.0 --delta 1e-5

Compliance Commands
^^^^^^^^^^^^^^^^^

Check compliance with privacy regulations:

.. code-block:: bash

    # Check dataset compliance
    secureml compliance-check --data customer_data.csv \
        --regulations gdpr,ccpa \
        --sensitive-attributes ssn,name,address,dob \
        --quasi-identifiers zipcode,age,gender \
        --output compliance_report.pdf
    
    # Check model compliance
    secureml compliance-check-model --model credit_model.pkl \
        --training-data training_data.csv \
        --regulations gdpr,hipaa \
        --output model_compliance_report.pdf
    
    # Generate compliance documentation
    secureml compliance-docs --data patient_data.csv \
        --model diagnosis_model.pkl \
        --regulations hipaa,gdpr \
        --documents privacy_notice,dpia,records_of_processing \
        --output-dir compliance_docs/

Work with Regulation Presets

.. code-block:: bash
    
    # List available presets
    secureml presets list

    # View a specific preset
    secureml presets show gdpr

    # Extract a specific field from a preset
    secureml presets show gdpr --field personal_data_identifiers

    # Save a preset to a file
    secureml presets show hipaa --output hipaa_preset.json

Key Management Commands
^^^^^^^^^^^^^^^^^^^^

Manage encryption keys:

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
    
    # Delete a key (with confirmation)
    secureml keys delete KEY_ID

Audit Commands
^^^^^^^^^^^^

Work with audit trails:

.. code-block:: bash

    # Initialize audit trails
    secureml audit init --storage-backend file --storage-path audit_logs/
    
    # Generate audit reports
    secureml audit report --start-date 2023-01-01 --end-date 2023-06-30 \
        --report-type activity --output audit_report_q2.pdf
    
    # Search audit logs
    secureml audit search --event-type data_access --user analyst_123 \
        --start-date 2023-01-01 --end-date 2023-06-30
    
    # Export audit logs
    secureml audit export --start-date 2023-01-01 --end-date 2023-06-30 \
        --format json --output audit_logs_q2_2023.json

Advanced Usage
------------

Automation and Scripting
^^^^^^^^^^^^^^^^^^^^^

Use the CLI in automation scripts and workflows:

.. code-block:: bash

    #!/bin/bash
    
    # Script to process new data files with SecureML
    
    # Anonymize the data
    secureml anonymize new_data.csv --output anonymized_data.csv \
        --config anonymization_config.json
    
    # Generate synthetic version
    secureml synthetic-generate --data anonymized_data.csv \
        --output synthetic_data.csv --method ctgan \
        --config synthetic_config.json
    
    # Train a differentially private model
    secureml dp-train --data synthetic_data.csv --output dp_model.pkl \
        --target-column target --model-type random_forest \
        --epsilon 3.0 --delta 1e-5
    
    # Check compliance
    secureml compliance-check --data anonymized_data.csv \
        --regulations gdpr,ccpa --output compliance_report.pdf
    
    echo "Processing complete!"

Configuration Files
^^^^^^^^^^^^^^^^

Use configuration files to store common settings:

.. code-block:: bash

    # Create a configuration template
    secureml config init --output secureml_config.json
    
    # Run commands with the configuration
    secureml --config secureml_config.json anonymize data.csv --output anonymized_data.csv

Example configuration file (secureml_config.json):

.. code-block:: json

    {
        "key_management": {
            "backend": "vault",
            "vault_url": "https://vault.example.com:8200",
            "mount_point": "secureml"
        },
        "audit": {
            "enabled": true,
            "storage_backend": "file",
            "storage_path": "audit_logs/"
        },
        "anonymization": {
            "default_k": 5,
            "default_l": 3,
            "default_t": 0.2
        },
        "differential_privacy": {
            "default_epsilon": 1.0,
            "default_delta": 1e-5
        }
    }

Pipeline Commands
^^^^^^^^^^^^^^

Define and run end-to-end privacy-preserving ML pipelines:

.. code-block:: bash

    # Create a pipeline template
    secureml pipeline init --output ml_pipeline.yaml
    
    # Edit the pipeline configuration file (ml_pipeline.yaml)
    # ...
    
    # Run the pipeline
    secureml pipeline run ml_pipeline.yaml
    
    # Get pipeline status
    secureml pipeline status pipeline_id_12345

Example pipeline configuration (ml_pipeline.yaml):

.. code-block:: yaml

    name: credit_risk_pipeline
    description: Privacy-preserving credit risk prediction pipeline
    
    steps:
      - name: data_loading
        type: load_data
        params:
          source: customer_data.csv
          output: raw_data
      
      - name: anonymization
        type: anonymize
        params:
          input: raw_data
          output: anonymized_data
          quasi_identifiers: [age, zipcode, gender]
          sensitive_attributes: [income, loan_history]
          k: 5
      
      - name: synthetic_generation
        type: generate_synthetic
        params:
          input: anonymized_data
          output: synthetic_data
          method: ctgan
          epochs: 300
      
      - name: model_training
        type: dp_train
        params:
          input: synthetic_data
          output: trained_model
          model_type: random_forest
          target_column: credit_risk
          epsilon: 3.0
      
      - name: compliance_check
        type: check_compliance
        params:
          data: anonymized_data
          model: trained_model
          regulations: [gdpr, ccpa]
          output: compliance_report.pdf

Environment Variables
^^^^^^^^^^^^^^^^^^

Configure the CLI behavior using environment variables:

.. code-block:: bash

    # Set environment variables
    export SECUREML_CONFIG_FILE=/path/to/secureml_config.json
    export SECUREML_VAULT_ADDR=https://vault.example.com:8200
    export SECUREML_VAULT_TOKEN=hvs.example
    export SECUREML_LOG_LEVEL=INFO
    
    # Run commands (will use environment variables)
    secureml keys list

Custom Command Plugins
^^^^^^^^^^^^^^^^^^^

Extend the CLI with custom plugins:

.. code-block:: bash

    # Install a plugin
    secureml plugins install custom_plugin.zip
    
    # List installed plugins
    secureml plugins list
    
    # Use a plugin command
    secureml custom-command --option value

Interactive Mode
^^^^^^^^^^^^^

Use the interactive shell for multiple commands:

.. code-block:: bash

    # Start the interactive shell
    secureml shell
    
    # In the shell
    secureml> anonymize data.csv --output anonymized_data.csv
    secureml> synthetic-generate --data anonymized_data.csv --output synthetic_data.csv
    secureml> exit

Best Practices
-------------

1. **Use configuration files**: Store common settings in configuration files to avoid repetitive command-line parameters

2. **Script automation**: Create shell scripts for common workflows

3. **Check logs**: Use the ``--verbose`` flag to see detailed logs when troubleshooting

4. **Secure environment variables**: Use environment variables for sensitive configuration like tokens and passwords

5. **Version control configurations**: Keep configuration files in version control but exclude files with secrets

6. **Command templates**: Save complex commands as templates for future use

7. **Include audit trails**: Enable audit trails for all CLI operations in production environments

8. **Use scheduled jobs**: Set up scheduled CLI commands for regular tasks like key rotation and compliance checks

Further Reading
-------------

* ``secureml --help`` - Comprehensive help documentation for all commands
* :doc:`/examples/cli` - More examples of CLI usage
* :doc:`/api/cli` - Reference for programmatically extending the CLI 