CLI Examples
===========

This section demonstrates how to use the SecureML command-line interface through practical examples. You can use these examples as a starting point for your own privacy-preserving data workflows.

Basic Setup
----------

To use the CLI, make sure you have SecureML installed:

.. code-block:: bash

    pip install secureml

Getting help information:

.. code-block:: bash

    # Show general help
    secureml --help
    
    # Show help for a specific command
    secureml anonymization --help
    
    # Show version information
    secureml --version

Anonymization Examples
-------------------

Applying k-anonymity to protect sensitive data:

.. code-block:: bash

    # Basic k-anonymity with k=3
    secureml anonymization k-anonymize patient_data.csv anonymized_data.csv \
        --quasi-id age --quasi-id zipcode \
        --sensitive diagnosis --sensitive income \
        --k 3
    
    # Using a different output format
    secureml anonymization k-anonymize patient_data.csv anonymized_data.json \
        --quasi-id age --quasi-id zipcode \
        --sensitive diagnosis \
        --k 2 \
        --format json

Compliance Checking Examples
-------------------------

Verifying compliance with privacy regulations:

.. code-block:: bash

    # Basic GDPR compliance check
    secureml compliance check patient_data.csv \
        --regulation GDPR
    
    # Compliance check with metadata and HTML output
    secureml compliance check patient_data.csv \
        --regulation GDPR \
        --metadata metadata.json \
        --output gdpr_report.html \
        --format html

Example metadata.json file:

.. code-block:: json

    {
        "description": "Patient health data",
        "data_owner": "Example Hospital",
        "data_retention_period": "5 years",
        "data_encrypted": true,
        "data_storage_location": "EU",
        "consent_obtained": true,
        "consent_date": "2023-01-15"
    }

Checking both dataset and model compliance:

.. code-block:: bash

    # Comprehensive HIPAA compliance check
    secureml compliance check patient_data.csv \
        --regulation HIPAA \
        --metadata metadata.json \
        --model-config model_config.json \
        --output hipaa_report.pdf \
        --format pdf

Example model_config.json file:

.. code-block:: json

    {
        "model_type": "RandomForestClassifier",
        "parameters": {
            "n_estimators": 100,
            "max_depth": 5
        },
        "supports_forget_request": true,
        "supports_deletion_request": true,
        "data_processing_purpose": "Medical diagnosis prediction",
        "model_storage_location": "EU"
    }

Synthetic Data Generation Examples
-------------------------------

Creating synthetic datasets based on real data:

.. code-block:: bash

    # Basic statistical synthesis
    secureml synthetic generate patient_data.csv synthetic_data.csv \
        --method statistical \
        --samples 1000
    
    # Auto-detecting sensitive columns
    secureml synthetic generate patient_data.csv synthetic_data.csv \
        --method statistical \
        --auto-detect-sensitive \
        --sensitivity-confidence 0.7 \
        --sensitivity-sample-size 200 \
        --samples 1000
    
    # Using GAN-based synthesis with specific sensitive columns
    secureml synthetic generate patient_data.csv synthetic_data.parquet \
        --method gan \
        --sensitive name --sensitive email --sensitive diagnosis \
        --epochs 300 --batch-size 32 \
        --samples 500 \
        --format parquet

Regulation Presets Examples
------------------------

Working with regulation presets:

.. code-block:: bash

    # List all available regulation presets
    secureml presets list
    
    # View the GDPR preset
    secureml presets show gdpr
    
    # Extract just the personal data identifiers field from GDPR
    secureml presets show gdpr --field personal_data_identifiers
    
    # Save the entire HIPAA preset to a file
    secureml presets show hipaa --output hipaa_preset.json

Isolated Environment Examples
--------------------------

Managing isolated environments for conflicting dependencies:

.. code-block:: bash

    # Set up the TensorFlow Privacy environment
    secureml environments setup-tf-privacy
    
    # Check if environments are properly configured
    secureml environments info
    
    # Force recreation of an environment
    secureml environments setup-tf-privacy --force

Key Management Examples
--------------------

Working with encryption keys (requires HashiCorp Vault):

.. code-block:: bash

    # Configure Vault connection
    secureml keys configure-vault \
        --vault-url https://vault.example.com:8200 \
        --vault-token hvs.example_token \
        --vault-path secureml
    
    # Test Vault connection
    secureml keys configure-vault --test-connection
    
    # Generate a new encryption key
    secureml keys generate-key \
        --key-name patient_data_key \
        --length 32 \
        --encoding hex
    
    # Retrieve a key
    secureml keys get-key \
        --key-name patient_data_key \
        --encoding base64

Using environment variables for safer key management:

.. code-block:: bash

    # Set environment variables instead of passing tokens directly
    export SECUREML_VAULT_URL=https://vault.example.com:8200
    export SECUREML_VAULT_TOKEN=hvs.example_token
    
    # The command now uses environment variables automatically
    secureml keys get-key --key-name patient_data_key

End-to-End Example Workflow
-------------------------

A complete workflow for processing sensitive health data:

.. code-block:: bash

    # 1. Check compliance of the original dataset
    secureml compliance check patient_data.csv \
        --regulation GDPR \
        --output compliance_original.html \
        --format html
    
    # 2. Anonymize the dataset for safe processing
    secureml anonymization k-anonymize patient_data.csv anonymized_data.csv \
        --quasi-id age --quasi-id zipcode \
        --sensitive diagnosis --sensitive income \
        --k 3
    
    # 3. Check compliance of the anonymized dataset
    secureml compliance check anonymized_data.csv \
        --regulation GDPR \
        --output compliance_anonymized.html \
        --format html
    
    # 4. Generate synthetic data for sharing with researchers
    secureml synthetic generate anonymized_data.csv synthetic_data.csv \
        --method statistical \
        --auto-detect-sensitive \
        --samples 1000
    
    # 5. Final compliance check on the synthetic data
    secureml compliance check synthetic_data.csv \
        --regulation GDPR \
        --output compliance_synthetic.html \
        --format html

Processing Multiple Files
-----------------------

Example shell script for batch processing:

.. code-block:: bash

    #!/bin/bash
    
    # Directory containing data files
    DATA_DIR="patient_data"
    
    # Process each CSV file in the directory
    for file in "$DATA_DIR"/*.csv; do
        filename=$(basename "$file" .csv)
        
        echo "Processing $filename..."
        
        # Check compliance
        secureml compliance check "$file" \
            --regulation GDPR \
            --output "reports/${filename}_compliance.html" \
            --format html
        
        # Anonymize data
        secureml anonymization k-anonymize "$file" \
            "anonymized/${filename}_anon.csv" \
            --quasi-id age --quasi-id zipcode \
            --sensitive diagnosis --sensitive income \
            --k 3
        
        # Generate synthetic data
        secureml synthetic generate "anonymized/${filename}_anon.csv" \
            "synthetic/${filename}_synth.csv" \
            --method statistical \
            --samples 1000
        
        echo "$filename completed."
    done
    
    echo "All files processed."

Performance Considerations
------------------------

For large datasets, consider these performance tips:

1. **Batch processing**: Process large files in batches rather than all at once
2. **Sample data first**: Test your commands on a small sample before processing the entire dataset
3. **Choose appropriate output formats**: For large datasets, parquet format may be more efficient
4. **Monitor resources**: Some operations (especially GAN-based synthetic data generation) can be resource-intensive

.. code-block:: bash

    # Process only a subset of records for testing
    head -n 1000 large_dataset.csv > sample_dataset.csv
    
    # Test your workflow on the sample
    secureml synthetic generate sample_dataset.csv synthetic_sample.csv \
        --method statistical \
        --samples 500
    
    # If satisfied, process the full dataset with parquet output
    secureml synthetic generate large_dataset.csv synthetic_full.parquet \
        --method statistical \
        --samples 10000 \
        --format parquet 