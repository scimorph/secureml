#!/bin/bash
# SecureML CLI Examples
# This script demonstrates various ways to use the SecureML command-line interface
# Note: This is an example script. Some commands require additional setup (like Vault).

# Set up colors for better readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print section headers
print_header() {
    echo -e "\n${BLUE}=============================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}=============================================${NC}\n"
}

# Function to print commands before executing
run_command() {
    echo -e "${GREEN}$1${NC}"
    echo ""
    # Uncomment the next line to actually execute the commands
    # eval "$1"
    echo "# Command output would appear here"
    echo ""
}

print_header "Basic Help Commands"
run_command "secureml --help"
run_command "secureml --version"
run_command "secureml anonymization --help"

# ---------------------------------------------
print_header "1. Anonymization Examples"

# Create a sample dataset for the examples
echo "# First, let's create a sample dataset"
cat << 'EOF' > sample_data.csv
id,name,age,zipcode,income,diagnosis,email
1,John Smith,32,12345,75000,Hypertension,john.smith@example.com
2,Jane Doe,45,12345,85000,Diabetes,jane.doe@example.com
3,Bob Johnson,28,23456,65000,Asthma,bob.j@example.net
4,Alice Williams,36,12345,95000,Migraine,alice.w@example.org
5,Charlie Brown,41,34567,55000,Allergies,charlie@example.com
6,Diana Prince,39,23456,125000,Anxiety,diana.p@example.net
7,Ethan Hunt,44,34567,110000,Hypertension,ethan.h@example.org
8,Fiona Green,29,45678,72000,Migraine,fiona.g@example.com
9,George Miller,52,12345,92000,Diabetes,george.m@example.net
10,Hannah Lee,33,23456,68000,Asthma,hannah.l@example.org
EOF
echo "sample_data.csv created."
echo ""

# Basic k-anonymity
run_command "secureml anonymization k-anonymize sample_data.csv anonymized_data.csv \
    --quasi-id age --quasi-id zipcode \
    --sensitive diagnosis --sensitive income \
    --k 3"

# Specify output format
run_command "secureml anonymization k-anonymize sample_data.csv anonymized_data.json \
    --quasi-id age --quasi-id zipcode \
    --sensitive name --sensitive email \
    --k 2 \
    --format json"

# ---------------------------------------------
print_header "2. Compliance Checking Examples"

# Basic compliance check
run_command "secureml compliance check sample_data.csv \
    --regulation GDPR"

# Create a metadata file
cat << 'EOF' > metadata.json
{
    "description": "Patient health data",
    "data_owner": "Example Hospital",
    "data_retention_period": "5 years",
    "data_encrypted": true,
    "data_storage_location": "EU",
    "contains_ca_residents": true,
    "consent_obtained": true,
    "consent_date": "2023-01-15"
}
EOF
echo "metadata.json created."
echo ""

# Compliance check with metadata
run_command "secureml compliance check sample_data.csv \
    --regulation GDPR \
    --metadata metadata.json \
    --output gdpr_report.html \
    --format html"

# Create a model config file
cat << 'EOF' > model_config.json
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
EOF
echo "model_config.json created."
echo ""

# Compliance check with model config
run_command "secureml compliance check sample_data.csv \
    --regulation HIPAA \
    --metadata metadata.json \
    --model-config model_config.json \
    --output hipaa_report.pdf \
    --format pdf"

# ---------------------------------------------
print_header "3. Synthetic Data Generation Examples"

# Basic synthetic data generation
run_command "secureml synthetic generate sample_data.csv synthetic_data.csv \
    --method statistical \
    --samples 100"

# Synthetic data with auto-detection of sensitive columns
run_command "secureml synthetic generate sample_data.csv synthetic_data2.csv \
    --method statistical \
    --auto-detect-sensitive \
    --sensitivity-confidence 0.7 \
    --samples 150"

# Synthetic data with explicit sensitive columns and GAN method
run_command "secureml synthetic generate sample_data.csv synthetic_data3.csv \
    --method gan \
    --sensitive name --sensitive email --sensitive diagnosis \
    --epochs 200 --batch-size 16 \
    --samples 200 \
    --format parquet"

# ---------------------------------------------
print_header "4. Regulation Presets Examples"

# List available presets
run_command "secureml presets list"

# Show GDPR preset
run_command "secureml presets show gdpr"

# Extract specific field from a preset
run_command "secureml presets show gdpr --field personal_data_identifiers"

# Save preset to a file
run_command "secureml presets show hipaa --output hipaa_preset.json"

# ---------------------------------------------
print_header "5. Isolated Environment Examples"

# Setup TensorFlow Privacy environment
run_command "secureml environments setup-tf-privacy"

# Get environment info
run_command "secureml environments info"

# ---------------------------------------------
print_header "6. Key Management Examples"

# These examples require HashiCorp Vault to be set up
# The commands below are for demonstration only

# Configure Vault connection
run_command "secureml keys configure-vault \
    --vault-url https://vault.example.com:8200 \
    --vault-token hvs.example_token \
    --vault-path secureml \
    --test-connection"

# Using environment variables (safer approach)
run_command "export SECUREML_VAULT_URL=https://vault.example.com:8200"
run_command "export SECUREML_VAULT_TOKEN=hvs.example_token"

# Generate a new encryption key
run_command "secureml keys generate-key \
    --key-name patient_data_key \
    --length 32 \
    --encoding hex"

# Retrieve a key
run_command "secureml keys get-key \
    --key-name patient_data_key \
    --encoding base64"

# ---------------------------------------------
print_header "7. End-to-End Example Workflow"

run_command "# 1. Check compliance of the original dataset"
run_command "secureml compliance check sample_data.csv --regulation GDPR --output compliance_original.html --format html"
run_command ""
run_command "# 2. Anonymize the dataset for safe processing"
run_command "secureml anonymization k-anonymize sample_data.csv anonymized_for_processing.csv --quasi-id age --quasi-id zipcode --sensitive diagnosis --sensitive income --k 3"
run_command ""
run_command "# 3. Check compliance of the anonymized dataset"
run_command "secureml compliance check anonymized_for_processing.csv --regulation GDPR --output compliance_anonymized.html --format html"
run_command ""
run_command "# 4. Generate synthetic data for sharing with researchers"
run_command "secureml synthetic generate anonymized_for_processing.csv synthetic_for_sharing.csv --method statistical --auto-detect-sensitive --samples 200"
run_command ""
run_command "# 5. Final compliance check on the synthetic data"
run_command "secureml compliance check synthetic_for_sharing.csv --regulation GDPR --output compliance_synthetic.html --format html"

# Cleanup temporary files (uncomment if actually running the script)
# echo "Cleaning up temporary files..."
# rm -f sample_data.csv metadata.json model_config.json

echo -e "\nAll examples completed."