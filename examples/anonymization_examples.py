import pandas as pd
from secureml.anonymization import anonymize, _identify_sensitive_columns

# Sample Data (adjust as needed for realism)
data_list = [
    {'id': 1, 'name': 'Alice Smith', 'age': 30, 'zipcode': '12345', 'diagnosis': 'Flu', 'email': 'alice.s@example.com', 'income': 60000, 'phone': '555-1234'},
    {'id': 2, 'name': 'Bob Johnson', 'age': 45, 'zipcode': '12345', 'diagnosis': 'Diabetes', 'email': 'b.johnson@email.net', 'income': 85000, 'phone': '555-5678'},
    {'id': 3, 'name': 'Charlie Brown', 'age': 30, 'zipcode': '67890', 'diagnosis': 'Allergy', 'email': 'charlie@domain.org', 'income': 55000, 'phone': '555-1122'},
    {'id': 4, 'name': 'Diana Prince', 'age': 35, 'zipcode': '12345', 'diagnosis': 'Flu', 'email': 'diana.p@sample.com', 'income': 72000, 'phone': '555-3344'},
    {'id': 5, 'name': 'Ethan Hunt', 'age': 50, 'zipcode': '98765', 'diagnosis': 'Hypertension', 'email': 'ethan.h@test.io', 'income': 120000, 'phone': '555-9900'},
    {'id': 6, 'name': 'Alice Miller', 'age': 30, 'zipcode': '12345', 'diagnosis': 'Cold', 'email': 'alice.m@example.com', 'income': 61000, 'phone': '555-1235'},
    {'id': 7, 'name': 'Bob Davis', 'age': 45, 'zipcode': '12345', 'diagnosis': 'Diabetes', 'email': 'b.davis@email.net', 'income': 86000, 'phone': '555-5679'},
    {'id': 8, 'name': 'Frank Castle', 'age': 40, 'zipcode': '67890', 'diagnosis': 'Headache', 'email': 'frank@domain.org', 'income': 70000, 'phone': '555-1123'},
    {'id': 9, 'name': 'Grace Hopper', 'age': 35, 'zipcode': '12345', 'diagnosis': 'Cold', 'email': 'grace.h@sample.com', 'income': 75000, 'phone': '555-3345'},
    {'id': 10, 'name': 'Henry Jones', 'age': 50, 'zipcode': '98765', 'diagnosis': 'Hypertension', 'email': 'henry.j@test.io', 'income': 125000, 'phone': '555-9901'},
]

df = pd.DataFrame(data_list)

print("Original Data:")
print(df)
print("\n" + "="*50 + "\n")

# --- Example 1: k-Anonymity --- 
print("Example 1: k-Anonymity (k=2)")
# Identify sensitive columns automatically or specify manually
sensitive_cols_auto = _identify_sensitive_columns(df)
print(f"Automatically identified sensitive columns: {sensitive_cols_auto}")

# Define quasi-identifiers (often age, zipcode)
quasi_identifiers = ['age', 'zipcode']

# Use 'id' and 'diagnosis' as sensitive for this example, others as quasi
sensitive_k_anon = ['id', 'diagnosis', 'income'] # Keep these less changed
quasi_k_anon = ['age', 'zipcode', 'name', 'email', 'phone'] # Generalize these

try:
    anonymized_k = anonymize(
        df.copy(), 
        method="k-anonymity", 
        k=2, 
        sensitive_columns=sensitive_k_anon,
        quasi_identifier_columns=quasi_k_anon, # Specify QIs for better control
        suppression_threshold=0.2 # Allow suppressing up to 20% if needed
    )
    print("\nAnonymized with k-Anonymity (k=2):")
    print(anonymized_k)
except Exception as e:
    print(f"\nError applying k-Anonymity: {e}")
print("\n" + "="*50 + "\n")


# --- Example 2: Pseudonymization --- 
print("Example 2: Pseudonymization")
sensitive_pseudo = ['name', 'email', 'phone']

# Strategy: Hash
anonymized_hash = anonymize(
    df.copy(),
    method="pseudonymization",
    sensitive_columns=sensitive_pseudo,
    strategy="hash",
    preserve_format=True # Try to keep email/phone structure
)
print("\nPseudonymized with Hash (Preserve Format):")
print(anonymized_hash)

# Strategy: Deterministic (requires a salt)
anonymized_determ = anonymize(
    df.copy(),
    method="pseudonymization",
    sensitive_columns=sensitive_pseudo,
    strategy="deterministic",
    salt="my-super-secret-salt-123",
    preserve_format=True
)
print("\nPseudonymized with Deterministic Hash (Salted, Preserve Format):")
print(anonymized_determ)

# Strategy: Format-Preserving Encryption (FPE)
# Note: Requires key management setup (secureml.key_management)
# This might need a key to be set up first. 
# For demonstration, we'll use the default internal handling.
try:
    anonymized_fpe = anonymize(
        df.copy(),
        method="pseudonymization",
        sensitive_columns=sensitive_pseudo,
        strategy="fpe", 
        # master_key_name='my_fpe_key' # Optional: specify key name if configured
    )
    print("\nPseudonymized with FPE:")
    print(anonymized_fpe)
except ImportError as e:
    print(f"\nCould not run FPE example: {e}. Make sure key management dependencies are installed.")
except Exception as e:
    print(f"\nError applying FPE Pseudonymization: {e}")

print("\n" + "="*50 + "\n")


# --- Example 3: Data Masking --- 
print("Example 3: Data Masking")
sensitive_mask = ['email', 'phone', 'income', 'name']

masking_rules = {
    "email": {"strategy": "regex", "pattern": r"(.*)(@.*)", "replacement": r"masked***\2"}, # Mask local part
    "phone": {"strategy": "character", "show_last": 4, "mask_char": "X"}, # Show last 4 digits
    "income": {"strategy": "fixed", "format": "******"}, # Fixed mask
    "name": {"strategy": "character", "show_first": 1, "mask_char": "."} # Initial + dots
}

anonymized_masking = anonymize(
    df.copy(),
    method="data-masking",
    sensitive_columns=sensitive_mask,
    masking_rules=masking_rules,
    preserve_format=False # Rules define the format here
)
print("\nMasked Data with Custom Rules:")
print(anonymized_masking)

# Example: Masking with statistic preservation (for numeric)
anonymized_masking_stats = anonymize(
    df.copy(),
    method="data-masking",
    sensitive_columns=['income'],
    masking_rules={"income": {"strategy": "random", "preserve_statistics": True}},
    preserve_statistics=True # Enable global flag as well
)
print("\nMasked Data (Preserving Income Statistics):")
print(anonymized_masking_stats)
print("\n" + "="*50 + "\n")


# --- Example 4: Generalization --- 
print("Example 4: Generalization")
sensitive_generalize = ['age', 'zipcode', 'diagnosis']

# Define a simple diagnosis hierarchy
dx_hierarchy = {
    "Flu": "Respiratory",
    "Cold": "Respiratory",
    "Allergy": "Immune",
    "Diabetes": "Endocrine",
    "Hypertension": "Cardiovascular",
    "Headache": "Neurological"
}

generalization_rules = {
    "age": {"method": "range", "range_size": 10}, # Group age into 10-year bands
    "zipcode": {"method": "topk", "k": 2, "other_value": "Other_Zip"}, # Keep top 2 zipcodes, generalize rest
    "diagnosis": {"method": "hierarchy", "taxonomy": dx_hierarchy, "level": 1} # Use defined hierarchy
}

anonymized_general = anonymize(
    df.copy(),
    method="generalization",
    sensitive_columns=sensitive_generalize,
    generalization_rules=generalization_rules
)
print("\nGeneralized Data:")
print(anonymized_general)
print("\n" + "="*50 + "\n")

print("Anonymization examples complete.") 