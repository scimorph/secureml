#!/usr/bin/env python
"""
Example script demonstrating how to use SecureML's compliance checker
with regulation-specific presets.
"""

import pandas as pd
from secureml import check_compliance
from secureml.presets import list_available_presets, load_preset

def main():
    """Run the compliance checking examples."""
    print("SecureML Compliance Checking Examples")
    print("=====================================\n")
    
    # List available regulatory presets
    print(f"Available regulation presets: {', '.join(list_available_presets())}\n")
    
    # Create a sample dataset with potential compliance issues
    data = pd.DataFrame({
        'patient_name': ['John Doe', 'Jane Smith', 'Robert Johnson'],
        'email_address': ['john@example.com', 'jane@example.com', 'robert@example.com'],
        'phone_number': ['555-123-4567', '555-234-5678', '555-345-6789'],
        'birth_date': ['1980-01-15', '1992-05-20', '1975-11-03'],
        'medical_condition': ['Hypertension', 'Diabetes', 'Asthma'],
        'treatment_plan': ['Medication A', 'Insulin therapy', 'Inhaler as needed'],
        'doctor_notes': [
            'Patient reports feeling better after treatment.',
            'Patient needs follow-up in 3 months.',
            'Patient shows improvement in respiratory function.'
        ],
        'home_address': [
            '123 Main St, Anytown, USA',
            '456 Oak Ave, Somewhere, USA',
            '789 Pine Rd, Nowhere, USA'
        ]
    })
    
    # Add metadata about the dataset
    metadata = {
        'dataset_name': 'Patient Records',
        'created_date': '2023-01-15',
        'data_source': 'Hospital System',
        'data_owner': 'Medical Research Department',
        'data_storage_location': 'US-East',
        'contains_ca_residents': True,
        'data_shared_with_third_parties': True,
        'ccpa_disclosure_provided': False,
        'data_encrypted': False
    }
    
    # Model configuration
    model_config = {
        'model_type': 'random_forest',
        'target_variable': 'treatment_plan',
        'features': ['medical_condition', 'birth_date'],
        'supports_forget_request': False,
        'supports_deletion_request': False,
        'access_controls': False
    }
    
    # Check compliance with each regulation
    for regulation in list_available_presets():
        print(f"\n\n{'='*50}")
        print(f"Checking compliance with {regulation.upper()}")
        print(f"{'='*50}\n")
        
        # Load and display some key information from the preset
        preset = load_preset(regulation)
        print(f"Regulation: {preset['regulation']['name']} - {preset['regulation']['description']}")
        
        # Special handling for HIPAA with multiple effective dates
        if regulation.lower() == 'hipaa':
            print(f"Privacy Rule Effective: {preset['regulation']['privacy_rule_effective']}")
            print(f"Security Rule Effective: {preset['regulation']['security_rule_effective']}")
            print(f"Breach Notification Effective: {preset['regulation']['breach_notification_effective']}\n")
        else:
            print(f"Effective Date: {preset['regulation']['effective_date']}\n")
        
        # Perform compliance check
        report = check_compliance(
            {'data': data, **metadata},
            model_config=model_config,
            regulation=regulation
        )
        
        # Display the report
        print(report)
        
        # Show a summary
        summary = report.summary()
        print(f"\nSummary: {summary['issues_count']} issues, {summary['warnings_count']} warnings")
        print(f"Compliant: {'Yes' if summary['compliant'] else 'No'}")


if __name__ == '__main__':
    main() 