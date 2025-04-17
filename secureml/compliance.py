"""
Compliance checking functionality for SecureML.

This module provides tools to verify that datasets and models
comply with privacy regulations like GDPR, CCPA, HIPAA, or LGPD.
"""

from typing import Any, Dict, List, Optional, Union
import re

import pandas as pd
import spacy
from spacy.language import Language

# Import presets functionality
from .presets import load_preset, get_preset_field, list_available_presets

# Load spaCy model - we'll use the en_core_web_sm model for NER
_NLP_MODEL: Optional[Language] = None


def get_nlp_model(model_name: str = "en_core_web_sm") -> Language:
    """
    Load and return a SpaCy NLP model.
    
    This function caches the model to avoid reloading it multiple times.
    If the model is not installed, it will attempt to download and install it.
    
    Args:
        model_name: Name of the SpaCy model to load
    
    Returns:
        Loaded SpaCy language model
        
    Raises:
        ImportError: If the specified model cannot be installed or loaded
    """
    global _NLP_MODEL
    
    if _NLP_MODEL is None:
        try:
            _NLP_MODEL = spacy.load(model_name)
        except OSError:
            # Model not found, attempt to install it
            print(f"SpaCy model '{model_name}' not found. Attempting to download and install...")
            try:
                import subprocess
                import sys
                
                # Run the spacy download command
                subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])
                
                # Try loading the model again
                _NLP_MODEL = spacy.load(model_name)
                print(f"Successfully installed and loaded model '{model_name}'.")
            except Exception as e:
                raise ImportError(
                    f"Failed to automatically install SpaCy model '{model_name}'. "
                    f"Error: {str(e)}. "
                    f"Please install it manually with: python -m spacy download {model_name}"
                )
    
    return _NLP_MODEL


class ComplianceReport:
    """
    A report containing the results of a compliance check.
    """

    def __init__(self, regulation: str):
        """
        Initialize a compliance report.

        Args:
            regulation: The regulation the check was performed against
        """
        self.regulation = regulation
        self.issues: List[Dict[str, Any]] = []
        self.warnings: List[Dict[str, Any]] = []
        self.passed_checks: List[str] = []

    def add_issue(
        self, component: str, issue: str, severity: str, recommendation: str
    ) -> None:
        """
        Add an issue to the report.

        Args:
            component: The component where the issue was found
            issue: Description of the issue
            severity: Severity level ('high', 'medium', 'low')
            recommendation: Recommended action to resolve the issue
        """
        self.issues.append(
            {
                "component": component,
                "issue": issue,
                "severity": severity,
                "recommendation": recommendation,
            }
        )

    def add_warning(
        self, component: str, warning: str, recommendation: str
    ) -> None:
        """
        Add a warning to the report.

        Args:
            component: The component where the warning was triggered
            warning: Description of the warning
            recommendation: Recommended action to address the warning
        """
        self.warnings.append(
            {
                "component": component,
                "warning": warning,
                "recommendation": recommendation,
            }
        )

    def add_passed_check(self, check_name: str) -> None:
        """
        Add a passed check to the report.

        Args:
            check_name: Name of the check that passed
        """
        self.passed_checks.append(check_name)

    def has_issues(self) -> bool:
        """
        Check if the report contains any issues.

        Returns:
            True if the report contains issues, False otherwise
        """
        return len(self.issues) > 0

    def has_warnings(self) -> bool:
        """
        Check if the report contains any warnings.

        Returns:
            True if the report contains warnings, False otherwise
        """
        return len(self.warnings) > 0

    def summary(self) -> Dict[str, Any]:
        """
        Get a summary of the compliance report.

        Returns:
            A dictionary containing the summary information
        """
        return {
            "regulation": self.regulation,
            "issues_count": len(self.issues),
            "warnings_count": len(self.warnings),
            "passed_checks_count": len(self.passed_checks),
            "compliant": not self.has_issues(),
        }

    def __str__(self) -> str:
        """
        Get a string representation of the compliance report.

        Returns:
            A formatted string representation of the report
        """
        summary = self.summary()
        result = [
            f"Compliance Report for {summary['regulation']}",
            f"Status: {'Compliant' if summary['compliant'] else 'Non-compliant'}",
            f"Passed Checks: {summary['passed_checks_count']}",
            f"Issues: {summary['issues_count']}",
            f"Warnings: {summary['warnings_count']}",
        ]

        if self.issues:
            result.append("\nIssues:")
            for i, issue in enumerate(self.issues, 1):
                result.append(
                    f"{i}. [{issue['severity'].upper()}] {issue['component']}: "
                    f"{issue['issue']}"
                )
                result.append(f"   Recommendation: {issue['recommendation']}")

        if self.warnings:
            result.append("\nWarnings:")
            for i, warning in enumerate(self.warnings, 1):
                result.append(
                    f"{i}. {warning['component']}: {warning['warning']}"
                )
                result.append(
                    f"   Recommendation: {warning['recommendation']}"
                )

        if self.passed_checks:
            result.append("\nPassed Checks:")
            for check in self.passed_checks:
                result.append(f"- {check}")

        return "\n".join(result)


def check_compliance(
    data: Union[pd.DataFrame, Dict[str, Any]],
    model_config: Optional[Dict[str, Any]] = None,
    regulation: str = "GDPR",
    max_samples: int = 100,
    **kwargs: Any,
) -> ComplianceReport:
    """
    Check a dataset and model configuration for compliance with privacy regulations.

    Uses NLP techniques to analyze dataset content for sensitive information.

    Args:
        data: The dataset to check (DataFrame or dict with metadata)
        model_config: Optional configuration of the model to check
        regulation: The regulation to check compliance against ('GDPR', 'CCPA', 
                   'HIPAA', 'LGPD')
        max_samples: Maximum number of samples to analyze for content
        **kwargs: Additional parameters for specific compliance checks

    Returns:
        A ComplianceReport with the results of the compliance check

    Raises:
        ValueError: If an unsupported regulation is specified
    """
    # Initialize the compliance report
    report = ComplianceReport(regulation)

    # Convert dict to DataFrame if necessary
    if isinstance(data, dict) and "data" in data and isinstance(
        data["data"], pd.DataFrame
    ):
        metadata = data.copy()
        data_df = metadata.pop("data")
    elif isinstance(data, pd.DataFrame):
        data_df = data
        metadata = {}
    else:
        data_df = pd.DataFrame()
        metadata = data if isinstance(data, dict) else {}

    # Check if regulation is supported
    supported_regulations = list_available_presets()
    
    if regulation.lower() not in [reg.lower() for reg in supported_regulations]:
        raise ValueError(
            f"Unsupported regulation: {regulation}. "
            f"Supported regulations are: {', '.join(supported_regulations)}."
        )

    # Apply the appropriate regulation checks
    if regulation.upper() == "GDPR":
        _check_gdpr_compliance(
            data_df, metadata, model_config, report, max_samples, **kwargs
        )
    elif regulation.upper() == "CCPA":
        _check_ccpa_compliance(
            data_df, metadata, model_config, report, max_samples, **kwargs
        )
    elif regulation.upper() == "HIPAA":
        _check_hipaa_compliance(
            data_df, metadata, model_config, report, max_samples, **kwargs
        )
    elif regulation.upper() == "LGPD":
        _check_lgpd_compliance(
            data_df, metadata, model_config, report, max_samples, **kwargs
        )
    else:
        # This should never happen because of the check above, but just in case
        raise ValueError(
            f"Unsupported regulation: {regulation}. "
            f"Supported regulations are: {', '.join(supported_regulations)}."
        )

    return report


def _check_gdpr_compliance(
    data: pd.DataFrame,
    metadata: Dict[str, Any],
    model_config: Optional[Dict[str, Any]],
    report: ComplianceReport,
    max_samples: int = 100,
    **kwargs: Any,
) -> None:
    """
    Check compliance with GDPR.

    Args:
        data: The dataset
        metadata: Metadata about the dataset
        model_config: Model configuration
        report: The compliance report to update
        max_samples: Maximum number of samples to analyze for content
        **kwargs: Additional parameters
    """
    # Load GDPR preset
    gdpr_preset = load_preset("gdpr")
    
    # Check for personal data in the dataset using both column names and content
    personal_data_identifiers = get_preset_field("gdpr", "personal_data_identifiers")
    special_categories = get_preset_field("gdpr", "special_categories")
    
    # Customize the personal data identification with the preset
    personal_data_info = identify_personal_data(
        data, 
        max_samples,
        personal_identifiers=personal_data_identifiers,
        sensitive_categories=special_categories
    )
    
    if personal_data_info["columns"]:
        cols_str = ", ".join(personal_data_info["columns"])
        report.add_issue(
            component="Dataset",
            issue=f"Personal data identified in columns: {cols_str}",
            severity="high",
            recommendation=(
                "Anonymize or pseudonymize these columns before processing."
            ),
        )
    else:
        report.add_passed_check("No personal data identified in dataset columns")
    
    if personal_data_info["content_findings"]:
        findings = personal_data_info["content_findings"]
        report.add_issue(
            component="Dataset Content",
            issue=f"Personal data found in content: {findings}",
            severity="high",
            recommendation=(
                "Apply text redaction or anonymization to text fields."
            ),
        )
    else:
        report.add_passed_check("No personal data identified in text content")

    # Check for data minimization
    max_recommended_columns = get_preset_field("gdpr", "data_minimization.max_recommended_columns")
    if max_recommended_columns and len(data.columns) > max_recommended_columns:
        report.add_warning(
            component="Dataset",
            warning="Large number of columns may violate data minimization principle",
            recommendation=(
                "Review dataset to ensure only necessary data is collected"
            ),
        )
    else:
        report.add_passed_check(
            "Dataset appears to follow data minimization principle"
        )

    # Check for explicit consent metadata
    consent_metadata_fields = get_preset_field("gdpr", "consent.metadata_fields")
    if consent_metadata_fields and not any(field in metadata for field in consent_metadata_fields):
        report.add_warning(
            component="Metadata",
            warning="No explicit record of consent found",
            recommendation=(
                "Add metadata indicating when and how consent was obtained"
            ),
        )
    else:
        report.add_passed_check("Consent metadata is present")

    # Check model for right to be forgotten capability
    if model_config is not None:
        if not model_config.get("supports_forget_request", False):
            report.add_issue(
                component="Model",
                issue="Model does not support 'right to be forgotten' requests",
                severity="medium",
                recommendation=(
                    "Implement a mechanism to remove specific data points"
                ),
            )
        else:
            report.add_passed_check("Model supports 'right to be forgotten'")

    # Check for cross-border data transfer
    allowed_locations = get_preset_field("gdpr", "cross_border_transfer.allowed_locations")
    if allowed_locations and metadata.get("data_storage_location") not in [None] + allowed_locations:
        report.add_warning(
            component="Data Storage",
            warning=f"Data may be stored outside the allowed locations: {', '.join(allowed_locations)}",
            recommendation=(
                "Ensure adequate safeguards for cross-border data transfers"
            ),
        )
    else:
        report.add_passed_check("Data storage location complies with GDPR")


def _check_ccpa_compliance(
    data: pd.DataFrame,
    metadata: Dict[str, Any],
    model_config: Optional[Dict[str, Any]],
    report: ComplianceReport,
    max_samples: int = 100,
    **kwargs: Any,
) -> None:
    """
    Check compliance with CCPA.

    Args:
        data: The dataset
        metadata: Metadata about the dataset
        model_config: Model configuration
        report: The compliance report to update
        max_samples: Maximum number of samples to analyze for content
        **kwargs: Additional parameters
    """
    # Load CCPA preset fields
    contains_ca_residents_field = get_preset_field("ccpa", "data_handling.contains_ca_residents_field")
    data_sharing_field = get_preset_field("ccpa", "data_handling.data_sharing_disclosure_field")
    ccpa_disclosure_field = get_preset_field("ccpa", "data_handling.ccpa_disclosure_field")
    
    # Check for California residents' data
    if metadata.get(contains_ca_residents_field, False):
        # Get personal information categories from the preset
        personal_info_categories = []
        all_personal_info = get_preset_field("ccpa", "personal_information")
        
        if all_personal_info:
            for category in all_personal_info:
                for cat_name, identifiers in category.items():
                    personal_info_categories.extend(identifiers)
        
        # Check for personal information using both column names and content
        personal_info = identify_personal_data(
            data, 
            max_samples,
            personal_identifiers=personal_info_categories
        )
        
        if personal_info["columns"]:
            cols_str = ", ".join(personal_info["columns"])
            report.add_issue(
                component="Dataset",
                issue=f"Personal information found in columns: {cols_str}",
                severity="medium",
                recommendation="Ensure proper disclosure to California residents",
            )
        else:
            report.add_passed_check("No personal information identified in dataset columns")
            
        if personal_info["content_findings"]:
            findings = personal_info["content_findings"]
            report.add_issue(
                component="Dataset Content",
                issue=(
                    f"Personal information found in content: "
                    f"{findings}"
                ),
                severity="medium",
                recommendation=(
                    "Ensure California residents can opt out of "
                    "data collection"
                ),
            )
        else:
            report.add_passed_check("No personal information identified in text content")

        # Check for sale of personal information
        if metadata.get(data_sharing_field, False):
            if not metadata.get(ccpa_disclosure_field, False):
                report.add_issue(
                    component="Metadata",
                    issue="Data shared with third parties without CCPA disclosure",
                    severity="high",
                    recommendation="Provide 'Do Not Sell My Personal Information' option",
                )
            else:
                report.add_passed_check(
                    "CCPA disclosure for third-party sharing provided"
                )
    else:
        report.add_passed_check(
            "Data does not contain California residents' information"
        )

    # Check for deletion capability
    if model_config is not None:
        deletion_requirement = get_preset_field("ccpa", "model_requirements.supports_deletion_request")
        if deletion_requirement and not model_config.get("supports_deletion_request", False):
            report.add_warning(
                component="Model",
                warning="Model may not support consumer deletion requests",
                recommendation="Implement mechanism to honor deletion requests",
            )
        else:
            report.add_passed_check("Model supports deletion requests")


def _check_hipaa_compliance(
    data: pd.DataFrame,
    metadata: Dict[str, Any],
    model_config: Optional[Dict[str, Any]],
    report: ComplianceReport,
    max_samples: int = 100,
    **kwargs: Any,
) -> None:
    """
    Check compliance with HIPAA.

    Args:
        data: The dataset
        metadata: Metadata about the dataset
        model_config: Model configuration
        report: The compliance report to update
        max_samples: Maximum number of samples to analyze for content
        **kwargs: Additional parameters
    """
    # Load HIPAA preset fields
    phi_identifiers = get_preset_field("hipaa", "phi_identifiers")
    deident_method_field = get_preset_field("hipaa", "data_management.deidentification_method_field")
    data_encrypted_field = get_preset_field("hipaa", "data_management.data_encrypted_field")
    deident_methods = [method["name"] for method in get_preset_field("hipaa", "deidentification_methods")]
    
    # Check for protected health information (PHI) using both column names and content analysis
    phi_info = identify_phi(
        data, 
        max_samples,
        phi_identifiers=phi_identifiers
    )
    
    if phi_info["columns"]:
        cols_str = ", ".join(phi_info["columns"])
        report.add_issue(
            component="Dataset",
            issue=f"Protected Health Information found in columns: {cols_str}",
            severity="high",
            recommendation="Apply de-identification techniques to these columns",
        )
    else:
        report.add_passed_check("No PHI identified in dataset columns")
        
    if phi_info["content_findings"]:
        findings = phi_info["content_findings"]
        report.add_issue(
            component="Dataset Content",
            issue=f"PHI found in content: {findings}",
            severity="high",
            recommendation="Apply text redaction or de-identification to text fields",
        )
    else:
        report.add_passed_check("No PHI identified in text content")

    # Check for proper de-identification
    if metadata.get(deident_method_field) not in deident_methods:
        report.add_warning(
            component="Metadata",
            warning="No recognized de-identification method specified",
            recommendation=(
                f"Apply either {' or '.join(deident_methods)} method"
            ),
        )
    else:
        report.add_passed_check("Recognized de-identification method used")

    # Check for data security
    if not metadata.get(data_encrypted_field, False):
        report.add_issue(
            component="Data Security",
            issue="Data is not encrypted",
            severity="high",
            recommendation=(
                "Implement encryption for PHI at rest and in transit"
            ),
        )
    else:
        report.add_passed_check("Data is properly encrypted")


def _check_lgpd_compliance(
    data: pd.DataFrame,
    metadata: Dict[str, Any],
    model_config: Optional[Dict[str, Any]],
    report: ComplianceReport,
    max_samples: int = 100,
    **kwargs: Any,
) -> None:
    """
    Check compliance with LGPD (Brazilian General Data Protection Law).

    Args:
        data: The dataset
        metadata: Metadata about the dataset
        model_config: Model configuration
        report: The compliance report to update
        max_samples: Maximum number of samples to analyze for content
        **kwargs: Additional parameters
    """
    # Load LGPD preset fields
    personal_data_identifiers = get_preset_field("lgpd", "personal_data_identifiers")
    sensitive_personal_data = get_preset_field("lgpd", "sensitive_personal_data")
    
    # Check for personal data in the dataset using both column names and content
    personal_data_info = identify_personal_data(
        data, 
        max_samples,
        personal_identifiers=personal_data_identifiers,
        sensitive_categories=sensitive_personal_data
    )
    
    if personal_data_info["columns"]:
        cols_str = ", ".join(personal_data_info["columns"])
        report.add_issue(
            component="Dataset",
            issue=f"Personal data identified in columns: {cols_str}",
            severity="high",
            recommendation=(
                "Ensure proper legal basis for processing this data and "
                "consider pseudonymization or anonymization techniques."
            ),
        )
    else:
        report.add_passed_check("No personal data identified in dataset columns")
    
    if personal_data_info["content_findings"]:
        findings = personal_data_info["content_findings"]
        report.add_issue(
            component="Dataset Content",
            issue=f"Personal data found in content: {findings}",
            severity="high",
            recommendation=(
                "Apply text redaction or anonymization to text fields containing personal data."
            ),
        )
    else:
        report.add_passed_check("No personal data identified in text content")

    # Check for data processing principles
    data_processing_requirements = get_preset_field("lgpd", "data_processing_principles.requirements")
    if data_processing_requirements and len(data.columns) > 20:  # Similar threshold as GDPR
        report.add_warning(
            component="Dataset",
            warning="Large number of columns may violate data necessity principle",
            recommendation=(
                "Review dataset to ensure only necessary data is collected "
                "for the specified purpose"
            ),
        )
    else:
        report.add_passed_check(
            "Dataset appears to follow data necessity principle"
        )

    # Check for legal basis metadata
    legal_basis_fields = get_preset_field("lgpd", "legal_bases.metadata_fields")
    if legal_basis_fields and not any(field in metadata for field in legal_basis_fields):
        report.add_issue(
            component="Metadata",
            warning="No legal basis for processing identified",
            severity="high",
            recommendation=(
                "Document the legal basis for processing personal data "
                "(e.g., consent, legitimate interest, contract execution)"
            ),
        )
    else:
        report.add_passed_check("Legal basis for processing is documented")

    # Check model for data subject rights support
    if model_config is not None:
        # Check for deletion support
        if not model_config.get("supports_deletion_request", False):
            report.add_issue(
                component="Model",
                issue="Model does not support deletion requests",
                severity="high",
                recommendation=(
                    "Implement a mechanism to delete or anonymize specific data "
                    "points when requested by data subjects"
                ),
            )
        else:
            report.add_passed_check("Model supports deletion requests")
        
        # Check for access and portability
        if not model_config.get("supports_access_request", False):
            report.add_warning(
                component="Model",
                warning="Model may not support data access requests",
                recommendation=(
                    "Implement a mechanism to provide data subjects with "
                    "access to their personal data"
                ),
            )
        else:
            report.add_passed_check("Model supports data access requests")
        
        # Check for automated decision making
        if model_config.get("uses_automated_decisions", False):
            if not model_config.get("allows_human_review", False):
                report.add_issue(
                    component="Model",
                    issue="Automated decision-making without human review option",
                    severity="high",
                    recommendation=(
                        "Implement a mechanism for human review of automated decisions "
                        "that affect data subjects"
                    ),
                )
            else:
                report.add_passed_check("Model allows human review of automated decisions")

    # Check for security measures
    data_security_field = get_preset_field("lgpd", "security.technical_administrative_measures")
    if data_security_field and not metadata.get("security_measures_implemented", False):
        report.add_warning(
            component="Data Security",
            warning="No documentation of security measures for data protection",
            recommendation=(
                "Implement and document technical and administrative security measures "
                "appropriate to the nature of the data being processed"
            ),
        )
    else:
        report.add_passed_check("Security measures are documented")
        
    # Check for encryption, which is recommended under LGPD
    if not metadata.get("data_encrypted", False):
        report.add_warning(
            component="Data Security",
            warning="Data may not be encrypted",
            recommendation=(
                "Consider implementing encryption for sensitive personal data"
            ),
        )
    else:
        report.add_passed_check("Data is properly encrypted")
        
    # Check for international transfers
    if metadata.get("international_transfer", False):
        international_transfer_mechanism = metadata.get("international_transfer_mechanism")
        allowed_mechanisms = get_preset_field("lgpd", "international_data_transfer.allowed_mechanisms")
        if allowed_mechanisms and international_transfer_mechanism not in allowed_mechanisms:
            report.add_issue(
                component="Data Transfer",
                issue="International data transfer without adequate safeguards",
                severity="high",
                recommendation=(
                    "Implement one of the approved mechanisms for international data transfers: "
                    f"{', '.join(allowed_mechanisms)}"
                ),
            )
        else:
            report.add_passed_check("International data transfers comply with LGPD")
            
    # Check for DPO appointment
    if get_preset_field("lgpd", "documentation.data_protection_officer_required"):
        if not metadata.get("data_protection_officer_appointed", False):
            report.add_warning(
                component="Organization",
                warning="No Data Protection Officer (DPO) appears to be appointed",
                recommendation=(
                    "Appoint a Data Protection Officer (Encarregado) responsible for "
                    "handling data subject requests and communications with the ANPD"
                ),
            )
        else:
            report.add_passed_check("Data Protection Officer has been appointed")


def identify_personal_data(
    data: pd.DataFrame, 
    max_samples: int = 100,
    personal_identifiers: Optional[List[str]] = None,
    sensitive_categories: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Identify personal data in a dataset.
    
    Args:
        data: The dataset to analyze
        max_samples: Maximum number of samples to analyze for content
        personal_identifiers: List of personal data identifiers to check for
        sensitive_categories: List of sensitive data categories to check for
        
    Returns:
        Dictionary with information about identified personal data
    """
    # Use default personal identifiers if none provided
    if personal_identifiers is None:
        personal_identifiers = [
            "name", "email", "phone", "address", "ip", "ssn", "passport",
            "birth", "age", "gender", "zip", "postal", "license"
        ]
    
    # Use default sensitive categories if none provided
    if sensitive_categories is None:
        sensitive_categories = [
            "race", "ethnicity", "religion", "health", "sexual", "political",
            "biometric", "genetic"
        ]
    
    result = {
        "columns": [],
        "content_findings": set(),
    }
    
    # 1. Check column names
    for col in data.columns:
        col_lower = col.lower()
        if any(pattern in col_lower for pattern in personal_identifiers):
            result["columns"].append(col)
    
    # 2. Analyze text content using SpaCy NER
    try:
        nlp = get_nlp_model()
        
        # Identify potential text columns for content analysis
        text_columns = [
            col for col in data.columns 
            if data[col].dtype == "object" and not col in result["columns"]
        ]
        
        if text_columns and len(data) > 0:
            # Sample the data to avoid processing the entire dataset
            sample_size = min(max_samples, len(data))
            samples = data.sample(n=sample_size) if sample_size < len(data) else data
            
            # Define entity types that are likely to be personal information
            personal_entity_types = {
                "PERSON", "ORG", "GPE", "LOC", "EMAIL", "PHONE", "URL",
                "CREDIT_CARD", "SSN", "ID", "PASSPORT", "DL", "DATE"
            }
            
            # Process text columns
            for col in text_columns:
                # Skip non-string values
                sample_texts = samples[col].dropna().astype(str)
                
                # Process each text sample
                for text in sample_texts:
                    if len(text) > 5:  # Skip very short texts
                        doc = nlp(text)
                        
                        # Check for named entities
                        for ent in doc.ents:
                            if ent.label_ in personal_entity_types:
                                result["content_findings"].add(
                                    f"{col}: contains {ent.label_} entity"
                                )
                                break  # No need to keep checking this sample
                
                # Check for patterns that SpaCy might miss
                patterns = {
                    "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                    "phone": r'\b(\+\d{1,3}[\s-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b',
                    "ssn": r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b',
                    "credit_card": r'\b(?:\d{4}[-\s]?){3}\d{4}\b|\b\d{16}\b'
                }
                
                for pattern_name, regex in patterns.items():
                    if any(re.search(regex, str(text)) for text in sample_texts):
                        result["content_findings"].add(
                            f"{col}: contains possible {pattern_name}"
                        )
    
    except (ImportError, Exception) as e:
        # If SpaCy analysis fails, fall back to just column name checking
        result["content_findings"].add(
            f"Warning: Text analysis couldn't complete: {str(e)}"
        )
    
    # Convert set to list for better serialization
    result["content_findings"] = list(result["content_findings"])
    
    return result


def identify_phi(
    data: pd.DataFrame, 
    max_samples: int = 100,
    phi_identifiers: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Identify Protected Health Information (PHI) in a dataset.
    
    Args:
        data: The dataset to analyze
        max_samples: Maximum number of samples to analyze for content
        phi_identifiers: List of PHI identifiers to check for
        
    Returns:
        Dictionary with information about identified PHI
    """
    # Use default PHI identifiers if none provided
    if phi_identifiers is None:
        phi_identifiers = [
            "name", "address", "date", "phone", "fax", "email", "ssn", "medical",
            "health", "beneficiary", "account", "certificate", "vehicle", "device",
            "url", "ip", "biometric", "photo", "identifier"
        ]
    
    result = {
        "columns": [],
        "content_findings": set(),
    }
    
    # 1. Check column names
    for col in data.columns:
        col_lower = col.lower()
        if any(pattern in col_lower for pattern in phi_identifiers):
            result["columns"].append(col)
    
    # 2. Analyze text content using SpaCy
    try:
        nlp = get_nlp_model()
        
        # Identify potential text columns for content analysis
        text_columns = [
            col for col in data.columns 
            if data[col].dtype == "object" and not col in result["columns"]
        ]
        
        if text_columns and len(data) > 0:
            # Sample the data to avoid processing the entire dataset
            sample_size = min(max_samples, len(data))
            samples = data.sample(n=sample_size) if sample_size < len(data) else data
            
            # Health-related keywords to look for
            health_keywords = [
                "diagnosis", "disease", "syndrome", "disorder", "condition",
                "treatment", "therapy", "medication", "prescription", "drug",
                "patient", "doctor", "physician", "hospital", "clinic", "medical",
                "health", "symptom", "pain", "surgery", "procedure", "test", 
                "blood", "exam", "x-ray", "mri", "ct scan", "cancer", "diabetes",
                "hypertension", "allergy", "immunization", "vaccine"
            ]
            
            # Process text columns
            for col in text_columns:
                # Skip non-string values
                sample_texts = samples[col].dropna().astype(str)
                
                # Process each text sample
                for text in sample_texts:
                    if len(text) > 5:  # Skip very short texts
                        # Check for health-related keywords
                        text_lower = text.lower()
                        for keyword in health_keywords:
                            if keyword in text_lower:
                                result["content_findings"].add(
                                    f"{col}: contains health-related term '{keyword}'"
                                )
                                break  # No need to keep checking this sample
                        
                        # Use NLP to check for medical entities
                        doc = nlp(text)
                        
                        # Basic entity recognition - note that the default SpaCy model
                        # doesn't have specialized medical entity recognition
                        for ent in doc.ents:
                            if ent.label_ in {"DISEASE", "CONDITION", "TREATMENT"}:
                                result["content_findings"].add(
                                    f"{col}: contains medical entity {ent.label_}"
                                )
                            # Also catch personal identifiers that could be PHI
                            elif ent.label_ in {"PERSON", "DATE", "GPE", "LOC"}:
                                result["content_findings"].add(
                                    f"{col}: contains potential PHI {ent.label_}"
                                )
                
                # Check for patterns that might be PHI
                patterns = {
                    "mrn": r'\b(MR|MRN)[-\s]?\d{5,10}\b',  # Medical Record Number
                    "date": r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b',  # Date
                    "npi": r'\b\d{10}\b',  # National Provider Identifier
                    "icd10": r'\b[A-Z]\d{2}(\.\d{1,2})?\b'  # ICD-10 code
                }
                
                for pattern_name, regex in patterns.items():
                    if any(re.search(regex, str(text)) for text in sample_texts):
                        result["content_findings"].add(
                            f"{col}: contains possible {pattern_name}"
                        )
    
    except (ImportError, Exception) as e:
        # If SpaCy analysis fails, fall back to just column name checking
        result["content_findings"].add(
            f"Warning: Text analysis couldn't complete: {str(e)}"
        )
    
    # Convert set to list for better serialization
    result["content_findings"] = list(result["content_findings"])
    
    return result


class ComplianceAuditor:
    """
    A class for auditing ML pipelines for compliance with privacy regulations.
    
    The ComplianceAuditor provides a higher-level interface for conducting
    compliance audits of ML pipelines, generating comprehensive audit trails,
    and producing detailed reports.
    """
    
    def __init__(
        self, 
        regulation: str = "GDPR",
        log_dir: Optional[str] = None
    ):
        """
        Initialize a compliance auditor.
        
        Args:
            regulation: The regulation to audit against
            log_dir: Directory to store audit logs
        """
        self.regulation = regulation
        self.log_dir = log_dir
        
        # Validate the regulation
        if regulation.lower() not in [reg.lower() for reg in list_available_presets()]:
            raise ValueError(
                f"Unsupported regulation: {regulation}. "
                f"Supported regulations are: {', '.join(list_available_presets())}."
            )
    
    def audit_dataset(
        self,
        dataset: Union[pd.DataFrame, Dict[str, Any]],
        dataset_name: str,
        metadata: Optional[Dict[str, Any]] = None,
        max_samples: int = 100
    ) -> ComplianceReport:
        """
        Audit a dataset for compliance.
        
        Args:
            dataset: The dataset to audit
            dataset_name: Name of the dataset
            metadata: Additional metadata about the dataset
            max_samples: Maximum number of samples to analyze for content
        
        Returns:
            A compliance report for the dataset
        """
        # Import the audit module
        from .audit import AuditTrail
        
        # Create an audit trail
        audit = AuditTrail(
            operation_name=f"dataset_audit_{dataset_name}",
            log_dir=self.log_dir,
            regulations=[self.regulation]
        )
        
        try:
            # Log the audit start
            audit.log_event(
                "audit_started",
                {
                    "audit_type": "dataset",
                    "dataset_name": dataset_name,
                    "regulation": self.regulation
                }
            )
            
            # Prepare data for check_compliance
            if isinstance(dataset, pd.DataFrame):
                data = dataset
                full_metadata = metadata or {}
            else:
                # Dataset is a dict with metadata
                data = dataset.get("data", pd.DataFrame())
                full_metadata = dataset.copy()
                if "data" in full_metadata:
                    del full_metadata["data"]
                
                # Update with additional metadata
                if metadata:
                    full_metadata.update(metadata)
            
            # Log dataset info
            audit.log_data_access(
                dataset_name=dataset_name,
                columns_accessed=list(data.columns),
                num_records=len(data),
                purpose=f"Compliance audit for {self.regulation}"
            )
            
            # Run compliance check
            report = check_compliance(
                data={"data": data, **full_metadata},
                regulation=self.regulation,
                max_samples=max_samples
            )
            
            # Log the results
            audit.log_compliance_check(
                check_type="dataset_compliance",
                regulation=self.regulation,
                result=not report.has_issues(),
                details={
                    "issues_count": len(report.issues),
                    "warnings_count": len(report.warnings),
                    "passed_checks_count": len(report.passed_checks)
                }
            )
            
            # Log issues and warnings
            for issue in report.issues:
                audit.log_event(
                    "compliance_issue",
                    {
                        "component": issue["component"],
                        "issue": issue["issue"],
                        "severity": issue["severity"],
                        "recommendation": issue["recommendation"]
                    }
                )
            
            for warning in report.warnings:
                audit.log_event(
                    "compliance_warning",
                    {
                        "component": warning["component"],
                        "warning": warning["warning"],
                        "recommendation": warning["recommendation"]
                    }
                )
            
            # Log passed checks
            for check in report.passed_checks:
                audit.log_event(
                    "compliance_passed",
                    {
                        "check": check
                    }
                )
            
            # Close the audit trail
            audit.close(
                status="completed",
                details={
                    "compliant": not report.has_issues()
                }
            )
            
            return report
            
        except Exception as e:
            # Log the error
            audit.log_error(
                error_type=type(e).__name__,
                message=str(e)
            )
            
            # Close the audit trail
            audit.close(
                status="error",
                details={
                    "error": str(e)
                }
            )
            
            # Re-raise the exception
            raise
    
    def audit_model(
        self,
        model_config: Dict[str, Any],
        model_name: str,
        model_documentation: Optional[Dict[str, Any]] = None
    ) -> ComplianceReport:
        """
        Audit a model for compliance.
        
        Args:
            model_config: Configuration of the model
            model_name: Name of the model
            model_documentation: Additional documentation about the model
        
        Returns:
            A compliance report for the model
        """
        # Import the audit module
        from .audit import AuditTrail
        
        # Create an audit trail
        audit = AuditTrail(
            operation_name=f"model_audit_{model_name}",
            log_dir=self.log_dir,
            regulations=[self.regulation]
        )
        
        try:
            # Log the audit start
            audit.log_event(
                "audit_started",
                {
                    "audit_type": "model",
                    "model_name": model_name,
                    "regulation": self.regulation
                }
            )
            
            # Combine model_config with documentation
            full_model_config = model_config.copy()
            if model_documentation:
                full_model_config["documentation"] = model_documentation
            
            # Create an empty dataframe for the check_compliance function
            # since we're only checking the model
            empty_df = pd.DataFrame()
            
            # Run compliance check
            report = check_compliance(
                data=empty_df,
                model_config=full_model_config,
                regulation=self.regulation
            )
            
            # Log the results
            audit.log_compliance_check(
                check_type="model_compliance",
                regulation=self.regulation,
                result=not report.has_issues(),
                details={
                    "issues_count": len(report.issues),
                    "warnings_count": len(report.warnings),
                    "passed_checks_count": len(report.passed_checks)
                }
            )
            
            # Log issues and warnings
            for issue in report.issues:
                audit.log_event(
                    "compliance_issue",
                    {
                        "component": issue["component"],
                        "issue": issue["issue"],
                        "severity": issue["severity"],
                        "recommendation": issue["recommendation"]
                    }
                )
            
            for warning in report.warnings:
                audit.log_event(
                    "compliance_warning",
                    {
                        "component": warning["component"],
                        "warning": warning["warning"],
                        "recommendation": warning["recommendation"]
                    }
                )
            
            # Log passed checks
            for check in report.passed_checks:
                audit.log_event(
                    "compliance_passed",
                    {
                        "check": check
                    }
                )
            
            # Close the audit trail
            audit.close(
                status="completed",
                details={
                    "compliant": not report.has_issues()
                }
            )
            
            return report
            
        except Exception as e:
            # Log the error
            audit.log_error(
                error_type=type(e).__name__,
                message=str(e)
            )
            
            # Close the audit trail
            audit.close(
                status="error",
                details={
                    "error": str(e)
                }
            )
            
            # Re-raise the exception
            raise
    
    def audit_pipeline(
        self,
        dataset: Optional[Union[pd.DataFrame, Dict[str, Any]]] = None,
        dataset_name: Optional[str] = None,
        model: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None,
        preprocessing_steps: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Audit an entire ML pipeline for compliance.
        
        Args:
            dataset: The dataset used in the pipeline
            dataset_name: Name of the dataset
            model: Model configuration or object
            model_name: Name of the model
            preprocessing_steps: List of preprocessing steps
            metadata: Additional metadata about the pipeline
        
        Returns:
            Dictionary containing compliance reports for each component
        """
        # Import the audit module
        from .audit import AuditTrail
        
        # Create an audit trail
        pipeline_name = f"{dataset_name or 'dataset'}_{model_name or 'model'}"
        audit = AuditTrail(
            operation_name=f"pipeline_audit_{pipeline_name}",
            log_dir=self.log_dir,
            regulations=[self.regulation]
        )
        
        try:
            # Log the audit start
            audit.log_event(
                "audit_started",
                {
                    "audit_type": "pipeline",
                    "pipeline_name": pipeline_name,
                    "regulation": self.regulation,
                    "components": {
                        "dataset": dataset_name is not None,
                        "model": model_name is not None,
                        "preprocessing": preprocessing_steps is not None
                    }
                }
            )
            
            results = {}
            
            # Audit dataset if provided
            if dataset is not None:
                dataset_audit_name = dataset_name or "dataset"
                audit.log_event(
                    "component_audit_started",
                    {
                        "component": "dataset",
                        "name": dataset_audit_name
                    }
                )
                
                dataset_report = self.audit_dataset(
                    dataset=dataset,
                    dataset_name=dataset_audit_name,
                    metadata=metadata
                )
                
                results["dataset"] = dataset_report
                
                audit.log_event(
                    "component_audit_completed",
                    {
                        "component": "dataset",
                        "name": dataset_audit_name,
                        "compliant": not dataset_report.has_issues()
                    }
                )
            
            # Audit preprocessing steps if provided
            if preprocessing_steps is not None:
                audit.log_event(
                    "component_audit_started",
                    {
                        "component": "preprocessing",
                        "steps_count": len(preprocessing_steps)
                    }
                )
                
                # Check preprocessing steps
                preprocessing_report = ComplianceReport(self.regulation)
                
                # Check for anonymization or privacy-preserving steps
                has_anonymization = False
                has_minimization = False
                
                for i, step in enumerate(preprocessing_steps):
                    step_name = step.get("name", f"step_{i}")
                    step_type = step.get("type", "unknown")
                    
                    # Log the preprocessing step
                    audit.log_data_transformation(
                        transformation_type=step_type,
                        input_data=step.get("input", "unknown"),
                        output_data=step.get("output", "unknown"),
                        parameters=step.get("parameters", {})
                    )
                    
                    # Check if this step performs anonymization
                    if any(keyword in step_type.lower() for keyword in [
                        "anonym", "pseudonym", "mask", "encrypt", "hash"
                    ]):
                        has_anonymization = True
                        preprocessing_report.add_passed_check(
                            f"Anonymization step found: {step_name}"
                        )
                    
                    # Check if this step performs data minimization
                    if any(keyword in step_type.lower() for keyword in [
                        "minimiz", "reduce", "select", "filter", "drop"
                    ]):
                        has_minimization = True
                        preprocessing_report.add_passed_check(
                            f"Data minimization step found: {step_name}"
                        )
                
                # Check for missing privacy steps based on regulation
                if self.regulation.upper() in ["GDPR", "CCPA", "HIPAA", "LGPD"]:
                    if not has_anonymization:
                        preprocessing_report.add_warning(
                            component="Preprocessing",
                            warning="No anonymization steps found in preprocessing pipeline",
                            recommendation="Consider adding anonymization techniques like k-anonymity or data masking"
                        )
                    
                    if not has_minimization and self.regulation.upper() in ["GDPR", "LGPD"]:
                        preprocessing_report.add_warning(
                            component="Preprocessing",
                            warning="No data minimization steps found in preprocessing pipeline",
                            recommendation=f"Consider adding data minimization steps to comply with {self.regulation}'s data minimization principle"
                        )
                
                results["preprocessing"] = preprocessing_report
                
                audit.log_event(
                    "component_audit_completed",
                    {
                        "component": "preprocessing",
                        "compliant": not preprocessing_report.has_issues()
                    }
                )
            
            # Audit model if provided
            if model is not None:
                model_audit_name = model_name or "model"
                audit.log_event(
                    "component_audit_started",
                    {
                        "component": "model",
                        "name": model_audit_name
                    }
                )
                
                # Extract model config if it's not already a dict
                if not isinstance(model, dict):
                    try:
                        # Try to get model configuration
                        model_config = {
                            "model_type": type(model).__name__,
                            "parameters": getattr(model, "get_params", lambda: {})()
                        }
                    except (AttributeError, TypeError):
                        model_config = {"model_type": type(model).__name__}
                else:
                    model_config = model
                
                model_report = self.audit_model(
                    model_config=model_config,
                    model_name=model_audit_name
                )
                
                results["model"] = model_report
                
                audit.log_event(
                    "component_audit_completed",
                    {
                        "component": "model",
                        "name": model_audit_name,
                        "compliant": not model_report.has_issues()
                    }
                )
            
            # Create overall compliance summary
            overall_compliance = True
            for component, report in results.items():
                if report.has_issues():
                    overall_compliance = False
                    break
            
            # Close the audit trail
            audit.close(
                status="completed",
                details={
                    "compliant": overall_compliance,
                    "components_audited": list(results.keys())
                }
            )
            
            return results
            
        except Exception as e:
            # Log the error
            audit.log_error(
                error_type=type(e).__name__,
                message=str(e)
            )
            
            # Close the audit trail
            audit.close(
                status="error",
                details={
                    "error": str(e)
                }
            )
            
            # Re-raise the exception
            raise
    
    def generate_pdf(
        self, 
        audit_result: Dict[str, Any],
        output_file: str,
        title: Optional[str] = None,
        logo_path: Optional[str] = None
    ) -> str:
        """
        Generate a PDF report from an audit result.
        
        Args:
            audit_result: The result of an audit_pipeline call
            output_file: Path to write the PDF report to
            title: Title for the report
            logo_path: Path to a logo image
        
        Returns:
            Path to the generated PDF
        """
        from .reporting import ReportGenerator
        
        # Create a report generator
        generator = ReportGenerator()
        
        # Generate a comprehensive report
        # First, create a combined compliance report
        combined_report = ComplianceReport(self.regulation)
        
        # Add all issues, warnings, and passed checks from each component
        for component_name, component_report in audit_result.items():
            # Add issues
            for issue in component_report.issues:
                # Modify the component to include which part of the pipeline it's from
                issue_copy = issue.copy()
                issue_copy["component"] = f"{component_name.capitalize()}: {issue_copy['component']}"
                combined_report.issues.append(issue_copy)
            
            # Add warnings
            for warning in component_report.warnings:
                warning_copy = warning.copy()
                warning_copy["component"] = f"{component_name.capitalize()}: {warning_copy['component']}"
                combined_report.warnings.append(warning_copy)
            
            # Add passed checks
            for check in component_report.passed_checks:
                combined_report.add_passed_check(f"{component_name.capitalize()}: {check}")
        
        # Generate the PDF
        return generator.generate_compliance_report(
            report=combined_report,
            output_file=output_file,
            logo_path=logo_path,
            include_charts=True,
            additional_context={
                "title": title or f"ML Pipeline Compliance Audit - {self.regulation}"
            }
        ) 