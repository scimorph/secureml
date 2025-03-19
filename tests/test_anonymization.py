"""
Tests for the anonymization module of SecureML.
"""

import unittest
import pandas as pd
import numpy as np

from secureml.anonymization import (
    anonymize,
    _identify_sensitive_columns,
    _apply_k_anonymity,
    _apply_pseudonymization,
    _apply_data_masking,
    _apply_generalization,
)


class TestAnonymization(unittest.TestCase):
    """Test cases for the anonymization module."""

    def setUp(self):
        """Set up test data."""
        # Create a sample DataFrame with sensitive information
        self.test_data = pd.DataFrame({
            "name": ["John Doe", "Jane Smith", "Bob Johnson", "Alice Brown", "Charlie Davis"],
            "email": ["john@example.com", "jane@example.com", "bob@example.com", 
                     "alice@example.com", "charlie@example.com"],
            "age": [35, 28, 42, 31, 25],
            "zip_code": ["12345", "23456", "34567", "45678", "56789"],
            "income": [85000, 72000, 95000, 63000, 78000],
            "purchase": [120.50, 85.75, 210.25, 55.30, 150.00],
        })

    def test_identify_sensitive_columns(self):
        """Test the identification of sensitive columns."""
        sensitive_cols = _identify_sensitive_columns(self.test_data)
        
        # These columns should be identified as sensitive
        expected_sensitive = ["name", "email", "zip_code", "income"]
        
        # Check that each expected column is identified
        for col in expected_sensitive:
            self.assertIn(col, sensitive_cols, f"Column '{col}' should be identified as sensitive")
        
        # Check that non-sensitive columns are not included
        self.assertNotIn("age", sensitive_cols, "Column 'age' should not be identified as sensitive")
        self.assertNotIn("purchase", sensitive_cols, "Column 'purchase' should not be identified as sensitive")

    def test_anonymize_basic(self):
        """Test basic anonymization functionality."""
        # Test with default parameters
        result = anonymize(self.test_data)
        
        # Check that the result is a DataFrame with the same shape
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape, self.test_data.shape)
        
        # Check that sensitive columns have been modified
        self.assertFalse(
            result["name"].equals(self.test_data["name"]), 
            "Names should be anonymized"
        )
        self.assertFalse(
            result["email"].equals(self.test_data["email"]), 
            "Emails should be anonymized"
        )

    def test_anonymize_k_anonymity(self):
        """Test k-anonymity anonymization."""
        # Test with k=2 (should group rare values)
        result = anonymize(
            self.test_data, 
            method="k-anonymity", 
            k=2,
            sensitive_columns=["name", "email", "income"]
        )
        
        # Check that the income column has been modified
        # In our simple implementation, values that appear less than k times
        # should be replaced with "[RARE_VALUE]"
        income_value_counts = result["income"].value_counts()
        for value, count in income_value_counts.items():
            self.assertGreaterEqual(
                count, 2,
                f"Income value {value} appears {count} times, should be at least 2"
            )

    def test_anonymize_pseudonymization(self):
        """Test pseudonymization."""
        result = anonymize(
            self.test_data, 
            method="pseudonymization",
            sensitive_columns=["name", "email"]
        )
        
        # Names should be replaced with consistent pseudonyms
        self.assertNotEqual(result["name"].iloc[0], self.test_data["name"].iloc[0])
        
        # Check that the same original values map to the same pseudonyms
        test_df = pd.DataFrame({
            "name": ["John Doe", "Jane Smith", "John Doe"],
            "value": [1, 2, 3]
        })
        
        result_df = anonymize(test_df, method="pseudonymization", sensitive_columns=["name"])
        
        # The first and third names (both "John Doe") should be the same pseudonym
        self.assertEqual(
            result_df["name"].iloc[0], 
            result_df["name"].iloc[2],
            "Same original values should map to same pseudonyms"
        )

    def test_anonymize_data_masking(self):
        """Test data masking."""
        result = anonymize(
            self.test_data, 
            method="data-masking",
            sensitive_columns=["email", "zip_code"]
        )
        
        # Check that emails are masked but still have the same pattern
        # In our implementation, emails like "user@example.com" would become "us**@ex*****.com"
        for original, masked in zip(self.test_data["email"], result["email"]):
            self.assertNotEqual(original, masked)
            self.assertIn("*", masked, "Masked email should contain asterisks")

    def test_anonymize_generalization(self):
        """Test generalization."""
        result = anonymize(
            self.test_data, 
            method="generalization",
            sensitive_columns=["income", "zip_code"]
        )
        
        # For numeric columns, we expect rounding to nearest 10
        # So all incomes should be divisible by 10
        for income in result["income"]:
            self.assertEqual(income % 10, 0, "Generalized income should be rounded to nearest 10")
        
        # For string columns like zip_code, we expect the first character + "..."
        for zip_code in result["zip_code"]:
            self.assertTrue(
                zip_code.endswith("...") or len(zip_code) <= 1,
                f"Generalized zip code '{zip_code}' should end with '...'"
            )

    def test_anonymize_list_input(self):
        """Test anonymization with list input."""
        # Convert DataFrame to list of dicts
        data_list = self.test_data.to_dict("records")
        
        # Anonymize the list
        result = anonymize(data_list, method="k-anonymity", k=2)
        
        # Check that the result is a list of dictionaries
        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], dict)
        self.assertEqual(len(result), len(data_list))


if __name__ == "__main__":
    unittest.main() 