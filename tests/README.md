# SecureML Tests

This directory contains comprehensive test suites for the SecureML library. The tests are designed to ensure that all components of the library work as expected and maintain compatibility as the project evolves.

## Test Structure

The tests are organized by module, with each module having its own test file:

- `test_anonymization.py`: Tests for the anonymization module (k-anonymity, l-diversity, etc.)
- `test_privacy.py`: Tests for the privacy module (differential privacy, etc.)
- `test_compliance.py`: Tests for the compliance module (GDPR, CCPA, HIPAA checks)
- `test_audit.py`: Tests for the audit logging and tracking module
- `test_synthetic.py`: Tests for the synthetic data generation module
- `test_federated.py`: Tests for the federated learning module
- `test_reporting.py`: Tests for the report generation module
- `test_cli.py`: Tests for the command-line interface
- `test_property_based.py`: Property-based tests for privacy guarantees across varied inputs

## Testing Approach

The test suite employs the following strategies:

1. **Unit Testing**: Each function and class is tested in isolation to ensure it works correctly.
2. **Mock Objects**: External dependencies and complex components are mocked to focus on the specific unit under test.
3. **Parametrization**: Tests use parametrization to cover multiple scenarios and edge cases.
4. **Fixtures**: Common test setups are created as fixtures to promote code reuse.
5. **Property-Based Testing**: Uses Hypothesis to verify that key properties of privacy-preserving methods hold true across a wide range of automatically generated inputs, helping to identify edge cases that manual testing might miss.

## Test Framework

The tests are implemented using:

- **pytest**: The primary test framework
- **unittest**: Used for certain test cases that benefit from the unittest.TestCase class
- **unittest.mock**: For mocking external dependencies
- **tempfile**: For creating temporary test files and directories
- **hypothesis**: For property-based testing, particularly for privacy-preserving methods

## Running the Tests

To run the entire test suite:

```bash
pytest
```

To run tests for a specific module:

```bash
pytest test_anonymization.py
```

To run a specific test:

```bash
pytest test_anonymization.py::TestAnonymization::test_k_anonymize
```

To run only the property-based tests:

```bash
pytest test_property_based.py
```

To run tests with coverage information:

```bash
pytest --cov=secureml
```

## Test Configurations

The test suite uses the following configuration:

- Test fixtures are defined in each test file
- Mocks are used to simulate the behavior of components not directly under test
- Temporary files and directories are used for testing I/O operations
- Random seeds are set for reproducibility when using random data
- Hypothesis settings are configured to balance thoroughness with execution time

## Contributing New Tests

When adding new functionality to SecureML, please also add corresponding tests. Follow these guidelines:

1. Create tests for both the happy path and error cases
2. Use parametrized tests to cover multiple scenarios
3. Use mocking for external dependencies
4. Follow the existing naming patterns
5. Add appropriate docstrings to test functions
6. Ensure tests are isolated and don't depend on external state
7. Consider adding property-based tests for privacy-preserving features

## Property-Based Testing with Hypothesis

The property-based tests verify that our privacy-preserving methods maintain their guarantees when faced with varied inputs. Unlike traditional unit tests that check specific examples, property-based tests define properties that should always hold true regardless of the input data.

For example, our property-based tests verify:

- K-anonymity methods ensure that each combination of quasi-identifiers appears at least k times
- Pseudonymization consistently maps the same input to the same output
- Data masking preserves the format of the original data (e.g., emails still contain @)
- Differential privacy maintains utility while providing privacy guarantees
- Synthetic data generation produces statistically similar datasets
- Federated learning preserves privacy by ensuring model updates don't leak sensitive information
- Federated learning maintains utility despite privacy measures (privacy-utility tradeoff)

These tests help catch edge cases and subtle bugs that might go unnoticed with manual testing approaches. For federated learning in particular, we test both:

1. **Privacy Preservation**: Verifying that private data cannot be reconstructed from model updates, especially when differential privacy is applied
2. **Utility Preservation**: Ensuring that despite privacy measures, the federated model still provides useful predictions comparable to centralized training

## Continuous Integration

These tests are automatically run as part of the CI/CD pipeline for SecureML. All tests must pass before changes are merged into the main codebase.

## Test Documentation

Each test file includes detailed docstrings explaining:

- The purpose of each test
- What functionality is being tested
- Expected behavior and outcomes
- Any special setup or configuration required

This documentation helps other developers understand what's being tested and why, making it easier to maintain and extend the test suite as the project evolves.