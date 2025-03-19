# SecureML Tests

This directory contains unit tests for the SecureML library. These tests ensure that the library's functionality works as expected and help prevent regressions when making changes.

## Running Tests

You can run the tests using the `pytest` command from the root directory of the project:

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run a specific test file
pytest tests/test_anonymization.py

# Run a specific test class
pytest tests/test_anonymization.py::TestAnonymization

# Run a specific test
pytest tests/test_anonymization.py::TestAnonymization::test_anonymize_basic
```

## Test Coverage

To run tests with coverage reporting:

```bash
# Run tests with coverage
pytest --cov=secureml

# Generate HTML coverage report
pytest --cov=secureml --cov-report=html
```

This will create a directory called `htmlcov` with an HTML report of the test coverage. You can open `htmlcov/index.html` in a web browser to view the report.

## Test Files

- `test_anonymization.py`: Tests for the anonymization module
- (Future tests to be added for privacy, compliance, and synthetic data modules)

## Writing Tests

When adding new features to SecureML, please also add corresponding tests. Each test should:

1. Test a specific functionality
2. Have a descriptive name
3. Include assertions that verify the expected behavior
4. Be independent from other tests (can run in isolation)

Here's an example of a well-structured test:

```python
def test_feature_x():
    """Test that feature X behaves as expected."""
    # Set up test data
    input_data = ...
    
    # Call the function being tested
    result = feature_x(input_data)
    
    # Assert the expected outcome
    assert result.property_a == expected_value
    assert result.property_b is True
``` 