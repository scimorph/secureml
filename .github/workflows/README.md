# SecureML GitHub Actions Workflows

This directory contains GitHub Actions workflows for automating the testing, building, and deployment of the SecureML project.

## CI/CD Workflow (`ci.yml`)

This workflow runs the full test suite, builds the package, and handles deployments when needed.

### Triggers

The workflow is triggered on:

- **Pull Requests**: When a PR is opened or updated targeting the `master` branch
- **Push to Main**: When code is pushed directly to the `master` branch
- **Releases**: When a GitHub release is created or published
- **Scheduled**: Runs nightly at midnight UTC (00:00)

### Jobs

The workflow consists of three main jobs:

#### 1. Test

- Runs on latest Ubuntu with Python 3.11
- Sets up Poetry and dependencies
- Performs code quality checks with isort and black
- Runs static type checking with mypy
- Executes the full test suite with pytest

#### 2. Build

- Depends on the successful completion of the test job
- Builds the Python package using Poetry
- Uploads the built artifacts
- Tests documentation building with Sphinx

#### 3. Deploy

- Only runs on GitHub releases
- Depends on the successful completion of both test and build jobs
- Publishes the package to PyPI using the configured API token

### Secrets

The following secrets need to be configured in your GitHub repository:

- `PYPI_API_TOKEN`: API token for publishing to PyPI (only needed for releases)

## Setup Instructions

1. Make sure your GitHub repository has the necessary secrets configured
2. Push this workflow file to your repository, and GitHub Actions will automatically pick it up
3. The first workflow run might take longer due to caching not being available yet 