"""
FastAPI server example demonstrating the SecureML library.

This API provides endpoints for:
1. Anonymizing data
2. Generating synthetic data
3. Checking compliance with privacy regulations
"""

from typing import Dict, List, Optional, Union

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from secureml import anonymize, check_compliance, generate_synthetic_data


# Define API models
class Column(BaseModel):
    name: str
    data: List[Union[str, int, float, bool, None]]


class DatasetRequest(BaseModel):
    columns: List[Column]
    sensitive_columns: Optional[List[str]] = None


class AnonymizationRequest(DatasetRequest):
    method: str = "k-anonymity"
    k: int = Field(default=5, ge=2)


class SyntheticDataRequest(DatasetRequest):
    method: str = "simple"
    num_samples: int = Field(default=100, ge=1, le=10000)
    seed: Optional[int] = None


class ComplianceRequest(DatasetRequest):
    regulation: str = "GDPR"
    model_config: Optional[Dict] = None
    metadata: Optional[Dict] = None


class DatasetResponse(BaseModel):
    columns: List[str]
    data: List[Dict[str, Union[str, int, float, bool, None]]]
    message: str


class ComplianceResponse(BaseModel):
    regulation: str
    compliant: bool
    issues_count: int
    warnings_count: int
    passed_checks_count: int
    issues: Optional[List[Dict]] = None
    warnings: Optional[List[Dict]] = None
    passed_checks: Optional[List[str]] = None
    message: str


# Initialize FastAPI app
app = FastAPI(
    title="SecureML API",
    description="API for privacy-preserving machine learning operations",
    version="0.1.0",
)


def request_to_dataframe(request: DatasetRequest) -> pd.DataFrame:
    """Convert an API request to a pandas DataFrame."""
    data = {}
    for column in request.columns:
        data[column.name] = column.data
    return pd.DataFrame(data)


def dataframe_to_response(df: pd.DataFrame, message: str) -> DatasetResponse:
    """Convert a pandas DataFrame to an API response."""
    return DatasetResponse(
        columns=df.columns.tolist(),
        data=df.to_dict(orient="records"),
        message=message,
    )


@app.post("/anonymize", response_model=DatasetResponse)
async def api_anonymize(request: AnonymizationRequest) -> DatasetResponse:
    """
    Anonymize a dataset using the specified method.
    
    Example:
    ```
    {
        "columns": [
            {"name": "name", "data": ["John Doe", "Jane Smith"]},
            {"name": "email", "data": ["john@example.com", "jane@example.com"]},
            {"name": "age", "data": [30, 25]},
            {"name": "income", "data": [50000, 60000]}
        ],
        "sensitive_columns": ["name", "email", "income"],
        "method": "k-anonymity",
        "k": 2
    }
    ```
    """
    try:
        # Convert request to DataFrame
        df = request_to_dataframe(request)
        
        # Apply anonymization
        result = anonymize(
            df,
            method=request.method,
            k=request.k,
            sensitive_columns=request.sensitive_columns,
        )
        
        return dataframe_to_response(
            result, f"Data anonymized using {request.method} method."
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/synthetic", response_model=DatasetResponse)
async def api_synthetic(request: SyntheticDataRequest) -> DatasetResponse:
    """
    Generate synthetic data based on the provided template dataset.
    
    Example:
    ```
    {
        "columns": [
            {"name": "age", "data": [30, 25, 40, 35, 50]},
            {"name": "income", "data": [50000, 60000, 70000, 55000, 65000]}
        ],
        "method": "statistical",
        "num_samples": 10,
        "seed": 42
    }
    ```
    """
    try:
        # Convert request to DataFrame
        df = request_to_dataframe(request)
        
        # Generate synthetic data
        result = generate_synthetic_data(
            df,
            method=request.method,
            num_samples=request.num_samples,
            sensitive_columns=request.sensitive_columns,
            seed=request.seed,
        )
        
        return dataframe_to_response(
            result, 
            f"Generated {request.num_samples} synthetic samples using {request.method} method."
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/compliance", response_model=ComplianceResponse)
async def api_compliance(request: ComplianceRequest) -> ComplianceResponse:
    """
    Check a dataset for compliance with privacy regulations.
    
    Example:
    ```
    {
        "columns": [
            {"name": "name", "data": ["John Doe", "Jane Smith"]},
            {"name": "email", "data": ["john@example.com", "jane@example.com"]},
            {"name": "age", "data": [30, 25]}
        ],
        "regulation": "GDPR",
        "model_config": {
            "supports_forget_request": true
        },
        "metadata": {
            "consent_obtained": true,
            "data_storage_location": "EU"
        }
    }
    ```
    """
    try:
        # Convert request to DataFrame
        df = request_to_dataframe(request)
        
        # Prepare data with metadata if provided
        if request.metadata:
            data_with_metadata = {
                "data": df,
                **request.metadata
            }
        else:
            data_with_metadata = df
        
        # Check compliance
        report = check_compliance(
            data_with_metadata,
            model_config=request.model_config,
            regulation=request.regulation,
        )
        
        # Convert report to response
        summary = report.summary()
        return ComplianceResponse(
            regulation=report.regulation,
            compliant=summary["compliant"],
            issues_count=summary["issues_count"],
            warnings_count=summary["warnings_count"],
            passed_checks_count=summary["passed_checks_count"],
            issues=report.issues if report.issues else None,
            warnings=report.warnings if report.warnings else None,
            passed_checks=report.passed_checks if report.passed_checks else None,
            message=f"Compliance check completed for {request.regulation}.",
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/")
async def root():
    """Return a welcome message and version information."""
    return {
        "message": "Welcome to the SecureML API",
        "version": "0.1.0",
        "endpoints": [
            "/anonymize - Anonymize a dataset",
            "/synthetic - Generate synthetic data",
            "/compliance - Check compliance with privacy regulations",
        ],
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(app, host="0.0.0.0", port=8000) 