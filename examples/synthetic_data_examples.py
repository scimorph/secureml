import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Import SecureML's synthetic data generation functionality
from secureml.synthetic import generate_synthetic_data

# Define a function to print example headers
def print_header(title):
    """Print a section header with formatting."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")

# Example 1: Basic Synthetic Data Generation
def example_simple_synthetic():
    """
    Simple synthetic data generation for quick prototyping.
    This example demonstrates the most basic method to generate synthetic data.
    """
    print_header("Example 1: Simple Synthetic Data Generation")
    
    # Create a synthetic dataset
    print("Creating a sample dataset with sensitive information...")
    
    # Sample data with sensitive information
    data = pd.DataFrame({
        'name': ['John Smith', 'Jane Doe', 'Robert Johnson', 'Emily Williams', 
                'Michael Brown', 'Sarah Davis', 'David Miller', 'Lisa Wilson'],
        'age': [34, 29, 42, 35, 51, 27, 38, 44],
        'gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
        'email': ['john.smith@example.com', 'jane.doe@example.com', 
                 'robert.j@example.com', 'e.williams@example.com',
                 'm.brown@example.com', 's.davis@example.com',
                 'david.m@example.com', 'lisa.wilson@example.com'],
        'income': [65000, 72000, 58000, 93000, 81000, 67000, 79000, 82000],
        'credit_score': [720, 750, 680, 790, 705, 740, 710, 760],
        'loan_amount': [250000, 320000, 180000, 420000, 340000, 275000, 310000, 290000],
        'zipcode': ['12345', '23456', '34567', '45678', '56789', '67890', '78901', '89012'],
        'phone': ['555-123-4567', '555-234-5678', '555-345-6789', '555-456-7890',
                 '555-567-8901', '555-678-9012', '555-789-0123', '555-890-1234']
    })
    
    print("Original data sample:")
    print(data.head(3))
    print(f"Shape: {data.shape}\n")
    
    # Generate synthetic data using simple method
    print("Generating synthetic data with the 'simple' method...")
    synthetic_data = generate_synthetic_data(
        template=data,
        num_samples=20,
        method="simple",
        sensitive_columns=['name', 'email', 'phone'],
        seed=42
    )
    
    print("Synthetic data sample:")
    print(synthetic_data.head(3))
    print(f"Shape: {synthetic_data.shape}\n")
    
    # Quick comparison of distributions for numeric columns
    print("Comparing distributions between original and synthetic data:")
    for col in ['age', 'income', 'credit_score', 'loan_amount']:
        print(f"\nColumn: {col}")
        print(f"Original - Mean: {data[col].mean():.2f}, Std: {data[col].std():.2f}")
        print(f"Synthetic - Mean: {synthetic_data[col].mean():.2f}, Std: {synthetic_data[col].std():.2f}")
    
    return data, synthetic_data

# Example 2: Statistical Synthetic Data Generation
def example_statistical_synthetic(data):
    """
    Generate synthetic data using the statistical method.
    This method preserves statistical properties like distributions and correlations.
    """
    print_header("Example 2: Statistical Synthetic Data Generation")
    
    print("Generating synthetic data with the 'statistical' method...")
    # Using the statistical method for better distribution preservation
    statistical_synthetic = generate_synthetic_data(
        template=data,
        num_samples=20,
        method="statistical",
        sensitive_columns=['name', 'email', 'phone'],
        preserve_dtypes=True,
        preserve_outliers=True,
        categorical_threshold=10,
        handle_skewness=True,
        seed=42
    )
    
    print("Statistical synthetic data sample:")
    print(statistical_synthetic.head(3))
    
    # Correlation comparison for numeric columns
    numeric_cols = ['age', 'income', 'credit_score', 'loan_amount']
    
    print("\nComparing correlations between original and synthetic data:")
    orig_corr = data[numeric_cols].corr()
    synth_corr = statistical_synthetic[numeric_cols].corr()
    
    print("\nOriginal correlation matrix:")
    print(orig_corr)
    print("\nSynthetic correlation matrix:")
    print(synth_corr)
    
    # Visualize distributions (optional - commented out)
    """
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numeric_cols):
        plt.subplot(2, 2, i+1)
        plt.hist(data[col], alpha=0.5, label='Original', bins=15)
        plt.hist(statistical_synthetic[col], alpha=0.5, label='Synthetic', bins=15)
        plt.title(f'Distribution of {col}')
        plt.legend()
    plt.tight_layout()
    plt.savefig('statistical_method_comparison.png')
    plt.close()
    print("\nDistribution comparison plot saved as 'statistical_method_comparison.png'")
    """
    
    return statistical_synthetic

# Example 3: Automatic Sensitive Column Detection
def example_auto_sensitive_detection(data):
    """
    Demonstrate automatic detection of sensitive columns in the data.
    """
    print_header("Example 3: Automatic Sensitive Column Detection")
    
    # Remove a few columns to show how the auto-detection works
    subset_data = data.drop(columns=['loan_amount', 'credit_score'])
    
    print("Original columns:", list(subset_data.columns))
    
    print("\nGenerating synthetic data with automatic sensitive column detection...")
    auto_synthetic = generate_synthetic_data(
        template=subset_data,
        num_samples=20,
        method="statistical",
        sensitivity_detection={
            "auto_detect": True,
            "confidence_threshold": 0.5,
            "sample_size": 8  # Using all rows in our small sample
        },
        seed=42
    )
    
    print("Synthetic data sample:")
    print(auto_synthetic.head(3))
    print(f"Shape: {auto_synthetic.shape}")
    
    # Indirectly check which columns were treated as sensitive by comparing with original
    print("\nExamining how columns were handled:")
    for col in subset_data.columns:
        if subset_data[col].dtype == object:
            # Check if values are completely different (likely treated as sensitive)
            overlap = set(subset_data[col]) & set(auto_synthetic[col])
            sensitive = len(overlap) == 0
            print(f"Column '{col}': {'Detected as sensitive' if sensitive else 'Not detected as sensitive'}")
    
    return auto_synthetic

# Example 4: Schema-Based Synthetic Data Generation
def example_schema_synthetic():
    """
    Generate synthetic data from a schema definition instead of a template DataFrame.
    """
    print_header("Example 4: Schema-Based Synthetic Data Generation")
    
    # Define a schema for financial customer data
    print("Defining a schema for customer financial data...")
    schema = {
        "columns": {
            "customer_id": "int",
            "age": "int",
            "income": "float",
            "credit_score": "int",
            "education_level": "category",
            "employment_status": "category",
            "has_mortgage": "bool",
            "has_loan": "bool",
            "account_balance": "float"
        }
    }
    
    print("Schema definition:")
    print(schema)
    
    print("\nGenerating synthetic data from schema...")
    schema_synthetic = generate_synthetic_data(
        template=schema,
        num_samples=20,
        method="statistical",
        seed=42
    )
    
    print("Schema-based synthetic data sample:")
    print(schema_synthetic.head(5))
    print(f"Shape: {schema_synthetic.shape}")
    
    return schema_synthetic

# Example 5: SDV Integration Methods
def example_sdv_integration(data):
    """
    Generate synthetic data using SDV integration methods.
    This example will gracefully handle cases where SDV is not installed.
    """
    print_header("Example 5: SDV Integration Methods")
    
    try:
        # Try to import SDV to check if it's available
        from sdv.single_table import GaussianCopulaSynthesizer
        has_sdv = True
    except ImportError:
        has_sdv = False
        print("SDV package is not installed. To use these methods, install SDV with:")
        print("pip install sdv")
        return None
    
    if has_sdv:
        print("SDV package is available. Generating synthetic data with SDV methods...")
        
        # Define constraints for SDV
        constraints = [
            {"type": "unique", "columns": ["zipcode"]}
        ]
        
        # Generate data with SDV-Copula
        print("\nUsing SDV-Copula method:")
        sdv_copula_synthetic = generate_synthetic_data(
            template=data,
            num_samples=20,
            method="sdv-copula",
            sensitive_columns=['name', 'email', 'phone'],
            anonymize_fields=True,
            constraints=constraints,
            seed=42
        )
        
        print("SDV-Copula synthetic data sample:")
        print(sdv_copula_synthetic.head(3))
        
        # Generate data with SDV-CTGAN
        print("\nUsing SDV-CTGAN method:")
        try:
            sdv_ctgan_synthetic = generate_synthetic_data(
                template=data,
                num_samples=20,
                method="sdv-ctgan",
                sensitive_columns=['name', 'email', 'phone'],
                anonymize_fields=True,
                epochs=100,  # Reduced for example
                batch_size=4,  # Small batch size for the example
                constraints=constraints,
                seed=42
            )
            
            print("SDV-CTGAN synthetic data sample:")
            print(sdv_ctgan_synthetic.head(3))
        except Exception as e:
            print(f"Error generating CTGAN synthetic data: {str(e)}")
            sdv_ctgan_synthetic = None
        
        return sdv_copula_synthetic, sdv_ctgan_synthetic
    
    return None

# Example 6: GAN-based Synthetic Data
def example_gan_synthetic(data):
    """
    Generate synthetic data using the GAN-based method.
    Note: This might fall back to SDV-CTGAN if TensorFlow is not available.
    """
    print_header("Example 6: GAN-based Synthetic Data")
    
    try:
        # Try to import TensorFlow to check if it's available
        import tensorflow as tf
        has_tf = True
    except ImportError:
        has_tf = False
        print("TensorFlow is not installed. The GAN method may fall back to SDV-CTGAN.")
    
    print("Generating synthetic data with the GAN-based method...")
    try:
        gan_synthetic = generate_synthetic_data(
            template=data,
            num_samples=20,
            method="gan",
            sensitive_columns=['name', 'email', 'phone'],
            epochs=50,  # Reduced for example
            batch_size=4,  # Small batch size for the example
            generator_dim=[64, 64],  # Smaller network for the example
            discriminator_dim=[64, 64],
            learning_rate=0.001,
            noise_dim=50,
            preserve_dtypes=True,
            seed=42
        )
        
        print("GAN-based synthetic data sample:")
        print(gan_synthetic.head(3))
        
        # PCA visualization to compare distributions
        numeric_cols = ['age', 'income', 'credit_score', 'loan_amount']
        
        # Scale the data
        scaler = StandardScaler()
        original_scaled = scaler.fit_transform(data[numeric_cols])
        synthetic_scaled = scaler.transform(gan_synthetic[numeric_cols])
        
        # Apply PCA
        pca = PCA(n_components=2)
        original_pca = pca.fit_transform(original_scaled)
        synthetic_pca = pca.transform(synthetic_scaled)
        
        # Display variance explained
        print(f"\nPCA variance explained: {pca.explained_variance_ratio_}")
        
        # Visualization commented out - would create a plot in a real environment
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(original_pca[:, 0], original_pca[:, 1], label='Original Data', alpha=0.7)
        plt.scatter(synthetic_pca[:, 0], synthetic_pca[:, 1], label='Synthetic Data', alpha=0.7)
        plt.legend()
        plt.title('PCA Comparison: Original vs GAN Synthetic Data')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.savefig('gan_pca_comparison.png')
        plt.close()
        print("\nPCA comparison plot saved as 'gan_pca_comparison.png'")
        """
        
        return gan_synthetic
    except Exception as e:
        print(f"Error generating GAN synthetic data: {str(e)}")
        return None

# Example 7: Copula-based Synthetic Data
def example_copula_synthetic(data):
    """
    Generate synthetic data using the copula-based method.
    """
    print_header("Example 7: Copula-based Synthetic Data")
    
    print("Generating synthetic data with the copula-based method...")
    try:
        copula_synthetic = generate_synthetic_data(
            template=data,
            num_samples=20,
            method="copula",
            sensitive_columns=['name', 'email', 'phone'],
            copula_type="gaussian",
            fit_method="ml",
            preserve_dtypes=True,
            handle_missing="mean",
            categorical_threshold=10,
            handle_skewness=True,
            seed=42
        )
        
        print("Copula-based synthetic data sample:")
        print(copula_synthetic.head(3))
        
        # Compare distributions of numeric columns
        numeric_cols = ['age', 'income', 'credit_score', 'loan_amount']
        
        print("\nComparing distributions of original and copula-based synthetic data:")
        for col in numeric_cols:
            print(f"\nColumn: {col}")
            orig_quantiles = np.quantile(data[col], [0.25, 0.5, 0.75])
            synth_quantiles = np.quantile(copula_synthetic[col], [0.25, 0.5, 0.75])
            
            print(f"Original quartiles (25%, 50%, 75%): {orig_quantiles}")
            print(f"Synthetic quartiles (25%, 50%, 75%): {synth_quantiles}")
        
        return copula_synthetic
    except Exception as e:
        print(f"Error generating copula synthetic data: {str(e)}")
        return None

# Example 8: Comparison of Methods
def example_methods_comparison(data):
    """
    Compare different synthetic data generation methods.
    """
    print_header("Example 8: Comparison of Synthetic Data Generation Methods")
    
    print("Generating synthetic data using different methods for comparison...")
    
    # Number of samples to generate
    n_samples = 20
    
    # Generate synthetic data with each method
    methods = ["simple", "statistical", "copula"]
    synthetic_datasets = {}
    
    for method in methods:
        print(f"\nGenerating data with method: {method}")
        try:
            synthetic_datasets[method] = generate_synthetic_data(
                template=data,
                num_samples=n_samples,
                method=method,
                sensitive_columns=['name', 'email', 'phone'],
                seed=42
            )
            print(f"Successfully generated {n_samples} samples")
        except Exception as e:
            print(f"Error with method {method}: {str(e)}")
    
    # Compare means and standard deviations of numeric columns
    numeric_cols = ['age', 'income', 'credit_score', 'loan_amount']
    
    print("\nComparing statistical properties across methods:")
    print(f"{'Column':<15} {'Metric':<10} {'Original':<10}", end="")
    for method in methods:
        if method in synthetic_datasets:
            print(f" {method.capitalize():<10}", end="")
    print()
    
    for col in numeric_cols:
        # Mean comparison
        print(f"{col:<15} {'Mean':<10} {data[col].mean():<10.2f}", end="")
        for method in methods:
            if method in synthetic_datasets:
                synthetic_mean = synthetic_datasets[method][col].mean()
                print(f" {synthetic_mean:<10.2f}", end="")
        print()
        
        # Std comparison
        print(f"{col:<15} {'Std':<10} {data[col].std():<10.2f}", end="")
        for method in methods:
            if method in synthetic_datasets:
                synthetic_std = synthetic_datasets[method][col].std()
                print(f" {synthetic_std:<10.2f}", end="")
        print()
    
    # Calculate overall statistical similarity (simplified metric)
    print("\nOverall statistical similarity score (lower is better):")
    for method in methods:
        if method in synthetic_datasets:
            mse = 0
            for col in numeric_cols:
                # Normalized mean difference
                mean_diff = (data[col].mean() - synthetic_datasets[method][col].mean()) / data[col].mean()
                # Normalized std difference
                std_diff = (data[col].std() - synthetic_datasets[method][col].std()) / data[col].std()
                mse += (mean_diff ** 2 + std_diff ** 2)
            mse /= (len(numeric_cols) * 2)  # Average across columns and metrics
            print(f"{method.capitalize()}: {mse:.4f}")
    
    return synthetic_datasets

# Main function to run all examples
def main():
    print("SecureML Synthetic Data Generation Examples")
    print("------------------------------------------")
    
    # Run the examples
    original_data, simple_synthetic = example_simple_synthetic()
    statistical_synthetic = example_statistical_synthetic(original_data)
    auto_sensitive = example_auto_sensitive_detection(original_data)
    schema_synthetic = example_schema_synthetic()
    sdv_results = example_sdv_integration(original_data)
    gan_synthetic = example_gan_synthetic(original_data)
    copula_synthetic = example_copula_synthetic(original_data)
    comparison_results = example_methods_comparison(original_data)
    
    print("\nAll synthetic data generation examples completed.")

if __name__ == "__main__":
    main() 