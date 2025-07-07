import pandas as pd
import numpy as np
import torch

##############################################################################
# 1) Load Real & Synthetic Data
##############################################################################

# Path to your real dataset CSV
REAL_DATA_PATH = r"C:\Users\ortho\OneDrive\Desktop\SDV\CTGAN\my_project\data\dataset_26219_16.csv"

# For demonstration, assume there's a synthetic CSV as well
# Replace with your actual synthetic data path or method of obtaining synthetic data
SYNTH_DATA_PATH = r"C:\Users\ortho\OneDrive\Desktop\SDV\CTGAN\my_project\output\synthetic_data.csv"

# Load the real dataset
real_df = pd.read_csv(REAL_DATA_PATH)

# Load the synthetic dataset
synthetic_df = pd.read_csv(SYNTH_DATA_PATH)

##############################################################################
# 2) Extract Numeric Columns (Manually Specified)
##############################################################################
# Manually specify numeric columns (exclude categorical columns like "Histologic Type ICD_O_3")
numeric_cols = [
    "Year of diagnosis",
    "Time from diagnosis to treatment in days recode",
    "Survival months"
]

# Filter both DataFrames to keep only numeric columns
real_numeric_df = real_df[numeric_cols].dropna()
synthetic_numeric_df = synthetic_df[numeric_cols].dropna()

##############################################################################
# 3) Compute Pearson Correlation Matrices in PyTorch
##############################################################################

def compute_correlation_matrix_torch(dataframe):
    """
    Convert the numeric DataFrame to a PyTorch tensor, then compute the Pearson correlation matrix.
    """
    # Convert to PyTorch tensor (float32)
    data_tensor = torch.tensor(dataframe.values, dtype=torch.float32)

    # Center the data
    data_centered = data_tensor - data_tensor.mean(dim=0, keepdim=True)
    # Covariance
    cov = (data_centered.t() @ data_centered) / (data_tensor.size(0) - 1)
    # Standard deviations
    diag = torch.diag(cov)
    stddev = torch.sqrt(torch.clamp(diag, min=1e-8))
    # Outer product of stddev to get denominators
    denom = stddev.unsqueeze(0) * stddev.unsqueeze(1)
    corr_matrix = cov / torch.clamp(denom, min=1e-8)
    return corr_matrix

# Compute correlation matrices
corr_real = compute_correlation_matrix_torch(real_numeric_df)
corr_synth = compute_correlation_matrix_torch(synthetic_numeric_df)

##############################################################################
# 4) Define the Regularization (Correlation) Loss
##############################################################################

def correlation_loss(corr_real, corr_synth, lambda_corr=1.0):
    """
    Compare two correlation matrices using the Frobenius norm of their difference,
    scaled by lambda_corr.
    """
    # Ensure both correlation matrices have the same shape
    if corr_real.shape != corr_synth.shape:
        raise ValueError("Real and synthetic correlation matrices must have the same shape.")
    
    # Frobenius norm of the difference
    diff = corr_real - corr_synth
    frob_norm = torch.norm(diff, p='fro')  # Frobenius norm
    return lambda_corr * frob_norm

# Set your hyperparameter for correlation enforcement
LAMBDA_CORR = 1.0

# Calculate the correlation loss
corr_loss_value = correlation_loss(corr_real, corr_synth, LAMBDA_CORR)

##############################################################################
# 5) Print Results
##############################################################################
print("Numeric columns used:", numeric_cols)
print("Real correlation matrix shape:", corr_real.shape)
print("Synthetic correlation matrix shape:", corr_synth.shape)
print(f"Correlation Loss (Frobenius norm * {LAMBDA_CORR}): {corr_loss_value.item():.4f}")
