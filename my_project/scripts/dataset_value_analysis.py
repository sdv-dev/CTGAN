import pandas as pd
import os

# Define the dataset path
dataset_path = r"C:\Users\ortho\OneDrive\Desktop\SDV\CTGAN\my_project\data\dataset_26219_16.csv"

# Load the dataset (adjust delimiter if needed)
df = pd.read_csv(dataset_path)

# Prepare a variable to accumulate the analysis results as text
output = "=== Overall Distribution Analysis for Each Column ===\n\n"

# Loop through each column in the dataset
for col in df.columns:
    output += f"Column: {col}\n"
    if pd.api.types.is_numeric_dtype(df[col]):
        output += "Numeric Column Summary:\n"
        output += df[col].describe().to_string() + "\n\n"
    else:
        # For categorical columns, calculate counts and percentages
        counts = df[col].value_counts(dropna=False)
        percentages = df[col].value_counts(normalize=True, dropna=False) * 100
        distribution = pd.DataFrame({'Count': counts, 'Percentage': percentages})
        output += "Categorical Distribution:\n"
        output += distribution.to_string() + "\n\n"

# Detailed analysis for a specific column, e.g., "Sex"
specific_column = "Sex"  # Change this to any other column name if needed
output += f"=== Detailed Analysis for Column: {specific_column} ===\n\n"
if specific_column in df.columns:
    counts = df[specific_column].value_counts(dropna=False)
    percentages = df[specific_column].value_counts(normalize=True, dropna=False) * 100
    specific_distribution = pd.DataFrame({'Count': counts, 'Percentage': percentages})
    output += specific_distribution.to_string() + "\n"
else:
    output += f"Column '{specific_column}' not found in the dataset.\n"

# Save the results as a text file in the same folder as the dataset
output_file = os.path.join(os.path.dirname(dataset_path), "distribution_analysis.txt")
with open(output_file, "w") as f:
    f.write(output)

print("Distribution analysis saved to:", output_file)
