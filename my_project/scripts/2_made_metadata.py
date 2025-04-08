import pandas as pd
from sdv.metadata import SingleTableMetadata

# === File Paths ===
dataset_path = r"C:\Users\ortho\OneDrive\Desktop\PendarLink AI\Registry\Dataset\Python_Analysis\data\dataset_26219_16.csv"
metadata_save_path = r"C:\Users\ortho\OneDrive\Desktop\PendarLink AI\Registry\Dataset\Python_Analysis\data\metadata\metadata.json"

# === Step 1: Load the Dataset ===
data = pd.read_csv(dataset_path)
print("✅ Dataset Loaded Successfully!")
print(data.head())

# === Step 2: Initialize Metadata and Auto-Detect ===
# Initialize SingleTableMetadata
metadata = SingleTableMetadata()

# Auto-detect metadata from the DataFrame
metadata.detect_from_dataframe(data)
print("✅ Metadata Auto-Detected.")

# === Step 3: Update Column Types Based on Provided Information ===

# Categorical Columns
categorical_columns = [
    'Age recode with less 1 year olds', 'Sex', 'Race recode White_Black_Other',
    'ICD_O_3 Hist_behav', 'ICCC site recode extended 3rd ed.',
    'Site recode ICD_O_3_WHO 2008', 'Primary Site _labeled',
    'Histologic Type ICD_O_3', 'Reason no cancer_directed surgery',
    'Radiation recode', 'Chemotherapy recode Yes_No_Unknownunk',
    'SEER cause_specific death classification', 'COD to site recode'
]

# Numeric Columns
numeric_columns = [
    'Year of diagnosis', 'Time from diagnosis to treatment in days recode',
    'Survival months'
]

# Update Categorical Columns
for column in categorical_columns:
    metadata.update_column(column_name=column, sdtype='categorical')

# Update Numeric Columns
for column in numeric_columns:
    metadata.update_column(column_name=column, sdtype='numerical')

# === Step 4: Validate Metadata ===
try:
    metadata.validate()
    print("✅ Metadata Validation Successful!")
except Exception as e:
    print("❌ Metadata Validation Failed:", e)

# === Step 5: Save Metadata to JSON ===
metadata.save_to_json(filepath=metadata_save_path)
print(f"📁 Metadata saved to: {metadata_save_path}")

# === (Optional) Print Metadata ===
print("📋 Final Metadata Structure:")
print(metadata.to_dict())
