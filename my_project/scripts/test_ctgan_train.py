import os
import pandas as pd

# SDV Imports
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer

# --------------------------------------------------------------------------------
# 1. Define File Paths
# --------------------------------------------------------------------------------
DATASET_PATH = r"C:\Users\ortho\OneDrive\Desktop\SDV\CTGAN\my_project\data\dataset_26219_16.csv"
OUTPUT_DIR = r"C:\Users\ortho\OneDrive\Desktop\SDV\CTGAN\my_project\output"
METADATA_PATH = os.path.join(OUTPUT_DIR, "metadata.json")
SYNTHETIC_DATA_PATH = os.path.join(OUTPUT_DIR, "synthetic_data.csv")
MODEL_PATH = os.path.join(OUTPUT_DIR, "my_synthesizer.pkl")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------------------------------------
# 2. Load the Dataset
# --------------------------------------------------------------------------------
df = pd.read_csv(DATASET_PATH)
print("✅ Dataset loaded successfully!")
print(df.head())
print(f"Dataset shape: {df.shape}")

# --------------------------------------------------------------------------------
# 3. Initialize Metadata and Auto-Detect
# --------------------------------------------------------------------------------
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df)
print("✅ Metadata auto‐detected.")

# --------------------------------------------------------------------------------
# 4. Update Column Types
#    (Adjust these lists if your column names differ in the CSV)
# --------------------------------------------------------------------------------
categorical_columns = [
    'Age recode with less 1 year olds',
    'Sex',
    'Race recode White_Black_Other',
    'ICD_O_3 Hist_behav',
    'ICCC site recode extended 3rd edition_IARC 2017',
    'Site recode ICD_O_3_WHO 2008',
    'Primary Site _labeled',
    'Histologic Type ICD_O_3',
    'Reason no cancer_directed surgery',
    'Radiation recode',
    'Chemotherapy recode Yes_ No_Unknownunk',
    'SEER cause_specific death classification',
    'COD to site recode'
]

numeric_columns = [
    'Year of diagnosis',
    'Time from diagnosis to treatment in days recode',
    'Survival months'
]

# Update categorical columns
for col in categorical_columns:
    metadata.update_column(column_name=col, sdtype='categorical')

# Update numeric columns
for col in numeric_columns:
    metadata.update_column(column_name=col, sdtype='numerical')

# --------------------------------------------------------------------------------
# 5. Validate and Save Metadata
# --------------------------------------------------------------------------------
try:
    metadata.validate()
    print("✅ Metadata validation successful!")
except Exception as e:
    print("❌ Metadata validation failed:", e)
    
# Remove existing metadata.json file if it exists
if os.path.exists(METADATA_PATH):
    os.remove(METADATA_PATH)
    print(f"🗑 Existing metadata file removed: {METADATA_PATH}")

metadata.save_to_json(filepath=METADATA_PATH)
print(f"📁 Metadata saved to: {METADATA_PATH}")

# --------------------------------------------------------------------------------
# 6. Load Metadata from JSON (Optional, but recommended for a clean workflow)
# --------------------------------------------------------------------------------
metadata = SingleTableMetadata.load_from_json(METADATA_PATH)
print("✅ Metadata reloaded from JSON.")

# --------------------------------------------------------------------------------
# 7. Train the CTGAN Synthesizer
# --------------------------------------------------------------------------------
ctgan = CTGANSynthesizer(metadata, verbose=True, epochs=800)
ctgan.fit(df)
print("🚀 CTGAN model trained successfully!")

# Save the trained model
ctgan.save(filepath=MODEL_PATH)
print(f"🚀 CTGAN model saved successfully to {MODEL_PATH}!")

# --------------------------------------------------------------------------------
# 8. Generate and Save Synthetic Data
# --------------------------------------------------------------------------------
synthetic_data = ctgan.sample(num_rows=len(df))
print("✅ Synthetic data generated!")
print(synthetic_data.head())

synthetic_data.to_csv(SYNTHETIC_DATA_PATH, index=False)
print(f"📁 Synthetic data saved to: {SYNTHETIC_DATA_PATH}")
