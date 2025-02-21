import pandas as pd
from sdv.metadata import SingleTableMetadata

# === File Paths ===
dataset_path = r"C:\Users\ortho\OneDrive\Desktop\PendarLink AI\Registry\Dataset\Python_Analysis\data\dataset_26219_16.csv"
metadata_path = r"C:\Users\ortho\OneDrive\Desktop\PendarLink AI\Registry\Dataset\Python_Analysis\data\metadata\metadata.json"

# === Step 1: Load Dataset ===
real_data = pd.read_csv(dataset_path)
print("✅ Dataset Loaded Successfully!")
print(real_data.head())

# === Step 2: Load Metadata ===
metadata = SingleTableMetadata.load_from_json(metadata_path)
print("✅ Metadata Loaded Successfully!")
print(metadata.to_dict())

# === (Optional) Validate Metadata ===
metadata.validate()

# === Example: Using SDV Synthesizer ===
from sdv.single_table import CTGANSynthesizer

# Initialize and Train CTGAN Synthesizer
ctgan = CTGANSynthesizer(metadata)
ctgan.fit(real_data)
print("🚀 CTGAN Model Trained Successfully!")

# Generate Synthetic Data
synthetic_data = ctgan.sample(num_rows=len(real_data))
print("✅ Synthetic Data Generated!")
print(synthetic_data.head())

# Save Synthetic Data
synthetic_data.to_csv(r"C:\Users\ortho\OneDrive\Desktop\PendarLink AI\Registry\Dataset\Python_Analysis\data\synthetic_data.csv", index=False)
print("📁 Synthetic data saved successfully.")
