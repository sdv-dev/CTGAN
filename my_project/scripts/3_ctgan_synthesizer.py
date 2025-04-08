import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
import matplotlib.pyplot as plt

# === File Paths ===
dataset_path = r"C:\Users\ortho\OneDrive\Desktop\PendarLink AI\Registry\Dataset\Python_Analysis\data\dataset_26219_16.csv"
metadata_path = r"C:\Users\ortho\OneDrive\Desktop\PendarLink AI\Registry\Dataset\Python_Analysis\data\metadata\metadata.json"
synthesizer_save_path = r"C:\Users\ortho\OneDrive\Desktop\PendarLink AI\Registry\Dataset\Python_Analysis\models\ctgan_synthesizer.pkl"
synthetic_data_save_path = r"C:\Users\ortho\OneDrive\Desktop\PendarLink AI\Registry\Dataset\Python_Analysis\data\synthetic_data.csv"

# === Step 1: Load Dataset and Metadata ===
data = pd.read_csv(dataset_path)
print("✅ Dataset Loaded Successfully!")
print(data.head())

metadata = SingleTableMetadata.load_from_json(metadata_path)
print("✅ Metadata Loaded Successfully!")

# === Step 2: Initialize CTGANSynthesizer ===
synthesizer = CTGANSynthesizer(
    metadata=metadata,
    enforce_rounding=False,  # Avoid rounding numeric values
    epochs=500,             # Number of training epochs
    verbose=True,           # Print progress during training
    cuda=True               # Use GPU if available
)
print("🚀 CTGANSynthesizer Initialized!")

# === Step 3: Train the CTGAN Model ===
synthesizer.fit(data)
print("✅ CTGAN Model Trained Successfully!")

# === Step 4: Evaluate Training Loss ===
loss_values = synthesizer.get_loss_values()
print("📉 Training Loss Values:")
print(loss_values.head())

# Plot Loss Values
loss_values.plot(x='Epoch', y=['Generator Loss', 'Discriminator Loss'])
plt.title('CTGAN Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Generator Loss', 'Discriminator Loss'])
plt.show()

# === Step 5: Generate Synthetic Data ===
synthetic_data = synthesizer.sample(num_rows=len(data))
print("✅ Synthetic Data Generated!")
print(synthetic_data.head())

# Save Synthetic Data
synthetic_data.to_csv(synthetic_data_save_path, index=False)
print(f"📁 Synthetic Data Saved to: {synthetic_data_save_path}")

# === Step 6: Save Trained Synthesizer ===
synthesizer.save(filepath=synthesizer_save_path)
print(f"💾 Trained CTGANSynthesizer Saved to: {synthesizer_save_path}")

# === Step 7: (Optional) Load Saved Synthesizer ===
# To reload the saved synthesizer for future use:
# loaded_synthesizer = CTGANSynthesizer.load(filepath=synthesizer_save_path)
# print("✅ Trained CTGANSynthesizer Loaded Successfully!")
