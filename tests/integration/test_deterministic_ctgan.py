# test the GitHub demo
import os
import numpy as np
import torch

# # NOTE: To test settings that would enable deterministic training
# See: https://pytorch.org/docs/stable/notes/randomness.html
torch.manual_seed(0)
np.random.seed(0)
# torch.set_deterministic(True)  # may require a newer version v1.7.

print(torch.__version__)

from ctgan import load_demo
from ctgan import CTGANSynthesizer

cwd = os.getcwd()
print("Current working directory is:", cwd)

# 1. Model the data
# Step 1: Prepare your data
data = load_demo()

discrete_columns = [
    'workclass',
    'education',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'native-country',
    'income'
]

# Step 2: Fit CTGAN to your data
ctgan = CTGANSynthesizer()
ctgan.fit(data, discrete_columns, epochs=5, model_summary=True)

# 2. Generate synthetic data
samples_1 = ctgan.sample(10)
print('size of sample_1', samples_1.shape)
print(samples_1)

