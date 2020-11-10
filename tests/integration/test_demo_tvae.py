# test the GitHub demo
import os
from ctgan import load_demo
from ctgan import TVAESynthesizer



# TEST REMARK

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

# Step 2: Fit TVAE to your data
tvae = TVAESynthesizer()
print('Training tvae is starting')
tvae.fit(data, discrete_columns, epochs=5, model_summary=True)
print('Training tvae is completed')

# 2. Generate synthetic data
samples_1 = tvae.sample(10)
print('size of sample_1', samples_1.shape)

