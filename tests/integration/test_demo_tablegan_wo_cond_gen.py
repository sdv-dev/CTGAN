import os
from ctgan import load_demo
from ctgan import TableganSynthesizerWOCondGen
import numpy as np
import pandas as pd

cwd = os.getcwd()
print("Current working directory is:", cwd)

# using a toy example to test tablegan
data = pd.DataFrame({
    'continuous1': np.random.random(1000),
    'discrete1': np.repeat([1, 2, 3], [950, 25, 25]),
    'discrete2': np.repeat(["a", "b"], [580, 420]),
    'discrete3': np.repeat([6, 7], [100, 900])
})

# index of columns
discrete_columns = ['discrete1', 'discrete2', 'discrete3']

# # 1. Model the data
# # Step 1: Prepare your data
# data = load_demo()
#
# discrete_columns = [
#     'workclass',
#     'education',
#     'marital-status',
#     'occupation',
#     'relationship',
#     'race',
#     'sex',
#     'native-country',
#     'income'
# ]

# Step 2: Fit tableGan to your data
tablegan = TableganSynthesizerWOCondGen()
print('Training tablegan is starting')
# NOTE: This runs much slower than ctgan and tvae
tablegan.fit(data, discrete_columns=discrete_columns, epochs=1, model_summary=True)
print('Training tablegan is completed')

# 2. Generate synthetic data
samples_1 = tablegan.sample(10)
print('size of sample_1', samples_1.shape)

