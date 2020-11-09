import os
# from ctgan import load_demo
from ctgan import TableganSynthesizerOriginal
import numpy as np
import pandas as pd

cwd = os.getcwd()
print("Current working directory is:", cwd)

# using a toy example to test tablegan
# discreet values must be numerical values.
data = pd.DataFrame({
    'continuous1': np.random.random(1000),
    'discrete1': np.repeat([1, 2, 3], [950, 25, 25]),
    'discrete2': np.repeat([4, 5], [580, 420]),
    'discrete3': np.repeat([6, 7], [100, 900])
})

# TableganTransformer requires a matrix and does not process a dataframe
data = data.values

# index of columns
discrete_columns = [1, 2, 3]

# Step 2: Fit tableGan to your data
tablegan = TableganSynthesizerOriginal()
print('Training tablegan is starting')
tablegan.fit(data, categorical_columns=discrete_columns, epochs=5)
print('Training tablegan is completed')

# 2. Generate synthetic data
samples_1 = tablegan.sample(10)
print('size of sample_1', samples_1.shape)

