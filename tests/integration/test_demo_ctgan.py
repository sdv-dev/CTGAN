# test the GitHub demo
import os
from ctgan import load_demo
from ctgan import CTGANSynthesizer
import pandas as pd
import numpy as np

cwd = os.getcwd()
print("Current working directory is:", cwd)

# # using a toy example to test tablegan
# data = pd.DataFrame({
#     'continuous1': np.random.random(1000),
#     'discrete1': np.repeat([1, 2, 3], [950, 25, 25]),
#     'discrete2': np.repeat(["a", "b"], [580, 420]),
#     'discrete3': np.repeat([6, 7], [100, 900])
# })
#
# # index of columns
# discrete_columns = ['discrete1', 'discrete2', 'discrete3']

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

# NOTE: the next test runs into an error currently.
# # 3. Generate synthetic data conditioning on one column
# samples_2 = ctgan.sample(10, 'workclass', ' Private')
# print('size of sample_2', samples_2.shape)

# #4. Save and load the synthesizer
# path_to_a_folder = cwd + "/test_model.pth"
# print("before saving, does file exist?", os.path.exists(path_to_a_folder))
#
# # To save a trained ctgan synthesizer
# ctgan.save(path_to_a_folder)
# # NOTE: We'll see warnings:
# # UserWarning: Couldn't retrieve source code for container of type ... . It won't be checked for correctness upon loading.
# #   "type " + obj.__name__ + ". It won't be checked "
#
# print("after saving, does file exist?", os.path.exists(path_to_a_folder))
#
# # NOTE: the next test runs into an error currently.
# # # To restore a saved synthesizer
# # ctgan_2 = CTGANSynthesizer()
# # ctgan_2.fit(data, discrete_columns, epochs=0, load_path=path_to_a_folder)
# # ctgan_2_sample = ctgan_2.sample(5)
# # print('ctgan_2_sample', ctgan_2_sample.shape)
#
# # cleaning up
# if os.path.exists(path_to_a_folder):
#     os.remove(path_to_a_folder)
