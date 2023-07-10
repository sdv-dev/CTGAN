from ctgan import CTGAN
from ctgan import load_demo
import pandas as pd
real_data = load_demo()

# Names of the columns that are discrete
discrete_columns = ['3','4','5','25','26','27','29','30','34','35','36','41','73','80','86','102']

ctgan = CTGAN(epochs=10)
ctgan.fit(real_data, discrete_columns)

# Create synthetic data
synthetic_data = ctgan.sample(1000)
df_out = pd.DataFrame(synthetic_data)
df_out.to_csv('output.csv', index = False)