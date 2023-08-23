
from time import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from ctgan.synthesizers.ctgan import CTGAN

gaussians = list(np.random.normal(loc=-2, size=10_000)) + \
    list(np.random.normal(loc=2, size=10_000))
neg_gaussians = [-i for i in gaussians]
data = pd.DataFrame({
    'continuous1': gaussians,
    'continuous2': neg_gaussians,
})
plt.hist(data, bins=100)

start = time()
transformer = CTGAN(epochs=100)
transformer.fit(data, [])
new_data = transformer.sample(1000)
print(time() - start)

new_data[:, 0] += np.random.uniform(low=-.2, high=.2, size=len(new_data))
reversed = transformer.inverse_transform(new_data)

plt.figure(2)
plt.hist(reversed, bins=100)
plt.show()

assert 0
