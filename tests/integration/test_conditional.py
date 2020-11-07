import numpy as np
import pandas as pd
from ctgan.transformer import DataTransformer
from ctgan.conditional import ConditionalGenerator


def test_conditional_generator():
    # create toy example with 3 categorical columns.
    # there are 7 total categories
    data = pd.DataFrame({
        'continuous1': np.random.random(1000),
        'discrete1': np.repeat(['a', 'b', 'c'], [950, 25, 25]),
        'discrete2': np.repeat(['d', 'e'], [580, 420]),
        'discrete3': np.repeat(['f', 'g'], [100, 900])
    })

    discrete_columns = ['discrete1', 'discrete2', 'discrete3']

    # we need transformer.output_info
    transformer = DataTransformer()
    transformer.fit(data, discrete_columns)
    data = transformer.transform(data)

    # Note that log_frequency=True converts the proportions of each categorical column to log scale.
    test_cond_gen = ConditionalGenerator(data, transformer.output_info, log_frequency=True)
    assert test_cond_gen.n_opt == 7  # sum of number of categories in all categorical columns
    assert test_cond_gen.n_col == 3  # number of columns

    print("n_opt", test_cond_gen.n_opt)
    print("n_col", test_cond_gen.n_col)

    # generate 5 samples.
    # output of .sample is vec1, mask1, idx, opt1prime
    test_sample = test_cond_gen.sample(5)

    # vec1 is one-hot encoded.
    # 1 category is selected from 1 of the categorical columns
    # In the paper, see Figure 2, the one-hot encoded vector of D1 and D2.
    print('vec1')
    print(test_sample[0])

    # mask1 shows which categorical column is selected.
    print('mask1')
    print(test_sample[1])

    # corresponding index of categorical column selected.
    print('idx')
    print(test_sample[2])

    # the selected category of corresponding categorical column.
    print('opt1prime')
    print(test_sample[3])

    # check distribution of idx.
    # this test shows that each categorical column is sampled evenly/uniformly
    # out of 3000 samples, each appearing about 1000 times.
    print('check distribution of idx')
    nsample = 3000
    test_sample = test_cond_gen.sample(nsample)

    # display the categorical column and corresponding counts
    (unique, counts) = np.unique(test_sample[2], return_counts=True)
    for i in range(len(unique)):
        print(unique[i], discrete_columns[i], counts[i])


if __name__ == "__main__":
    test_conditional_generator()
