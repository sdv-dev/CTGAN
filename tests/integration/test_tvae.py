#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Integration tests for tvae.

These tests only ensure that the software does not crash and that
the API works as expected in terms of input and output data formats,
but correctness of the data values and the internal behavior of the
model are not checked.
"""

import pandas as pd
from sklearn import datasets

from ctgan.synthesizers.tvae import TVAESynthesizer


def test_tvae(tmpdir):
    iris = datasets.load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    data['class'] = pd.Series(iris.target).map(iris.target_names.__getitem__)

    tvae = TVAESynthesizer(epochs=10)
    tvae.fit(data, ['class'])

    path = str(tmpdir / 'test_tvae.pkl')
    tvae.save(path)
    tvae = TVAESynthesizer.load(path)

    sampled = tvae.sample(100)

    assert sampled.shape == (100, 5)
    assert isinstance(sampled, pd.DataFrame)
    assert set(sampled.columns) == set(data.columns)
    assert set(sampled.dtypes) == set(data.dtypes)


def test_drop_last_false():
    data = pd.DataFrame({
        '1': ['a', 'b', 'c'] * 150,
        '2': ['a', 'b', 'c'] * 150
    })

    tvae = TVAESynthesizer(epochs=300)
    tvae.fit(data, ['1', '2'])

    sampled = tvae.sample(100)
    correct = 0
    for _, row in sampled.iterrows():
        if row['1'] == row['2']:
            correct += 1

    assert correct >= 95
