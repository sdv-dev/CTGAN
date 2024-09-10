from ctgan import CTGAN, load_demo


def test_load_demo():
    """End-to-end test to load and synthesize data."""
    # Setup
    discrete_columns = [
        'workclass',
        'education',
        'marital-status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'native-country',
        'income',
    ]
    ctgan = CTGAN(epochs=1)

    # Run
    data = load_demo()
    ctgan.fit(data, discrete_columns)
    samples = ctgan.sample(1000, condition_column='native-country', condition_value='United-States')

    # Assert
    assert samples.shape == (1000, 15)
    assert all([col[0] != ' ' for col in samples.columns])
    assert not samples.isna().any().any()
