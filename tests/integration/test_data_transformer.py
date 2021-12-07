"""Data transformer intergration testing module."""


# Data Transformer tests that should be implemented in the future.
def test_constant():
    """Test transforming a dataframe containing constant values."""


def test_df_continuous():
    """Test transforming a dataframe containing only continuous values."""
    # validate output ranges [0, 1]
    # validate output shape (# samples, # output dims)
    # validate that forward transform is **not** deterministic
    # make sure it can be inverted


def test_df_categorical():
    """Test transforming a dataframe containing only categorical values."""
    # validate output ranges [0, 1]
    # validate output shape (# samples, # output dims)
    # validate that forward transform is deterministic
    # make sure it can be inverted


def test_df_mixed():
    """Test transforming a dataframe containing mixed data types."""


def test_df_mixed_nan():
    """Test transforming a dataframe containing mixed data types + NaN for categoricals."""


def test_np_continuous():
    """Test transforming a np.array containing only continuous values."""


def test_np_categorical():
    """Test transforming a np.array containing only categorical values."""


def test_np_mixed():
    """Test transforming a np.array containing mixed data types."""
