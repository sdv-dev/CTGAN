from unittest import TestCase
from ctgan.data_transformer import DataTransformer

class TestDataTransformer(TestCase):

    def test___fit_continuous_(self):
        """Test '_fit_continuous_' on a simple continuous column.

        A 'BayesianGaussianMixture' will be created and fit with the
        'raw_column_data'.

        Setup:
            - Create DataTransformer with weight_threshold
            - Mock the BayesianGaussianMixture
            - Provide fit method (no-op)
            - Provide weights_ attribute, some above threshold, some below
        
        Input:
            - column_name = string
            - raw_column_data = numpy array of continuous values

        Output:
            - ColumnTransformInfo
              - Check column name
              - Check that output_dims matches expected (1 + # weights above threshold)

        Side Effects:
            - fit should be called with the data
        """
        pass
    
    def test___fit_discrete_(self):
        """Test '_fit_discrete_' on a simple discrete column.

        A 'OneHotEncodingTransformer' will be created and fit with the
        'raw_column_data'.
        
        Setup:
            - Create DataTransformer
            - Mock the OneHotEncodingTransformer
            - Provide fit method (no-op)
        
        Input:
            - column_name = string
            - raw_column_data = numpy array of discrete values
        
        Output:
            - ColumnTransformInfo
              - Check column name
              - Check that output_dims matches expected (number of categories)
        
        Side Effects:
            - fit should be called with the data
        """
        pass

    def test_fit(self):
        """Test 'fit' on a np.ndarray with one continuous and one discrete columns.
        
        The 'fit' method should:
            - Set 'self.dataframe' to 'False'
            - Set 'self._column_raw_dtypes' to the appropirate dtypes
            - Use the appropriate '_fit' type for each column'
            - Update 'self.output_info_list', 'self.output_dimensions' and 
            'self._column_transform_info_list' appropriately

        Setup:
            - Create DataTransformer
            - Mock _fit_discrete
            - Mock _fit_continuous

        Input:
            - raw_data = a table with one continuous and one discrete columns.
            - discrete_columns = list with the name of the discrete column
        
        Output:
            - None

        Side Effects:
            - _fit_discrete and _fit_continuous should each be called once
            - Assigns 'self._column_raw_dtypes' the appropriate dtypes
            - Assigns 'self.output_info_list' the appropriate 'output_info'.
            - Assigns 'self.output_dimensions' the appropriate 'output_dimensions'.
            - Assigns 'self._column_transform_info_list' the appropriate 'column_transform_info'.
        """
        pass

    def test__transform_continuous(self):
        """Test '_transform_continuous'.

        The `transform_continuous` method first computes the probability that the 
        continuous value came from each component, then samples a component based
        on that probability and finally returns a (value, onehot) tuple, where the
        onehot vector indicates the component that was selected and the value 
        is a normalized representation of the continuous value based on the mean
        and std of the selected component.

        This test mocks the gaussian mixture model used to compute the probabilities
        as well as the sampling method; it returns deterministic values to avoid
        randomness in the test. Then, it tests to make sure the probabilities 
        are computed correctly and that the normalized value is computed correctly.

        Setup:
            - Create column_transform_info with mocked transformer
            - Mock the BayesianGaussianMixture transformer
               - specify means, covariances, predict_proba
               - means = [0, 10]
               - covariances = [1.0, 11.0]
            - Mock np.random.choice to choose maximum likelihood

        Input:
            - column_transform_info
            - raw_column_data = np.array([0.001, 11.9999, 13.001])
        
        Output:
            - normalized_value (assert between -1.0, 1.0)
              - assert approx = [0.0, -1.0, 1.0]
            - onehot (assert that it's a one-hot encoding)
              - assert = [[0, 1], [1, 0], [1, 0]]

        Side Effects:
            - assert predict_proba called
            - assert np.random.choice with appropriate probabilities
        """
        pass

    def test_transform(self):
        """Test 'transform' on a dataframe with one continuous and one discrete columns.
        
        It should use the appropriate '_transform' type for each column and should return
        them concanenated appropriately.

        Setup:
            - Mock _column_transform_info_list
            - Mock _transform_discrete
            - Mock _trarnsform_continuous

        Input:
            - raw_data = a table with one continuous and one discrete columns.
        
        Output:
            - numpy array containing the transformed two columns

        Side Effects:
            - _transform_discrete and _transform_continuous should each be called once.
        """
        pass

    def test__inverse_transform_continuous(self):
        """Test '_inverse_transform_continuous' with sigmas != None.

        The '_inverse_transform_continuous' method should be able to return np.ndarray
        to the appropriate continuous column. However, it currently cannot do so because
        of the way sigmas/st is being passed around. We should look into a less hacky way
        of using this function for TVAE...
        
        Setup:
            - Mock column_transform_info

        Input:
            - column_data = np.ndarray
              - the first column contains the normalized value
              - the remaining columns correspond to the one-hot
            - sigmas = np.ndarray of floats
            - st = index of the sigmas ndarray
        
        Output:
            - numpy array containing a single column of continuous values

        Side Effects:
            - None
        """
        pass

    def test_inverse_transform(self):
        """Test 'inverse_transform' on a np.ndarray representing one continuous and one
        discrete columns.

        It should use the appropriate '_fit' type for each column and should return
        the corresponding columns. Since we are using the same example as the 'test_transform',
        and these two functions are inverse of each other, the returned value here should
        match the input of that function.
        
        Setup:
            - Mock _column_transform_info_list
            - Mock _inverse_transform_discrete
            - Mock _inverse_trarnsform_continuous

        Input:
            - column_data = a concatenation of two np.ndarrays
              - the first one refers to the continuous values 
                - the first column contains the normalized values
                - the remaining columns correspond to the a one-hot
              - the second one refers to the discrete values
                - the columns correspond to a one-hot
        Output:
            - numpy array containing a discrete column and a continuous column

        Side Effects:
            - _transform_discrete and _transform_continuous should each be called once.
        """
        pass


    def test_convert_column_name_value_to_id(self):
        """Test 'convert_column_name_value_to_id' on a simple '_column_transform_info_list'.

        Tests that the appropriate indexes are returned when a table of three columns,
        discrete, continuous, discrete, is passed as '_column_transform_info_list'.
        
        Setup:
            - Mock _column_transform_info_list
        
        Input:
            - column_name = the name of a discrete column 
            - value = the categorical value

        Output:
            - dictionary containing:
              - "discrete_column_id" = the index of the target column, considering only discrete columns
              - "column_id" = the index of the target column (e.g. 3 = the third column in the data)
              - "value_id" = the index of the indicator value in the one-hot encoding
        """
        pass




    def test_constant(self):
        """Test transforming a dataframe containing constant values."""
        pass

    def test_df_continuous(self):
        """Test transforming a dataframe containing only continuous values."""
        # validate output ranges [0, 1]
        # validate output shape (# samples, # output dims)
        # validate that forward transform is **not** deterministic
        # make sure it can be inverted
        pass

    def test_df_categorical(self):
        """Test transforming a dataframe containing only categorical values."""
        # validate output ranges [0, 1]
        # validate output shape (# samples, # output dims)
        # validate that forward transform is deterministic
        # make sure it can be inverted
        pass

    def test_df_mixed(self):
        """Test transforming a dataframe containing mixed data types."""
        pass

    def test_df_mixed_nan(self):
        """Test transforming a dataframe containing mixed data types + NaN for categoricals."""
        pass

    def test_np_continuous(self):
        """Test transforming a np.array containing only continuous values."""
        pass

    def test_np_categorical(self):
        """Test transforming a np.array containing only categorical values."""
        pass

    def test_np_mixed(self):
        """Test transforming a np.array containing mixed data types."""
        pass
