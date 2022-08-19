"""Data transformer unit testing module."""

from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from ctgan.data_transformer import ColumnTransformInfo, DataTransformer, SpanInfo


class TestDataTransformer(TestCase):

    @patch('ctgan.data_transformer.ClusterBasedNormalizer')
    def test___fit_continuous(self, MockCBN):
        """Test ``_fit_continuous`` on a simple continuous column.

        A ``ClusterBasedNormalizer`` will be created and fit with some ``data``.

        Setup:
            - Mock the ``ClusterBasedNormalizer`` with ``valid_component_indicator`` as
              ``[True, False, True]``.
            - Initialize a ``DataTransformer``.

        Input:
            - A dataframe with only one column containing random float values.

        Output:
            - A ``ColumnTransformInfo`` object where:
              - ``column_name`` matches the column of the data.
              - ``transform`` is the ``ClusterBasedNormalizer`` instance.
              - ``output_dimensions`` is 3 (matches size of ``valid_component_indicator``).
              - ``output_info`` assigns the correct activation functions.

        Side Effects:
            - ``fit`` should be called with the data.
        """
        # Setup
        cbn_instance = MockCBN.return_value
        cbn_instance.valid_component_indicator = [True, False, True]
        transformer = DataTransformer()
        data = pd.DataFrame(np.random.normal((100, 1)), columns=['column'])

        # Run
        info = transformer._fit_continuous(data)

        # Assert
        assert info.column_name == 'column'
        assert info.transform == cbn_instance
        assert info.output_dimensions == 3
        assert info.output_info[0].dim == 1
        assert info.output_info[0].activation_fn == 'tanh'
        assert info.output_info[1].dim == 2
        assert info.output_info[1].activation_fn == 'softmax'

    @patch('ctgan.data_transformer.ClusterBasedNormalizer')
    def test__fit_continuous_max_clusters(self, MockCBN):
        """Test ``_fit_continuous`` with data that has less than 10 rows.

        Expect that a ``ClusterBasedNormalizer`` is created with the max number of clusters
        set to the length of the data.

        Input:
        - Data with less than 10 rows.

        Side Effects:
        - A ``ClusterBasedNormalizer`` is created with the max number of clusters set to the
          length of the data.
        """
        # Setup
        data = pd.DataFrame(np.random.normal((7, 1)), columns=['column'])
        transformer = DataTransformer()

        # Run
        transformer._fit_continuous(data)

        # Assert
        MockCBN.assert_called_once_with(model_missing_values=True, max_clusters=len(data))

    @patch('ctgan.data_transformer.OneHotEncoder')
    def test___fit_discrete(self, MockOHE):
        """Test ``_fit_discrete_`` on a simple discrete column.

        A ``OneHotEncoder`` will be created and fit with the ``data``.

        Setup:
            - Mock the ``OneHotEncoder``.
            - Create ``DataTransformer``.

        Input:
            - A dataframe with only one column containing ``['a', 'b']`` values.

        Output:
            - A ``ColumnTransformInfo`` object where:
              - ``column_name`` matches the column of the data.
              - ``transform`` is the ``OneHotEncoder`` instance.
              - ``output_dimensions`` is 2.
              - ``output_info`` assigns the correct activation function.

        Side Effects:
            - ``fit`` should be called with the data.
        """
        # Setup
        ohe_instance = MockOHE.return_value
        ohe_instance.dummies = ['a', 'b']
        transformer = DataTransformer()
        data = pd.DataFrame(np.array(['a', 'b'] * 100), columns=['column'])

        # Run
        info = transformer._fit_discrete(data)

        # Assert
        assert info.column_name == 'column'
        assert info.transform == ohe_instance
        assert info.output_dimensions == 2
        assert info.output_info[0].dim == 2
        assert info.output_info[0].activation_fn == 'softmax'

    def test_fit(self):
        """Test ``fit`` on a np.ndarray with one continuous and one discrete columns.

        The ``fit`` method should:
            - Set ``self.dataframe`` to ``False``.
            - Set ``self._column_raw_dtypes`` to the appropirate dtypes.
            - Use the appropriate ``_fit`` type for each column.
            - Update ``self.output_info_list``, ``self.output_dimensions`` and
            ``self._column_transform_info_list`` appropriately.

        Setup:
            - Create ``DataTransformer``.
            - Mock ``_fit_discrete``.
            - Mock ``_fit_continuous``.

        Input:
            - A table with one continuous and one discrete columns.
            - A list with the name of the discrete column.

        Side Effects:
            - ``_fit_discrete`` and ``_fit_continuous`` should each be called once.
            - Assigns ``self._column_raw_dtypes`` the appropriate dtypes.
            - Assigns ``self.output_info_list`` the appropriate ``output_info``.
            - Assigns ``self.output_dimensions`` the appropriate ``output_dimensions``.
            - Assigns ``self._column_transform_info_list`` the appropriate
            ``column_transform_info``.
        """
        # Setup
        transformer = DataTransformer()
        transformer._fit_continuous = Mock()
        transformer._fit_continuous.return_value = ColumnTransformInfo(
            column_name='x', column_type='continuous', transform=None,
            output_info=[SpanInfo(1, 'tanh'), SpanInfo(3, 'softmax')],
            output_dimensions=1 + 3
        )

        transformer._fit_discrete = Mock()
        transformer._fit_discrete.return_value = ColumnTransformInfo(
            column_name='y', column_type='discrete', transform=None,
            output_info=[SpanInfo(2, 'softmax')],
            output_dimensions=2
        )

        data = pd.DataFrame({
            'x': np.random.random(size=100),
            'y': np.random.choice(['yes', 'no'], size=100)
        })

        # Run
        transformer.fit(data, discrete_columns=['y'])

        # Assert
        transformer._fit_discrete.assert_called_once()
        transformer._fit_continuous.assert_called_once()
        assert transformer.output_dimensions == 6

    @patch('ctgan.data_transformer.ClusterBasedNormalizer')
    def test__transform_continuous(self, MockCBN):
        """Test ``_transform_continuous``.

        Setup:
            - Mock the ``ClusterBasedNormalizer`` with the transform method returning
            some dataframe.
            - Create ``DataTransformer``.

        Input:
            - ``ColumnTransformInfo`` object.
            - A dataframe containing a continuous column.

        Output:
            - A np.array where the first column contains the normalized part
            of the mocked transform, and the other columns are a one hot encoding
            representation of the component part of the mocked transform.
        """
        # Setup
        cbn_instance = MockCBN.return_value
        cbn_instance.transform.return_value = pd.DataFrame({
            'x.normalized': [0.1, 0.2, 0.3],
            'x.component': [0.0, 1.0, 1.0]
        })

        transformer = DataTransformer()
        data = pd.DataFrame({'x': np.array([0.1, 0.3, 0.5])})
        column_transform_info = ColumnTransformInfo(
            column_name='x', column_type='continuous', transform=cbn_instance,
            output_info=[SpanInfo(1, 'tanh'), SpanInfo(3, 'softmax')],
            output_dimensions=1 + 3
        )

        # Run
        result = transformer._transform_continuous(column_transform_info, data)

        # Assert
        expected = np.array([
            [0.1, 1, 0, 0],
            [0.2, 0, 1, 0],
            [0.3, 0, 1, 0],
        ])
        np.testing.assert_array_equal(result, expected)

    def test_transform(self):
        """Test ``transform`` on a dataframe with one continuous and one discrete columns.

        It should use the appropriate ``_transform`` type for each column and should return
        them concanenated appropriately.

        Setup:
            - Initialize a ``DataTransformer`` with a ``column_transform_info`` detailing
            a continuous and a discrete columns.
            - Mock the ``_transform_discrete`` and ``_transform_continuous`` methods.

        Input:
            - A table with one continuous and one discrete columns.

        Output:
            - np.array containing the transformed columns.
        """
        # Setup
        data = pd.DataFrame({
            'x': np.array([0.1, 0.3, 0.5]),
            'y': np.array(['yes', 'yes', 'no'])
        })

        transformer = DataTransformer()
        transformer._column_transform_info_list = [
            ColumnTransformInfo(
                column_name='x', column_type='continuous', transform=None,
                output_info=[SpanInfo(1, 'tanh'), SpanInfo(3, 'softmax')],
                output_dimensions=1 + 3
            ),
            ColumnTransformInfo(
                column_name='y', column_type='discrete', transform=None,
                output_info=[SpanInfo(2, 'softmax')],
                output_dimensions=2
            )
        ]

        transformer._transform_continuous = Mock()
        selected_normalized_value = np.array([[0.1], [0.3], [0.5]])
        selected_component_onehot = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
        ])
        return_value = np.concatenate(
            (selected_normalized_value, selected_component_onehot), axis=1)
        transformer._transform_continuous.return_value = return_value

        transformer._transform_discrete = Mock()
        transformer._transform_discrete.return_value = np.array([
            [0, 1],
            [0, 1],
            [1, 0],
        ])

        # Run
        result = transformer.transform(data)

        # Assert
        expected = np.array([
            [0.1, 1, 0, 0, 0, 1],
            [0.3, 0, 1, 0, 0, 1],
            [0.5, 0, 1, 0, 1, 0],
        ])
        assert result.shape == (3, 6)
        assert (result[:, 0] == expected[:, 0]).all(), 'continuous-cdf'
        assert (result[:, 1:4] == expected[:, 1:4]).all(), 'continuous-softmax'
        assert (result[:, 4:6] == expected[:, 4:6]).all(), 'discrete'

    def test_parallel_sync_transform_same_output(self):
        """Test ``_parallel_transform`` and ``_synchronous_transform`` on a dataframe.

        The output of ``_parallel_transform`` should be the same as the output of
        ``_synchronous_transform``.

        Setup:
            - Initialize a ``DataTransformer`` with a ``column_transform_info`` detailing
            a continuous and a discrete columns.
            - Mock the ``_transform_discrete`` and ``_transform_continuous`` methods.

        Input:
            - A table with one continuous and one discrete columns.

        Output:
            - A list containing the transformed columns.
        """
        # Setup
        data = pd.DataFrame({
            'x': np.array([0.1, 0.3, 0.5]),
            'y': np.array(['yes', 'yes', 'no'])
        })

        transformer = DataTransformer()
        transformer._column_transform_info_list = [
            ColumnTransformInfo(
                column_name='x', column_type='continuous', transform=None,
                output_info=[SpanInfo(1, 'tanh'), SpanInfo(3, 'softmax')],
                output_dimensions=1 + 3
            ),
            ColumnTransformInfo(
                column_name='y', column_type='discrete', transform=None,
                output_info=[SpanInfo(2, 'softmax')],
                output_dimensions=2
            )
        ]

        transformer._transform_continuous = Mock()
        selected_normalized_value = np.array([[0.1], [0.3], [0.5]])
        selected_component_onehot = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
        ])
        return_value = np.concatenate(
            (selected_normalized_value, selected_component_onehot), axis=1)
        transformer._transform_continuous.return_value = return_value

        transformer._transform_discrete = Mock()
        transformer._transform_discrete.return_value = np.array([
            [0, 1],
            [0, 1],
            [1, 0],
        ])

        # Run
        parallel_result = transformer._parallel_transform(
            data,
            transformer._column_transform_info_list
        )
        sync_result = transformer._synchronous_transform(
            data,
            transformer._column_transform_info_list
        )
        parallel_result_np = np.concatenate(parallel_result, axis=1).astype(float)
        sync_result_np = np.concatenate(sync_result, axis=1).astype(float)

        # Assert
        assert len(parallel_result) == len(sync_result)
        np.testing.assert_array_equal(parallel_result_np, sync_result_np)

    @patch('ctgan.data_transformer.ClusterBasedNormalizer')
    def test__inverse_transform_continuous(self, MockCBN):
        """Test ``_inverse_transform_continuous``.

        Setup:
            - Create ``DataTransformer``.
            - Mock the ``ClusterBasedNormalizer`` where:
                - ``get_output_sdtypes`` returns the appropriate dictionary.
                - ``reverse_transform`` returns some dataframe.

        Input:
            - A ``ColumnTransformInfo`` object.
            - A np.ndarray where:
              - The first column contains the normalized value
              - The remaining columns correspond to the one-hot
            - sigmas = np.ndarray of floats
            - st = index of the sigmas ndarray

        Output:
            - Dataframe where the first column are floats and the second is a lable encoding.

        Side Effects:
            - The ``reverse_transform`` method should be called with a dataframe
            where the first column are floats and the second is a lable encoding.
        """
        # Setup
        cbn_instance = MockCBN.return_value
        cbn_instance.get_output_sdtypes.return_value = {
            'x.normalized': 'numerical',
            'x.component': 'numerical'
        }

        cbn_instance.reverse_transform.return_value = pd.DataFrame({
            'x.normalized': [0.1, 0.2, 0.3],
            'x.component': [0.0, 1.0, 1.0]
        })

        transformer = DataTransformer()
        column_data = np.array([
            [0.1, 1, 0, 0],
            [0.3, 0, 1, 0],
            [0.5, 0, 1, 0],
        ])

        column_transform_info = ColumnTransformInfo(
            column_name='x', column_type='continuous', transform=cbn_instance,
            output_info=[SpanInfo(1, 'tanh'), SpanInfo(3, 'softmax')],
            output_dimensions=1 + 3
        )

        # Run
        result = transformer._inverse_transform_continuous(
            column_transform_info, column_data, None, None)

        # Assert
        expected = pd.DataFrame({
            'x.normalized': [0.1, 0.2, 0.3],
            'x.component': [0.0, 1.0, 1.0]
        })

        np.testing.assert_array_equal(result, expected)

        expected_data = pd.DataFrame({
            'x.normalized': [0.1, 0.3, 0.5],
            'x.component': [0, 1, 1]
        })

        pd.testing.assert_frame_equal(
            cbn_instance.reverse_transform.call_args[0][0],
            expected_data
        )

    def test_inverse_transform(self):
        """Test ``inverse_transform`` on a np.ndarray with continuous and discrete columns.

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

    def test_convert_column_name_value_to_id(self):
        """Test ``convert_column_name_value_to_id`` on a simple ``_column_transform_info_list``.

        Tests that the appropriate indexes are returned when a table of three columns,
        discrete, continuous, discrete, is passed as '_column_transform_info_list'.

        Setup:
            - Mock ``_column_transform_info_list``.

        Input:
            - column_name = the name of a discrete column
            - value = the categorical value

        Output:
            - dictionary containing:
              - ``discrete_column_id`` = the index of the target column,
                when considering only discrete columns
              - ``column_id`` = the index of the target column
                (e.g. 3 = the third column in the data)
              - ``value_id`` = the index of the indicator value in the one-hot encoding
        """
        # Setup
        ohe = Mock()
        ohe.transform.return_value = pd.DataFrame([
            [0, 1]  # one hot encoding, second dimension
        ])
        transformer = DataTransformer()
        transformer._column_transform_info_list = [
            ColumnTransformInfo(
                column_name='x', column_type='continuous', transform=None,
                output_info=[SpanInfo(1, 'tanh'), SpanInfo(3, 'softmax')],
                output_dimensions=1 + 3
            ),
            ColumnTransformInfo(
                column_name='y', column_type='discrete', transform=ohe,
                output_info=[SpanInfo(2, 'softmax')],
                output_dimensions=2
            )
        ]

        # Run
        result = transformer.convert_column_name_value_to_id('y', 'yes')

        # Assert
        assert result['column_id'] == 1  # this is the 2nd column
        assert result['discrete_column_id'] == 0  # this is the 1st discrete column
        assert result['value_id'] == 1  # this is the 2nd dimension in the one hot encoding

    def test_convert_column_name_value_to_id_multiple(self):
        """Test ``convert_column_name_value_to_id``."""
        # Setup
        ohe = Mock()
        ohe.transform.return_value = pd.DataFrame([
            [0, 1, 0]  # one hot encoding, second dimension
        ])
        transformer = DataTransformer()
        transformer._column_transform_info_list = [
            ColumnTransformInfo(
                column_name='x', column_type='continuous', transform=None,
                output_info=[SpanInfo(1, 'tanh'), SpanInfo(3, 'softmax')],
                output_dimensions=1 + 3
            ),
            ColumnTransformInfo(
                column_name='y', column_type='discrete', transform=ohe,
                output_info=[SpanInfo(2, 'softmax')],
                output_dimensions=2
            ),
            ColumnTransformInfo(
                column_name='z', column_type='discrete', transform=ohe,
                output_info=[SpanInfo(2, 'softmax')],
                output_dimensions=2
            )
        ]

        # Run
        result = transformer.convert_column_name_value_to_id('z', 'yes')

        # Assert
        assert result['column_id'] == 2  # this is the 3rd column
        assert result['discrete_column_id'] == 1  # this is the 2nd discrete column
        assert result['value_id'] == 1  # this is the 1st dimension in the one hot encoding
