"""DataTransformer module."""

from collections import namedtuple
import pickle
import time

import numpy as np
import pandas as pd
from rdt.transformers import BayesGMMTransformer, OneHotEncodingTransformer

SpanInfo = namedtuple('SpanInfo', ['dim', 'activation_fn'])
ColumnTransformInfo = namedtuple(
    'ColumnTransformInfo', [
        'column_name', 'column_type', 'transform', 'output_info', 'output_dimensions'
    ]
)


class DataTransformer_v2(object):
    """Data Transformer.

    Model continuous columns with a BayesianGMM and normalized to a scalar [0, 1] and a vector.
    Discrete columns are encoded using a scikit-learn OneHotEncoder.
    """

    def __init__(self, max_clusters=10, weight_threshold=0.005):
        """Create a data transformer.

        Args:
            max_clusters (int):
                Maximum number of Gaussian distributions in Bayesian GMM.
            weight_threshold (float):
                Weight threshold for a Gaussian distribution to be kept.
        """
        self._max_clusters = max_clusters
        self._weight_threshold = weight_threshold

    def _fit_continuous(self, data):
        """Train Bayesian GMM for continuous columns.

        Args:
            data (pd.DataFrame):
                A dataframe containing a column.

        Returns:
            namedtuple:
                A ``ColumnTransformInfo`` object.
        """
        column_name = data.columns[0]
        gm = BayesGMMTransformer(max_clusters=min(len(data), 10))
        gm.fit(data, [column_name])
        num_components = sum(gm.valid_component_indicator)

        return ColumnTransformInfo(
            column_name=column_name, column_type='continuous', transform=gm,
            output_info=[SpanInfo(1, 'tanh'), SpanInfo(num_components, 'softmax')],
            output_dimensions=1 + num_components)

    def _fit_discrete(self, data):
        """Fit one hot encoder for discrete column.

        Args:
            data (pd.DataFrame):
                A dataframe containing a column.

        Returns:
            namedtuple:
                A ``ColumnTransformInfo`` object.
        """
        column_name = data.columns[0]
        ohe = OneHotEncodingTransformer()
        ohe.fit(data, [column_name])
        num_categories = len(ohe.dummies)

        return ColumnTransformInfo(
            column_name=column_name, column_type='discrete', transform=ohe,
            output_info=[SpanInfo(num_categories, 'softmax')],
            output_dimensions=num_categories)

    def fit(self, raw_data_path: str, raw_data_info_path: str):
        """Fit the ``DataTransformer``.

        Fits a ``BayesGMMTransformer`` for continuous columns and a
        ``OneHotEncodingTransformer`` for discrete columns.

        This step also counts the #columns in matrix data and span information.
        """
        self.output_info_list = []
        self.output_dimensions = 0
        self.dataframe = False

        # work around for RDT issue #328 Fitting with numerical column names fails

        data_info = pd.read_csv(raw_data_info_path)
        assert np.isin(["features", "dtype"], list(data_info.columns)).all()

        cat_dtype = ["object", "category"]
        discrete_columns = data_info.features[data_info["dtype"].isin(cat_dtype)].values
        discrete_columns = [str(column) for column in discrete_columns]

        self._column_raw_dtypes = data_info["dtype"].values
        self._column_transform_info_list = []
        for column_name in data_info["features"].values:

            start = time.time()
            tmp_dtype = df_info.loc[df_info.features == column_name,"dtype"].values[0]
            tmp_data = pd.concat(pd.read_csv(raw_data_path, 
                                            chunksize=50000, 
                                            usecols=[column_name], 
                                            dtype=tmp_dtype))
            end = time.time()
            print((end-start)/60, " mins to load ", column_name)

            start = time.time()
            if column_name in discrete_columns:
                column_transform_info = self._fit_discrete(tmp_data)
            else:
                column_transform_info = self._fit_continuous(tmp_data)
            
            end = time.time()
            print((end-start)/60, " mins to train encoder for ", column_name)

            self.output_info_list.append(column_transform_info.output_info)
            self.output_dimensions += column_transform_info.output_dimensions
            self._column_transform_info_list.append(column_transform_info)

    def _transform_continuous(self, column_transform_info, data):
        column_name = data.columns[0]
        data[column_name] = data[column_name].to_numpy().flatten()
        gm = column_transform_info.transform
        transformed = gm.transform(data, [column_name])

        #  Converts the transformed data to the appropriate output format.
        #  The first column (ending in '.normalized') stays the same,
        #  but the lable encoded column (ending in '.component') is one hot encoded.
        output = np.zeros((len(transformed), column_transform_info.output_dimensions))
        output[:, 0] = transformed[f'{column_name}.normalized'].to_numpy()
        index = transformed[f'{column_name}.component'].to_numpy().astype(int)
        output[np.arange(index.size), index + 1] = 1.0

        return output

    def _transform_discrete(self, column_transform_info, data):
        ohe = column_transform_info.transform
        return ohe.transform(data).to_numpy()

    def transform(self, raw_data_path: str, raw_data_info_path: str):
        """Take raw data and output a matrix data."""

        data_info = pd.read_csv(raw_data_info_path)
        assert np.isin(["features", "dtype"], list(data_info.columns)).all()

        column_data_list = []
        for column_transform_info in self._column_transform_info_list:
            
            column_name = column_transform_info.column_name
            tmp_dtype = df_info.loc[df_info.features == column_name,"dtype"].values[0]
            data = pd.concat(pd.read_csv(raw_data_path, 
                                            chunksize=50000, 
                                            usecols=[column_name], 
                                            dtype=tmp_dtype))

            if column_transform_info.column_type == 'continuous':
                column_data_list.append(self._transform_continuous(column_transform_info, data))
            else:
                column_data_list.append(self._transform_discrete(column_transform_info, data))

        del data
        with open(raw_data_path[:-4]+"_list", 'wb') as fp:
            pickle.dump(column_data_list, fp)

        return np.concatenate(column_data_list, axis=1).astype(float)

    def _inverse_transform_continuous(self, column_transform_info, column_data, sigmas, st):
        gm = column_transform_info.transform
        data = pd.DataFrame(column_data[:, :2], columns=list(gm.get_output_types()))
        data.iloc[:, 1] = np.argmax(column_data[:, 1:], axis=1)
        if sigmas is not None:
            selected_normalized_value = np.random.normal(data.iloc[:, 0], sigmas[st])
            data.iloc[:, 0] = selected_normalized_value

        return gm.reverse_transform(data, [column_transform_info.column_name])

    def _inverse_transform_discrete(self, column_transform_info, column_data):
        ohe = column_transform_info.transform
        data = pd.DataFrame(column_data, columns=list(ohe.get_output_types()))
        return ohe.reverse_transform(data)[column_transform_info.column_name]

    def inverse_transform(self, data, sigmas=None):
        """Take matrix data and output raw data.

        Output uses the same type as input to the transform function.
        Either np array or pd dataframe.
        """
        st = 0
        recovered_column_data_list = []
        column_names = []
        for column_transform_info in self._column_transform_info_list:
            dim = column_transform_info.output_dimensions
            column_data = data[:, st:st + dim]
            if column_transform_info.column_type == 'continuous':
                recovered_column_data = self._inverse_transform_continuous(
                    column_transform_info, column_data, sigmas, st)
            else:
                recovered_column_data = self._inverse_transform_discrete(
                    column_transform_info, column_data)

            recovered_column_data_list.append(recovered_column_data)
            column_names.append(column_transform_info.column_name)
            st += dim

        recovered_data = np.column_stack(recovered_column_data_list)
        recovered_data = (pd.DataFrame(recovered_data, columns=column_names)
                          .astype(self._column_raw_dtypes))
        if not self.dataframe:
            recovered_data = recovered_data.to_numpy()

        return recovered_data

    def convert_column_name_value_to_id(self, column_name, value):
        """Get the ids of the given `column_name`."""
        discrete_counter = 0
        column_id = 0
        for column_transform_info in self._column_transform_info_list:
            if column_transform_info.column_name == column_name:
                break
            if column_transform_info.column_type == 'discrete':
                discrete_counter += 1

            column_id += 1

        else:
            raise ValueError(f"The column_name `{column_name}` doesn't exist in the data.")

        ohe = column_transform_info.transform
        data = pd.DataFrame([value], columns=[column_transform_info.column_name])
        one_hot = ohe.transform(data).to_numpy()[0]
        if sum(one_hot) == 0:
            raise ValueError(f"The value `{value}` doesn't exist in the column `{column_name}`.")

        return {
            'discrete_column_id': discrete_counter,
            'column_id': column_id,
            'value_id': np.argmax(one_hot)
        }


if __name__ == '__main__':

    #Testing Large data set with 

    TEST_DATA_PATH = "X:\\4xtra\\4xtra_LBS_model\\model_layer\\Users\\lbs_all\\Data\\final_extra.csv"
    import pandas as pd
    import numpy as np
    import time

    df_info = pd.read_csv(TEST_DATA_PATH[:-4]+"_info.csv")

    df = pd.concat(pd.read_csv( TEST_DATA_PATH, chunksize=60000, 
                                usecols=df_info["features"].values, 
                                dtype=df_info["dtype"].values))

    categorical = df_info["features"][df_info.dtype == "object"].values
    transformer = DataTransformer_v2()

    start = time.time()
    transformer.fit(TEST_DATA_PATH, TEST_DATA_PATH[:-4]+"_info.csv")
    end = time.time()
    dif = end - start
    print(dif//60, "mins")

    # with open(TEST_DATA_PATH[:-4]+"_DTv2.pickle", 'wb') as fp:
    #     pickle.dump(transformer, fp)


    with open(TEST_DATA_PATH[:-4]+"_DTv2.pickle", 'rb') as fp:
        tmp = pickle.load(fp)


    tmp._column_transform_info_list
