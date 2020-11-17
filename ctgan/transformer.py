import pickle

import numpy as np
import pandas as pd
from rdt.transformers import OneHotEncodingTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import BayesianGaussianMixture

# NOTE: 2020-11-09. To fix FutureWarning. Use filterwarnings instead of ignore_warnings.
# from sklearn.utils.testing import ignore_warnings
import warnings
warnings.filterwarnings('ignore', category=ConvergenceWarning)


class DataTransformer(object):
    """Data Transformer.

    Model continuous columns with a BayesianGMM and normalized to a scalar
    [0, 1] and a vector.
    Discrete columns are encoded using a scikit-learn OneHotEncoder.

    Args:
        n_cluster (int):
            Number of modes.
        epsilon (float):
            Epsilon value.
    """

    def __init__(self, n_clusters=10, epsilon=0.005):
        self.n_clusters = n_clusters
        self.epsilon = epsilon
        self.side = 0  # for tablegan
        self.datalen = 0  # for tablegan

    # @ignore_warnings(category=ConvergenceWarning)
    def _fit_continuous(self, column, data):
        # (JY)
        # 'gm' gives approximate parameters of a Gaussian mixture distribution
        # 'epsilon' is threshold to prevent mixtures with many negligible components
        # 'weight_concentration_prior': the higher concentration puts more mass in
        # the centre and will lead to more components being active,
        # a lower concentration parameter will lead to more mass at the edge
        # of the mixture weights simplex
        gm = BayesianGaussianMixture(
            self.n_clusters,
            weight_concentration_prior_type='dirichlet_process',
            weight_concentration_prior=0.001,
            n_init=1
        )
        # (JY)
        # 'gm.fit(data)' estimates model parameters using 'data' and predict labels
        # for 'data'; fits the model 'n_init' times and sets parameters with which
        # the model has largest likelihood or lower bound; after fitting, predicts
        # the most probable label for the input data points
        gm.fit(data)
        components = gm.weights_ > self.epsilon
        # (JY) 'num_components' is the optimal number of modes identified
        num_components = components.sum()

        return {
            'name': column,
            'model': gm,
            'components': components,
            'output_info': [(1, 'tanh'), (num_components, 'softmax')],
            'output_dimensions': 1 + num_components,
        }

    def _fit_discrete(self, column, data):
        ohe = OneHotEncodingTransformer()
        data = data[:, 0]
        ohe.fit(data)
        categories = len(set(data))

        return {
            'name': column,
            'encoder': ohe,
            'output_info': [(categories, 'softmax')],
            'output_dimensions': categories
        }

    def fit(self, data, discrete_columns=tuple()):
        self.output_info = []
        self.output_dimensions = 0

        if not isinstance(data, pd.DataFrame):
            self.dataframe = False
            data = pd.DataFrame(data)
        else:
            self.dataframe = True

        self.dtypes = data.infer_objects().dtypes
        self.meta = []
        for column in data.columns:
            column_data = data[[column]].values
            if column in discrete_columns:
                meta = self._fit_discrete(column, column_data)
            else:
                meta = self._fit_continuous(column, column_data)

            self.output_info += meta['output_info']
            self.output_dimensions += meta['output_dimensions']
            self.meta.append(meta)

    def _transform_continuous(self, column_meta, data):
        components = column_meta['components']
        model = column_meta['model']

        means = model.means_.reshape((1, self.n_clusters))
        stds = np.sqrt(model.covariances_).reshape((1, self.n_clusters))
        features = (data - means) / (4 * stds)

        probs = model.predict_proba(data)

        n_opts = components.sum()
        features = features[:, components]
        probs = probs[:, components]

        opt_sel = np.zeros(len(data), dtype='int')
        for i in range(len(data)):
            pp = probs[i] + 1e-6
            pp = pp / pp.sum()
            opt_sel[i] = np.random.choice(np.arange(n_opts), p=pp)

        idx = np.arange((len(features)))
        features = features[idx, opt_sel].reshape([-1, 1])
        features = np.clip(features, -.99, .99)

        probs_onehot = np.zeros_like(probs)
        probs_onehot[np.arange(len(probs)), opt_sel] = 1
        return [features, probs_onehot]

    def _transform_discrete(self, column_meta, data):
        encoder = column_meta['encoder']
        return encoder.transform(data[:, 0])

    def transform(self, data):
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        values = []
        for meta in self.meta:
            column_data = data[[meta['name']]].values
            if 'model' in meta:
                values += self._transform_continuous(meta, column_data)
            else:
                values.append(self._transform_discrete(meta, column_data))

        return np.concatenate(values, axis=1).astype(float)

    # For tablegan, there is an additional transformation of training data to square matrices.
    def transform_tablegan(self, data):
        print('shape of input data:', data.shape)
        data = self.transform(data)
        print('shape of data after VGM transformation:', data.shape)
        self.datalen = data.shape[1]

        sides = [4, 8, 16, 24, 32, 48]  # added 48 to accommodate OVS dataset
        for i in sides:
            if i * i >= self.datalen:
                self.side = i
                break

        data = data.copy().astype('float32')
        if self.side * self.side > len(data[1]):
            padding = np.zeros((len(data), self.side * self.side - len(data[1])))
            data = np.concatenate([data, padding], axis=1)
        return data.reshape(-1, 1, self.side, self.side)

    def _inverse_transform_continuous(self, meta, data, sigma):
        model = meta['model']
        components = meta['components']

        u = data[:, 0]
        v = data[:, 1:]

        if sigma is not None:
            u = np.random.normal(u, sigma)

        u = np.clip(u, -1, 1)
        v_t = np.ones((len(data), self.n_clusters)) * -100
        v_t[:, components] = v
        v = v_t
        means = model.means_.reshape([-1])
        stds = np.sqrt(model.covariances_).reshape([-1])
        p_argmax = np.argmax(v, axis=1)
        std_t = stds[p_argmax]
        mean_t = means[p_argmax]
        column = u * 4 * std_t + mean_t

        return column

    def _inverse_transform_discrete(self, meta, data):
        encoder = meta['encoder']
        return encoder.reverse_transform(data)

    def inverse_transform(self, data, sigmas):
        start = 0
        output = []
        column_names = []
        for meta in self.meta:
            dimensions = meta['output_dimensions']
            columns_data = data[:, start:start + dimensions]

            if 'model' in meta:
                # NOTE: 2020-11-09.
                # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
                # sigma = sigmas[start] if sigmas else None
                if sigmas is not None:
                    sigma = sigmas[start]
                else:
                    sigma = None
                inverted = self._inverse_transform_continuous(meta, columns_data, sigma)
            else:
                inverted = self._inverse_transform_discrete(meta, columns_data)

            output.append(inverted)
            column_names.append(meta['name'])
            start += dimensions

        output = np.column_stack(output)
        output = pd.DataFrame(output, columns=column_names).astype(self.dtypes)
        if not self.dataframe:
            output = output.values

        return output

    def save(self, path):
        with open(path + "/data_transform.pl", "wb") as f:
            pickle.dump(self, f)

    def covert_column_name_value_to_id(self, column_name, value):
        discrete_counter = 0
        column_id = 0
        for info in self.meta:
            if info["name"] == column_name:
                break
            if len(info["output_info"]) == 1:  # is discrete column
                discrete_counter += 1
            column_id += 1

        return {
            "discrete_column_id": discrete_counter,
            "column_id": column_id,
            "value_id": np.argmax(info["encoder"].transform(np.array([value]))[0])
        }

    @classmethod
    def load(cls, path):
        with open(path + "/data_transform.pl", "rb") as f:
            return pickle.load(f)
