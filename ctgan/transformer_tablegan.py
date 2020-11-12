import numpy as np
import pandas as pd

from rdt.transformers import OneHotEncodingTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import BayesianGaussianMixture
#from sklearn.utils.testing import ignore_warnings

# NOTE: 2020-11-09. To fix FutureWarning. Use filterwarnings instead of ignore_warnings.
# from sklearn.utils.testing import ignore_warnings
import warnings
warnings.filterwarnings('ignore', category=ConvergenceWarning)



CATEGORICAL = "categorical"
CONTINUOUS = "continuous"
## ORDINAL = "ordinal" remove all parts about "ordinal"
## change all "categorical_columns" to "discrete_columns"



class Transformer:

    @staticmethod
    #def get_metadata(data, categorical_columns=tuple(), ordinal_columns=tuple()):
    def get_metadata(data, discrete_columns=tuple()):
        meta = []

        df = pd.DataFrame(data)
        for index in df: ##index is the column name
            column = df[index]

            # if index in categorical_columns:
            if index in discrete_columns: ##check whether column is categorical
                mapper = column.value_counts().index.tolist()
                meta.append({
                    "name": index,
                    "type": CATEGORICAL,
                    "size": len(mapper),
                    "i2s": mapper
                })
            # elif index in ordinal_columns:
            #     value_count = list(dict(column.value_counts()).items())
            #     value_count = sorted(value_count, key=lambda x: -x[1])
            #     mapper = list(map(lambda x: x[0], value_count))
            #     meta.append({
            #         "name": index,
            #         "type": ORDINAL,
            #         "size": len(mapper),
            #         "i2s": mapper
            #     })
            else:
                meta.append({
                    "name": index,
                    "type": CONTINUOUS,
                    "min": column.min(),
                    "max": column.max(),
                })

        return meta

    #def fit(self, data, categorical_columns=tuple(), ordinal_columns=tuple()):
    def fit(self, data, discrete_columns=tuple()):
        raise NotImplementedError

    def transform(self, data):
        raise NotImplementedError

    def inverse_transform(self, data):
        raise NotImplementedError


###This is min-max transformation
class TableganTransformer(Transformer):

    def __init__(self, side):
        self.height = side

    #def fit(self, data, categorical_columns=tuple(), ordinal_columns=tuple()):
    def fit(self, data, discrete_columns=tuple()):
        self.meta = self.get_metadata(data, discrete_columns)
        self.minn = np.zeros(len(self.meta))
        self.maxx = np.zeros(len(self.meta))
        for i in range(len(self.meta)):
            if self.meta[i]['type'] == CONTINUOUS:
                self.minn[i] = self.meta[i]['min'] - 1e-3
                self.maxx[i] = self.meta[i]['max'] + 1e-3
            else:
                self.minn[i] = -1e-3
                self.maxx[i] = self.meta[i]['size'] - 1 + 1e-3

    def transform(self, data):
        data = data.copy().astype('float32')
        data = (data - self.minn) / (self.maxx - self.minn) * 2 - 1 ##range [-1,1]
        if self.height * self.height > len(data[0]): ##decide the size of the square matrix
            padding = np.zeros((len(data), self.height * self.height - len(data[0])))
            data = np.concatenate([data, padding], axis=1)
        return data.reshape(-1, 1, self.height, self.height)

    def inverse_transform(self, data):
        data = data.reshape(-1, self.height * self.height)

        data_t = np.zeros([len(data), len(self.meta)])

        for id_, info in enumerate(self.meta):
            numerator = (data[:, id_].reshape([-1]) + 1)
            data_t[:, id_] = (numerator / 2) * (self.maxx[id_] - self.minn[id_]) + self.minn[id_]
            #if info['type'] in [CATEGORICAL, ORDINAL]:
            if info['type'] in CATEGORICAL:
                data_t[:, id_] = np.round(data_t[:, id_])

        return data_t


### VGM Transformation
class VGMTransformer(object):
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

    def __init__(self, n_clusters=10, epsilon=0.05):  ##change epsilon = 0.05
        self.n_clusters = n_clusters
        self.epsilon = epsilon

    # @ignore_warnings(category=ConvergenceWarning)
    def _fit_continuous(self, column, data):
        gm = BayesianGaussianMixture(
            self.n_clusters,
            weight_concentration_prior_type='dirichlet_process',
            weight_concentration_prior=0.001,
            n_init=1
        )
        gm.fit(data)
        components = gm.weights_ > self.epsilon
        num_components = components.sum()

        return {
            'name': column,
            'model': gm,
            "type": CONTINUOUS, ##added "type" since classifier will check the "type"
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
            "type": CATEGORICAL, ##added "type" since classifier will check the "type"
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

        data = np.concatenate(values, axis=1).astype(float)
        sides = [4, 8, 16, 24, 32, 40, 48] ##add 40 and 48 to accommodate OVS dataset

        for i in sides:
            if i * i >= data.shape[1]:  ##transform to square matrix
                self.height = i
                padding = np.zeros((len(data), self.height * self.height - len(data[0])))
                data = np.concatenate([data, padding], axis=1)
                break
        ## return square matrix and its height
        return data.reshape(-1, 1, self.height, self.height), self.height

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

    def inverse_transform(self, data):
        data = data.reshape(-1, self.height * self.height) ##needed to be further investigated
        start = 0
        output = []
        column_names = []
        sigmas = None
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


