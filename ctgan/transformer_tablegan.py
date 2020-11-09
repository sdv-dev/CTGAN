import numpy as np
import pandas as pd

CATEGORICAL = "categorical"
CONTINUOUS = "continuous"
ORDINAL = "ordinal"


class Transformer:

    @staticmethod
    def get_metadata(data, categorical_columns=tuple(), ordinal_columns=tuple()):
        meta = []

        df = pd.DataFrame(data)
        for index in df:
            column = df[index]

            # if index in categorical_columns:
            if index in categorical_columns:
                mapper = column.value_counts().index.tolist()
                meta.append({
                    "name": index,
                    "type": CATEGORICAL,
                    "size": len(mapper),
                    "i2s": mapper
                })
            elif index in ordinal_columns:
                value_count = list(dict(column.value_counts()).items())
                value_count = sorted(value_count, key=lambda x: -x[1])
                mapper = list(map(lambda x: x[0], value_count))
                meta.append({
                    "name": index,
                    "type": ORDINAL,
                    "size": len(mapper),
                    "i2s": mapper
                })
            else:
                meta.append({
                    "name": index,
                    "type": CONTINUOUS,
                    "min": column.min(),
                    "max": column.max(),
                })

        return meta

    def fit(self, data, categorical_columns=tuple(), ordinal_columns=tuple()):
        raise NotImplementedError

    def transform(self, data):
        raise NotImplementedError

    def inverse_transform(self, data):
        raise NotImplementedError


class TableganTransformer(Transformer):

    def __init__(self, side):
        self.height = side

    def fit(self, data, categorical_columns=tuple(), ordinal_columns=tuple()):
        self.meta = self.get_metadata(data, categorical_columns, ordinal_columns)
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
        data = (data - self.minn) / (self.maxx - self.minn) * 2 - 1
        if self.height * self.height > len(data[0]):
            padding = np.zeros((len(data), self.height * self.height - len(data[0])))
            data = np.concatenate([data, padding], axis=1)
        return data.reshape(-1, 1, self.height, self.height)

    def inverse_transform(self, data):
        data = data.reshape(-1, self.height * self.height)

        data_t = np.zeros([len(data), len(self.meta)])

        for id_, info in enumerate(self.meta):
            numerator = (data[:, id_].reshape([-1]) + 1)
            data_t[:, id_] = (numerator / 2) * (self.maxx[id_] - self.minn[id_]) + self.minn[id_]
            if info['type'] in [CATEGORICAL, ORDINAL]:
                data_t[:, id_] = np.round(data_t[:, id_])

        return data_t
