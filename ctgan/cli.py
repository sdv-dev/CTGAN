import numpy as np
from absl import app, flags

from ctgan.model import CTGANSynthesizer

flags.DEFINE_string("data", "", "Filename of training data.")
flags.DEFINE_string("meta", "", "Filename of meta data.")
flags.DEFINE_string("model_dir", "", "Path to save model.")
flags.DEFINE_string("output", "", "Output filename.")
flags.DEFINE_integer("max_epoch", 100, "Epoches to train.")
flags.DEFINE_integer("sample", 1000, "Number of rows to generate.")

FLAGS = flags.FLAGS


def read_data(data_filename, meta_filename):
    with open(meta_filename) as f:
        column_info = f.readlines()

    column_info_raw = [
        x.replace("{", " ").replace("}", " ").split()
        for x in column_info
    ]

    discrete = []
    continuous = []
    column_info = []

    for idx, item in enumerate(column_info_raw):
        if item[0] == 'C':
            continuous.append(idx)
            column_info.append((float(item[1]), float(item[2])))
        else:
            assert item[0] == 'D'
            discrete.append(idx)
            column_info.append(item[1:])

    meta = {
        "continuous_columns": continuous,
        "discrete_columns": discrete,
        "column_info": column_info
    }

    with open(data_filename) as f:
        lines = f.readlines()

    data = []
    for row in lines:
        row_raw = row.split()
        row = []
        for idx, col in enumerate(row_raw):
            if idx in continuous:
                row.append(col)
            else:
                assert idx in discrete
                row.append(column_info[idx].index(col))

        data.append(row)

    return np.asarray(data, dtype='float32'), meta


def write_data(data, meta, output_filename):
    with open(output_filename, "w") as f:
        for row in data:
            for idx, col in enumerate(row):
                if idx in meta['continuous_columns']:
                    print(col, end=' ', file=f)
                else:
                    assert idx in meta['discrete_columns']
                    print(meta['column_info'][idx][int(col)], end=' ', file=f)

            print(file=f)


def _main(_):
    data, meta = read_data(FLAGS.data, FLAGS.meta)
    model = CTGANSynthesizer(epochs=FLAGS.max_epoch)
    model.fit(data, meta['discrete_columns'], tuple())
    data_syn = model.sample(FLAGS.sample)
    write_data(data_syn, meta, FLAGS.output)


def main():
    app.run(_main)


if __name__ == '__main__':
    main()
