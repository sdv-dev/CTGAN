from absl import app, flags

from ctgan.data import read_data, write_data
from ctgan.synthesizer import CTGANSynthesizer

flags.DEFINE_string("data", "", "Filename of training data.")
flags.DEFINE_string("meta", "", "Filename of meta data.")
flags.DEFINE_string("model_dir", "", "Path to save model.")
flags.DEFINE_string("output", "", "Output filename.")
flags.DEFINE_integer("max_epoch", 100, "Epoches to train.")
flags.DEFINE_integer("sample", 1000, "Number of rows to generate.")

FLAGS = flags.FLAGS


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
