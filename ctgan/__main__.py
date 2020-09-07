import argparse

from ctgan.data import read_csv, read_tsv, write_tsv
from ctgan.synthesizer import CTGANSynthesizer


def _parse_args():
    parser = argparse.ArgumentParser(description='CTGAN Command Line Interface')
    parser.add_argument('-e', '--epochs', default=300, type=int,
                        help='Number of training epochs')
    parser.add_argument('-t', '--tsv', action='store_true',
                        help='Load data in TSV format instead of CSV')
    parser.add_argument('--no-header', dest='header', action='store_false',
                        help='The CSV file has no header. Discrete columns will be indices.')

    parser.add_argument('-m', '--metadata', help='Path to the metadata')
    parser.add_argument('-d', '--discrete',
                        help='Comma separated list of discrete columns, no whitespaces')

    parser.add_argument('-n', '--num-samples', type=int,
                        help='Number of rows to sample. Defaults to the training data size')

    parser.add_argument('--save', default=None, type=str,
                        help='A filename to save the trained synthesizer.')
    parser.add_argument('--load', default=None, type=str,
                        help='A filename to load a trained synthesizer.')

    parser.add_argument("--sample_condition_column", default=None, type=str,
                        help="Select a discrete column name.")
    parser.add_argument("--sample_condition_column_value", default=None, type=str,
                        help="Specify the value of the selected discrete column.")

    parser.add_argument('data', help='Path to training data')
    parser.add_argument('output', help='Path of the output file')

    return parser.parse_args()


def main():
    args = _parse_args()

    if args.tsv:
        data, discrete_columns = read_tsv(args.data, args.metadata)
    else:
        data, discrete_columns = read_csv(args.data, args.metadata, args.header, args.discrete)

    if args.load:
        model = CTGANSynthesizer.load(args.load)
    else:
        model = CTGANSynthesizer()
    model.fit(data, discrete_columns, args.epochs)

    if args.save is not None:
        model.save(args.save)

    num_samples = args.num_samples or len(data)

    if args.sample_condition_column is not None:
        assert args.sample_condition_column_value is not None

    sampled = model.sample(
        num_samples,
        args.sample_condition_column,
        args.sample_condition_column_value)

    if args.tsv:
        write_tsv(sampled, args.metadata, args.output)
    else:
        sampled.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
