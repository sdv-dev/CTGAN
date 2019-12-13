import argparse
import json

import pandas as pd

from ctgan.data import read_data, write_data
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

    parser.add_argument('data', help='Path to training data')
    parser.add_argument('output', help='Path of the output file')

    return parser.parse_args()


def main():
    args = _parse_args()

    if args.tsv:
        data, metadata = read_data(args.data, args.metadata)
        discrete_columns = metadata['discrete_columns']
    else:
        if args.metadata:
            with open(args.metadata) as f:
                metadata = json.load(f)

            discrete_columns = [
                column['name']
                for column in metadata['columns']
                if column['type'] != 'continuous'
            ]

        elif args.discrete:
            discrete_columns = args.discrete.split(',')
            if not args.header:
                discrete_columns = [int(i) for i in discrete_columns]

        else:
            discrete_columns = []

        header = 'infer' if args.header else None
        data = pd.read_csv(args.data, header=header)

    model = CTGANSynthesizer()
    model.fit(data, discrete_columns, args.epochs)

    num_samples = args.num_samples or len(data)
    sampled = model.sample(num_samples)

    if args.tsv:
        write_data(sampled, metadata, args.output)
    else:
        sampled.to_csv(args.output, index=False)
