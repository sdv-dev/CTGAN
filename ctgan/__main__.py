import argparse

from .data import read_csv, read_tsv, write_tsv
from .synthesizer import CTGANSynthesizer


def _parse_args():
    parser = argparse.ArgumentParser(description='CTGAN Command Line Interface')
    parser.add_argument('-e', '--epochs', default=300, type=int,
                        help='Number of training epochs')
    parser.add_argument('-t', '--tsv', action='store_true',
                        help='Load data in TSV format instead of CSV')
    parser.add_argument('--no-header', dest='header', action='store_false',
                        help='The CSV file has no header. '
                        'Discrete columns will be indices.')

    parser.add_argument('-m', '--metadata', help='Path to the metadata')
    parser.add_argument('-d', '--discrete',
                        help='Comma separated list of discrete columns without '
                        'whitespaces.')

    parser.add_argument('-n', '--num-samples', type=int,
                        help='Number of rows to sample. Defaults to the '
                        'training data size.')

    parser.add_argument('--gen_lr', type=float, default=2e-4,
                        help='Learning rate for the generator.')
    parser.add_argument('--dis_lr', type=float, default=2e-4,
                        help='Learning rate for the discriminator.')

    parser.add_argument('--gen_decay', type=float, default=1e-6,
                        help='Weight decay for the generator.')
    parser.add_argument('--dis_decay', type=float, default=0,
                        help='Weight decay for the discriminator.')

    parser.add_argument('--z_dim', type=int, default=128,
                        help='Dimension of input z to the generator.')
    parser.add_argument('--gen_dims', type=str, default='256,256',
                        help='Dimension of each generator layer. '
                        'Comma separated integers with no whitespaces.')
    parser.add_argument('--dis_dims', type=str, default='256,256',
                        help='Dimension of each discriminator layer. '
                        'Comma separated integers with no whitespaces.')

    parser.add_argument('--bs', type=int, default=500,
                        help='Batch size. Must be an even number.')

    parser.add_argument('data', help='Path to training data')
    parser.add_argument('output', help='Path of the output file')

    return parser.parse_args()


def main():
    args = _parse_args()
    if args.tsv:
        data, discrete_columns = read_tsv(args.data, args.metadata)
    else:
        data, discrete_columns = read_csv(
            args.data, args.metadata, args.header, args.discrete)

    gen_dims = [int(x) for x in args.gen_dims.split(',')]
    dis_dims = [int(x) for x in args.dis_dims.split(',')]
    model = CTGANSynthesizer(
        z_dim=args.z_dim, gen_dims=gen_dims, dis_dims=dis_dims,
        gen_lr=args.gen_lr, gen_decay=args.gen_decay,
        dis_lr=args.dis_lr, dis_decay=args.dis_decay,
        batch_size=args.bs)
    model.fit(data, discrete_columns, args.epochs)

    num_samples = args.num_samples or len(data)
    sampled = model.sample(num_samples)

    if args.tsv:
        write_tsv(sampled, args.metadata, args.output)
    else:
        sampled.to_csv(args.output, index=False)


if __name__ == '__main__':
    main()
