from argparse import ArgumentParser

from exp.__main__ import build_config
from exp.utility import read_dataset
from exp.validation import Validation
from plot import plot_results, plot_graph

OPT_GRAPH = 'graph'
OPT_TABLE = 'table'


def shared_args(parser: ArgumentParser, path_help):
    parser.add_argument(
        dest='path',
        action='store',
        help=path_help,
    )
    parser.add_argument(
        '-o', '--out',
        action="store",
        dest="out",
        help="output directory path",
        default="result"
    )


def graph_plot(parser: ArgumentParser):
    parser.set_defaults(which=OPT_GRAPH)
    shared_args(parser, 'Configuration file')


def table_plot(parser: ArgumentParser):
    parser.set_defaults(which=OPT_TABLE)
    shared_args(parser, 'Results directory')


def parse_args(parser: ArgumentParser):
    """Setup available program arguments."""
    subparsers = parser.add_subparsers()
    graph_plot(subparsers.add_parser(
        name=OPT_GRAPH, help='plot graphs'))
    table_plot(subparsers.add_parser(
        name=OPT_TABLE, help='plot tables'))
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args(ArgumentParser())

    if args.which is OPT_TABLE:
        plot_results(args.path, args.out)

    elif args.which is OPT_GRAPH:
        conf = build_config(args)
        attrs = read_dataset(conf.dataset)[0]
        plot_graph(conf, attrs)
