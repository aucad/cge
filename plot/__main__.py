from argparse import ArgumentParser

from plot import plot_results, plot_graph

OPT_GRAPH = 'graph'
OPT_TABLE = 'table'


def shared_args(parser: ArgumentParser):
    parser.add_argument(
        '-o', '--out',
        action="store",
        dest="out",
        help="output directory path",
        default="result"
    )


def graph_plot(parser: ArgumentParser):
    parser.set_defaults(which=OPT_GRAPH)
    parser.add_argument(
        dest='path',
        action='store',
        help='Configuration file',
    )
    shared_args(parser)


def table_plot(parser: ArgumentParser):
    parser.set_defaults(which=OPT_TABLE)
    parser.add_argument(
        dest='path',
        action='store',
        help='Results directory',
    )
    shared_args(parser)


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
        from exp.experiment import Validation, Experiment
        from exp.__main__ import build_config

        exp = Experiment(build_config(args))
        exp.prepare_input_data()
        validation = Validation(
            exp.config.constraints, exp.attrs,
            exp.config.reset_strategy)
        plot_graph(validation, exp.config, exp.attrs)
