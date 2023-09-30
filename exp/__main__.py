import sys

from argparse import ArgumentParser

from exp import Experiment, AttackPicker, ClsPicker
from exp.preproc import pred_parse
from exp.utility import to_namedtuple, read_yaml
from exp.plot import plot_results


def parse_args(parser: ArgumentParser):
    """Setup available program arguments."""
    parser.add_argument(
        dest='path',
        action='store',
        help='Configuration file or results directory',
    )
    parser.add_argument(
        '-g', '--graph',
        action='store_true',
        help='Plot a constraint graph (without running experiment)'
    )
    parser.add_argument(
        '-p', '--plot',
        action='store_true',
        help='Plot results'
    )
    parser.add_argument(
        '-v', '--validate',
        action='store_true',
        help='enforce constraints during attack'
    )
    parser.add_argument(
        '-a',
        action='store',
        choices=AttackPicker.list_attacks(),
        help='evasion attack'
    )
    parser.add_argument(
        '-c',
        action='store',
        choices=ClsPicker.list_cls(),
        help='classifier'
    )
    parser.add_argument(
        '-i',
        type=int,
        choices=range(1, 501),
        metavar="1-500",
        help='max attack iterations, [default: 0]',
        default=-1
    )
    parser.add_argument(
        '--reset',
        type=int,
        choices=[1, 2],
        help='reset strategy: 1=all, 2=dependencies [default: 2]'
    )
    parser.add_argument(
        '--fn',
        type=str,
        action='store',
        help='expression to append to results file name',
        default=None,
    )
    return parser.parse_args()


def check_params(conf):
    def invalid(*msg):
        print(*msg, '-> terminating')
        sys.exit(0)

    if conf.cls == ClsPicker.XGB and conf.attack in \
            [AttackPicker.PDG, AttackPicker.CPGD]:
        invalid('Unsupported configuration:', conf.cls, conf.attack)
    if conf.attack == AttackPicker.CPGD and not conf.cpgd['feat_file']:
        invalid('Missing required configuration "cpgd.feat_file"')
    return True


def build_config(args):
    c = read_yaml('./config/default.yaml')
    params = read_yaml(args.path)

    # merge the default config, experiment config, from files
    for k, v in params.items():
        c[k] = {**((c[k] or {}) if k in c else {}), **v} \
            if type(v) is dict else v
    config = {**c, 'config_path': args.path}

    # if defined, override file configs with command arguments
    if args.validate:
        config['validate'] = True
    attack_name = config['attack'] = \
        args.a if args.a else config['attack']
    if args.i > 0:
        config[attack_name]['max_iter'] = args.i
    config['cls'] = args.c or config['cls']
    if args.reset and int(args.reset) > 0:
        config['reset_strategy'] = args.reset
    config['fn_pattern'] = args.fn if args.fn else None
    config = to_namedtuple(pred_parse(config))
    check_params(config)
    return config


if __name__ == '__main__':
    args_ = parse_args(ArgumentParser())

    if args_.plot:
        plot_results(args_.path)
        sys.exit(0)

    config_ = build_config(args_)
    if args_.graph:
        Experiment(config_).graph()
        sys.exit(0)

    Experiment(config_).run()
