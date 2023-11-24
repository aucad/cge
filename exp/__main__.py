import sys
from argparse import ArgumentParser

from cge.validation import ALL as RESET_ALL
from exp import Experiment, AttackPicker, ClsPicker
from exp.preproc import pred_parse
from exp.utility import to_namedtuple, read_yaml


def parse_args(parser: ArgumentParser):
    """Setup available program arguments."""
    parser.add_argument(
        dest='path',
        action='store',
        help='Configuration file',
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
        dest="attack",
        help='evasion attack'
    )
    parser.add_argument(
        '-c',
        action='store',
        choices=ClsPicker.list_cls(),
        dest="cls",
        help='classifier'
    )
    parser.add_argument(
        '-i',
        type=int,
        choices=range(0, 501),
        metavar="0-500",
        help='max attack iterations, [default: 0]',
        dest="iter",
        default=-1
    )
    parser.add_argument(
        '--reset_all',
        action='store_true',
        help='for non-valid record, reset all features to original'
    )
    parser.add_argument(
        '--fn',
        type=str,
        action='store',
        help='expression to append to results file name',
        default=None,
    )
    parser.add_argument(
        '-o', '--out',
        action="store",
        dest="out",
        help="output directory path",
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
    if 'validate' in args and args.validate:
        config['validate'] = True
    config['cpgd']['args']['enable_constraints'] = \
        config['validate']
    attack_name = config['attack'] = \
        (args.attack if 'attack' in args and args.attack
         else config['attack'])
    if 'iter' in args and args.iter > 0:
        config[attack_name]['max_iter'] = args.iter
    if 'cls' in args:
        config['cls'] = args.cls or config['cls']
    config['out'] = args.out or config['out']
    if 'reset_all' in args and args.reset_all:
        config['reset_strategy'] = RESET_ALL
    config['fn_pattern'] = args.fn if 'fn' in args and args.fn else None
    config = to_namedtuple(pred_parse(config))
    check_params(config)
    return config


if __name__ == '__main__':
    args_ = parse_args(ArgumentParser())
    config_ = build_config(args_)
    Experiment(config_).run()
