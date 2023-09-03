import sys

import yaml
from pathlib import Path
from argparse import ArgumentParser

from exp import Experiment, AttackPicker, ClsPicker
from exp.preproc import pred_parse
from exp.utility import to_namedtuple


def parse_args(parser: ArgumentParser):
    """Setup available program arguments."""
    parser.add_argument(
        dest='config',
        action='store',
        help='Experiment configuration file path',
    )
    parser.add_argument(
        '-v', '--validate',
        action='store_true',
        help='enforce constraints during attack'
    )
    parser.add_argument(
        '-a', '--attack',
        action='store',
        choices=AttackPicker.list_attacks(),
        help='evasion attack'
    )
    parser.add_argument(
        '-c', '--cls',
        action='store',
        choices=ClsPicker.list_cls(),
        help='classifier'
    )
    parser.add_argument(
        '-i', '--iter',
        type=int,
        choices=range(0, 501),
        metavar="0-500",
        help='max attack iterations, 0=default',
        default=-1
    )
    parser.add_argument(
        '-r', '--reset',
        type=int,
        choices=[1, 2],
        help='reset strategy: 1=all, 2=dependencies'
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


def read_yaml(file_path):
    with open(Path(file_path), 'r', encoding='utf-8') as open_yml:
        return yaml.safe_load(open_yml)


def build_config(args):
    c = read_yaml('./config/default.yaml')
    params = read_yaml(args.config)

    # merge the default config, experiment config, from files
    for k, v in params.items():
        c[k] = {**((c[k] or {}) if k in c else {}), **v} \
            if type(v) is dict else v
    config = {**c, 'config_path': args.config}

    # if defined, override file configs with command arguments
    if args.validate:
        config['validate'] = True
    attack_name = config['attack'] = \
        args.attack if args.attack else config['attack']
    if args.iter > -1:
        config[attack_name]['max_iter'] = args.iter
    config['cls'] = args.cls or config['cls']
    if args.reset > 0:
        config['reset_strategy'] = args.reset
    config = to_namedtuple(pred_parse(config))
    check_params(config)
    return config


if __name__ == '__main__':
    args_ = parse_args(ArgumentParser())
    config_ = build_config(args_)
    Experiment(config_).run()
