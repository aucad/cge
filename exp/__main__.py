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
        choices=range(0, 500),
        metavar="1-500",
        help='max attack iterations',
        default=0
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
    base_config = './config/default.yaml'

    # merge the default config, experiment config, from files
    with open(Path(base_config), 'r', encoding='utf-8') as open_yml:
        c = (yaml.safe_load(open_yml))
    with open(Path(args.config), 'r', encoding='utf-8') as open_yml:
        params = (yaml.safe_load(open_yml))
    for k, v in params.items():
        c[k] = {**((c[k] or {}) if k in c else {}), **v} \
            if type(v) is dict else v
    config = {**c, 'config_path': args.config}

    # if defined, override file configs with command arguments
    if args.validate:
        config['validate'] = True
    attack_name = config['attack'] = \
        args.attack if args.attack else config['attack']
    if args.iter:
        config[attack_name]['max_iter'] = args.iter
    if args.cls:
        config['cls'] = args.cls
    config = to_namedtuple(pred_parse(config))
    check_params(config)
    return config


if __name__ == '__main__':
    args_ = parse_args(ArgumentParser())
    config_ = build_config(args_)
    Experiment(config_).run()
