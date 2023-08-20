import yaml
from pathlib import Path
from argparse import ArgumentParser

from exp import Experiment, AttackPicker
from exp.preproc import pred_parse


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
        help=f'evasion attack'
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


if __name__ == '__main__':
    BASE_CONFIG = './config/default.yaml'
    args = parse_args(ArgumentParser())
    # merge the default config, experiment config, from files
    c = yaml.safe_load(Path(BASE_CONFIG).read_text())
    params = yaml.safe_load(Path(args.config).read_text())
    for k, v in params.items():
        c[k] = {**((c[k] or {}) if k in c else {}), **v} \
            if type(v) is dict else v
    config = {**c, 'config_path': args.config,
              'validate': args.validate}
    # if defined, override file configs with command arguments
    attack_name = config['attack'] = \
        args.attack if args.attack else config['attack']
    if args.iter:
        config[attack_name]['max_iter'] = args.iter
    Experiment(pred_parse(config)).run()
