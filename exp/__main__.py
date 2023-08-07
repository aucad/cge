import yaml
from pathlib import Path
from argparse import ArgumentParser

from exp import Experiment
from exp.utility import pred_convert, read_dataset

BASE_CONFIG = './config/default.yaml'
DEFAULT_EXP = './config/iot23.yaml'


def parse_args(parser: ArgumentParser):
    """Setup available program arguments."""
    parser.add_argument(
        dest='config',
        action='store',
        default=DEFAULT_EXP,
        help='Experiment configuration file path',
    )
    parser.add_argument(
        '-v', '--validate',
        action='store_true',
        help='Actively enforce constraints during attack'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args(ArgumentParser())

    fp = args.config
    def_args = yaml.safe_load(Path(BASE_CONFIG).read_text())
    exp_args = yaml.safe_load(Path(fp).read_text())
    config = {**def_args, **exp_args, 'config_path': fp,
              'validate': args.validate}

    const = 'constraints'
    if const in config.keys():
        attrs, _ = read_dataset(config['dataset'])
        config['str_' + const] = config[const]
        config[const], config['str_func'] = (
            pred_convert(config[const], attrs) if
            config[const] is not None else ({}, {}))
    Experiment({**config}).run()
