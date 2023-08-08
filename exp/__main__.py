import yaml
from pathlib import Path
from argparse import ArgumentParser

from exp import Experiment
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
        help='Actively enforce constraints during attack'
    )
    return parser.parse_args()


if __name__ == '__main__':
    BASE_CONFIG = './config/default.yaml'
    args = parse_args(ArgumentParser())
    def_args = yaml.safe_load(Path(BASE_CONFIG).read_text())
    exp_args = yaml.safe_load(Path(args.config).read_text())
    config = pred_parse({
        **def_args, **exp_args,
        'config_path': args.config,
        'validate': args.validate})
    Experiment(config).run()
