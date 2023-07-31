import yaml
from pathlib import Path
from argparse import ArgumentParser

from exp import Experiment, Utility

BASE_CONFIG = './config/base.yaml'
DEFAULT_EXP = './config/large.yaml'


def parse_args(parser: ArgumentParser, args=None):
    """Setup available program arguments."""
    parser.add_argument(
        dest="config",
        action="store",
        default=DEFAULT_EXP,
        help=f'Experiment configuration file',
    )
    parser.add_argument(
        "-v", "--validate",
        action='store_true',
        help="Enforce constraints during attack"
    )
    return parser.parse_args(args)


if __name__ == '__main__':
    parser = ArgumentParser()
    args = parse_args(parser)

    fp = args.config
    def_args = yaml.safe_load(Path(BASE_CONFIG).read_text())
    exp_args = yaml.safe_load(Path(fp).read_text())
    config = {**def_args, **exp_args, 'config_path': fp,
              'validate': args.validate}

    # parse the predicates, because they are text
    const = 'constraints'
    if const in config.keys():
        config['str_' + const] = config[const]
        config[const] = Utility.parse_pred(config[const]) \
            if config[const] is not None else {}

    Experiment({**config}).run()
