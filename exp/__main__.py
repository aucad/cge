import sys
import yaml
from pathlib import Path

from exp import Experiment, Utility

BASE_CONFIG = './config/base.yaml'
DEFAULT_EXP = './config/config_lgT.yaml'

if __name__ == '__main__':
    fp = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_EXP
    def_args = yaml.safe_load(Path(BASE_CONFIG).read_text())
    exp_args = yaml.safe_load(Path(fp).read_text())
    config = {**def_args, **exp_args, 'config_path': fp}

    # parse the predicates, because they are text
    const = 'constraints'
    if const in config.keys():
        config['str_' + const] = config[const]
        config[const] = Utility.parse_pred(config[const]) \
            if config[const] is not None else {}

    Experiment({**config}).run()
