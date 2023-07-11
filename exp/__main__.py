import sys
import yaml
from pathlib import Path

from exp import Experiment, Utility

DEFAULT_CONFIG = './config/config_sm.yaml'

if __name__ == '__main__':
    fp = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG
    config = yaml.safe_load(Path(fp).read_text())

    # parse the predicates, because they are text
    const = 'constraints'
    if const in config.keys():
        config[const] = Utility.parse_pred(config[const]) \
            if config[const] is not None else {}

    Experiment({**config}).run()
