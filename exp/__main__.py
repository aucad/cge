import yaml
from pathlib import Path

from exp import Experiment, Validation

if __name__ == '__main__':
    fp = './config/config_sm.yaml'
    config = yaml.safe_load(Path(fp).read_text())

    # because they are text
    if 'constraints' in config.keys():
        config['constraints'] = Validation.parse_pred(config['constraints'])

    Experiment({**config}).run()
