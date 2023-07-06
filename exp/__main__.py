import yaml
from pathlib import Path

from exp import Experiment

if __name__ == '__main__':
    c_args = {**(yaml.safe_load(Path('./config/config.yaml').read_text()))}
    Experiment(c_args).run()
