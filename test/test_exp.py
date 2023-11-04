import numpy as np

from exp import Experiment
from exp.preproc import pred_convert
from exp.utility import to_namedtuple, read_yaml


def test_run(mocker):
    config = to_namedtuple({
        **read_yaml('./config/default.yaml'),
        **{"name": "test_run",
           "dataset": "data/ctu_1-8-34.csv",
           "desc": "description",
           "constraints": pred_convert([], [], [])[0]}})

    mocker.patch('exp.model.BaseModel.train')
    mocker.patch('exp.attack.AttackRunner.run')
    mocker.patch('exp.scoring.ModelScore.calculate')
    mocker.patch('yaml.dump')
    mocker.patch('sys.print')

    res = Experiment(config).run().to_dict()

    assert res["experiment"]["name"] is "test_run"
    assert res["experiment"]["dataset"] is "data/ctu_1-8-34.csv"
    assert res["experiment"]["description"] is "description"
