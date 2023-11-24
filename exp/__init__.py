# flake8: noqa: F401

"""
Adversarial machine learning with constraints.
"""

__title__ = 'Constrained AML attacks'
__author__ = "@nkrusch and @Nour-Alhussien"
__license__ = 'MIT'
__version__ = '1.0.0'

import os
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# noinspection PyPep8
from cge import Validatable, CONSTR_DICT, Validation
from exp.machine import machine_details
from exp.utility import ensure_dir
from exp.scoring import Result, ModelScore, AttackScore, score_valid
from exp.model import BaseModel
from exp.xgb import XGBoost
from exp.dnn import DeepNeuralNetwork
from exp.classifier import ClsPicker
from exp.cpgd import CPGD, cpgd_apply_and_predict
from exp.zoo import VZoo
from exp.pgd import VPGD
from exp.hopskip import VHSJ
from exp.attack import AttackRunner, AttackPicker
from exp.experiment import Experiment
