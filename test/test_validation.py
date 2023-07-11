import numpy as np
from exp.validation import Validation


def test_no_constraints():
    ori = np.array([[1, 2, 3], [5, 6, 7]])
    adv = np.array([[6, 7, 1], [3, 1, 2]])
    result = Validation().reset(ori).enforce(adv)
    assert (result == adv).all()


def test_immutable_all():
    ori = np.array([[.6, .4, .3, .3], [.3, .4, .5, .1]])
    adv = np.array([[.2, .4, .6, .0], [.7, .6, .6, .3]])

    result = Validation(immutable=[0, 1, 2, 3]).reset(ori).enforce(adv)
    assert (result == ori).all()


def test_immutable_1():
    ori = np.array([[.2, .5, .6], [.6, .5, .4], [.1, .2, .3]])
    adv = np.array([[.2, .4, .6], [.0, .5, .2], [.6, .6, .3]])
    exp = np.array([[.2, .5, .6], [.0, .5, .2], [.6, .2, .3]])

    result = Validation(immutable=[1]).reset(ori).enforce(adv)
    assert (result == exp).all()


def test_immutable_2():
    ori = np.array([[.0, .3, .4, .5, .6, .7], [.2, .8, .4, .7, .6, .1]])
    adv = np.array([[.5, .7, .8, .1, .9, .1], [.8, .3, .2, .5, .3, .2]])
    exp = np.array([[.5, .7, .4, .1, .9, .7], [.8, .3, .4, .5, .3, .1]])

    result = Validation(immutable=[2, 5]).reset(ori).enforce(adv)
    assert (result == exp).all()


def test_bin_feature():
    ori = np.array([[1.], [0.], [1.], [1.], [0.], [1.]])
    adv = np.array([[.8], [1.], [.4], [0.], [.2], [1.]])
    exp = np.array([[1.], [1.], [1.], [0.], [0.], [1.]])
    constraints = {0: lambda x: x == 0 or x == 1}
    result = Validation(constraints=constraints).reset(ori).enforce(adv)
    assert (result == exp).all()


def test_single_feature():
    ori = np.array([[.0, .2, .4], [.5, .7, .2], [.3, .2, .6], [.9, .8, .8]])
    adv = np.array([[.0, .1, .2], [.5, .9, .6], [.3, .4, .9], [.9, .6, .7]])
    exp = np.array([[.0, .1, .2], [.5, .9, .6], [.3, .2, .6], [.9, .8, .8]])

    constraints = {
        1: lambda x: x < .3 or x > .6,
        2: lambda x: (x * 10) % 2 == 0
    }
    result = Validation(constraints=constraints).reset(ori).enforce(adv)
    assert (result == exp).all()
