import numpy as np
from exp.validation import Validation


def test_no_constraints():
    ori = np.array([[1, 2, 3], [5, 6, 7]])
    adv = np.array([[6, 7, 1], [3, 1, 2]])
    result = Validation({}, np.array([1] * ori.shape[0])) \
        .enforce(ori, adv)
    assert (result == adv).all()


def test_immutable_all():
    ori = np.array([[.6, .4, .3, .3], [.3, .4, .5, .1]])
    adv = np.array([[.2, .4, .6, .0], [.7, .6, .6, .3]])
    constraints = {
        0: ((0,), False),
        1: ((1,), False),
        2: ((2,), False),
        3: ((3,), False)
    }
    ar = np.array([1] * ori.shape[0])
    result = Validation(constraints, ar).enforce(ori, adv)
    assert (result == ori).all()


def test_immutable_1():
    ori = np.array([[.2, .5, .6], [.6, .5, .4], [.1, .2, .3]])
    adv = np.array([[.2, .4, .6], [.0, .5, .2], [.6, .6, .3]])
    exp = np.array([[.2, .5, .6], [.0, .5, .2], [.6, .2, .3]])
    constraints = {1: ((1,), False)}
    ar = np.array([1] * ori.shape[0])
    result = Validation(constraints, ar).enforce(ori, adv)
    assert (result == exp).all()


def test_immutable_2():
    ori = np.array([[.0, .3, .4, .5, .6, .7], [.2, .8, .4, .7, .6, .1]])
    adv = np.array([[.5, .7, .8, .1, .9, .1], [.8, .3, .2, .5, .3, .2]])
    exp = np.array([[.5, .7, .4, .1, .9, .7], [.8, .3, .4, .5, .3, .1]])
    constraints = {2: ((2,), False), 5: ((5,), False)}
    ar = np.array([1] * ori.shape[0])
    result = Validation(constraints, ar) \
        .enforce(ori, adv)
    assert (result == exp).all()


def test_bin_feature():
    ori = np.array([[1.], [0.], [1.], [1.], [0.], [1.]])
    adv = np.array([[.8], [1.], [.4], [0.], [.2], [1.]])
    exp = np.array([[1.], [1.], [1.], [0.], [0.], [1.]])
    constraints = {0: ((0,), lambda x: x == 0 or x == 1)}
    result = Validation(constraints, np.array([1])) \
        .enforce(ori, adv)
    assert (result == exp).all()


def test_single_feature():
    ori = np.array(
        [[.0, .2, .4], [.5, .7, .2], [.3, .2, .6], [.9, .8, .8]])
    adv = np.array(
        [[.0, .1, .2], [.5, .9, .6], [.3, .4, .9], [.9, .6, .7]])
    exp = np.array(
        [[.0, .1, .2], [.5, .9, .6], [.3, .2, .6], [.9, .8, .8]])

    constraints = {
        1: ((1,), lambda x: x < .3 or x > .6),
        2: ((2,), lambda x: (x * 10) % 2 == 0)
    }
    ar = np.array([1] * ori.shape[0])
    result = Validation(constraints, ar).enforce(ori, adv)
    assert (result == exp).all()


def test_multi_feature():
    ori = np.array([[1, 0, 0, 1, 0, 1, 1], [1, 0, 0, 1, 0, 1, 1]])
    adv = np.array([[0, 1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 1]])
    exp = np.array([[1, 0, 0, 1, 0, 0, 1], [1, 0, 0, 1, 0, 0, 1]])

    constraints = {
        0: ((0, 1, 2), lambda arr: sum(arr) == 1),
        1: ((1, 0, 2,), lambda arr: sum(arr) == 1),
        2: ((2, 0, 1,), lambda arr: sum(arr) == 1),
        3: ((3, 0), lambda arr: arr[1] == 0 or arr[0] == 1),
        5: ((5, 4), lambda arr: arr[1] == 0 or arr[0] == 0),
        6: ((6, 4), lambda arr: arr[1] == 1 and arr[0] == 0)
    }
    ar = np.array([max(ori[:, i]) for i in range(ori.shape[1])])
    result = Validation(constraints, ar).enforce(ori, adv)
    assert (result == exp).all()
