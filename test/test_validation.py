import numpy as np
from cge.validation import Validation, ALL, DEP
from plot.graph import dep_graph


def test_no_constraints():
    ori = np.array([[1, 2, 3], [5, 6, 7]])
    adv = np.array([[6, 7, 1], [3, 1, 2]])
    mn = np.array([0] * ori.shape[1])
    mx = np.array([1] * ori.shape[1])
    ranges = list(zip(mn, mx))
    result = Validation({}, ranges).enforce(ori, adv)
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
    mn = np.array([0] * ori.shape[1])
    mx = np.array([1] * ori.shape[1])
    ranges = list(zip(mn, mx))
    result = Validation(constraints, ranges).enforce(ori, adv)
    assert (result == ori).all()


def test_immutable_1():
    ori = np.array([[.2, .5, .6], [.6, .5, .4], [.1, .2, .3]])
    adv = np.array([[.2, .4, .6], [.0, .5, .2], [.6, .6, .3]])
    exp = np.array([[.2, .5, .6], [.0, .5, .2], [.6, .2, .3]])
    constraints = {1: ((1,), False)}
    mn = np.array([0] * ori.shape[0])
    mx = np.array([1] * ori.shape[0])
    ranges = list(zip(mn, mx))
    result = Validation(constraints, ranges).enforce(ori, adv)
    assert (result == exp).all()


def test_immutable_2():
    ori = np.array([[.0, .3, .4, .5, .6, .7], [.2, .8, .4, .7, .6, .1]])
    adv = np.array([[.5, .7, .8, .1, .9, .1], [.8, .3, .2, .5, .3, .2]])
    exp = np.array([[.5, .7, .4, .1, .9, .7], [.8, .3, .4, .5, .3, .1]])
    constraints = {2: ((2,), False), 5: ((5,), False)}
    mn = np.array([0] * ori.shape[1])
    mx = np.array([1] * ori.shape[1])
    ranges = list(zip(mn, mx))
    result = Validation(constraints, ranges).enforce(ori, adv)
    assert (result == exp).all()


def test_bin_feature():
    ori = np.array([[1.], [0.], [1.], [1.], [0.], [1.]])
    adv = np.array([[.8], [1.], [.4], [0.], [.2], [1.]])
    exp = np.array([[1.], [1.], [1.], [0.], [0.], [1.]])
    constraints = {0: ((0,), lambda x: x[0] == 0 or x[0] == 1)}
    mn = np.array([0] * ori.shape[1])
    mx = np.array([1] * ori.shape[1])
    ranges = list(zip(mn, mx))
    result = Validation(constraints, ranges).enforce(ori, adv)
    assert (result == exp).all()


def test_mutable_simple():
    ori = np.array(
        [[.0, .2, .4], [.5, .7, .2], [.3, .2, .6], [.9, .8, .8]])
    adv = np.array(
        [[.0, .1, .2], [.5, .9, .6], [.3, .4, .9], [.9, .6, .7]])
    exp = np.array(
        [[.0, .1, .2], [.5, .9, .6], [.3, .2, .6], [.9, .8, .8]])

    constraints = {
        1: ((1,), lambda x: x[0] < .3 or x[0] > .6),
        2: ((2,), lambda x: (x[0] * 10) % 2 == 0)
    }
    mn = np.array([0] * ori.shape[1])
    mx = np.array([1] * ori.shape[1])
    ranges = list(zip(mn, mx))
    result = Validation(constraints, ranges).enforce(ori, adv)
    assert (result == exp).all()


def test_nominal():
    ori = np.array([[1, 0, 0, 0, 0]])
    adv1 = np.array([[0, 1, 0, 0, 0]])
    adv2 = np.array([[0, 1, 1, 0, 0]])
    sources = tuple(range(ori.shape[1]))
    test_f = lambda arr: sum(arr) == 1
    mn = np.array([0] * ori.shape[1])
    mx = np.array([1] * ori.shape[1])
    ranges = list(zip(mn, mx))
    v_model = Validation({'A': (sources, test_f)}, ranges)
    assert test_f(ori[0])
    assert test_f(adv1[0])
    assert not test_f(adv2[0])
    assert (v_model.enforce(ori, adv1) == adv1).all()
    assert (v_model.enforce(ori, adv2) == ori).all()


# noinspection DuplicatedCode
def test_mutable_dep_reset():
    ori = np.array([[1, 0, 0, 1, 0, 1, 1], [1, 0, 0, 1, 0, 1, 1]])
    adv = np.array([[0, 0, 1, 1, 1, 1, 1], [0, 1, 1, 0, 0, 0, 0]])
    exp = np.array([[0, 0, 1, 1, 0, 1, 1], [1, 0, 0, 1, 0, 1, 1]])
    constraints = {
        'A': ((0, 1, 2), lambda arr: sum(arr) == 1),
        'B': ((0, 3), lambda arr: arr[0] == 0 or arr[1] == 1),
        'C': ((4, 5), lambda arr: arr[0] == 0 or arr[1] == 0),
        'D': ((4, 6), lambda arr: arr[0] == 0 and arr[1] == 1)
    }
    mn = np.array([0] * ori.shape[1])
    mx = np.array([1] * ori.shape[1])
    ranges = list(zip(mn, mx))

    v_model = Validation(constraints, ranges, DEP)
    assert (v_model.enforce(ori, ori) == ori).all()
    assert (v_model.enforce(ori, adv) == exp).all()


# noinspection DuplicatedCode
def test_mutable_reset_all():
    ori = np.array([[1, 0, 0, 1, 0, 1, 1], [1, 0, 0, 1, 0, 1, 1]])
    adv = np.array([[0, 0, 1, 1, 1, 1, 1], [0, 0, 1, 0, 0, 0, 1]])
    exp = np.array([[1, 0, 0, 1, 0, 1, 1], [0, 0, 1, 0, 0, 0, 1]])
    constraints = {
        'A': ((0, 1, 2), lambda arr: sum(arr) == 1),
        'B': ((0, 3), lambda arr: arr[0] == 0 or arr[1] == 1),
        'C': ((4, 5), lambda arr: arr[0] == 0 or arr[1] == 0),
        'D': ((4, 6), lambda arr: arr[0] == 0 and arr[1] == 1)
    }
    mn = np.array([0] * ori.shape[1])
    mx = np.array([1] * ori.shape[1])
    ranges = list(zip(mn, mx))

    v_model = Validation(constraints, ranges, ALL)
    assert (v_model.enforce(ori, ori) == ori).all()
    assert (v_model.enforce(ori, adv) == exp).all()


def test_dep_generation():
    constraints = {
        'A': ((0, 1, 2), lambda arr: sum(arr) == 1),
        'B': ((0, 3), lambda arr: arr[0] == 0 or arr[1] == 1),
        'C': ((4, 5), lambda arr: arr[0] == 0 or arr[1] == 0),
        'D': ((4, 6), lambda arr: arr[0] == 0 and arr[1] == 1),
        'E': ((7,), False)
    }
    imm, mut = Validation.categorize(constraints)
    assert Validation.dep_map(mut) == dep_graph(imm, mut)[1]
