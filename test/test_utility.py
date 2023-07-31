from exp.utility import Utility


def test_const_parser():
    cdict = {1: "lambda x: x == 0 or x == 1"}
    parsed = Utility.parse_pred(cdict)
    assert 1 in parsed.keys()
    assert parsed[1][0] == (1,)


def test_const_parser2():
    cdict = {0: "lambda x: x > 0.5", 2: "lambda x: x == 0"}
    parsed = Utility.parse_pred(cdict)
    assert 2 == len(parsed.keys())
    assert parsed[0][0] == (0,)
    assert parsed[2][0] == (2,)


def test_const_parser_multi():
    cdict = {
        0: [[1, 2], "lambda x: x[0] + x[1] + x[2] == 1"],
        1: [[0, 2], "lambda x: x[0] + x[1] + x[2] == 1"],
        2: [[0, 1], "lambda x: x[0] + x[1] + x[2] == 1"],
        3: "lambda x: x == 1",
        4: [[0], "lambda x: x[0] != x[1]"]
    }
    parsed = Utility.parse_pred(cdict)

    assert len(parsed.keys()) == 5
    assert parsed[0][0] == (0, 1, 2)
    assert parsed[1][0] == (1, 0, 2)
    assert parsed[2][0] == (2, 0, 1)
    assert parsed[3][0] == (3,)
    assert parsed[4][0] == (4, 0)

    f2, f3, f4 = parsed[2][1], parsed[3][1], parsed[4][1]
    assert f2([0, 1, 0]) is True
    assert f2([1, 1, 0]) is False
    assert f3(1) is True
    assert f4([1, 0]) is True
    assert f4([0, 0]) is False
