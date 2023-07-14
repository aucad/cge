from exp.utility import Utility


def test_const_parser():
    cdict = {1: "lambda x: x == 0 or x == 1"}
    parsed = Utility.parse_pred(cdict)
    assert 1 in parsed.keys()
    assert parsed[1][0] == (1,)


def test_const_parser2():
    cdict = {0: "lambda x: x>0.5", 2: "lambda x: x == 0"}
    parsed = Utility.parse_pred(cdict)
    assert 2 == len(parsed.keys())
    assert parsed[0][0] == (0,)
    assert parsed[2][0] == (2,)


def test_const_parser_multi():
    cdict = {
        0: [[1, 2], "lambda c1, c2, c3: c1 + c2 + c3 == 1"],
        1: [[0, 2], "lambda c1, c2, c3: c1 + c2 + c3 == 1"],
        2: [[0, 1], "lambda c1, c2, c3: c1 + c2 + c3 == 1"],
        3: "lambda x: x == 1",
        4: [[0], "lambda x, y: x != y"]
    }
    parsed = Utility.parse_pred(cdict)
    assert len(parsed.keys()) == 5
    assert parsed[0][0] == (0, 1, 2)
    assert parsed[1][0] == (1, 0, 2)
    assert parsed[2][0] == (2, 0, 1)
    assert parsed[3][0] == (3,)
    assert parsed[4][0] == (4, 0)

