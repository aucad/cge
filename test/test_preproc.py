from exp.preproc import pred_convert


def test_constraint_parser():
    cdict = {"my_att": "my_att == 0 or my_att == 1"}
    parsed, _ = pred_convert(cdict, ["any", "my_att"])
    s, f = parsed[1]
    assert s == (1,)
    assert f([0.5]) is False
    assert f([1]) is True
    assert f([0]) is True


def test_constraint_parser2():
    cdict = {'any': "any > 0.5", 'blah': "blah == 0"}
    parsed, _ = pred_convert(cdict, ["any", "my_att", "blah"])
    assert 2 == len(parsed.keys())
    assert parsed[0][0] == (0,)
    assert parsed[2][0] == (2,)
    assert parsed[0][1]([.75]) is True
    assert parsed[0][1]([.5]) is False
    assert parsed[2][1]([0]) is True
    assert parsed[2][1]([0.001]) is False


def test_constraint_parser_multi():
    cdict = {
        "a": "a + b + c == 1",
        "b": "a + b + c == 1",
        "c": "a + b + c == 1",
        "d": "d == 1",
        "e": "e != a"
    }
    parsed, _ = pred_convert(cdict, ["a", "b", "c", "d", "e"])

    assert len(parsed.keys()) == 5
    assert parsed[0][0] == (0, 1, 2)
    assert parsed[1][0] == (1, 0, 2)
    assert parsed[2][0] == (2, 0, 1)
    assert parsed[3][0] == (3,)
    assert parsed[4][0] == (4, 0)

    f2, f3, f4 = parsed[2][1], parsed[3][1], parsed[4][1]
    assert f2((0, 1, 0)) is True
    assert f2((1, 1, 0)) is False
    assert f3((1,)) is True
    assert f4((0, 1)) is True
    assert f4((0, 0)) is False
