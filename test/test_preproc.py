from exp.preproc import pred_convert


def test_constraint_parser():
    imm, pred = [], ["my_att == 0 or my_att == 1"]
    attr = ["any", "my_att"]
    parsed, _ = pred_convert(imm, pred, attr)
    s, f = parsed[list(parsed.keys())[0]]
    assert s == (1,)
    assert f([0.5]) is False
    assert f([1]) is True
    assert f([0]) is True


def test_constraint_parser2():
    imm, pred = [], ["any > 0.5", "blah == 0"]
    attr = ["any", "my_att", "blah"]
    parsed, _ = pred_convert(imm, pred, attr)
    fk, sk = tuple(parsed.keys())
    assert parsed[fk][0] == (0,)
    assert parsed[sk][0] == (2,)
    assert parsed[fk][1]([.75]) is True
    assert parsed[fk][1]([.5]) is False
    assert parsed[sk][1]([0]) is True
    assert parsed[sk][1]([0.001]) is False


def test_constraint_parser_multi():
    attr = ["a", "b", "c", "d", "e"]
    imm, pred = [], [
        "a + b + c == 1",
        "d == 1",
        "e != a"]
    parsed, _ = pred_convert([], pred, attr)
    k0, k1, k2 = tuple(parsed.keys())
    f1, f2, f3 = [parsed[k][1] for k in [k0, k1, k2]]

    assert set(parsed[k0][0]) == {0, 1, 2}
    assert parsed[k1][0] == (3,)
    assert set(parsed[k2][0]) == {4, 0}

    assert f1((0, 1, 0)) is True
    assert f1((1, 1, 0)) is False
    assert f2((1,)) is True
    assert f3((0, 1)) is True
    assert f3((0, 0)) is False
