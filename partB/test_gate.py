from eta_mlflow import decide

PROD = {"mae": 10.0, "mae_rush": 12.0, "mae_rain": 20.0, "p95_ms": 5.0}
INSIDE = {"mae": 9.69, "mae_rush": 12.59, "mae_rain": 20.99, "p95_ms": 9.99}

def failed(result):
    return [k for k, ok in result["criteria"].items() if not ok]

def test_just_inside_every_threshold():
    r = decide(PROD, INSIDE)
    assert r["promote"] is True
    assert failed(r) == []

def test_gain_just_outside():
    r = decide(PROD, {**INSIDE, "mae": 9.71})
    assert r["promote"] is False
    assert failed(r) == ["mae_gain"]

def test_p95_just_outside():
    r = decide(PROD, {**INSIDE, "p95_ms": 10.01})
    assert r["promote"] is False
    assert failed(r) == ["p95_ms"]

def test_rush_slice_just_outside():
    r = decide(PROD, {**INSIDE, "mae_rush": 12.61})
    assert r["promote"] is False
    assert failed(r) == ["rush_slice"]

def test_rain_slice_just_outside():
    r = decide(PROD, {**INSIDE, "mae_rain": 21.01})
    assert r["promote"] is False
    assert failed(r) == ["rain_slice"]

def test_custom_thresholds_are_used():
    assert decide(PROD, INSIDE, min_rel_gain=0.60)["promote"] is False
    assert decide(PROD, INSIDE, max_p95_ms=9.0)["promote"] is False
    assert decide(PROD, INSIDE, max_slice_regression=0.01)["promote"] is False
