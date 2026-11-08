import re, uuid
import pandas as pd
from fastapi.testclient import TestClient
import eta_app

URL = "/v1/models/eta:predict"
client = TestClient(eta_app.app)
ROW = {"distance_km": 20.0, "n_stops": 5, "hour": 12, "weekday": 2, "traffic_index": 0.5, "rain": 0, "weight_kg": 10.0}
FAR = {**ROW, "distance_km": 120.0, "n_stops": 15}

def post(rows, **kw):
    return client.post(URL, json={"instances": rows}, **kw)

def counter(path, status):
    m = re.search(rf'eta_requests_total\{{path="{re.escape(path)}",status="{status}"\}} (\S+)', client.get("/metrics").text)
    return float(m.group(1)) if m else 0.0

def test_valid_request():
    r = post([ROW])
    expected = eta_app.model.predict(pd.DataFrame([ROW])[eta_app.FEATURES])[0]
    assert r.status_code == 200
    assert r.json()["predictions"] == [round(float(expected), 1)]
    assert r.json()["model_version"] == "v1"

def test_batch_order_preserved():
    a, b = post([ROW, FAR]).json()["predictions"], post([FAR, ROW]).json()["predictions"]
    assert a[0] != a[1]
    assert a == b[::-1]

def test_out_of_range():
    r = post([{**ROW, "hour": 24}])
    assert r.status_code == 422
    assert "hour" in r.json()["details"][0]["field"]

def test_missing_field():
    r = post([{k: v for k, v in ROW.items() if k != "hour"}])
    assert r.status_code == 422
    assert r.json()["error"] == "validation_failed"
    assert "hour" in r.json()["details"][0]["field"]

def test_wrong_type():
    body = post([{**ROW, "distance_km": "far"}]).json()
    assert set(body) == {"error", "details", "request_id"}
    assert set(body["details"][0]) == {"field", "msg"}
    assert "distance_km" in body["details"][0]["field"]

def test_cross_field_rule():
    r = post([{**ROW, "distance_km": 151, "n_stops": 21}])
    assert r.status_code == 422
    assert "infeasible" in r.json()["details"][0]["msg"]
    assert post([{**ROW, "distance_km": 150, "n_stops": 60}]).status_code == 200

def test_empty_batch_rejected():
    assert post([]).status_code == 422

def test_batch_limit():
    assert post([ROW] * 501).status_code == 422
    assert len(post([ROW] * 500).json()["predictions"]) == 500

def test_request_id_echoed():
    r = post([ROW], headers={"X-Request-ID": "abc-123"})
    assert r.headers["X-Request-ID"] == "abc-123"
    assert r.json()["request_id"] == "abc-123"

def test_request_id_generated():
    r = post([ROW])
    assert str(uuid.UUID(r.headers["X-Request-ID"])) == r.headers["X-Request-ID"]
    assert r.json()["request_id"] == r.headers["X-Request-ID"]

def test_request_id_on_error():
    r = post([{**ROW, "hour": 99}], headers={"X-Request-ID": "err-9"})
    assert r.headers["X-Request-ID"] == "err-9"
    assert r.json()["request_id"] == "err-9"
    r = post([{**ROW, "hour": 99}])
    assert uuid.UUID(r.headers["X-Request-ID"])
    assert r.json()["request_id"] == r.headers["X-Request-ID"]

def test_metrics():
    ok, bad = counter(URL, 200), counter(URL, 422)
    post([ROW])
    post([{**ROW, "hour": 99}])
    assert counter(URL, 200) == ok + 1
    assert counter(URL, 422) == bad + 1
    text = client.get("/metrics").text
    assert len(re.findall(r'eta_request_latency_seconds_bucket\{le="', text)) > 1
    assert 'eta_request_latency_seconds_bucket{le="+Inf"}' in text
    assert "eta_request_latency_seconds_sum" in text
    assert "eta_request_latency_seconds_count" in text
    assert 'path="/metrics"' not in text

def test_generic_500(monkeypatch):
    def boom(X):
        raise RuntimeError("secret-internal-detail")
    monkeypatch.setattr(eta_app.model, "predict", boom)
    r = post([ROW], headers={"X-Request-ID": "fail-1"})
    assert r.status_code == 500
    assert "secret" not in r.text and "RuntimeError" not in r.text
    assert r.headers["X-Request-ID"] == "fail-1"
    assert r.json()["request_id"] == "fail-1"
