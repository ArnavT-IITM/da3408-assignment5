import argparse, sys, time, requests
import numpy as np, pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--url", required=True)
p.add_argument("--expect-version", required=True)
p.add_argument("--max-p95-ms", type=float, default=80)
args = p.parse_args()

ROWS = pd.read_csv("data/eta_sample.csv").head(20).to_dict(orient="records")
results = []

def predict(rows):
    return requests.post(args.url, json={"instances": rows})

def check(name, ok, detail):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")

r = predict(ROWS)
body = r.json()
base = body.get("predictions", [])
check("valid batch", r.status_code == 200 and len(base) == len(ROWS) and all(x > 0 for x in base),
      f"status {r.status_code}, {len(base)} positive predictions for {len(ROWS)} rows")
check("model version", body.get("model_version") == args.expect_version,
      f"got {body.get('model_version')}, expected {args.expect_version}")

r = predict([{**ROWS[0], "hour": 25}])
check("invalid input", r.status_code == 422, f"hour=25 gave status {r.status_code}")

far = predict([{**x, "distance_km": x["distance_km"] * 1.5} for x in ROWS]).json()["predictions"]
n = sum(f < b for f, b in zip(far, base))
check("longer distance", n == 0, f"{n} of {len(ROWS)} rows got a shorter ETA with 50% more distance")

busy = predict([{**x, "traffic_index": min(x["traffic_index"] + 0.3, 1.0)} for x in ROWS]).json()["predictions"]
n = sum(f < b for f, b in zip(busy, base))
check("heavier traffic", n == 0, f"{n} of {len(ROWS)} rows got a shorter ETA with traffic_index +0.3")

lat = []
for x in ROWS * 5:
    t0 = time.perf_counter()
    predict([x])
    lat.append((time.perf_counter() - t0) * 1000)
p95 = np.percentile(lat, 95)
check("p95 latency", p95 <= args.max_p95_ms, f"{p95:.1f} ms over {len(lat)} single-row requests, limit {args.max_p95_ms} ms")

print(f"{sum(results)} of {len(results)} checks passed")
sys.exit(0 if all(results) else 1)
