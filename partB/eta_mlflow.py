import os, time, joblib, mlflow, mlflow.pyfunc, requests
import numpy as np, pandas as pd
from importlib.metadata import version
from mlflow import MlflowClient
from mlflow.models import infer_signature

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
NAME = "eta-model"
RUSH_HOURS = [7, 8, 9, 16, 17, 18, 19]
PINS = [f"{p}=={version(p)}" for p in ("mlflow", "scikit-learn", "pandas", "numpy", "joblib")]

class EtaPyfunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        b = joblib.load(context.artifacts["bundle"])
        self.model, self.cols, self.version = b["model"], b["feature_columns"], b["version"]

    def predict(self, context, model_input, params=None):
        X = pd.DataFrame(model_input)[self.cols]
        return pd.DataFrame({"eta_minutes": self.model.predict(X).round(1), "model_version": self.version})

def register():
    mlflow.set_experiment("eta-serving")
    client = MlflowClient()
    for alias, ver in (("prod", "v1"), ("candidate", "v2")):
        bundle = f"data/eta_{ver}.joblib"
        cols = joblib.load(bundle)["feature_columns"]
        X = pd.read_csv("data/eta_fresh.csv")[cols].astype("float64").head(5)
        sig = infer_signature(X, pd.DataFrame({"eta_minutes": [10.0], "model_version": [ver]}))
        with mlflow.start_run(run_name=ver):
            info = mlflow.pyfunc.log_model(name="model", python_model=EtaPyfunc(), artifacts={"bundle": bundle},
                                           signature=sig, input_example=X.head(2), pip_requirements=PINS,
                                           code_paths=[__file__], registered_model_name=NAME)
        client.set_registered_model_alias(NAME, alias, info.registered_model_version)
    aliases = client.get_registered_model(NAME).aliases
    print("version alias")
    for v in sorted(client.search_model_versions(f"name='{NAME}'"), key=lambda v: int(v.version)):
        print(v.version, [a for a, n in aliases.items() if str(n) == str(v.version)])

def decide(prod, cand, min_rel_gain=0.03, max_p95_ms=10.0, max_slice_regression=0.05):
    criteria = {"mae_gain": cand["mae"] <= prod["mae"] * (1 - min_rel_gain),
                "p95_ms": cand["p95_ms"] <= max_p95_ms,
                "rush_slice": cand["mae_rush"] <= prod["mae_rush"] * (1 + max_slice_regression),
                "rain_slice": cand["mae_rain"] <= prod["mae_rain"] * (1 + max_slice_regression)}
    return {"criteria": criteria, "promote": all(criteria.values())}

def evaluate(alias, fresh, cols):
    m = mlflow.pyfunc.load_model(f"models:/{NAME}@{alias}")
    X = fresh[cols].astype("float64")
    err = (m.predict(X)["eta_minutes"] - fresh["eta_minutes"]).abs()
    lat = []
    for i in range(200):
        row = X.iloc[[i]]
        t0 = time.perf_counter()
        m.predict(row)
        lat.append((time.perf_counter() - t0) * 1000)
    return {"mae": round(float(err.mean()), 3), "mae_rush": round(float(err[fresh["hour"].isin(RUSH_HOURS)].mean()), 3),
            "mae_rain": round(float(err[fresh["rain"] == 1].mean()), 3), "p95_ms": round(float(np.percentile(lat, 95)), 3)}

def promotion_gate(**thresholds):
    import eta_data
    fresh = eta_data.make_data(2000, seed=99)
    prod, cand = evaluate("prod", fresh, eta_data.FEATURES), evaluate("candidate", fresh, eta_data.FEATURES)
    return {"prod": prod, "candidate": cand, **decide(prod, cand, **thresholds)}

def shadow():
    rows = pd.read_csv("data/eta_fresh.csv").drop(columns="eta_minutes").head(200).to_dict(orient="records")
    preds, lat = {5101: [], 5102: []}, {5101: [], 5102: []}
    for row in rows:
        for port in preds:
            t0 = time.perf_counter()
            r = requests.post(f"http://localhost:{port}/invocations", json={"dataframe_records": [row]})
            lat[port].append((time.perf_counter() - t0) * 1000)
            preds[port].append(r.json()["predictions"][0]["eta_minutes"])
    diff = np.abs(np.array(preds[5101]) - np.array(preds[5102]))
    print(f"requests sent to each server: {len(rows)}")
    print(f"mean absolute difference: {diff.mean():.2f} min")
    print(f"differ by more than 10 min: {(diff > 10).mean():.1%}")
    print(f"p95 latency prod (5101): {np.percentile(lat[5101], 95):.1f} ms")
    print(f"p95 latency candidate (5102): {np.percentile(lat[5102], 95):.1f} ms")
