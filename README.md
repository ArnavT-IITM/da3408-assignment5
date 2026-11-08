# DA3408 Assignment 5: SwiftParcel ETA service

Serving a delivery-time model four ways: a REST API, MLflow, a Ray Serve canary and a smoke test release gate. By Arnav Thorat (DA24B027). All results and written answers are in `REPORT.pdf`.

## Repository contents

- `REPORT.pdf`: report with all tables, written answers, machine details and the not-tested list.
- `AI_DISCLOSURE.md`: how I used AI tools.
- `eta_data.py`, `load_test.py`, `requirements.txt`: starter files. I only added `prometheus_client` and `gunicorn` to `requirements.txt`.
- `partA/`: REST service (`eta_app.py`), its tests (`test_eta_app.py`) and load test results (`loadtest_results.json`).
- `partB/`: MLflow registry, promotion gate and shadow test (`eta_mlflow.py`), gate tests (`test_gate.py`) and their outputs.
- `partC/`: Ray Serve sticky canary (`eta_ray.py`), verification script (`verify_ray.py`) and its output.
- `partD/`: smoke test (`smoke_eta.py`).
- `evidence/`: screenshots of each step.

## Setup

Run every command from the repository root. Python 3.10 or newer.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python eta_data.py
python -c "import pandas as pd, eta_data; pd.read_csv('data/eta_fresh.csv')[eta_data.FEATURES].head(200).to_csv('data/eta_sample.csv', index=False)"
export PYTHONPATH=.:partA:partB:partC:partD
```

`eta_data.py` uses a fixed seed, so it rebuilds the same data and models I used. Every new terminal needs `source .venv/bin/activate` and the `export` line again.

## Part A: REST service

Tests:

```bash
python -m pytest partA
```

Start the server in terminal 1:

```bash
python -m uvicorn eta_app:app --port 8000
```

Run the load tests in terminal 2:

```bash
U=http://localhost:8000/v1/models/eta:predict
python load_test.py --url $U --data data/eta_sample.csv --concurrency 1 2 4 8 16 24 32 48 --duration 4 --out data/lt_sweep.json
python load_test.py --url $U --data data/eta_sample.csv --concurrency 1 --batch 1 --duration 4 --out data/lt_batch1.json
python load_test.py --url $U --data data/eta_sample.csv --concurrency 1 --batch 100 --duration 4 --out data/lt_batch100.json
```

For the two-worker change, stop the server (Ctrl+C) and start it with uvicorn's own workers:

```bash
python -m uvicorn eta_app:app --port 8000 --workers 2
```

```bash
python load_test.py --url $U --data data/eta_sample.csv --concurrency 1 2 4 8 16 --duration 4 --out data/lt_uvicorn2.json
```

Then stop it again and start the same two workers under gunicorn:

```bash
python -m gunicorn eta_app:app -w 2 -k uvicorn.workers.UvicornWorker -b 127.0.0.1:8000
```

```bash
python load_test.py --url $U --data data/eta_sample.csv --concurrency 1 2 4 8 16 24 32 48 64 --duration 4 --out data/lt_gunicorn2.json
```

Combine the five runs into `partA/loadtest_results.json`:

```bash
python -c "import json; runs = {'v1_uvicorn_1worker': 'sweep', 'batch_1': 'batch1', 'batch_100': 'batch100', 'uvicorn_workers2': 'uvicorn2', 'gunicorn_workers2': 'gunicorn2'}; json.dump([{'run': k, **r} for k, f in runs.items() for r in json.load(open(f'data/lt_{f}.json'))], open('partA/loadtest_results.json', 'w'), indent=2)"
```

Throughput and latency depend on the machine, so your numbers will differ from mine.

## Part B: MLflow registry, gate and shadow

```bash
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
python -m pytest partB
python -c "import eta_mlflow as e; e.register()"
python -c "import eta_mlflow as e; print('default:', e.promotion_gate()); print('min_rel_gain=0.60:', e.promotion_gate(min_rel_gain=0.60))" | tee partB/gate_output.txt
```

For the shadow test, serve each alias in its own terminal (both need `MLFLOW_TRACKING_URI` set):

```bash
mlflow models serve -m "models:/eta-model@prod" -p 5101 --env-manager local
```

```bash
mlflow models serve -m "models:/eta-model@candidate" -p 5102 --env-manager local
```

Then, in a third terminal:

```bash
python -c "import eta_mlflow as e; e.shadow()" | tee partB/shadow_output.txt
```

## Part C: Ray Serve sticky canary

This starts Ray with 2 CPUs on port 8100, runs the five checks and takes about a minute:

```bash
python partC/verify_ray.py | tee partC/verify_output.txt
```

## Part D: smoke test

Serve v2 with the Part A app in terminal 1:

```bash
ETA_MODEL=data/eta_v2.joblib python -m uvicorn eta_app:app --port 8000
```

In terminal 2, this should pass all 6 checks and exit with 0:

```bash
python partD/smoke_eta.py --url http://localhost:8000/v1/models/eta:predict --expect-version v2; echo "exit code $?"
```

And this should fail the version check and exit with 1:

```bash
python partD/smoke_eta.py --url http://localhost:8000/v1/models/eta:predict --expect-version v1; echo "exit code $?"
```
