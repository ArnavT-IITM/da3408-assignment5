import ray, requests
import pandas as pd
from ray import serve
from eta_ray import build

URL = "http://127.0.0.1:8100/v1/models/eta:predict"
ROW = pd.read_csv("data/eta_sample.csv").to_dict(orient="records")[0]
IDS = [f"customer-{i}" for i in range(1000)]

def ask(customer_id, headers=None):
    r = requests.post(URL, json={"customer_id": customer_id, "instances": [ROW]}, headers=headers)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["arm"] == body["model_version"], body
    return body["arm"]

def deploy(weight):
    serve.run(build(weight=weight), route_prefix="/", name="default")

ray.init(num_cpus=2, include_dashboard=False, logging_level="ERROR", log_to_driver=False)
serve.start(http_options={"host": "127.0.0.1", "port": 8100}, logging_config={"log_level": "WARNING"})
deploy(0.2)
first = {c: ask(c) for c in IDS}
print("Ray CPUs:", ray.cluster_resources()["CPU"])

share = sum(a == "v2" for a in first.values()) / len(IDS)
print(f"(a) weight 0.2, {len(IDS)} customers: share on candidate = {share:.3f}, allowed 0.16 to 0.24")
assert abs(share - 0.2) <= 0.04

sample = IDS[:200]
changed = sum(ask(c) != first[c] for c in sample)
print(f"(b) second call, {len(sample)} customers: {changed} changed arm")
assert changed == 0

serve.delete("default")
deploy(0.2)
changed = sum(ask(c) != first[c] for c in sample)
print(f"(c) after deleting and redeploying the same app, {len(sample)} customers: {changed} changed arm")
assert changed == 0

deploy(0.5)
second = {c: ask(c) for c in IDS}
moved_back = sum(first[c] == "v2" and second[c] == "v1" for c in IDS)
moved_up = sum(first[c] == "v1" and second[c] == "v2" for c in IDS)
share = sum(a == "v2" for a in second.values()) / len(IDS)
print(f"(d) weight 0.2 to 0.5: {moved_back} moved back to prod, {moved_up} moved to candidate, share now {share:.3f}")
assert moved_back == 0

on_prod = [c for c in IDS if second[c] == "v1"][:50]
forced = sum(ask(c, {"X-Beta": "1"}) == "v2" for c in on_prod)
after = sum(ask(c) == "v1" for c in on_prod)
print(f"(e) X-Beta: 1 on {len(on_prod)} prod customers: {forced} served by candidate, {after} back on prod without the header")
assert len(on_prod) == 50 and forced == 50 and after == 50

print("ALL CHECKS PASSED")
ray.shutdown()
