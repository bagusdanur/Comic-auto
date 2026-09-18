import json
import os
import urllib.request

base = os.environ["NINEROUTER_BASE_URL"].rstrip("/")
request = urllib.request.Request(
    base + "/models",
    headers={"Authorization": "Bearer " + os.environ["NINEROUTER_API_KEY"]},
)
with urllib.request.urlopen(request, timeout=30) as response:
    payload = json.load(response)
models = payload.get("data", payload if isinstance(payload, list) else [])
for item in models:
    if isinstance(item, str):
        print(item)
    elif isinstance(item, dict) and item.get("id"):
        print(item["id"])

