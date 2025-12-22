import json
import statistics
from pathlib import Path

LOG_FILE = Path("logs/metrics.log")

records = []
with LOG_FILE.open() as f:
    for line in f:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue

if not records:
    raise ValueError("metrics.log exists but contains no valid records")

# Extract safely
latencies = [r["latency_ms"] for r in records if "latency_ms" in r]
cache_hits = [r for r in records if r.get("cache_hit") is True]
llm_calls = [r for r in records if r.get("used_llm") is True]
symbolic_calls = [r for r in records if r.get("symbolic") is True]

total_requests = len(records)

if not latencies:
    raise ValueError("No latency values found in metrics.log")

# Metrics
avg_latency = statistics.mean(latencies)
p95_latency = statistics.quantiles(latencies, n=20)[18]

cache_hit_rate = len(cache_hits) / total_requests
llm_usage_rate = len(llm_calls) / total_requests
symbolic_rate = len(symbolic_calls) / total_requests

print("\n📊 SYSTEM METRICS")
print("=" * 40)
print(f"Total Requests       : {total_requests}")
print(f"Cache Hit Rate       : {cache_hit_rate:.2%}")
print(f"LLM Usage Rate       : {llm_usage_rate:.2%}")
print(f"Symbolic Exec Rate   : {symbolic_rate:.2%}")
print(f"Avg Latency (ms)     : {avg_latency:.2f}")
print(f"P95 Latency (ms)     : {p95_latency:.2f}")

# Cost estimation
COST_PER_LLM_CALL = 0.03  # USD (example)
estimated_cost = len(llm_calls) * COST_PER_LLM_CALL
saved_cost = (total_requests - len(llm_calls)) * COST_PER_LLM_CALL

print("\n💰 COST ESTIMATION")
print("=" * 40)
print(f"Estimated LLM Cost   : ${estimated_cost:.2f}")
print(f"Estimated Cost Saved : ${saved_cost:.2f}")
