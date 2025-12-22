from pathlib import Path
import json
import time

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "metrics.log"

def log_metric(
    *,
    cache_hit: bool,
    latency_ms: float,
    used_llm: bool,
    symbolic: bool = False
):
    record = {
        "timestamp": time.time(),
        "cache_hit": cache_hit,
        "latency_ms": round(latency_ms, 2),
        "used_llm": used_llm,
        "symbolic": symbolic
    }

    with LOG_FILE.open("a") as f:
        f.write(json.dumps(record) + "\n")
