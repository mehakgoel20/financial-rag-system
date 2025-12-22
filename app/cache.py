import redis
import json

redis_client = redis.Redis(
    host="redis",
    port=6379,
    decode_responses=True
)

def get_cache(key: str):
    val = redis_client.get(key)
    if val:
        return json.loads(val)
    return None

def set_cache(key: str, value, ttl=300):
    redis_client.setex(key, ttl, json.dumps(value))


