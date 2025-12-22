class QueryIntent:
    ARITHMETIC = "arithmetic"
    COMPARISON = "comparison"
    DESCRIPTIVE = "descriptive"


def detect_intent(query: str) -> dict:
    q = query.lower()

    
    if any(x in q for x in ["average", "per transaction", "per card"]):
        return {"type": "arithmetic", "subtype": "average"}

    if any(x in q for x in ["total", "sum"]):
        return {"type": "arithmetic", "subtype": "sum"}

    if any(x in q for x in ["ratio", "percentage", "percent"]):
        return {"type": "arithmetic", "subtype": "ratio"}

    return {"type": "descriptive", "subtype": None}
