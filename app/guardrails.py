def should_refuse(intent, evidence, score_threshold=1.5):
    if not evidence:
        return True

    if evidence[0]["rerank_score"] < score_threshold:
        return True

    if intent == "arithmetic" and not any(char.isdigit() for char in evidence[0]["text"]):
        return True

    return False
