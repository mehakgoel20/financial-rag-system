def arithmetic_confidence(facts_used: dict, required_facts: list[str]) -> float:
    if not required_facts:
        return 0.0

    present = sum(1 for f in required_facts if f in facts_used)
    return round(present / len(required_facts), 2)
