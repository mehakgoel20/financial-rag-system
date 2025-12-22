def verify_required_facts(facts, required_keys):
    return all(k in facts for k in required_keys)
