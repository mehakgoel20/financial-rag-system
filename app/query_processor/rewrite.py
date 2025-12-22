def rewrite_query(query: str, intent: str) -> list[str]:
    rewrites = [query]

    if intent == "arithmetic":
        if "average" in query.lower():
            rewrites.append(query + " payments volume")
            rewrites.append(query + " total transactions")

    return rewrites
