from app.math.specs import ARITHMETIC_SPECS
from app.math.confidence import arithmetic_confidence

def execute_math(subtype: str, facts: dict):
    if subtype not in ARITHMETIC_SPECS:
        return None, "Unsupported arithmetic type"

    spec = ARITHMETIC_SPECS[subtype]
    required = spec["required_facts"]

    missing = [f for f in required if f not in facts]
    if missing:
        return None, f"Missing required facts: {missing}"

    try:
        # Prevent division by zero
        if subtype in ("average", "percentage", "ratio"):
            if facts.get(required[1]) == 0:
                return None, "Division by zero detected"

        value = spec["compute"](facts)

    except Exception as e:
        return None, f"Arithmetic execution failed: {str(e)}"

    confidence = arithmetic_confidence(
        facts_used=facts,
        required_facts=required
    )

    return {
        "value": round(value, 2),
        "unit": spec["unit"],
        "confidence": confidence,
        "facts_used": facts
    }, None
