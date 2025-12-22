import re
from app.math.specs import ARITHMETIC_SPECS

def extract_numbers(text: str):
    return [float(x) for x in re.findall(r"\d+\.?\d*", text)]

# def extract_facts(subtype: str, chunks: list[dict]):
#     if subtype not in ARITHMETIC_SPECS:
#         return None, ["unsupported_subtype"]

#     spec = ARITHMETIC_SPECS[subtype]
#     required = spec["required_facts"]

#     text = " ".join(c["text"] for c in chunks)
#     numbers = extract_numbers(text)

#     facts = {}
#     missing = []

#     # Generic slot filling
#     for i, key in enumerate(required):
#         if i < len(numbers):
#             facts[key] = numbers[i]
#         else:
#             missing.append(key)

#     if missing:
#         return None, missing

#     return facts, None
def extract_facts(subtype: str, chunks: list[dict]):
    if subtype not in ARITHMETIC_SPECS:
        return None, ["unsupported_subtype"]

    text = " ".join(c["text"].lower() for c in chunks)

    facts = {}
    missing = []

    if subtype == "average":
        # semantic grounding
        pv = re.search(r"payments volume.*?(\d+\.?\d*)", text)
        tx = re.search(r"total transactions.*?(\d+\.?\d*)", text)

        if pv:
            facts["numerator"] = float(pv.group(1))
        else:
            missing.append("numerator")

        if tx:
            facts["denominator"] = float(tx.group(1))
        else:
            missing.append("denominator")

    elif subtype == "percentage":
        nums = extract_numbers(text)
        if len(nums) >= 2:
            facts["part"], facts["whole"] = nums[:2]
        else:
            missing += ["part", "whole"]

    else:
        return None, ["unsupported_subtype"]

    if missing:
        return None, missing

    return facts, None
