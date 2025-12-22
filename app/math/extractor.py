import re

def extract_financial_facts(chunks):
    facts = {}

    for c in chunks:
        text = c["text"].lower()

        pv = re.search(r"payments volume .*? is (\d+\.?\d*)", text)
        tx = re.search(r"total transactions .*? is (\d+\.?\d*)", text)

        if pv:
            facts["payment_volume_billion"] = float(pv.group(1))

        if tx:
            facts["transactions_billion"] = float(tx.group(1))

    return facts
