def compute_average_payment(facts):
    pv = facts["payment_volume_billion"] * 1e9
    tx = facts["transactions_billion"] * 1e9

    return pv / tx
