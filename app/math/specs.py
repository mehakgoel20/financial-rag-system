ARITHMETIC_SPECS = {
    "average": {
        "required_facts": ["numerator", "denominator"],
        "unit": "USD",
        "keywords": {
            "numerator": ["payment", "volume", "total"],
            "denominator": ["transaction", "transactions"]
        },
        "compute": lambda f: f["numerator"] / f["denominator"]
    },

    "percentage": {
        "required_facts": ["part", "whole"],
        "unit": "%",
        "keywords": {
            "part": ["portion", "part"],
            "whole": ["total", "overall"]
        },
        "compute": lambda f: (f["part"] / f["whole"]) * 100
    },
    "ratio": {
        "required_facts": ["numerator", "denominator"],
        "unit": None,
        "compute": lambda f: f["numerator"] / f["denominator"]
    },
    "difference": {
        "required_facts": ["a", "b"],
        "unit": "USD",
        "compute": lambda f: f["a"] - f["b"]
    },
    "sum": {
        "required_facts": ["values"],
        "unit": None,
        "compute": lambda f: sum(f["values"])
    }
}
