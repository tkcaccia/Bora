"""Versioned all-case contracts; historical cycles retain their 149-image gate."""
from collections import Counter
import json
from pathlib import Path

EXPECTED = {"bcss": 132, "tiger": 80, "wsss4luad": 40, "rings-tumor": 40,
            "ocelot": 40, "beetle": 10, "camelyon16-subset": 7}


def validate_scope(protocol, *, pilot=False):
    scope = protocol.get("benchmark_scope")
    if scope is None:
        if protocol.get("cycle", 1) >= 8:
            raise ValueError("Cycle8 and later require the expanded benchmark contract")
        if not pilot and len(protocol["cases"]) != 149:
            raise ValueError("Historical full cycles require149 images")
        return
    if protocol["cycle"] < 8 or scope["version"] != "expanded349-v1":
        raise ValueError("Expanded scope is prospective from cycle8")
    required, historical = scope["required_cases"], scope["historical_cases"]
    if (len(required) != 349 or len(set(required)) != 349 or len(historical) != 149
            or len(set(historical)) != 149 or not set(historical) <= set(required)):
        raise ValueError("Expanded scope must retain149 historical and200 added cases")
    if scope["dataset_counts"] != EXPECTED or scope["plan_sha256"] != protocol["input_plan_sha256"]:
        raise ValueError("Expanded dataset counts/plan identity changed")
    if not set(protocol["cases"]) <= set(required):
        raise ValueError("Unexpected benchmark case")
    if len(protocol["cases"]) != len(set(protocol["cases"])):
        raise ValueError("Duplicate benchmark case")
    if not pilot and set(protocol["cases"]) != set(required):
        raise ValueError("Full expanded cycle requires every349 case")


def requires_parent_parity(protocol, case):
    """New images have no previous mask; old images retain exact parity checks."""
    scope = protocol.get("benchmark_scope")
    return scope is None or case in scope["historical_cases"] or protocol["cycle"] > 8


def load_scope(source, cycle):
    from .run import digest
    source = Path(source)
    if cycle < 8:
        return None
    plan = json.loads((source/"PLAN.json").read_text())
    frozen = json.loads((source/"ALL_INPUTS_FROZEN.json").read_text())
    ids = [s["case"] for s in plan["samples"]]
    counts = dict(Counter(s["dataset"] for s in plan["samples"]))
    sha = digest(source/"PLAN.json")
    if frozen["plan_sha256"] != sha or set(frozen["manifests"]) != set(ids) or frozen["cases"] != 349:
        raise ValueError("Expanded all-input freeze missing or changed")
    for case, expected in frozen["manifests"].items():
        if digest(source/"cases"/case/"INPUTS_FROZEN.json") != expected:
            raise ValueError("Expanded case input manifest changed")
    scope = dict(version="expanded349-v1", required_cases=ids,
        historical_cases=plan["historical_cases"], dataset_counts=counts,
        plan_sha256=sha, all_inputs_sha256=digest(source/"ALL_INPUTS_FROZEN.json"))
    validate_scope(dict(cycle=cycle, cases=ids, benchmark_scope=scope, input_plan_sha256=sha))
    return scope
