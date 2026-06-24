"""STEP 3 — Multi-fidelity, cost-aware EHVI."""
import pytest

import checks
import adapter


def _assert(check_result):
    ok, detail = check_result
    assert ok, detail


def test_fidelity_costs_are_sane():
    # cheap proxies small-nonzero (no /0), wet-lab strictly more expensive.
    try:
        cost_map = adapter.get_cost_map()
    except NotImplementedError as e:
        pytest.skip(str(e))
    _assert(checks.costs_avoid_zero_division(cost_map))


def test_cost_weighting_prefers_cheaper_fidelity():
    try:
        scorer = adapter.get_cost_aware_scorer()
    except NotImplementedError as e:
        pytest.skip(str(e))
    _assert(checks.cost_weighting_prefers_cheaper(scorer))


def test_cost_weighting_is_monotone_at_fixed_cost():
    # Cost transform must not scramble ordering among equal-cost candidates.
    try:
        scorer = adapter.get_cost_aware_scorer()
    except NotImplementedError as e:
        pytest.skip(str(e))
    _assert(checks.cost_weighting_is_monotone(scorer))


def test_no_division_by_zero_in_scorer():
    # Even if someone passes the literal cheap cost, scorer must be finite.
    try:
        scorer = adapter.get_cost_aware_scorer()
        cost_map = adapter.get_cost_map()
    except NotImplementedError as e:
        pytest.skip(str(e))
    import math
    for k, c in cost_map.items():
        val = scorer(1.0, c)
        assert math.isfinite(val), f"scorer(1.0, cost[{k}]={c}) = {val} (non-finite)"
