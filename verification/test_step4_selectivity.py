"""STEP 4 — Selectivity Index & output schema."""
import pytest

import checks
import adapter


def _assert(check_result):
    ok, detail = check_result
    assert ok, detail


def test_selectivity_index_formula_direction():
    # SI = Predicted(Off_Target) / Predicted(Target). Inverting it is a classic,
    # silent, plot-ruining bug — this asserts the direction explicitly.
    try:
        si_fn = adapter.get_selectivity_fn()
    except NotImplementedError as e:
        pytest.skip(str(e))
    _assert(checks.selectivity_formula_ok(si_fn))


def test_output_dataframe_has_required_columns():
    # SMILES, Selectivity Index, ADMET, calibrated uncertainty bounds.
    try:
        df = adapter.build_output_dataframe()
    except NotImplementedError as e:
        pytest.skip(str(e))
    _assert(checks.output_dataframe_schema_ok(df))


def test_output_dataframe_is_ranked_and_nonempty():
    try:
        df = adapter.build_output_dataframe()
    except NotImplementedError as e:
        pytest.skip(str(e))
    assert len(df) > 0, "output DataFrame is empty"
    # selectivity column should be monotonically sorted (ranked)
    import pandas as pd
    si_cols = [c for c in df.columns if "selectiv" in str(c).lower() or str(c).lower() == "si"]
    if si_cols:
        col = df[si_cols[0]]
        is_sorted = col.is_monotonic_increasing or col.is_monotonic_decreasing
        assert is_sorted, f"candidates not ranked by {si_cols[0]}: {list(col)}"
