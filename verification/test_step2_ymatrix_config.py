"""STEP 2 — Y-matrix structure & dynamic target config."""
import pytest

import checks
import adapter


def _assert(check_result):
    ok, detail = check_result
    assert ok, detail


def test_config_has_dynamic_target_and_offtarget():
    try:
        cfg = adapter.load_config()
    except NotImplementedError as e:
        pytest.skip(str(e))
    _assert(checks.config_targets_dynamic(cfg))


def test_y_matrix_has_four_columns():
    try:
        Y, _ = adapter.load_y_matrix_and_columns()
    except NotImplementedError as e:
        pytest.skip(str(e))
    _assert(checks.y_matrix_shape_ok(Y))


def test_y_columns_match_functional_order():
    # Y = [Target_IC50, OffTarget_IC50, LF_Affinity_Proxy, ADMET_Score]
    try:
        _, cols = adapter.load_y_matrix_and_columns()
    except NotImplementedError as e:
        pytest.skip(str(e))
    _assert(checks.y_columns_ok(cols))


def test_model_num_tasks_matches_y_width():
    # The MOGP must map to exactly the 4 outputs.
    try:
        _, _, _, T = adapter.build_and_fit_model()
    except NotImplementedError as e:
        pytest.skip(str(e))
    assert T == 4, f"model num_tasks={T}, expected 4 to match Y width"
