"""
================================================================================
THE ONLY FILE YOU EDIT.
================================================================================
This wires the verification suite to whatever Claude Code actually produced in
the BioMOBO repo. The tests are written against the *contracts* below, not
against specific file/class names, so you just fill in the hooks.

Run modes:
  - Leave a hook returning None / raising NotImplementedError  -> that test is
    SKIPPED (reported, not failed). Wire hooks in one at a time as Claude
    finishes each step.
  - Set BIOMOBO_SELFTEST=1 to run the whole suite against the bundled reference
    implementation (sanity check that the harness itself is healthy).

Point Python at your repo before running, e.g.:
    export PYTHONPATH=/path/to/GP-MOBO
    pytest -v
"""
from __future__ import annotations

import os
import torch

SELFTEST = os.environ.get("BIOMOBO_SELFTEST") == "1"


# ============================================================================ #
# STEP 1 — model under test
# ============================================================================ #
def build_and_fit_model():
    """Return (fitted_model, train_x, train_y, num_tasks).

    `fitted_model` must be a gpytorch model whose `likelihood(model(x))` yields a
    MultitaskMultivariateNormal. Replace the body with your repo's training path,
    e.g.:

        from biomobo.model import BioMOGP
        from biomobo.data import load_training_data
        X, Y = load_training_data(...)
        model = BioMOGP(X, Y, num_tasks=4); train(model, X, Y)
        return model, X, Y, 4
    """
    if SELFTEST:
        import reference_icm as ref
        return ref.build_reference("icm", iters=60)
    raise NotImplementedError("Wire build_and_fit_model() to your BioMOBO model.")


def predict_joint_single(model, x_single):
    """Return the MultitaskMultivariateNormal posterior at ONE input row.
    Default works for a standard gpytorch ExactGP; override if your API differs.
    """
    import gpytorch
    model.eval()
    if hasattr(model, "likelihood"):
        model.likelihood.eval()
    with torch.no_grad():
        out = model(x_single)
        if hasattr(model, "likelihood"):
            out = model.likelihood(out)
        return out


# ============================================================================ #
# STEP 2 — data / config under test
# ============================================================================ #
def load_config():
    """Return the config object/dict exposing dynamic Target / Off_Target."""
    if SELFTEST:
        return {"target": "Cruzain", "off_target": "Cathepsin_L"}
    raise NotImplementedError("Wire load_config() to your config.py / setup.yaml loader.")


def load_y_matrix_and_columns():
    """Return (Y, column_names). Y is (n, 4); columns name each output in order."""
    if SELFTEST:
        from checks import EXPECTED_Y_ORDER
        return torch.randn(10, 4), list(EXPECTED_Y_ORDER)
    raise NotImplementedError("Wire load_y_matrix_and_columns() to your data loader.")


# ============================================================================ #
# STEP 3 — acquisition under test
# ============================================================================ #
def get_cost_map():
    """Return {objective_name: cost}. Cheap proxies small-nonzero, wet-lab >= 1."""
    if SELFTEST:
        return {"Target_IC50": 1.0, "OffTarget_IC50": 1.0,
                "LF_Affinity_Proxy": 0.01, "ADMET_Score": 0.01}
    raise NotImplementedError("Wire get_cost_map() to your fidelity cost definitions.")


def get_cost_aware_scorer():
    """Return a callable f(raw_ehvi, cost) -> cost-aware score (raw / cost)."""
    if SELFTEST:
        return lambda raw_ehvi, cost: raw_ehvi / cost
    raise NotImplementedError("Wire get_cost_aware_scorer() to your cost_aware_ehvi.")


# ============================================================================ #
# STEP 4 — output under test
# ============================================================================ #
def get_selectivity_fn():
    """Return a callable computing SI = predicted_off_target / predicted_target.
    May accept (target=, off_target=) kwargs or positional (off, target)."""
    if SELFTEST:
        return lambda target, off_target: off_target / target
    raise NotImplementedError("Wire get_selectivity_fn() to your selectivity calc.")


def build_output_dataframe():
    """Return the final ranked pandas DataFrame of top Pareto candidates."""
    if SELFTEST:
        import pandas as pd
        return pd.DataFrame({
            "SMILES": ["CCO", "c1ccccc1"],
            "Selectivity Index": [5.0, 3.2],
            "ADMET_Score": [0.8, 0.6],
            "uncertainty_lower": [0.1, 0.2],
            "uncertainty_upper": [0.9, 1.1],
        })
    raise NotImplementedError("Wire build_output_dataframe() to your ranking/output script.")
