"""
Naming-agnostic correctness checks.

These functions encode WHAT MUST BE TRUE for each step, independent of how the
modified repo names its files/classes. The pytest files call into these; so can
you, interactively, against your own model objects.

Each check returns (passed: bool, detail: str) so failures are self-explaining.
"""
from __future__ import annotations

import torch
import gpytorch


# --------------------------------------------------------------------------- #
# STEP 1 — Coregionalization (ICM)
# --------------------------------------------------------------------------- #

def find_kernels_of_type(module, kernel_type):
    """Walk a gpytorch module tree and collect submodules of a given type."""
    found = []
    for m in module.modules():
        if isinstance(m, kernel_type):
            found.append(m)
    return found


def uses_index_kernel(model):
    """ICM must contain an IndexKernel (the task/coregionalization kernel).
    MultitaskKernel wraps one internally as `.task_covar_module`."""
    has_index = len(find_kernels_of_type(model, gpytorch.kernels.IndexKernel)) > 0
    has_multitask = len(find_kernels_of_type(model, gpytorch.kernels.MultitaskKernel)) > 0
    ok = has_index or has_multitask
    return ok, f"IndexKernel found={has_index}, MultitaskKernel found={has_multitask}"


def uses_tanimoto(model, name_contains="tanimoto"):
    """Step 1 requires KEEPING the Tanimoto data kernel. We can't import the
    repo's class here, so we detect it by class name (robust to module path)."""
    names = [type(m).__name__.lower() for m in model.modules()]
    hits = [n for n in names if name_contains in n]
    return len(hits) > 0, f"kernel class names containing '{name_contains}': {hits}"


def task_covar_is_not_diagonal(model, tol=1e-4):
    """The learned task covariance B must be ABLE to be non-diagonal (rank>=1).
    A diagonal B == independent tasks == the bug we are hunting.

    We reconstruct B = W Wᵀ + diag(v) from the first IndexKernel found.
    """
    idx_kernels = find_kernels_of_type(model, gpytorch.kernels.IndexKernel)
    if not idx_kernels:
        return False, "no IndexKernel present, cannot evaluate task covariance"
    ik = idx_kernels[0]
    B = ik.covar_matrix.to_dense().detach()
    off = B - torch.diag(torch.diagonal(B))
    max_off = off.abs().max().item()
    ok = max_off > tol
    return ok, f"max |off-diagonal| of task covar B = {max_off:.3e} (tol {tol})"


def posterior_is_multitask(posterior):
    """forward()/posterior must yield a MultitaskMultivariateNormal."""
    ok = isinstance(posterior, gpytorch.distributions.MultitaskMultivariateNormal)
    return ok, f"posterior type = {type(posterior).__name__}"


def cross_task_posterior_covariance(posterior, tol=1e-5):
    """THE defining property of coregionalization.

    At a single test input the joint posterior over T tasks has a TxT covariance.
    For INDEPENDENT GPs it is diagonal. For a correct ICM the off-diagonal terms
    are nonzero (observing one task informs the others). This is impossible to
    fake without real cross-task structure.

    `posterior` must be a MultitaskMultivariateNormal evaluated at ONE point
    (shape (1, T)) or we take the first point's task-block.
    """
    if not isinstance(posterior, gpytorch.distributions.MultitaskMultivariateNormal):
        return False, f"not a MultitaskMultivariateNormal: {type(posterior).__name__}"
    cov = posterior.covariance_matrix.detach()          # (n*T, n*T) interleaved
    T = posterior.event_shape[-1]
    block = cov[:T, :T]                                  # task covariance at point 0
    off = block - torch.diag(torch.diagonal(block))
    max_off = off.abs().max().item()
    ok = max_off > tol
    return ok, f"max |cross-task posterior covariance| at a point = {max_off:.3e} (tol {tol})"


# --------------------------------------------------------------------------- #
# STEP 2 — Y-matrix structure & config
# --------------------------------------------------------------------------- #

EXPECTED_Y_ORDER = ["Target_IC50", "OffTarget_IC50", "LF_Affinity_Proxy", "ADMET_Score"]


def y_matrix_shape_ok(Y):
    """Y must be (n, 4)."""
    shape = tuple(getattr(Y, "shape", ()))
    ok = len(shape) == 2 and shape[-1] == 4
    return ok, f"Y shape = {shape}, expected (n, 4)"


def y_columns_ok(columns):
    """Column names/order must match the agreed functional structure."""
    cols = list(columns)
    ok = cols == EXPECTED_Y_ORDER
    return ok, f"got {cols}\nexpected {EXPECTED_Y_ORDER}"


def config_targets_dynamic(config_obj, target_attr_candidates=("target", "Target", "TARGET"),
                           offtarget_attr_candidates=("off_target", "Off_Target", "OFF_TARGET", "offtarget")):
    """Config must expose dynamic target / off-target identifiers (strings)."""
    def first_attr(obj, names):
        for n in names:
            if hasattr(obj, n):
                return n, getattr(obj, n)
            if isinstance(obj, dict) and n in obj:
                return n, obj[n]
        return None, None

    tname, tval = first_attr(config_obj, target_attr_candidates)
    oname, oval = first_attr(config_obj, offtarget_attr_candidates)
    ok = isinstance(tval, str) and isinstance(oval, str) and tval and oval
    return ok, f"target={tname!r}={tval!r}, off_target={oname!r}={oval!r}"


# --------------------------------------------------------------------------- #
# STEP 3 — Multi-fidelity cost-aware EHVI
# --------------------------------------------------------------------------- #

def costs_avoid_zero_division(cost_map, cheap_keys=("ADMET_Score", "LF_Affinity_Proxy"),
                              expensive_keys=("Target_IC50", "OffTarget_IC50")):
    """Cheap (computational) fidelities must have small NONZERO cost; expensive
    (wet-lab) fidelities must cost strictly more. Literal 0 is a bug (div by 0)."""
    msgs = []
    ok = True
    for k in cheap_keys:
        c = cost_map.get(k)
        if c is None:
            ok = False; msgs.append(f"{k}: MISSING")
        elif c <= 0:
            ok = False; msgs.append(f"{k}: cost={c} (<=0 => division by zero risk)")
        else:
            msgs.append(f"{k}: cost={c} OK")
    for k in expensive_keys:
        c = cost_map.get(k)
        if c is None:
            ok = False; msgs.append(f"{k}: MISSING")
        else:
            msgs.append(f"{k}: cost={c}")
    # expensive must be strictly greater than cheap
    try:
        if max(cost_map[k] for k in cheap_keys) >= min(cost_map[k] for k in expensive_keys):
            ok = False; msgs.append("cheap cost >= expensive cost (ordering wrong)")
    except (KeyError, TypeError):
        ok = False; msgs.append("could not compare cheap vs expensive ordering")
    return ok, "; ".join(msgs)


def cost_weighting_prefers_cheaper(cost_aware_fn, raw_ehvi=1.0):
    """Given equal raw EHVI, a cheaper fidelity must score HIGHER.
    `cost_aware_fn(raw_ehvi, cost)` should return raw_ehvi / cost (or equivalent)."""
    cheap = cost_aware_fn(raw_ehvi, 0.01)
    expensive = cost_aware_fn(raw_ehvi, 1.0)
    ok = cheap > expensive
    return ok, f"score(cheap=0.01)={cheap:.4f} should be > score(expensive=1.0)={expensive:.4f}"


def cost_weighting_is_monotone(cost_aware_fn, cost=1.0):
    """At fixed cost, ranking by cost-aware score must agree with ranking by raw
    EHVI (the cost transform must not scramble same-cost candidates)."""
    a = cost_aware_fn(0.2, cost)
    b = cost_aware_fn(0.8, cost)
    ok = b > a
    return ok, f"at fixed cost, score(0.8)={b:.4f} should be > score(0.2)={a:.4f}"


# --------------------------------------------------------------------------- #
# STEP 4 — Selectivity Index & output schema
# --------------------------------------------------------------------------- #

def selectivity_formula_ok(selectivity_fn, off=10.0, target=2.0):
    """SI must equal Predicted(Off_Target) / Predicted(Target).
    With off=10, target=2 => SI should be 5.0 (not 0.2)."""
    val = float(selectivity_fn(target=target, off_target=off)) if _accepts_kwargs(selectivity_fn) \
        else float(selectivity_fn(off, target))
    expected = off / target
    inverted = target / off
    if abs(val - expected) < 1e-6:
        return True, f"SI={val} == Off/Target ({expected}) ✓"
    if abs(val - inverted) < 1e-6:
        return False, f"SI={val} is INVERTED (Target/Off). Expected Off/Target={expected}"
    return False, f"SI={val} matches neither Off/Target ({expected}) nor inverse ({inverted})"


def _accepts_kwargs(fn):
    import inspect
    try:
        params = inspect.signature(fn).parameters
        return "target" in params and ("off_target" in params or "off" in params)
    except (TypeError, ValueError):
        return False


REQUIRED_OUTPUT_COLUMNS = {
    "SMILES": ("smiles",),
    "Selectivity Index": ("selectivity", "selectivity_index", "si"),
    "ADMET": ("admet", "admet_score"),
    "uncertainty": ("uncertainty", "std", "lower", "upper", "ci", "confidence"),
}


def output_dataframe_schema_ok(df):
    """Output DataFrame must rank candidates and expose SMILES, Selectivity
    Index, ADMET, and calibrated uncertainty bounds."""
    cols_lower = [str(c).lower() for c in df.columns]
    msgs = []
    ok = True
    for label, needles in REQUIRED_OUTPUT_COLUMNS.items():
        present = any(any(nd in c for nd in needles) for c in cols_lower)
        msgs.append(f"{label}: {'present' if present else 'MISSING'}")
        ok = ok and present
    return ok, "; ".join(msgs) + f" | columns={list(df.columns)}"
