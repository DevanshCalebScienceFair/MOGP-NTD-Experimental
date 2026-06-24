"""Meta-tests: prove the harness itself is trustworthy.

These do NOT test BioMOBO. They test that the Step-1 coregionalization checks
actually distinguish a real ICM from an independent-GP imposter. If these ever
fail, the harness has been weakened and its green checks mean nothing.
"""
import torch
import gpytorch

import reference_icm as ref
import checks


def _joint_single(model, x):
    model.eval(); model.likelihood.eval()
    with torch.no_grad():
        return model.likelihood(model(x))


def test_positive_control_icm_passes_cross_task():
    model, X, _, _ = ref.build_reference("icm", iters=60)
    ok, detail = checks.cross_task_posterior_covariance(_joint_single(model, X[:1]))
    assert ok, f"correct ICM wrongly flagged as independent: {detail}"
    ok, detail = checks.task_covar_is_not_diagonal(model)
    assert ok, detail


def test_negative_control_independent_is_caught():
    # rank-0 coregionalization == independent tasks. The harness MUST flag it.
    model, X, _, _ = ref.build_reference("independent", iters=60)
    ok, detail = checks.cross_task_posterior_covariance(_joint_single(model, X[:1]))
    assert not ok, f"harness FAILED to catch independent-GP imposter: {detail}"
    ok, detail = checks.task_covar_is_not_diagonal(model)
    assert not ok, f"harness FAILED to catch diagonal task covariance: {detail}"


def test_selectivity_inversion_is_caught():
    # An inverted SI (Target/Off) must be rejected.
    inverted = lambda target, off_target: target / off_target
    ok, _ = checks.selectivity_formula_ok(inverted)
    assert not ok, "harness FAILED to catch inverted selectivity index"
