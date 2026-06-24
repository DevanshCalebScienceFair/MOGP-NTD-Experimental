"""STEP 1 — Coregionalization (ICM) correctness.

Catches the #1 hallucination: a model that is multitask-*shaped* but still treats
tasks as independent (block-diagonal / rank-0 coregionalization).
"""
import pytest
import torch

import checks
import adapter


@pytest.fixture(scope="module")
def fitted():
    try:
        model, X, Y, T = adapter.build_and_fit_model()
    except NotImplementedError as e:
        pytest.skip(str(e))
    return model, X, Y, T


@pytest.fixture(scope="module")
def posterior(fitted):
    model, X, _, _ = fitted
    return adapter.predict_joint_single(model, X[:1])


def _assert(check_result):
    ok, detail = check_result
    assert ok, detail


def test_model_contains_coregionalization_kernel(fitted):
    _assert(checks.uses_index_kernel(fitted[0]))


def test_tanimoto_kernel_is_kept(fitted):
    # Step 1 explicitly requires preserving the Tanimoto data kernel.
    _assert(checks.uses_tanimoto(fitted[0]))


def test_task_covariance_is_not_diagonal(fitted):
    # rank-0 / diagonal B == independent tasks == the bug.
    _assert(checks.task_covar_is_not_diagonal(fitted[0]))


def test_output_is_multitask_mvn(posterior):
    _assert(checks.posterior_is_multitask(posterior))


def test_cross_task_posterior_covariance_is_nonzero(posterior):
    # The defining, hard-to-fake property of coregionalization.
    _assert(checks.cross_task_posterior_covariance(posterior))
