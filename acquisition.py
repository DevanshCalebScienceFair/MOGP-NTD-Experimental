"""
acquisition.py
==============

qNEHVI acquisition for CONTINUOUS latent-space multi-objective Bayesian
optimization (LSBO).

This module was rewritten for the De-Novo / Latent-Space migration. The old
version ran a **discrete** candidate scan: it scored a fixed library of
fingerprints in chunks and ranked them, using a grey-box composite objective that
folded in known-exact ADMET values. That entire mechanism
(``X_candidates`` chunking, ``DockingPosteriorModel``,
``CompositeKnownADMETObjective``, ``_augment_with_admet``) is gone.

The loop now searches a bounded, continuous 50-D latent box. The GP
(``mogp.ModelListGP``) models ALL five objectives directly, so acquisition is a
clean, textbook multi-objective BoTorch setup:

  * **qNEHVI** (``qLogNoisyExpectedHypervolumeImprovement``) over the 5-output
    ``ModelListGP`` posterior.
  * A ``WeightedMCMultiOutputObjective`` whose weights are the per-objective
    signs (``DEFAULT_OBJECTIVE_SIGNS``), so the two "lower is better" objectives
    are flipped and qNEHVI can treat everything as maximization.
  * **``optimize_acqf``** does gradient-based (L-BFGS-B) optimization of the
    acquisition directly over the latent ``bounds``, returning the winning latent
    vector(s) ``z``. Because the ``ModelListGP`` posterior is differentiable
    w.r.t. its input, gradients flow from the acquisition value all the way back
    to ``z`` — the property the mock was built to verify.

``compute_pareto_front`` / ``get_reference_point`` (ORIGINAL units) and
``get_active_objectives`` are retained as shared helpers used across
``evaluation.py`` and ``loop.py``.
"""

import numpy as np
import torch

from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.objective import (
    WeightedMCMultiOutputObjective,
)
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler

from mogp import TASK_NAMES, N_TASKS


# Per-objective optimization direction in TASK_NAMES order: +1 = higher better,
# -1 = lower better. Single source of truth for objective signs; MUST stay
# aligned with mogp.TASK_NAMES (same length, same order).
#   PfDHFR_Docking      -1  (minimize: strong parasite binding)
#   hDHFR_Docking       +1  (maximize: weak human binding -> selectivity)
#   hERG_Toxicity_Prob  -1  (minimize: cardiac safety)
#   Caco2_logPapp       +1  (maximize: intestinal permeability / absorption)
#   Half_Life_hours     +1  (maximize: metabolic stability)
DEFAULT_OBJECTIVE_SIGNS = [-1, +1, -1, +1, +1]

# qNEHVI quasi-Monte-Carlo posterior sample count.
N_MC_SAMPLES = 128

# optimize_acqf multi-start settings: number of L-BFGS-B restarts and the raw
# random points used to seed them. Modest values keep the mock fast while still
# exercising the real gradient-based optimizer.
NUM_RESTARTS = 10
RAW_SAMPLES = 256

# BoTorch multi-objective utilities work in double precision.
_DTYPE = torch.double


def _default_signs(num_objectives):
    """Return the default objective signs truncated to ``num_objectives``."""
    return list(DEFAULT_OBJECTIVE_SIGNS[:num_objectives])


def compute_pareto_front(Y, signs=None):
    """Find the Pareto-optimal rows of an objective matrix.

    Objectives may have mixed directions; the "lower is better" columns are
    negated internally so the dominance test is a uniform "higher is better"
    comparison. A point is dominated if another point is >= on every objective
    and strictly > on at least one.

    Args:
        Y: Objective matrix of shape ``(N, num_objectives)`` in ORIGINAL units.
        signs: Optional list of +1/-1 per objective. Defaults to
            ``DEFAULT_OBJECTIVE_SIGNS`` truncated to the number of columns.

    Returns:
        A tuple ``(pareto_mask, pareto_Y)`` where ``pareto_mask`` is a boolean
        array of shape ``(N,)`` (True for Pareto-optimal rows) and ``pareto_Y``
        is the array of Pareto-front rows in ORIGINAL units.
    """
    Y = np.asarray(Y, dtype=float)
    n, m = Y.shape
    if signs is None:
        signs = _default_signs(m)
    signs = np.asarray(signs, dtype=float)

    # Convert to a pure maximization frame: higher is better on every column.
    Y_max = Y * signs

    pareto_mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not pareto_mask[i]:
            continue
        ge_all = np.all(Y_max >= Y_max[i], axis=1)
        gt_any = np.any(Y_max > Y_max[i], axis=1)
        dominators = ge_all & gt_any
        if np.any(dominators):
            pareto_mask[i] = False

    return pareto_mask, Y[pareto_mask]


def get_reference_point(Y, signs=None):
    """Compute a hypervolume reference point from evaluated objectives.

    The reference point sits just past the worst observed value on each
    objective, in ORIGINAL units:

        higher-is-better column -> min(col) - 0.1 * range(col)
        lower-is-better column  -> max(col) + 0.1 * range(col)

    Args:
        Y: Objective matrix of shape ``(N, num_objectives)`` in ORIGINAL units.
        signs: Optional list of +1/-1 per objective.

    Returns:
        Reference point array of shape ``(num_objectives,)`` in ORIGINAL units.
    """
    Y = np.asarray(Y, dtype=float)
    m = Y.shape[1]
    if signs is None:
        signs = _default_signs(m)
    signs = np.asarray(signs, dtype=float)

    col_min = Y.min(axis=0)
    col_max = Y.max(axis=0)
    col_range = col_max - col_min

    ref = np.where(
        signs > 0,
        col_min - 0.1 * col_range,
        col_max + 0.1 * col_range,
    )
    return ref.astype(float)


def get_active_objectives(Y_evaluated):
    """Return indices of objective columns that have real (non all-NaN) data.

    Args:
        Y_evaluated: Objective matrix of shape ``(N, num_objectives)``.

    Returns:
        List of column indices that contain at least one finite value.
    """
    Y = np.asarray(Y_evaluated, dtype=float)
    return [j for j in range(Y.shape[1]) if np.isfinite(Y[:, j]).any()]


def _weighted_reference_point(Y, weights):
    """A qNEHVI reference point in the WEIGHTED (all-maximization) frame.

    ``Y`` is in original units; ``weights`` are the per-objective signs. In the
    weighted frame ``Y * weights`` every objective is "higher is better", so the
    reference sits just below the worst weighted value on each objective:
    ``min - (0.1 * range + eps)``. The small ``eps`` guarantees the reference is
    strictly dominated even when an objective is (near-)constant — as it is in the
    degenerate mock case where every decoded molecule scores identically.

    Returns a ``(num_objectives,)`` double tensor.
    """
    Yw = np.asarray(Y, dtype=float) * np.asarray(weights, dtype=float)
    col_min = Yw.min(axis=0)
    col_range = Yw.max(axis=0) - col_min
    ref = col_min - (0.1 * col_range + 1e-6)
    return torch.as_tensor(ref, dtype=_DTYPE)


def compute_qnehvi(model, train_x, train_y, bounds,
                   batch_size=1, objective_signs=None,
                   n_mc_samples=N_MC_SAMPLES,
                   num_restarts=NUM_RESTARTS, raw_samples=RAW_SAMPLES,
                   ref_point=None):
    """Optimize qNEHVI over the continuous latent box; return winning vector(s) z.

    Builds a noise-robust qNEHVI acquisition on the 5-output ``ModelListGP``
    posterior, wraps the per-objective directions into a weighted (all-
    maximization) objective, and runs ``optimize_acqf`` — gradient-based L-BFGS-B
    with multi-start restarts — over ``bounds`` to find the latent vector(s) that
    maximize expected hypervolume improvement over the current evaluated front.

    Args:
        model: A fitted ``mogp.ModelListGP`` (5 outputs, TASK_NAMES order).
        train_x: Evaluated latent vectors, shape ``(B, latent_dim)`` — the qNEHVI
            baseline (current front) ``X_baseline``.
        train_y: Evaluated objectives, shape ``(B, N_TASKS)`` in ORIGINAL units,
            used only to place the (weighted) reference point.
        bounds: Latent search box as a ``(2, latent_dim)`` tensor (row 0 lower,
            row 1 upper) — typically ``LatentSpaceBridge.bounds``.
        batch_size: ``q`` for ``optimize_acqf``; number of latent vectors to
            return jointly (they are naturally diverse — qNEHVI optimizes the
            batch's *joint* hypervolume, so no separate diversity filter is needed).
        objective_signs: +1/-1 per objective; defaults to ``DEFAULT_OBJECTIVE_SIGNS``.
        n_mc_samples: qNEHVI quasi-MC sample count.
        num_restarts, raw_samples: ``optimize_acqf`` multi-start settings.
        ref_point: Optional explicit reference point (weighted frame); defaults to
            ``_weighted_reference_point(train_y, signs)``.

    Returns:
        A tuple ``(candidates, acq_value)`` where ``candidates`` is a
        ``(batch_size, latent_dim)`` double tensor of latent vectors within
        ``bounds`` and ``acq_value`` is the scalar acquisition value achieved.
    """
    if objective_signs is None:
        objective_signs = _default_signs(N_TASKS)
    weights = torch.as_tensor(objective_signs, dtype=_DTYPE)

    X_baseline = torch.as_tensor(train_x, dtype=_DTYPE)
    Y = np.asarray(train_y, dtype=float)
    if X_baseline.shape[0] == 0:
        raise ValueError(
            "compute_qnehvi: empty baseline; need at least one evaluated "
            "molecule to define the current front."
        )
    bounds = torch.as_tensor(bounds, dtype=_DTYPE)

    # Flip minimize->maximize so qNEHVI (a maximizer) optimizes every objective in
    # the correct direction. The reference point lives in this same weighted frame.
    objective = WeightedMCMultiOutputObjective(weights=weights)
    ref = (_weighted_reference_point(Y, objective_signs)
           if ref_point is None else torch.as_tensor(ref_point, dtype=_DTYPE))

    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([int(n_mc_samples)]))
    acqf = qLogNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref,
        X_baseline=X_baseline,
        sampler=sampler,
        objective=objective,
        prune_baseline=True,
    )

    candidates, acq_value = optimize_acqf(
        acq_function=acqf,
        bounds=bounds,
        q=int(batch_size),
        num_restarts=int(num_restarts),
        raw_samples=int(raw_samples),
    )
    return candidates.detach(), acq_value.detach()


if __name__ == "__main__":
    # Self-contained smoke test of the continuous acquisition path: fit the latent
    # GP on random data, then verify optimize_acqf returns in-bounds latent
    # vectors AND that the acquisition is differentiable w.r.t. z (gradients flow).
    from mogp import train_mogp
    from vae_bridge import LatentSpaceBridge

    torch.manual_seed(0)
    np.random.seed(0)

    bridge = LatentSpaceBridge(latent_dim=50)
    latent_dim = bridge.latent_dim
    n_train = 16

    Z = np.random.uniform(-1.0, 1.0, size=(n_train, latent_dim))
    Y = np.zeros((n_train, N_TASKS), dtype=float)
    for j in range(N_TASKS):
        w = np.random.uniform(-1.0, 1.0, size=latent_dim)
        Y[:, j] = Z @ w + 0.05 * np.random.randn(n_train)

    print(f"Fitting latent GP on {n_train} points (dim={latent_dim})...")
    model = train_mogp(Z, Y)

    q = 3
    print(f"Optimizing qNEHVI with optimize_acqf (q={q})...")
    candidates, acq_value = compute_qnehvi(model, Z, Y, bridge.bounds, batch_size=q)

    print(f"candidates shape = {tuple(candidates.shape)} (expected ({q}, {latent_dim}))")
    assert candidates.shape == (q, latent_dim)
    lo, hi = bridge.bounds[0], bridge.bounds[1]
    assert torch.all(candidates >= lo - 1e-6) and torch.all(candidates <= hi + 1e-6), \
        "optimize_acqf returned an out-of-bounds latent vector"
    print(f"acq_value = {float(acq_value):.6f}")

    # --- Explicit gradient-flow check: rebuild the acqf and differentiate it
    # w.r.t. a latent input, confirming the numpy bridge no longer severs autograd.
    objective = WeightedMCMultiOutputObjective(
        weights=torch.tensor(DEFAULT_OBJECTIVE_SIGNS, dtype=_DTYPE)
    )
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([N_MC_SAMPLES]))
    acqf = qLogNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=_weighted_reference_point(Y, DEFAULT_OBJECTIVE_SIGNS),
        X_baseline=torch.as_tensor(Z, dtype=_DTYPE),
        sampler=sampler,
        objective=objective,
        prune_baseline=True,
    )
    z = torch.as_tensor(
        np.random.uniform(-1.0, 1.0, size=(1, 1, latent_dim)), dtype=_DTYPE
    ).requires_grad_(True)
    val = acqf(z)
    val.backward()
    assert z.grad is not None and torch.isfinite(z.grad).all(), \
        "gradient did not flow from qNEHVI back to the latent vector z"
    grad_norm = float(z.grad.norm())
    print(f"grad ||dAcq/dz|| = {grad_norm:.6e} (finite, non-None -> autograd intact)")

    print("\nCONTINUOUS ACQUISITION + GRADIENT-FLOW TEST PASSED")
