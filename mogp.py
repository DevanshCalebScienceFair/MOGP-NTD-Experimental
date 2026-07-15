"""Multi-output Gaussian Process over a CONTINUOUS latent molecular space.

This module was rebuilt for the De-Novo / Latent-Space Bayesian Optimization
(LSBO) migration. It no longer models fingerprints with a Tanimoto kernel; it
models the VAE **latent vectors** (``vae_bridge.LatentSpaceBridge``) with a
smooth stationary kernel, which is what a gradient-based acquisition optimizer
(``botorch.optim.optimize_acqf``) needs to differentiate through.

Two architectural shifts from the old grey-box design:

  * **All 5 objectives are modelled.** The old code modelled only the 2 docking
    objectives and folded in known-exact ADMET at acquisition time. In the
    generative setting molecules are invented, so their ADMET is *not* known in
    advance — every objective (2 docking + 3 ADMET) is now predicted by the GP.
  * **Native BoTorch model with coregionalized docking.** The model is a
    ``ModelListGP`` whose 5 outputs come from: one multi-task ``MultiTaskGP`` that
    models the two correlated docking objectives JOINTLY (an ICM/coregionalization
    kernel lets an observation of one DHFR target inform the other), plus three
    independent ``SingleTaskGP``s for the mechanistically-unrelated ADMET
    objectives (each ``ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=latent_dim))``
    + ``Standardize(1)``). All fit with ``fit_gpytorch_mll``. Because it is a
    first-class BoTorch model, its posterior is differentiable w.r.t. the latent
    input, so ``optimize_acqf`` can push gradients from the acquisition all the way
    back to the latent vector ``z``. See ``build_model`` for the layout contract.

Predictions are returned in the fixed ``TASK_NAMES`` order
``[PfDHFR_Docking, hDHFR_Docking, hERG_Toxicity_Prob, Caco2_logPapp,
Half_Life_hours]`` — now ALL five columns carry real predictions.

Run ``python mogp.py`` for a self-contained demo on random latent data.
"""

import numpy as np
import torch

from botorch.models import SingleTaskGP, ModelListGP, MultiTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import SumMarginalLogLikelihood
from gpytorch.kernels import ScaleKernel, MaternKernel


# The fixed column order every Y matrix / prediction in this module adheres to.
# This is the SINGLE SOURCE OF TRUTH for objective order everywhere in the
# project (train_mogp, predict, acquisition, loop, evaluation, dashboard).
# Import TASK_NAMES rather than hard-coding column strings.
#
# 5-objective POTENCY / SELECTIVITY / SAFETY / ADMET problem:
#   PfDHFR_Docking      minimize (strong parasite binding)
#   hDHFR_Docking       MAXIMIZE (weak *human* binding -> selectivity)
#   hERG_Toxicity_Prob  minimize (cardiac safety)
#   Caco2_logPapp       MAXIMIZE (intestinal permeability / absorption)
#   Half_Life_hours     MAXIMIZE (metabolic stability)
#
# Each name maps to its evaluation-time source in OBJECTIVE_SOURCES below, and to
# its optimization direction in acquisition.DEFAULT_OBJECTIVE_SIGNS (order-aligned
# with this list). In the LSBO setting all five are GP-modelled.
TASK_NAMES = [
    "PfDHFR_Docking",
    "hDHFR_Docking",
    "hERG_Toxicity_Prob",
    "Caco2_logPapp",
    "Half_Life_hours",
]

N_TASKS = len(TASK_NAMES)


# How each objective is PRODUCED at evaluation time, for a newly DECODED molecule:
#   ("dock", "<TargetName>")    -> docking.batch_dock_targets against that named
#                                  receptor (see docking.TARGETS).
#   ("admet", "<column>")       -> a column of admet_oracle.ADMETOracle.predict().
# In the generative setting the ADMET values are NOT precomputed (the molecule did
# not exist until it was decoded), so they are produced on demand from the oracle
# exactly like the docking scores. loop.py uses this to assemble the 5-objective
# row for each decoded SMILES. Extra entries for objectives not in TASK_NAMES are
# harmless and support swapping the objective set later.
OBJECTIVE_SOURCES = {
    "PfDHFR_Docking":     ("dock", "PfDHFR"),
    "hDHFR_Docking":      ("dock", "hDHFR"),
    "hERG_Toxicity_Prob": ("admet", "hERG_Toxicity_Prob"),
    "Caco2_logPapp":      ("admet", "Caco2_logPapp"),
    "Half_Life_hours":    ("admet", "Half_Life_hours"),
}


# Task indices (into TASK_NAMES) produced by docking vs. the ADMET oracle.
# Derived from OBJECTIVE_SOURCES so nothing keys off a hard-coded column position.
DOCKING_TASK_INDICES = [
    j for j, name in enumerate(TASK_NAMES) if OBJECTIVE_SOURCES[name][0] == "dock"
]
ADMET_TASK_INDICES = [
    j for j, name in enumerate(TASK_NAMES) if OBJECTIVE_SOURCES[name][0] == "admet"
]

# BoTorch multi-objective math runs in double precision throughout.
_DTYPE = torch.double


def resolve_objective_layout():
    """Map ``TASK_NAMES`` onto their evaluation-time data sources.

    Returns:
        A tuple ``(admet_tasks, docking_tasks, docking_targets)`` where:
          * ``admet_tasks``     = list of ``(task_index, admet_column_name)``
          * ``docking_tasks``   = list of ``(task_index, target_name)``
          * ``docking_targets`` = ordered unique target names to dock.

    Unlike the pre-LSBO version this takes no ``admet_columns`` argument: ADMET
    objectives now name their oracle output column directly (the oracle returns
    columns by the same ``TASK_NAMES`` strings), so there is no positional
    ``admet_scores`` array to resolve against.
    """
    admet_tasks = []
    docking_tasks = []
    for j, name in enumerate(TASK_NAMES):
        kind, ref = OBJECTIVE_SOURCES[name]
        if kind == "admet":
            admet_tasks.append((j, ref))
        elif kind == "dock":
            docking_tasks.append((j, ref))
        else:
            raise ValueError(f"Unknown objective source {kind!r} for {name!r}.")
    docking_targets = []
    for _, target in docking_tasks:
        if target not in docking_targets:
            docking_targets.append(target)
    return admet_tasks, docking_tasks, docking_targets


def _single_task_gp(X, y_col, latent_dim):
    """One independent ``SingleTaskGP`` for a single objective column ``(N, 1)``."""
    return SingleTaskGP(
        X,
        y_col,
        covar_module=ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=latent_dim)),
        outcome_transform=Standardize(m=1),
    )


def build_model(train_x, train_y):
    """Construct (untrained) the coregionalized ``ModelListGP`` over latent space.

    The two DOCKING objectives (PfDHFR vs. hDHFR) are biologically correlated —
    both measure binding to a dihydrofolate reductase — so they are modelled
    JOINTLY by a single coregionalized (multi-task) GP that shares information
    across the two targets through a learned inter-task covariance. Concretely a
    ``MultiTaskGP`` with an ICM/coregionalization kernel
    ``data_kernel(x) * task_kernel(task)``: a Matern kernel over the latent inputs
    multiplied by a positive-correlation ``IndexKernel`` over the two task ids, so
    an observation of one target informs the posterior of the other.

    The three ADMET objectives (hERG, Caco2, half-life) are mechanistically
    unrelated to each other and to docking, so each stays an INDEPENDENT
    ``SingleTaskGP`` exactly as before.

    All submodels are bundled into one ``ModelListGP`` whose concatenated outputs
    stay in ``TASK_NAMES`` order, so ``optimize_acqf`` / qNEHVI see a single
    5-output model. This requires the docking tasks to occupy the LEADING
    contiguous block of ``TASK_NAMES`` and ADMET the trailing block (true in this
    repo: docking = 0,1; ADMET = 2,3,4); a reorder is rejected loudly below.

    Args:
        train_x: Latent-vector tensor/array of shape ``(N, latent_dim)``.
        train_y: Target tensor/array of shape ``(N, N_TASKS)``, columns in
            ``TASK_NAMES`` order. Must be fully observed (no NaN); the loop
            filters failed evaluations before calling.

    Returns:
        An unfitted ``ModelListGP``: ``[MultiTaskGP(docking, 2-output)]`` followed
        by one ``SingleTaskGP`` per ADMET objective. When there are not exactly two
        docking tasks it degrades gracefully to all-independent ``SingleTaskGP``s.
    """
    X = torch.as_tensor(train_x, dtype=_DTYPE)
    Y = torch.as_tensor(train_y, dtype=_DTYPE)
    if X.ndim != 2:
        raise ValueError(f"train_x must be 2-D (N, latent_dim); got {tuple(X.shape)}.")
    if Y.ndim != 2 or Y.shape[1] != N_TASKS:
        raise ValueError(
            f"train_y must be (N, {N_TASKS}) in TASK_NAMES order; got {tuple(Y.shape)}."
        )
    if not torch.isfinite(Y).all():
        raise ValueError(
            "build_model: train_y contains non-finite values; filter failed "
            "evaluations before fitting (each GP needs observed targets)."
        )

    latent_dim = X.shape[-1]
    dock_idx = DOCKING_TASK_INDICES
    admet_idx = ADMET_TASK_INDICES

    # Fall back to the original all-independent design unless there are exactly
    # two docking tasks forming the leading block (with ADMET as the trailing
    # block), which is the only layout for which the coregionalized ModelListGP
    # output order provably matches TASK_NAMES.
    coregionalize = (
        len(dock_idx) == 2
        and dock_idx == list(range(len(dock_idx)))
        and admet_idx == list(range(len(dock_idx), N_TASKS))
    )
    if not coregionalize:
        models = [_single_task_gp(X, Y[:, i : i + 1], latent_dim)
                  for i in range(N_TASKS)]
        return ModelListGP(*models)

    # Coregionalized GP over the two correlated docking tasks. MultiTaskGP wants
    # task-augmented training data: each latent point appears once per task with a
    # trailing task-id column. Outputs are requested via output_tasks=[0, 1], so
    # the 2-output posterior lands in [PfDHFR, hDHFR] == TASK_NAMES[0:2] order.
    #
    # NOTE: outcome_transform=None. MultiTaskGP would otherwise apply a DEFAULT
    # Standardize, whose untransformed posterior does not expose the ``.distribution``
    # that ModelListGP's multi-output combine path needs; raw docking kcal (~-10..0)
    # is already well-scaled and the per-task mean absorbs the offset, matching the
    # Standardize-untransformed ADMET GPs (predictions stay in ORIGINAL units).
    Y_dock = Y[:, dock_idx]                                    # (N, 2)
    n = X.shape[0]
    task_col_0 = torch.zeros(n, 1, dtype=_DTYPE)
    task_col_1 = torch.ones(n, 1, dtype=_DTYPE)
    X_aug = torch.cat(
        [torch.cat([X, task_col_0], dim=-1), torch.cat([X, task_col_1], dim=-1)],
        dim=0,
    )                                                          # (2N, latent_dim+1)
    Y_aug = torch.cat([Y_dock[:, 0:1], Y_dock[:, 1:2]], dim=0)  # (2N, 1)
    dock_gp = MultiTaskGP(
        X_aug,
        Y_aug,
        task_feature=latent_dim,          # the appended task-id column
        output_tasks=[0, 1],
        rank=len(dock_idx),               # full-rank 2x2 inter-task covariance
        covar_module=MaternKernel(nu=2.5, ard_num_dims=latent_dim),
        outcome_transform=None,
    )

    # Independent GPs for the three ADMET tasks (outputs land at indices 2,3,4).
    admet_gps = [_single_task_gp(X, Y[:, j : j + 1], latent_dim) for j in admet_idx]

    # ModelListGP concatenates submodel outputs in order -> [0,1] then [2,3,4],
    # i.e. exactly TASK_NAMES order for the downstream acquisition.
    return ModelListGP(dock_gp, *admet_gps)


def train_mogp(train_x, train_y, n_iterations=None, lr=None):
    """Fit the latent-space ``ModelListGP`` with ``fit_gpytorch_mll``.

    Args:
        train_x: Latent vectors, shape ``(N, latent_dim)``.
        train_y: Targets, shape ``(N, N_TASKS)`` in ``TASK_NAMES`` order,
            fully observed.
        n_iterations, lr: Accepted for signature compatibility with the old
            Adam-based trainer and IGNORED — ``fit_gpytorch_mll`` uses BoTorch's
            default (scipy L-BFGS-B) fit, which needs neither.

    Returns:
        The fitted ``ModelListGP``. (The old ``(model, likelihood, y_mean, y_std)``
        4-tuple is gone: the likelihood lives inside the model and normalization is
        handled by each model's ``Standardize`` transform.)
    """
    model = build_model(train_x, train_y)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


def predict(model, X_new):
    """Predict all 5 objectives (original units) for new latent vectors.

    Args:
        model: A fitted ``ModelListGP`` from ``train_mogp``.
        X_new: Latent vectors of shape ``(M, latent_dim)``.

    Returns:
        A tuple ``(mean, variance)`` of numpy arrays, each shape ``(M, N_TASKS)``,
        columns in ``TASK_NAMES`` order on the original (de-standardized) scale.
        The ``Standardize`` outcome transforms are inverted by the posterior, so
        these are already in real units.
    """
    X = torch.as_tensor(X_new, dtype=_DTYPE)
    model.eval()
    with torch.no_grad():
        posterior = model.posterior(X)
        mean = posterior.mean.cpu().numpy()
        variance = posterior.variance.cpu().numpy()
    return mean, variance


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    latent_dim = 50
    n_train = 16

    # Random latent training set in the [-1, 1] box, with synthetic but smoothly
    # varying targets so each per-objective GP has real signal to fit.
    Z = np.random.uniform(-1.0, 1.0, size=(n_train, latent_dim))
    Y = np.zeros((n_train, N_TASKS), dtype=float)
    for j in range(N_TASKS):
        w = np.random.uniform(-1.0, 1.0, size=latent_dim)
        Y[:, j] = Z @ w + 0.05 * np.random.randn(n_train)

    print(f"Fitting ModelListGP: {n_train} latent points, dim={latent_dim}, "
          f"{N_TASKS} objectives ({len(DOCKING_TASK_INDICES)} docking, "
          f"{len(ADMET_TASK_INDICES)} ADMET)...")
    model = train_mogp(Z, Y)
    print("Fit complete.")

    # Predict on fresh latent points.
    Z_new = np.random.uniform(-1.0, 1.0, size=(3, latent_dim))
    mean, variance = predict(model, Z_new)
    std = np.sqrt(variance)

    assert mean.shape == (3, N_TASKS), mean.shape
    assert np.isfinite(mean).all(), "predictions must be finite for all 5 objectives"

    print(f"\nPredictions (shape {mean.shape}, all 5 objectives modelled):")
    print(f"{'point':>6}" + "".join(f"{h:>22}" for h in TASK_NAMES))
    for i in range(mean.shape[0]):
        print(f"{i:>6}" + "".join(
            f"{mean[i, j]:>12.3f}±{std[i, j]:<9.3f}" for j in range(N_TASKS)
        ))

    print("\nMOGP (ModelListGP over latent space) OK")
