"""
loop.py
=======

De-Novo / Latent-Space Bayesian Optimization (LSBO) loop for antimalarial drug
discovery against *Plasmodium falciparum* dihydrofolate reductase (PfDHFR).

This replaces the old **virtual-screening** loop (which searched a fixed library
by discrete index) with **generative** search over a continuous latent space.
Each iteration:

    1. Train the latent-space multi-output GP (``mogp.ModelListGP`` — 5
       ``SingleTaskGP``s with Matern-2.5 kernels) on every (latent vector,
       5-objective) pair evaluated so far, across the objectives in ``TASK_NAMES``.
    2. Optimize qNEHVI over the continuous latent box with ``optimize_acqf``
       (``acquisition.compute_qnehvi``) to obtain the winning latent vector(s)
       ``z`` — the batch that most expands the Pareto front.
    3. **Decode** ``z`` into brand-new SMILES via the VAE bridge
       (``vae_bridge.LatentSpaceBridge.decode``), then evaluate those molecules
       with the real oracles: ``admet_oracle.ADMETOracle.predict`` (3 ADMET
       objectives) and ``docking.batch_dock_targets`` (2 docking objectives, one
       dock per target).
    4. Append the new (z, y) pairs, recompute the Pareto front and hypervolume,
       and record history.

Unlike the virtual-screening loop, NOTHING is precomputed per candidate: a
decoded molecule did not exist until this iteration, so all five objectives — not
just docking — are produced on demand and modelled by the GP.

NOTE (mock phase): the docking objective columns hold the RAW Vina affinity
(kcal/mol), not the size-corrected ligand efficiency the virtual-screening loop
used. The ligand-efficiency normalization was tied to the fixed-library
hypervolume comparison, which does not apply to a continuous generative search;
raw kcal keeps the mock self-contained. Hypervolume here is computed directly in
the signed (all-maximization) objective frame against a reference point fixed at
initialization, so it is monotone within a run.

Run ``python loop.py --help`` for the command-line options.
"""

import os
import time
import argparse

import numpy as np
import torch
import pandas as pd

from botorch.utils.multi_objective.hypervolume import Hypervolume

from vae_bridge import LatentSpaceBridge
from admet_oracle import ADMETOracle
from mogp import train_mogp, TASK_NAMES, N_TASKS, resolve_objective_layout
from acquisition import (
    compute_qnehvi,
    compute_pareto_front,
    get_active_objectives,
    DEFAULT_OBJECTIVE_SIGNS,
)
import evaluation
from docking import batch_dock_targets, docked_summary


# Objective -> data-source layout, resolved once from TASK_NAMES. In the
# generative setting every objective is produced on demand for a decoded
# molecule: two docking targets (expensive) + three ADMET oracle columns (cheap).
ADMET_TASKS, DOCKING_TASKS, DOCKING_TARGETS = resolve_objective_layout()
_SIGNS = np.asarray(DEFAULT_OBJECTIVE_SIGNS, dtype=float)


class BOLoop:
    """Latent-space multi-objective Bayesian optimization loop.

    Searches a bounded continuous latent space (via a mock VAE bridge) rather
    than a fixed molecule library. Each iteration trains a 5-output latent GP,
    optimizes qNEHVI with ``optimize_acqf`` to propose latent vectors, decodes
    them to SMILES, evaluates all five objectives with the real oracles, and
    folds the results back in.
    """

    def __init__(self, seed=42, latent_dim=50,
                 n_init=10, batch_size=5, n_iterations=10,
                 mogp_train_iters=200, mogp_lr=0.1,
                 bridge=None, oracle=None, train_fn=None):
        # --- Reproducibility ---
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        # --- VAE bridge (latent <-> SMILES) and latent geometry ---
        self.bridge = bridge if bridge is not None else LatentSpaceBridge(latent_dim)
        self.latent_dim = self.bridge.latent_dim
        self.bounds = self.bridge.bounds            # (2, latent_dim) tensor

        # --- Evaluation oracles ---
        self.oracle = oracle if oracle is not None else ADMETOracle()
        self.admet_tasks = ADMET_TASKS
        self.docking_tasks = DOCKING_TASKS
        self.docking_targets = DOCKING_TARGETS

        # --- GP training function (the latent ModelListGP by default) ---
        self.train_fn = train_fn if train_fn is not None else train_mogp

        # --- Hyperparameters ---
        self.n_init = n_init
        self.batch_size = batch_size
        self.n_iterations = n_iterations
        self.mogp_train_iters = mogp_train_iters
        self.mogp_lr = mogp_lr

        # --- Tracking state ---
        # Z_evaluated (N, latent_dim) is the GP's training X; smiles[i] is the
        # molecule decoded from Z_evaluated[i]; Y_evaluated (N, N_TASKS) its scores.
        self.Z_evaluated = np.empty((0, self.latent_dim), dtype=np.float64)
        self.Y_evaluated = np.empty((0, N_TASKS), dtype=np.float64)
        self.smiles = []
        self.history = []
        self.hv_ref_point = None   # fixed at initialize(), signed/maximization frame

    # ------------------------------------------------------------------ #
    # Evaluation helper
    # ------------------------------------------------------------------ #
    def _evaluate(self, smiles_list):
        """Score decoded SMILES on all 5 objectives (ORIGINAL units).

        Runs the ADMET oracle (3 objectives) and docks against every required
        target (2 objectives). Failed docks / un-featurizable molecules leave NaN
        in the affected columns.

        Returns:
            A tuple ``(Y, docking_by_target)`` where ``Y`` has shape
            ``(len(smiles_list), N_TASKS)`` with columns in ``TASK_NAMES`` order,
            and ``docking_by_target`` maps each target name to its ``(k,)`` raw
            affinity vector.
        """
        smiles_list = list(smiles_list)
        admet_df = self.oracle.predict(smiles_list)
        docking_by_target = batch_dock_targets(smiles_list, self.docking_targets)

        Y = np.full((len(smiles_list), N_TASKS), np.nan, dtype=np.float64)
        for j, col in self.admet_tasks:
            Y[:, j] = admet_df[col].to_numpy(dtype=float)
        for j, target in self.docking_tasks:
            Y[:, j] = np.asarray(docking_by_target[target], dtype=float)
        return Y, docking_by_target

    def _append(self, Z_new, smiles_new, Y_new):
        """Append a batch of (latent, SMILES, objective) records to loop state."""
        self.Z_evaluated = np.vstack([self.Z_evaluated, np.asarray(Z_new)])
        self.Y_evaluated = np.vstack([self.Y_evaluated, np.asarray(Y_new)])
        self.smiles.extend(smiles_new)

    # ------------------------------------------------------------------ #
    # Pareto / hypervolume helpers
    # ------------------------------------------------------------------ #
    def _finite_mask(self):
        """Rows fully observed across all 5 objectives (a failed eval leaves NaN)."""
        if len(self.Y_evaluated) == 0:
            return np.zeros(0, dtype=bool)
        return np.isfinite(self.Y_evaluated).all(axis=1)

    def _pareto_mask(self):
        """Boolean mask over evaluated rows: True for Pareto-optimal molecules."""
        Y = self.Y_evaluated
        full_mask = np.zeros(len(Y), dtype=bool)
        finite = self._finite_mask()
        if finite.any():
            sub_mask, _ = compute_pareto_front(Y[finite], _SIGNS)
            full_mask[np.where(finite)[0]] = sub_mask
        return full_mask

    def _hypervolume(self):
        """Dominated hypervolume in the signed (maximization) frame.

        Uses the reference point fixed at ``initialize`` so the metric is
        comparable across iterations of a single run. Points that do not dominate
        the reference contribute nothing.
        """
        finite = self._finite_mask()
        if self.hv_ref_point is None or not finite.any():
            return 0.0
        Yw = self.Y_evaluated[finite] * _SIGNS
        ref = self.hv_ref_point
        dominating = np.all(Yw > ref, axis=1)
        if not dominating.any():
            return 0.0
        hv = Hypervolume(torch.as_tensor(ref, dtype=torch.double))
        return float(hv.compute(torch.as_tensor(Yw[dominating], dtype=torch.double)))

    # ------------------------------------------------------------------ #
    # Main loop stages
    # ------------------------------------------------------------------ #
    def initialize(self):
        """Seed the loop with ``n_init`` random latent vectors, decoded & evaluated."""
        lo = self.bounds[0].cpu().numpy()
        hi = self.bounds[1].cpu().numpy()
        Z0 = np.random.uniform(lo, hi, size=(self.n_init, self.latent_dim))
        smiles0 = self.bridge.decode(torch.as_tensor(Z0))

        print(f"Initializing with {self.n_init} random latent vectors "
              f"(dim={self.latent_dim})...")
        Y0, docking = self._evaluate(smiles0)
        self._append(Z0, smiles0, Y0)

        # Fix the hypervolume reference point from the initial batch, in the signed
        # (maximization) frame: just below the worst value on each objective.
        finite = np.isfinite(Y0).all(axis=1)
        Yw = Y0[finite] * _SIGNS if finite.any() else Y0 * _SIGNS
        col_min = Yw.min(axis=0)
        col_range = Yw.max(axis=0) - col_min
        self.hv_ref_point = col_min - (0.1 * col_range + 1e-6)

        print(f"Initialized {self.n_init} molecules; "
              f"docked {docked_summary(docking, self.n_init)}.")

    def step(self):
        """Run one BO iteration: train, optimize qNEHVI, decode, evaluate, record."""
        iteration = len(self.history) + 1

        # --- Train the latent GP on everything fully evaluated so far ---
        finite = self._finite_mask()
        Z_train = self.Z_evaluated[finite]
        Y_train = self.Y_evaluated[finite]
        print(f"\n[Iteration {iteration}] Training latent GP on "
              f"{int(finite.sum())}/{len(self.Y_evaluated)} fully-evaluated molecules...")
        model = self.train_fn(
            Z_train, Y_train,
            n_iterations=self.mogp_train_iters, lr=self.mogp_lr,
        )

        # --- Optimize qNEHVI over the continuous latent box ---
        # optimize_acqf returns a (batch_size, latent_dim) tensor of latent
        # vectors that jointly maximize expected hypervolume improvement.
        candidates, acq_value = compute_qnehvi(
            model, Z_train, Y_train, self.bounds, batch_size=self.batch_size,
        )
        Z_new = candidates.cpu().numpy()

        # --- Decode the winning latent vectors into brand-new molecules ---
        smiles_new = self.bridge.decode(candidates)

        # --- Evaluate the decoded molecules on all 5 objectives ---
        Y_new, docking_new = self._evaluate(smiles_new)
        batch_docked = docked_summary(docking_new, len(smiles_new))
        self._append(Z_new, smiles_new, Y_new)

        # --- Track Pareto front + hypervolume ---
        pareto_mask = self._pareto_mask()
        pareto_size = int(pareto_mask.sum())
        hypervolume = self._hypervolume()

        self.history.append({
            "iteration": iteration,
            "n_evaluated": len(self.Y_evaluated),
            "pareto_size": pareto_size,
            "hypervolume": hypervolume,
            "acq_value": float(acq_value),
            "batch_smiles": list(smiles_new),
        })

        print(f"[Iteration {iteration}] "
              f"evaluated={len(self.Y_evaluated)}, "
              f"batch={len(smiles_new)}, "
              f"docked_this_batch=[{batch_docked}], "
              f"pareto_size={pareto_size}, hypervolume={hypervolume:.4f}, "
              f"acq_value={float(acq_value):.4f}")

    def run(self):
        """Run the complete loop: initialize, then ``n_iterations`` steps."""
        self.initialize()
        for _ in range(1, self.n_iterations + 1):
            self.step()

        final = self.history[-1] if self.history else {}
        print("\n=== LSBO run complete ===")
        print(f"  Total molecules evaluated: {len(self.Y_evaluated)}")
        print(f"  Final Pareto front size:   {final.get('pareto_size', 0)}")
        print(f"  Final hypervolume:         {final.get('hypervolume', 0.0):.4f}")
        return self.history

    # ------------------------------------------------------------------ #
    # Outputs
    # ------------------------------------------------------------------ #
    def get_pareto_front(self):
        """Return the current Pareto front as a dict of aligned fields."""
        mask = self._pareto_mask()
        rows = np.where(mask)[0]
        return {
            "rows": rows,
            "smiles": [self.smiles[r] for r in rows],
            "objectives": self.Y_evaluated[rows],
            "latent": self.Z_evaluated[rows],
            "task_names": TASK_NAMES,
        }

    def save_results(self, output_dir="results"):
        """Write history, all evaluations, and the Pareto front to ``output_dir``."""
        os.makedirs(output_dir, exist_ok=True)

        # history.csv
        history_df = pd.DataFrame([
            {
                "iteration": h["iteration"],
                "n_evaluated": h["n_evaluated"],
                "pareto_size": h["pareto_size"],
                "hypervolume": h["hypervolume"],
                "acq_value": h.get("acq_value", float("nan")),
            }
            for h in self.history
        ])
        history_path = os.path.join(output_dir, "history.csv")
        history_df.to_csv(history_path, index=False)

        # evaluated.csv — every evaluated molecule with all 5 objectives, plus the
        # reported-only Selectivity Index (hDHFR - PfDHFR).
        evaluated_df = pd.DataFrame({"SMILES": list(self.smiles)})
        for j, name in enumerate(TASK_NAMES):
            evaluated_df[name] = self.Y_evaluated[:, j]
        evaluation.add_selectivity_index(evaluated_df)
        evaluated_path = os.path.join(output_dir, "evaluated.csv")
        evaluated_df.to_csv(evaluated_path, index=False)

        # pareto_front.csv — only the Pareto-optimal molecules.
        pareto = self.get_pareto_front()
        pareto_df = pd.DataFrame({"SMILES": pareto["smiles"]})
        for j, name in enumerate(TASK_NAMES):
            pareto_df[name] = pareto["objectives"][:, j]
        evaluation.add_selectivity_index(pareto_df)
        pareto_path = os.path.join(output_dir, "pareto_front.csv")
        pareto_df.to_csv(pareto_path, index=False)

        print(f"Saved results to {output_dir}/:")
        print(f"  {history_path}")
        print(f"  {evaluated_path}")
        print(f"  {pareto_path}")
        return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the latent-space (de-novo) BO loop for PfDHFR drug discovery."
    )
    parser.add_argument("--latent-dim", type=int, default=50)
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--n-iterations", type=int, default=10)
    parser.add_argument("--mogp-iters", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    start = time.time()

    print("Running latent-space BO loop (mock VAE bridge).")
    loop = BOLoop(
        seed=args.seed,
        latent_dim=args.latent_dim,
        n_init=args.n_init,
        batch_size=args.batch_size,
        n_iterations=args.n_iterations,
        mogp_train_iters=args.mogp_iters,
    )
    loop.run()

    pareto = loop.get_pareto_front()
    print(f"\nPareto-optimal molecules: {len(pareto['smiles'])}")
    print(f"{'SMILES':<50}" + "".join(f"{n:>22}" for n in pareto["task_names"]))
    for smiles, row in zip(pareto["smiles"], pareto["objectives"]):
        print(f"{smiles:<50}" + "".join(f"{v:22.4f}" for v in row))

    loop.save_results(output_dir=args.output_dir)

    elapsed = time.time() - start
    print(f"\nTotal wall-clock time: {elapsed:.1f}s")
