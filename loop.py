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
import sys
import time
import argparse

import numpy as np
import torch
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, QED
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

# PAINS (Pan-Assay INterference compoundS) substructure catalog, built once. These
# promiscuous scaffolds hit many unrelated assays and are classic false-positive
# "hits"; any decoded molecule matching one is vetoed before docking so the Pareto
# front holds only biologically credible candidates.
_pains_params = FilterCatalogParams()
_pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
_PAINS_CATALOG = FilterCatalog(_pains_params)

# --- Synthesizability metric (contrib import, handled cleanly) --------------
# Prefer Ertl's Synthetic Accessibility (SA) score (1 = easy .. 10 = hard),
# the standard "can a chemist actually make this?" heuristic. It lives in
# RDKit's *contrib* tree, so it needs an explicit sys.path hookup that can fail
# on some installs; we guard the import and fall back to QED (RDKit core,
# 0 = poor .. 1 = excellent drug-likeness) if SA is unavailable. Either way the
# gate below is deliberately generous.
try:
    from rdkit.Chem import RDConfig
    _sa_dir = os.path.join(RDConfig.RDContribDir, "SA_Score")
    if _sa_dir not in sys.path:
        sys.path.append(_sa_dir)
    import sascorer  # noqa: E402  (RDKit contrib module, added to path above)
    _SYNTH_METRIC = "SA"
except Exception:
    sascorer = None
    _SYNTH_METRIC = "QED"

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


# --- Pre-dock structural filter (Lipinski Ro5 + a heavy-atom cap) ------------
# A cheap RDKit screen applied to every decoded molecule BEFORE the expensive
# dock. Oversized / non-drug-like molecules (which a generative VAE readily
# proposes, and which are slow to dock and out of the ADMET oracle's domain) are
# rejected here, never docked, and given a penalty objective row instead.
MAX_HEAVY_ATOMS = 45    # widened 35->45: give the optimizer room for larger,
                        # high-potency scaffolds before penalties bite
MW_MAX = 500.0          # Lipinski molecular weight (retained)
LOGP_MAX = 5.0          # Lipinski cLogP (retained)
HBD_MAX = 5             # Lipinski H-bond donors (retained)
HBA_MAX = 10            # Lipinski H-bond acceptors (retained)

# Synthesizability / drug-likeness gate — deliberately GENEROUS so we never throw
# away a highly potent scaffold just because it is a bit awkward to synthesize.
# Only one of these applies, depending on which metric imported above:
#   SA score : reject if > SA_SCORE_MAX (typical marketed drugs sit ~2-5; 7.2 is
#              very lax, letting highly active but slightly complex scaffolds survive)
#   QED      : reject if < QED_MIN      (0.3 keeps all but clearly non-drug-like mols)
SA_SCORE_MAX = 7.2
QED_MIN = 0.3

# Penalty objective row assigned to a REJECTED molecule (TASK_NAMES order). Each
# value is directionally the WORST outcome for its objective given
# DEFAULT_OBJECTIVE_SIGNS = [-1, +1, -1, +1, +1], so a rejected point is dominated
# by any real molecule and never joins the Pareto front / earns hypervolume. The
# magnitudes are deliberately "worse than any realistic molecule" but BOUNDED
# (not +/-inf): the GP fits on these rows so the MOGP learns to avoid the latent
# region, and wildly extreme values would blow up each objective's Standardize
# normalization and swamp the signal from real molecules.
#   PfDHFR_Docking (min)  ->  0.0    : ~no parasite binding (worst)
#   hDHFR_Docking  (max)  -> -14.0   : very strong human binding (worst selectivity)
#   hERG_Tox_Prob  (min)  ->  1.0    : certain hERG blocker
#   Caco2_logPapp  (max)  -> -8.0    : very poor permeability
#   Half_Life_hours(max)  ->  0.0    : ~zero half-life
REJECTION_PENALTY = np.array([0.0, -14.0, 1.0, -8.0, 0.0], dtype=np.float64)

# Mode-collapse guard, molecule level. The latent-distance guard in
# acquisition.compute_qnehvi keeps proposed z's apart, but the VAE decoder is
# MANY-TO-ONE: distinct z's can still decode to a molecule we already evaluated.
# Each BO step therefore over-generates from qNEHVI and keeps only decodes that
# are novel (vs. every prior molecule and vs. the batch so far), re-querying for
# any shortfall up to this many rounds before proceeding with what it has.
MAX_PROPOSAL_ROUNDS = 5

# epsilon-greedy wildcard. With this probability a BO step SKIPS qNEHVI entirely
# and proposes a batch of purely random (in-bounds) latent vectors. This injects
# unconditioned exploration so the search can never get permanently pinned in a
# single basin of the acquisition surface (a local optimum). The molecule-novelty
# guard still applies to wildcard batches; only the *proposal* mechanism changes.
EPSILON_GREEDY = 0.1


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
                 mogp_train_iters=80, mogp_lr=0.1, epsilon=EPSILON_GREEDY,
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
        self.epsilon = epsilon

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
    @staticmethod
    def _screen(smiles):
        """Cheap RDKit drug-likeness screen. Returns ``(passed, reason)``.

        Rejects molecules that are unparseable, exceed ``MAX_HEAVY_ATOMS``,
        violate the Lipinski Rule-of-Five thresholds (MW, cLogP, HBD, HBA), match a
        PAINS interference substructure, or fail a generous synthesizability gate
        (SA score, or QED as a fallback).
        """
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None:
            return False, "unparseable"
        if mol.GetNumHeavyAtoms() > MAX_HEAVY_ATOMS:
            return False, f"heavy_atoms>{MAX_HEAVY_ATOMS}"
        if Descriptors.MolWt(mol) > MW_MAX:
            return False, f"MW>{MW_MAX:g}"
        if Crippen.MolLogP(mol) > LOGP_MAX:
            return False, f"LogP>{LOGP_MAX:g}"
        if Lipinski.NumHDonors(mol) > HBD_MAX:
            return False, f"HBD>{HBD_MAX}"
        if Lipinski.NumHAcceptors(mol) > HBA_MAX:
            return False, f"HBA>{HBA_MAX}"
        # Chemist's veto: reject pan-assay interference (PAINS) substructures.
        if _PAINS_CATALOG.HasMatch(mol):
            return False, "PAINS"
        # Generous synthesizability / drug-likeness gate (SA preferred, QED fallback).
        if sascorer is not None:
            if sascorer.calculateScore(mol) > SA_SCORE_MAX:
                return False, f"SA>{SA_SCORE_MAX:g}"
        else:
            if QED.qed(mol) < QED_MIN:
                return False, f"QED<{QED_MIN:g}"
        return True, "ok"

    def _evaluate(self, smiles_list):
        """Screen, then score surviving decoded SMILES on all 5 objectives.

        Each molecule is first passed through the cheap ``_screen`` filter.
        REJECTED molecules are NOT docked (saving the expensive step); they get
        the fixed ``REJECTION_PENALTY`` objective row so the GP learns to avoid
        that latent region. SURVIVORS are scored normally: ADMET oracle (3
        objectives) + docking against every target (2 objectives); a failed dock /
        un-featurizable survivor leaves NaN (excluded from GP training later).

        Returns:
            A tuple ``(Y, docking_by_target, rejections)`` where ``Y`` has shape
            ``(len(smiles_list), N_TASKS)`` in ``TASK_NAMES`` order,
            ``docking_by_target`` maps each target to its ``(N,)`` raw affinity
            vector (NaN for rejected/failed), and ``rejections`` is a list of
            ``(index, smiles, reason)`` for the filtered-out molecules.
        """
        smiles_list = list(smiles_list)
        n = len(smiles_list)
        Y = np.full((n, N_TASKS), np.nan, dtype=np.float64)
        docking_by_target = {
            t: np.full(n, np.nan, dtype=np.float64) for t in self.docking_targets
        }

        # --- Cheap structural screen: partition into survivors vs rejections ---
        passed_local = []
        rejections = []
        for i, smi in enumerate(smiles_list):
            ok, reason = self._screen(smi)
            if ok:
                passed_local.append(i)
            else:
                Y[i, :] = REJECTION_PENALTY   # dominated penalty row; no docking
                rejections.append((i, smi, reason))

        # --- Only survivors reach the expensive oracles ---
        if passed_local:
            passed_smiles = [smiles_list[i] for i in passed_local]
            admet_df = self.oracle.predict(passed_smiles)
            admet_vals = {
                col: admet_df[col].to_numpy(dtype=float) for _, col in self.admet_tasks
            }
            dock_pass = batch_dock_targets(passed_smiles, self.docking_targets)
            for k, i in enumerate(passed_local):
                for j, col in self.admet_tasks:
                    Y[i, j] = admet_vals[col][k]
                for j, target in self.docking_tasks:
                    val = float(dock_pass[target][k])
                    Y[i, j] = val
                    docking_by_target[target][i] = val
        return Y, docking_by_target, rejections

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
        Y0, docking, rejections = self._evaluate(smiles0)
        self._append(Z0, smiles0, Y0)

        # Fix the hypervolume reference point from the initial batch, in the signed
        # (maximization) frame: just below the worst value on each objective.
        finite = np.isfinite(Y0).all(axis=1)
        Yw = Y0[finite] * _SIGNS if finite.any() else Y0 * _SIGNS
        col_min = Yw.min(axis=0)
        col_range = Yw.max(axis=0) - col_min
        self.hv_ref_point = col_min - (0.1 * col_range + 1e-6)

        print(f"Initialized {self.n_init} molecules; "
              f"{len(rejections)} rejected pre-dock, "
              f"docked {docked_summary(docking, self.n_init)}.")

    def _random_latents(self, n):
        """Draw ``n`` uniform in-bounds latent vectors as a ``(n, latent_dim)`` tensor."""
        lo = self.bounds[0]
        hi = self.bounds[1]
        u = torch.rand(int(n), self.latent_dim, dtype=lo.dtype)
        return lo + (hi - lo) * u

    def _propose_batch(self, model, Z_train, Y_train):
        """Propose the next batch of (latent vector, decoded SMILES) pairs.

        With probability ``self.epsilon`` this is an epsilon-greedy WILDCARD step:
        qNEHVI is bypassed entirely and the batch is drawn from purely random
        in-bounds latent vectors (unconditioned exploration, so the search can
        never get permanently trapped in one basin). Otherwise the batch comes from
        qNEHVI as usual.

        Either way diversity is enforced at TWO levels: ``compute_qnehvi`` enforces
        LATENT distance (no two z's on the same coordinate), and here we enforce
        MOLECULE novelty, because the many-to-one VAE decoder can map distinct z's
        to a molecule already evaluated. We over-generate and greedily keep only
        novel decodes (vs. every prior molecule and vs. the batch so far),
        re-querying for any shortfall up to ``MAX_PROPOSAL_ROUNDS`` rounds.

        Returns ``(Z_new, smiles_new, acq_value)``; ``acq_value`` is ``nan`` on a
        wildcard step (no acquisition was optimized). If no novel molecule can be
        found the batch may be smaller than ``batch_size`` (and, in the degenerate
        case, falls back to a raw batch) so the campaign never stalls.
        """
        wildcard = float(self.epsilon) > 0.0 and np.random.rand() < float(self.epsilon)
        if wildcard:
            print(f"[Iteration {len(self.history) + 1}] epsilon-greedy WILDCARD: "
                  f"bypassing qNEHVI, injecting random latent exploration.")

        seen = set(self.smiles)
        chosen_z, chosen_smiles = [], []
        last_acq = float("nan") if wildcard else 0.0
        for _ in range(MAX_PROPOSAL_ROUNDS):
            need = self.batch_size - len(chosen_z)
            if need <= 0:
                break
            if wildcard:
                candidates = self._random_latents(need)
            else:
                # Avoid every latent point spent so far AND those chosen this step.
                avoid = self.Z_evaluated
                if chosen_z:
                    avoid = np.vstack([avoid, np.asarray(chosen_z)])
                candidates, acq_value = compute_qnehvi(
                    model, Z_train, Y_train, self.bounds, batch_size=need,
                    avoid_points=avoid,
                )
                last_acq = float(acq_value)
            smi_batch = self.bridge.decode(candidates)
            for z_i, smi_i in zip(candidates.cpu().numpy(), smi_batch):
                if smi_i in seen:
                    continue
                chosen_z.append(z_i)
                chosen_smiles.append(smi_i)
                seen.add(smi_i)
                if len(chosen_z) >= self.batch_size:
                    break

        if not chosen_z:
            # Decoder fully collapsed onto known molecules; proceed with a raw
            # batch rather than stalling (these will be re-evaluated).
            if wildcard:
                candidates = self._random_latents(self.batch_size)
                return (candidates.cpu().numpy(),
                        list(self.bridge.decode(candidates)), float("nan"))
            candidates, acq_value = compute_qnehvi(
                model, Z_train, Y_train, self.bounds, batch_size=self.batch_size,
                avoid_points=self.Z_evaluated,
            )
            return (candidates.cpu().numpy(),
                    list(self.bridge.decode(candidates)), float(acq_value))

        return np.asarray(chosen_z), chosen_smiles, last_acq

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

        # --- Propose a diverse, novel batch (latent-distance + molecule novelty) ---
        Z_new, smiles_new, acq_value = self._propose_batch(model, Z_train, Y_train)

        # --- Evaluate the decoded molecules on all 5 objectives (screen first) ---
        Y_new, docking_new, rejections = self._evaluate(smiles_new)
        n_rejected = len(rejections)
        batch_docked = docked_summary(docking_new, len(smiles_new))
        self._append(Z_new, smiles_new, Y_new)

        # --- Track Pareto front + hypervolume ---
        pareto_mask = self._pareto_mask()
        pareto_size = int(pareto_mask.sum())
        hypervolume = self._hypervolume()

        self.history.append({
            "iteration": iteration,
            "n_evaluated": len(self.Y_evaluated),
            "n_rejected": n_rejected,
            "pareto_size": pareto_size,
            "hypervolume": hypervolume,
            "acq_value": float(acq_value),
            "batch_smiles": list(smiles_new),
        })

        print(f"[Iteration {iteration}] "
              f"evaluated={len(self.Y_evaluated)}, "
              f"batch={len(smiles_new)}, rejected_pre_dock={n_rejected}, "
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
                "n_rejected": h.get("n_rejected", 0),
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
    # "Overnight Grand Campaign" production defaults: a deep 50-iteration de-novo
    # search sized for a manageable multi-hour autonomous run. n_init=40 seeds the
    # GP, then n_iterations=50 x batch_size=5 drives the search; mogp_iters=80 caps
    # each GP fit's optimizer iterations to accelerate per-iteration training.
    parser.add_argument("--n-init", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--n-iterations", type=int, default=50)
    parser.add_argument("--mogp-iters", type=int, default=80)
    parser.add_argument("--epsilon", type=float, default=EPSILON_GREEDY,
                        help="epsilon-greedy wildcard probability per BO step "
                             "(0 disables random-exploration steps)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    start = time.time()

    print("Running latent-space (de-novo) BO loop with the SELFIES-VAE bridge.")
    loop = BOLoop(
        seed=args.seed,
        latent_dim=args.latent_dim,
        n_init=args.n_init,
        batch_size=args.batch_size,
        n_iterations=args.n_iterations,
        mogp_train_iters=args.mogp_iters,
        epsilon=args.epsilon,
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
