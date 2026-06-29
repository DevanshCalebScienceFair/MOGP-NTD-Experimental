# CLAUDE.md — MOGP-NTD

Guidance for working in this repo. MOGP-NTD is a **multi-objective Bayesian
optimization (BO)** pipeline for antimalarial drug discovery against
*Plasmodium falciparum* dihydrofolate reductase (**PfDHFR**, PDB 1J3I).

## What it does

Searches a fixed library of drug-like molecules for compounds that are good
across **4 objectives at once**, in this fixed order (`mogp.TASK_NAMES` is the
single source of truth):

| # | Objective | Direction | Source |
|---|---|---|---|
| 0 | `Caco2_Permeability` | ↑ higher better | ADMET oracle (cheap, precomputed) |
| 1 | `Half_Life` | ↑ higher better | ADMET oracle (cheap, precomputed) |
| 2 | `hERG_Toxicity_Prob` | ↓ lower better | ADMET oracle (cheap, precomputed) |
| 3 | `PfDHFR_Docking` | ↓ lower better (kcal/mol) | AutoDock Vina (**expensive**, on the fly) |

Direction signs: `acquisition.DEFAULT_OBJECTIVE_SIGNS = [+1, +1, -1, -1]`.

Core idea: the 3 ADMET objectives are cheap and precomputed for the whole
library; docking is expensive, so the EHVI acquisition spends docking only on
the molecules most likely to expand the Pareto front. The docking column is
all-NaN until evaluated, so the GP/acquisition/loop handle a **dynamic objective
count** (3 active until docking fills the 4th).

## Environment

- Conda env **`mogp-drug`** (Python 3.11). `vina` is also available in a separate
  `vina-cli` env.
- NumPy is **2.x**; `scikit-learn` is **pinned to 1.9.0** to match the serialized
  ADMET models in `models/pretrained_admet/` (loading under another version can
  silently produce invalid predictions).
- Docking deps: `vina` + `openbabel` (conda: `conda install -c conda-forge vina
  openbabel`), `meeko` + `biopython` (pip). `vina` must be on PATH.
- If you hit **OMP Error #15** (duplicate libomp) on import, set
  `KMP_DUPLICATE_LIB_OK=TRUE` (run.py already does this at module top).

## Key files

| File | Role |
|---|---|
| `data.py` | Build & cache the molecule library (ChEMBL → Lipinski filter → fingerprints → ADMET scores). `load_library()` reloads it. |
| `admet_oracle.py` | Inference wrapper for the 3 pretrained HistGradientBoosting ADMET models, with per-model Tanimoto applicability-domain flags. |
| `train_admet_oracle.py` | Retrain the 3 ADMET models from TDC datasets (`--refit-on-full` for production). |
| `utils/featurize.py` | SMILES → 2048-bit Morgan fingerprints. |
| `kernel.py` | `TanimotoKernel` for GPyTorch. |
| `mogp.py` | Multi-output Tanimoto GP (one independent scaled-Tanimoto GP per objective). Owns `TASK_NAMES`. |
| `acquisition.py` | Monte-Carlo EHVI, Pareto front / hypervolume / reference-point helpers, diverse `select_batch`. |
| `docking.py` | PfDHFR docking oracle: SMILES → 3D conformer → Vina → kcal/mol. `batch_dock()` returns NaN on failure. |
| `loop.py` | The multi-objective BO loop (`BOLoop`): train MOGP → EHVI select → dock → update Pareto/hypervolume → save. |
| `dashboard.py` | Streamlit results viewer (reads the 3 result CSVs). |
| `run.py` | Interactive end-to-end runner (train → build library → BO loop → launch dashboard). |

## Running

Interactive (auto-detects what's already computed):
```bash
python run.py
```
Or the stages individually:
```bash
python train_admet_oracle.py --refit-on-full     # (re)train ADMET models
python data.py --n-molecules 1000                 # build/cache the library
python loop.py --n-init 10 --batch-size 10 --n-iterations 10 --mogp-iters 200
streamlit run dashboard.py                         # view results/
```

Results land in `results/` as `history.csv`, `evaluated.csv`, `pareto_front.csv`.
The cached library lives in `data/library/` (`smiles.csv`, `fingerprints.npy`,
`admet_scores.csv`, row-aligned). Both are generated artifacts — not committed.

## Baselines / experiments

Controls that measure how much the MOGP + EHVI loop buys. Each mirrors `loop.py`,
writes the same 3 CSVs to its own `*_results/` dir, and saves a `comparison.png`
(hypervolume vs molecules evaluated) against the MOGP run in `results/`:

| Script | Acquisition | Tests |
|---|---|---|
| `baseline_random.py` | none (uniform random batch) | how much BO beats naive sampling |
| `baseline_single_obj.py` | single-output GP + Expected Improvement on **docking only** | whether optimizing docking alone yields strong binders with poor ADMET, i.e. a worse 4-objective hypervolume |

For a fair comparison, run the MOGP `loop.py` and the baselines on the **same
library and same scale** (n_init / batch_size / n_iterations).

## Conventions / gotchas

- **Objective order is fixed** by `mogp.TASK_NAMES`; all result CSV columns use
  those names. New code optimizing objectives should import `TASK_NAMES` rather
  than hard-coding column strings.
- `data.ADMET_COLUMNS` uses the oracle's *unit-named* columns
  (`Caco2_logPapp`, `Half_Life_hours`, `hERG_Toxicity_Prob`) — that's what the
  cached `admet_scores.csv` stores. `load_library()` returns those as a
  **positional** array, and `loop.py` relabels them with `TASK_NAMES`. The two
  naming schemes refer to the same quantities.
- `verification/` is a "BioMOBO" correctness harness that targets a *different*
  design (ICM coregionalization, off-target selectivity, cost-aware multi-
  fidelity) than this repo implements; its tests intentionally skip (see
  `verification/README.md`).

## Recent work (2026-06-29)

- Verified the full pipeline runs end-to-end in `mogp-drug`
  (`train_admet_oracle.py --refit-on-full` → `data.py` → `loop.py` →
  `dashboard.py`); docking validated (pyrimethamine ≈ −7 kcal/mol).
- Added `baseline_single_obj.py` (docking-only single-objective BO control).
- More baselines / tests planned (overnight comparison runs).
