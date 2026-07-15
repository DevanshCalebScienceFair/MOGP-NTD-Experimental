# MOGP-NTD — De Novo Antimalarial Drug Discovery Pipeline

**Goal:** Design *new* drug-like molecules that potently and selectively inhibit
*Plasmodium falciparum* dihydrofolate reductase (**PfDHFR**, PDB 1J3I) — the
antimalarial target — while sparing the human enzyme and staying safe and
drug-like.

Instead of screening a fixed catalog of molecules, this pipeline **invents
molecules from scratch** by searching a continuous chemical "latent space" with
multi-objective Bayesian Optimization (**Latent-Space BO / LSBO**).

---

## The 5 objectives (optimized simultaneously)

The pipeline balances **potency, selectivity, safety, and ADMET** at once. It
never collapses them into a single score — it searches for the **Pareto front**
of best-possible trade-offs.

| # | Objective | Direction | Source |
|---|-----------|-----------|--------|
| 0 | `PfDHFR_Docking` | ↓ **lower better** (kcal/mol) | AutoDock Vina vs. PfDHFR (1J3I) — *potency* |
| 1 | `hDHFR_Docking`  | ↑ **higher better** (kcal/mol) | AutoDock Vina vs. human DHFR (1U72) — *selectivity* |
| 2 | `hERG_Toxicity_Prob` | ↓ **lower better** | ADMET oracle — *cardiac safety* |
| 3 | `Caco2_logPapp` | ↑ **higher better** | ADMET oracle — *gut permeability* |
| 4 | `Half_Life_hours` | ↑ **higher better** | ADMET oracle — *metabolic stability* |

Optimization sign vector: `[-1, +1, -1, +1, +1]`.

> **Selectivity insight:** we *minimize* PfDHFR binding energy (tighter binding
> to the parasite) but *maximize* hDHFR binding energy (weaker binding to the
> human enzyme). The derived **Selectivity Index** = `hDHFR − PfDHFR` is
> reported but is not itself an optimized objective.

---

## Pipeline architecture (end to end)

```
                 ┌──────────────────────────────────────────────┐
                 │  1. SELFIES-VAE  (generative chemistry model) │
                 │  Latent vector z  ⇄  valid drug-like molecule │
                 └──────────────────────────────────────────────┘
                        ▲ decode(z)→SMILES        ▲ encode(SMILES)→z
                        │                          │
   ┌────────────────────┴──────────┐               │
   │ 4. qNEHVI acquisition          │              │
   │    optimize_acqf over the      │              │
   │    latent box → next z's       │              │
   └────────────────────┬──────────┘               │
                        │ proposes z                │
                        ▼                           │
   ┌───────────────────────────────┐                │
   │ 3. BoTorch ModelListGP         │  trains on ───┘
   │    (5 independent Matérn GPs)  │  (z, objective) pairs
   └───────────────────────────────┘
                        ▲ observed objective values
                        │
   ┌────────────────────┴───────────────────────────────────────┐
   │ 2. Pre-dock penalty filter → Oracles                        │
   │    RDKit screen ─(pass)→ Vina docking + ADMET oracle        │
   │                 └(fail)→ bounded penalty row (no docking)   │
   └─────────────────────────────────────────────────────────────┘
```

### 1. The SELFIES-VAE (generative chemistry engine) — `vae_bridge.py`

A self-contained, domain-specific **Variational Autoencoder** that learns a
smooth, continuous 50-dimensional map of chemical space.

- **Representation:** [SELFIES](https://github.com/aspuru-guzik-group/selfies)
  strings, which guarantee **100% syntactically valid molecules** on decode
  (unlike raw SMILES).
- **Architecture:** GRU encoder → `(μ, logσ²)` bottleneck → sampled latent
  `z ∈ ℝ⁵⁰` → GRU autoregressive decoder. Embedding dim 64, hidden dim 256.
- **Structural budget:** `max_len = 45` SELFIES tokens — deliberately capped to
  bias generation toward small, drug-like molecules and keep docking tractable.
- **Trained in-house** on the project's `data/library` (601 molecules), cached
  to `models/vae/selfies_vae.pt`, and reloaded on subsequent runs.
- **Posterior-collapse fixes:** *word-dropout* (0.3) + *free-bits KL*
  (0.05 nats/dim) keep the latent space informative (KL ≈ 2.5, fully distinct
  decodes) instead of the decoder ignoring `z`.
- **Robust decode:** every decoded molecule is RDKit-sanitized; the rare invalid
  case falls back to benzene so the loop never crashes.

> **Why we built our own** instead of downloading a pretrained VAE: external
> chemical VAEs (MOSES / DeepChem) would have forced a downgrade of the pinned
> NumPy 2.x / scikit-learn 1.9.0 stack and silently corrupted the ADMET oracle;
> most HuggingFace chemical models are encoder-only language models with no fixed
> latent space suitable for LSBO.

### 2. Pre-dock penalty filter + Oracles — `loop.py`, `docking.py`, `admet_oracle.py`

Docking is the expensive step (~30 s per molecule per target), so every decoded
molecule is **screened before docking**:

- **RDKit screen (`_screen`)** rejects a molecule if heavy-atom count > 35, or
  it violates Lipinski's Rule of Five (MW > 500, LogP > 5, HBD > 5, HBA > 10).
- **Rejected** molecules **skip docking entirely** and receive a *bounded,
  directionally-worst* penalty row `[0, −14, +1, −8, 0]`. This teaches the GP to
  **avoid that region of latent space** without extreme values destabilizing the
  GP's output standardization.
- **Survivors** are sent to the two oracles:
  - **Docking oracle** (`docking.py`): SMILES → 3D conformer → **AutoDock Vina**
    against both PfDHFR (1J3I) and hDHFR (1U72), returning binding energy in
    kcal/mol.
  - **ADMET oracle** (`admet_oracle.py`): three pretrained HistGradientBoosting
    models (trained on TDC datasets) predict hERG toxicity, Caco-2 permeability,
    and half-life — cheap and precomputed, each with a Tanimoto
    applicability-domain flag.

### 3. The surrogate model — `mogp.py`

A **BoTorch `ModelListGP`** of **five independent `SingleTaskGP`s**, one per
objective:

- Kernel: `ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=50))` — a Matérn-5/2
  kernel with **automatic relevance determination** over all 50 latent
  dimensions.
- Output transform: `Standardize(m=1)` per objective.
- Fit by exact marginal likelihood via `fit_gpytorch_mll`.

Each GP learns a probabilistic map from latent vector `z` to one objective,
with calibrated uncertainty that drives exploration.

> **Critical design decision:** the model is a **native, differentiable BoTorch
> GP**. An earlier design routed predictions through a NumPy bridge, which
> severed the autograd graph and made gradient-based acquisition impossible. The
> native `ModelListGP` keeps a differentiable posterior — verified gradient flow
> `∂Acq/∂z`, finite and non-null — which is what makes the next step work.

### 4. The acquisition function — `acquisition.py`

**q-Log Noisy Expected Hypervolume Improvement (qNEHVI)** over the continuous
latent box:

- `optimize_acqf` runs gradient-based (L-BFGS-B) optimization **directly in the
  50-D latent space** to propose the batch of `z` vectors most likely to expand
  the 5-objective Pareto front.
- `WeightedMCMultiOutputObjective` applies the sign vector `[-1,+1,-1,+1,+1]` so
  every objective is treated as "maximize" in the internal frame.
- Monte-Carlo hypervolume estimation with a `SobolQMCNormalSampler`.

This is the "brain": it decides **which new molecules to invent and test next**,
spending expensive docking only where it most improves the trade-off frontier.

---

## The optimization loop — `loop.py`

1. **Initialize** — sample `n_init` random latent vectors, decode, screen, and
   evaluate (dock survivors / penalize rejects) to seed the GP.
2. **Train** the 5-GP `ModelListGP` on all `(z, objectives)` observed so far.
3. **Acquire** — `optimize_acqf` proposes a batch of `batch_size` new latent
   vectors via qNEHVI.
4. **Decode → screen → evaluate** the proposed molecules.
5. **Update** the Pareto front and hypervolume; save results.
6. Repeat for `n_iterations`.

**Emergent behavior observed:** after seeding on a mostly-penalized initial set,
the GP + qNEHVI learned to **avoid the penalized latent regions**, and by the
first optimization iteration proposed a batch with a **0% rejection rate** — the
model taught itself to generate valid, drug-like molecules.

---

## Production campaign parameters (locked in)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `n_init` | **40** | ~15% random pre-dock pass rate → ~5–6 real docked anchor molecules to ground the GP |
| `n_iterations` | **20** | Substantive de-novo search horizon |
| `batch_size` | **3** | Molecules proposed per iteration |
| `latent_dim` | 50 | VAE latent dimensionality |
| `mogp_iters` | 200 | GP fitting iterations |

Run: `python loop.py` (defaults above), results written to `results/`
(`history.csv`, `evaluated.csv`, `pareto_front.csv`).

---

## Environment

Conda env `mogp-drug` (Python 3.11): NumPy 2.x, scikit-learn **pinned 1.9.0**
(matches the serialized ADMET models), PyTorch 2.12, BoTorch 0.18, GPyTorch 1.15,
`selfies`, RDKit, AutoDock Vina + OpenBabel + Meeko. Requires
`KMP_DUPLICATE_LIB_OK=TRUE` (duplicate libomp).

---

*One-line summary for the board:* **A generative AI (SELFIES-VAE) invents
candidate antimalarial molecules, and a multi-objective Bayesian optimizer
(5 Gaussian Processes + qNEHVI) steers that invention toward compounds that are
potent, human-selective, safe, and drug-like — all at once.**
