# BioMOBO correctness rubric

What "correct" means for each step of your friend's master prompt, the **subtle
ways Claude Code can pass a naive eye but be wrong**, and which automated check
catches it. Use this for manual review *and* to understand any red test.

Legend: ✅ = automated check exists · 👁 = needs a human look (the suite can't
fully judge it).

---

## STEP 1 — Independent GPs → Intrinsic Coregionalization (ICM)

| Requirement | Correct | Common failure | Check |
|---|---|---|---|
| Multi-output posterior | `forward`/`posterior` returns `MultitaskMultivariateNormal` | Returns plain `MultivariateNormal`, or a Python list of single-task GPs | ✅ `test_output_is_multitask_mvn` |
| Cross-task learning | Task covariance `B = WWᵀ + diag(v)` with **rank ≥ 1** (off-diagonal terms) | Uses `MultitaskKernel(rank=0)`, batched independent GPs, or `IndependentMultitaskGaussianLikelihood` — "multitask-shaped" but tasks never talk | ✅ `test_task_covariance_is_not_diagonal` |
| **Real coregionalization** | Observing one task shifts the posterior of the others ⇒ nonzero cross-task posterior covariance | Block-diagonal joint covariance (zero off-block) — the headline bug | ✅ `test_cross_task_posterior_covariance_is_nonzero` |
| Tanimoto kept | The data (X–X) kernel is still the repo's `TanimotoKernel`, wired *inside* the coregionalized covar | Claude swaps in an RBF/Matérn "because it's easier", silently breaking fingerprint similarity | ✅ `test_tanimoto_kernel_is_kept` |
| ICM uses an `IndexKernel` | Task covar via `IndexKernel`/`MultitaskKernel` | Hand-rolled correlation that isn't PSD | ✅ `test_model_contains_coregionalization_kernel` |
| Numerics | Fits without Cholesky/PSD errors; predictions finite | Jitter/`add_jitter` removed; NaNs at predict | 👁 (watch training logs) |

The cross-task posterior covariance test is the one that matters most: it is
**impossible to pass without genuine coregionalization** and is what makes the
"learns biological correlations between targets" claim true rather than narrated.

---

## STEP 2 — Generalize the Y-matrix & task config

| Requirement | Correct | Common failure | Check |
|---|---|---|---|
| Dynamic targets | Config exposes string `Target` and `Off_Target` (e.g. "Cruzain"/"Cathepsin_L") | Names hard-coded in the model/loader, not in config | ✅ `test_config_has_dynamic_target_and_offtarget` |
| Y width | `Y` is `(n, 4)` | Left at the repo's original objective count | ✅ `test_y_matrix_has_four_columns` |
| Y order | `[Target_IC50, OffTarget_IC50, LF_Affinity_Proxy, ADMET_Score]` exactly | Columns reordered ⇒ Step 4 selectivity reads the wrong tasks | ✅ `test_y_columns_match_functional_order` |
| Model fan-out | `num_tasks == 4`, matching Y | Model still has old task count | ✅ `test_model_num_tasks_matches_y_width` |
| Units sanity | IC50 in consistent units / log-scale; ADMET normalized | Mixed raw + log IC50 across rows | 👁 |

---

## STEP 3 — Multi-fidelity, cost-aware EHVI

| Requirement | Correct | Common failure | Check |
|---|---|---|---|
| Cost map | cheap proxies (`ADMET_Score`, `LF_Affinity_Proxy`) = small nonzero (0.01); wet-lab (`Target_IC50`, `OffTarget_IC50`) ≥ 1.0 | Cheap cost set to literal `0` ⇒ divide-by-zero / inf | ✅ `test_fidelity_costs_are_sane`, `test_no_division_by_zero_in_scorer` |
| Cost weighting | score = EHVI / cost ⇒ cheaper fidelity ranks higher at equal EHVI | Multiplies by cost, or ignores cost entirely (plain EHVI) | ✅ `test_cost_weighting_prefers_cheaper_fidelity` |
| Order-preserving | At fixed cost, ranking matches raw EHVI ranking | A bad transform scrambles same-cost candidates | ✅ `test_cost_weighting_is_monotone_at_fixed_cost` |
| EHVI still valid | Reference point & Pareto set computed before cost weighting; EHVI ≥ 0 | Cost applied to a broken/negative EHVI | 👁 |

---

## STEP 4 — Selectivity Index & output

| Requirement | Correct | Common failure | Check |
|---|---|---|---|
| Formula direction | `SI = Predicted(Off_Target) / Predicted(Target)` | **Inverted** (`Target/Off`) — silent, flips your whole ranking | ✅ `test_selectivity_index_formula_direction` |
| Post-hoc, not a GP output | SI computed from posterior means *after* prediction | GP forced to regress the ratio directly (prompt explicitly forbids) | 👁 (inspect: there must be 4 GP tasks, not a 5th "ratio" task) |
| Output schema | DataFrame with SMILES, Selectivity Index, ADMET, calibrated uncertainty bounds | Missing uncertainty, or SI without SMILES | ✅ `test_output_dataframe_has_required_columns` |
| Ranked Pareto front | Rows sorted; only Pareto-front candidates | Whole library dumped unsorted | ✅ `test_output_dataframe_is_ranked_and_nonempty` |
| Calibrated uncertainty | Bounds come from posterior variance (± k·σ) | "Uncertainty" is a constant or a placeholder | 👁 |

---

## How to read results
- **Green in self-test, red when wired** ⇒ Claude's code is wrong for that step.
  Read the assertion message — every check prints the actual vs expected value.
- **Skipped** ⇒ you haven't wired that hook in `adapter.py` yet (expected while
  work is in progress).
- The `test_harness_selfcheck.py` meta-tests must always pass; if they don't, the
  harness has been broken and you can't trust the other greens.
