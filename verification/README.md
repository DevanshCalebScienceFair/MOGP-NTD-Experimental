# BioMOBO verification harness

A test suite that checks whether a Claude-Code-modified GP-MOBO repo actually
implements the four steps of the "BioMOBO" master prompt **correctly** — not
just whether files exist. It tests mathematical contracts (real coregionalization,
cost-aware acquisition, selectivity direction) that are hard to fake.

> Built and self-validated with torch 2.x / gpytorch 1.15. No botorch required.

## What's here
| File | Role |
|---|---|
| `adapter.py` | **The only file you edit.** Hooks that point the tests at the real BioMOBO code. |
| `checks.py` | Naming-agnostic correctness checks (the actual logic). |
| `reference_icm.py` | A correct ICM + an independent-GP imposter, used as controls. |
| `test_step1..4_*.py` | One pytest file per step of the master prompt. |
| `test_harness_selfcheck.py` | Meta-tests proving the harness catches the key bugs. |
| `RUBRIC.md` | What "correct" means per step + the subtle failure modes. |

## Install
```bash
pip install pytest torch gpytorch pandas
```

## 1. Confirm the harness is healthy (no repo needed)
```bash
cd biomobo-verification
BIOMOBO_SELFTEST=1 pytest -q        # 16 pass: checks run against a correct reference
pytest test_harness_selfcheck.py -q # proves it catches the independent-GP imposter
```

## 2. Point it at the modified repo
```bash
export PYTHONPATH=/path/to/GP-MOBO    # so `import biomobo...` works
```
Open `adapter.py` and fill in each hook with your repo's real entry points
(import your model, config, cost map, selectivity fn, output DataFrame). Each
hook left unimplemented makes its tests **skip** (reported, not failed), so you
can wire in one step at a time as Claude finishes it.

## 3. Run
```bash
pytest -v                                  # everything wired so far
pytest test_step1_coregionalization.py -v  # just Step 1
pytest -rs                                  # show which tests skipped and why
```

## How to read it
- **PASS** — that contract holds.
- **FAIL** — the assertion message prints actual vs expected (e.g. `max
  |cross-task posterior covariance| = 0.000e+00` means tasks are still
  independent → Step 1 is wrong). See `RUBRIC.md` for the meaning + fix.
- **SKIP** — hook not wired yet.

The single most important test is
`test_step1...::test_cross_task_posterior_covariance_is_nonzero`: it passes only
if the model genuinely learned cross-task structure, which is the core claim of
the whole project. Items marked 👁 in `RUBRIC.md` still need a human eye.
