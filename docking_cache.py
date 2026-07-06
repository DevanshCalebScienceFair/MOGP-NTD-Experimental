"""
docking_cache.py
================

A small persistent on-disk cache for docking scores, keyed by
``(canonical SMILES, target name)`` -> affinity (kcal/mol).

Why this exists: docking is by far the most expensive step in the pipeline, and
``run_all.py`` / ``run_benchmark_seeds.py`` dock heavily *overlapping* library
molecules — the same molecule is re-selected across the four methods and, in the
multi-seed harness, across many seeds. Caching turns every repeat dock into a
dictionary lookup. This is only sound because the docking oracle is now
deterministic: the RDKit conformer is seeded (``0xF00D``) and the Vina call is
seeded (``docking.DEFAULT_VINA_SEED``), so a given ``(SMILES, target)`` always
produces the same score.

Design notes:

  * **Key canonicalization.** SMILES are canonicalized with RDKit
    (``Chem.MolToSmiles(Chem.MolFromSmiles(s))``) before use as a key, so two
    spellings of the same molecule share one cache entry. A SMILES RDKit cannot
    parse is left unchanged (it will fail docking anyway and be cached as a
    failure).
  * **Failures are cached, but marked.** A failed / NaN dock is stored with
    ``status = 'fail'`` (affinity NULL) rather than silently dropped, so we do
    not keep re-attempting a molecule that reliably fails. Because Vina is now
    deterministic a failure is reproducible; the escape hatch for retrying is to
    clear the cache (``clear()`` / the ``--clear-cache`` CLI flag), which drops
    every entry including the failures.
  * **Backing store.** SQLite (stdlib ``sqlite3``). One row per
    ``(smiles, target)``; writes are atomic and it tolerates many small writes
    far better than rewriting a whole JSON file per dock.
"""

import os
import sqlite3
import threading

from rdkit import Chem


# Default location for the shared cache. Lives under data/ next to the cached
# library; it is a generated artifact and is not committed.
DEFAULT_CACHE_DIR = os.path.join("data", "docking_cache")
DEFAULT_CACHE_PATH = os.path.join(DEFAULT_CACHE_DIR, "docking_cache.sqlite")

# Row status markers.
STATUS_OK = "ok"       # affinity is a finite float
STATUS_FAIL = "fail"   # dock failed / returned NaN; affinity is NULL


def canonicalize_smiles(smiles):
    """Return the RDKit canonical SMILES, or the input unchanged if unparseable.

    Canonicalizing before keying means ``"C1=CC=CC=C1"`` and ``"c1ccccc1"`` map
    to the same cache entry. If RDKit cannot parse the string we return it as-is
    so the caller can still proceed (docking will fail and be cached as such).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    return Chem.MolToSmiles(mol)


class DockingCache:
    """SQLite-backed ``(canonical SMILES, target) -> affinity`` docking cache.

    Thread-safe for the sequential-with-occasional-parallelism use in this repo:
    a single connection guarded by a lock, opened with a busy timeout so
    concurrent processes sharing the file block briefly rather than erroring.
    """

    def __init__(self, path=DEFAULT_CACHE_PATH):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        # check_same_thread=False + an explicit lock lets one cache instance be
        # shared across threads; timeout makes cross-process access wait on the
        # file lock instead of raising "database is locked".
        self._conn = sqlite3.connect(path, timeout=30.0, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self):
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS docking_scores (
                    smiles   TEXT NOT NULL,
                    target   TEXT NOT NULL,
                    affinity REAL,
                    status   TEXT NOT NULL,
                    seed     INTEGER,
                    PRIMARY KEY (smiles, target)
                )
                """
            )
            self._conn.commit()

    def get(self, canonical_smiles, target):
        """Look up a cached dock.

        Returns:
            ``None`` on a cache miss, otherwise ``(status, affinity)`` where
            ``status`` is :data:`STATUS_OK` (``affinity`` a float) or
            :data:`STATUS_FAIL` (``affinity`` is ``None``).
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT status, affinity FROM docking_scores "
                "WHERE smiles = ? AND target = ?",
                (canonical_smiles, target),
            ).fetchone()
        if row is None:
            return None
        status, affinity = row
        return status, affinity

    def put(self, canonical_smiles, target, affinity, status, seed=None):
        """Insert or replace the cached score for ``(canonical_smiles, target)``."""
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO docking_scores "
                "(smiles, target, affinity, status, seed) VALUES (?, ?, ?, ?, ?)",
                (canonical_smiles, target, affinity, status, seed),
            )
            self._conn.commit()

    def clear(self):
        """Drop every cached entry (the escape hatch to force re-docking)."""
        with self._lock:
            self._conn.execute("DELETE FROM docking_scores;")
            self._conn.commit()

    def size(self):
        """Return the number of cached ``(smiles, target)`` entries."""
        with self._lock:
            (n,) = self._conn.execute(
                "SELECT COUNT(*) FROM docking_scores"
            ).fetchone()
        return int(n)

    def close(self):
        with self._lock:
            self._conn.close()


if __name__ == "__main__":
    # Tiny self-check: put/get round-trips a hit and a marked failure, and
    # canonicalization collapses two spellings of benzene onto one key.
    import tempfile

    tmp = os.path.join(tempfile.mkdtemp(), "selfcheck.sqlite")
    cache = DockingCache(tmp)

    a = canonicalize_smiles("C1=CC=CC=C1")
    b = canonicalize_smiles("c1ccccc1")
    assert a == b, f"canonicalization mismatch: {a!r} != {b!r}"

    cache.put(a, "PfDHFR", -8.3, STATUS_OK, seed=42)
    assert cache.get(b, "PfDHFR") == (STATUS_OK, -8.3)
    assert cache.get(a, "hDHFR") is None            # different target -> miss

    cache.put("bad", "PfDHFR", None, STATUS_FAIL)
    assert cache.get("bad", "PfDHFR") == (STATUS_FAIL, None)

    assert cache.size() == 2
    cache.clear()
    assert cache.size() == 0
    print("docking_cache self-check PASSED")
