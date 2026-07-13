"""
vae_bridge.py
=============

Bridge between the continuous LATENT search space that the Bayesian-optimization
loop now searches over and the discrete SMILES space that the physical oracles
(``docking.py``, ``admet_oracle.py``) actually evaluate.

This is a **MOCK** implementation. Its only job right now is to let us verify the
high-dimensional continuous plumbing — tensor shapes, latent bounds, BoTorch
``optimize_acqf`` gradient flow — *before* we download and wire in a heavy
pre-trained molecular VAE (e.g. a JT-VAE / SELFIES-VAE). Every method has the
exact signature and tensor contract the real bridge will have, so swapping the
mock for real neural weights later touches nothing outside this file:

  * ``encode(smiles) -> (N, latent_dim)``   real: VAE encoder mean
  * ``decode(z)      -> list[str]``          real: VAE decoder / argmax detokenize
  * ``bounds``       ``(2, latent_dim)``     real: empirical latent box (e.g. +/-
                                             a few standard deviations)

Mock specifics:
  * ``encode`` returns standard-normal noise (shape-correct, content-meaningless).
  * ``decode`` maps each latent vector to one of a few tiny, guaranteed-valid
    SMILES. It is deterministic in the vector so the same ``z`` always decodes to
    the same molecule (as a real decoder would), and spreads across a handful of
    molecules so the mock loop sees *some* variation in the oracle scores rather
    than a single degenerate point. The chemistry is meaningless — only the
    plumbing is under test.
"""

import numpy as np
import torch


# A tiny palette of trivially-valid, fast-to-dock SMILES the mock decoder maps
# into. Real molecules (so RDKit featurization + Vina + the ADMET oracle all
# succeed), but chemically meaningless for optimization — the point is only that
# distinct latent vectors can decode to distinct molecules so the mock loop is
# not a single degenerate point. The real VAE decoder replaces this wholesale.
_MOCK_SMILES_PALETTE = [
    "C",          # methane
    "CCO",        # ethanol
    "CCC",        # propane
    "CCN",        # ethylamine
    "CCCC",       # butane
    "c1ccccc1",   # benzene
    "CC(=O)O",    # acetic acid
    "CCOCC",      # diethyl ether
]


class LatentSpaceBridge:
    """Mock encoder/decoder between latent vectors and SMILES.

    Args:
        latent_dim: Dimensionality of the continuous latent space the BO loop
            searches over (default 50).
        low, high: Per-dimension bounds of the latent search box. The real
            bridge derives these from the trained VAE's latent distribution; the
            mock fixes them to a symmetric cube ``[-1, 1]`` per the migration
            plan.
        dtype: Torch dtype for emitted tensors (BoTorch multi-objective math runs
            in double precision, so default ``torch.double``).
    """

    def __init__(self, latent_dim=50, low=-1.0, high=1.0, dtype=torch.double):
        self.latent_dim = int(latent_dim)
        self.low = float(low)
        self.high = float(high)
        self.dtype = dtype

    # ------------------------------------------------------------------ #
    # Latent-space geometry
    # ------------------------------------------------------------------ #
    @property
    def bounds(self):
        """The latent search box as a ``(2, latent_dim)`` tensor.

        Row 0 is the per-dimension lower bound, row 1 the upper bound — exactly
        the layout ``botorch.optim.optimize_acqf`` expects for its ``bounds``
        argument.
        """
        lo = torch.full((self.latent_dim,), self.low, dtype=self.dtype)
        hi = torch.full((self.latent_dim,), self.high, dtype=self.dtype)
        return torch.stack([lo, hi], dim=0)

    # ------------------------------------------------------------------ #
    # encode / decode
    # ------------------------------------------------------------------ #
    def encode(self, smiles_list):
        """Encode SMILES into latent vectors (MOCK: standard-normal noise).

        Args:
            smiles_list: Iterable of SMILES strings (only its length is used by
                the mock).

        Returns:
            A ``(N, latent_dim)`` ``torch`` tensor of dummy latent vectors, where
            ``N == len(smiles_list)``.
        """
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]
        n = len(list(smiles_list))
        return torch.randn(n, self.latent_dim, dtype=self.dtype)

    def decode(self, latent_vectors):
        """Decode latent vectors into SMILES (MOCK: deterministic tiny molecules).

        Accepts a ``(N, latent_dim)`` or ``(latent_dim,)`` tensor/array. Each row
        is deterministically hashed to one entry of ``_MOCK_SMILES_PALETTE`` so a
        given latent vector always decodes to the same molecule.

        Args:
            latent_vectors: Tensor/array of shape ``(N, latent_dim)`` (a single
                ``(latent_dim,)`` vector is accepted and treated as ``N == 1``).

        Returns:
            A list of ``N`` SMILES strings.
        """
        z = torch.as_tensor(latent_vectors, dtype=self.dtype)
        if z.ndim == 1:
            z = z.unsqueeze(0)
        z_np = z.detach().cpu().numpy()

        smiles = []
        for row in z_np:
            # Deterministic, vector-dependent selector: sign-pattern of the first
            # few dims folded into an index. Meaningless chemically; it only makes
            # decode() a stable function of z that spreads over the palette.
            key = int(np.floor(np.abs(row).sum() * 1000)) % len(_MOCK_SMILES_PALETTE)
            smiles.append(_MOCK_SMILES_PALETTE[key])
        return smiles


if __name__ == "__main__":
    bridge = LatentSpaceBridge(latent_dim=50)

    print(f"latent_dim = {bridge.latent_dim}")
    print(f"bounds shape = {tuple(bridge.bounds.shape)} "
          f"(expected (2, {bridge.latent_dim}))")
    assert bridge.bounds.shape == (2, 50)
    assert torch.all(bridge.bounds[0] == -1.0) and torch.all(bridge.bounds[1] == 1.0)

    z = bridge.encode(["CCO", "c1ccccc1", "CC(=O)O"])
    print(f"encode(3 smiles) -> {tuple(z.shape)} (expected (3, 50))")
    assert z.shape == (3, 50)

    smi = bridge.decode(z)
    print(f"decode(z) -> {smi}")
    assert len(smi) == 3

    # Determinism: same vector -> same SMILES.
    assert bridge.decode(z) == smi
    # Single-vector convenience path.
    assert len(bridge.decode(z[0])) == 1

    print("\nMOCK VAE BRIDGE OK")
