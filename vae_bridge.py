"""
vae_bridge.py
=============

Bridge between the continuous LATENT search space that the Bayesian-optimization
loop searches over and the discrete SMILES space that the physical oracles
(``docking.py``, ``admet_oracle.py``) evaluate.

Phase 2 (REAL): a self-contained, domain-specific **SELFIES Variational
Autoencoder**, trained on this project's own molecule library. It replaces the
Phase-1 mock (which returned canned SMILES) with a genuine learned latent space,
so latent-space BO explores real chemistry.

Design choices (see the migration discussion):

  * **SELFIES**, not SMILES. Every SELFIES string maps to a syntactically valid
    molecule, so *every* point in the latent space decodes to a real molecule —
    the "100% valid decode" guarantee. We still RDKit-sanitize the decoder output
    as a belt-and-suspenders check and to canonicalize; on the rare semantic
    failure we fall back to a safe baseline molecule so the BO loop never crashes.
  * **Lightweight GRU VAE.** A single-layer GRU encoder to a fixed
    ``latent_dim`` bottleneck (mu / logvar), and a GRU decoder. Small enough to
    train in a couple of minutes on an M-series CPU.
  * **Train-once, cache, reload.** The first construction trains on
    ``data/library`` and caches weights + vocab to ``models/vae/selfies_vae.pt``.
    Subsequent constructions just load that checkpoint. No external weights are
    downloaded — the model is ours.

The bridge's public contract is unchanged from the mock, so nothing downstream
(``loop.py``, ``acquisition.py``) needs to change:

  * ``encode(smiles) -> (N, latent_dim)`` tensor (posterior mean ``mu``)
  * ``decode(z)      -> list[str]``       RDKit-valid SMILES, one per row
  * ``bounds``       ``(2, latent_dim)``  the ``[-1, 1]`` search box

The ``[-1, 1]`` box covers the high-density core of the (unit-normal-prior)
latent space; SELFIES guarantees validity everywhere inside it.
"""

import os
import warnings

import numpy as np
import torch
import torch.nn as nn

import selfies as sf
from rdkit import Chem
from rdkit import RDLogger

# RDKit is noisy about sanitization failures we handle ourselves; silence it.
RDLogger.DisableLog("rdApp.*")


# Default location of the cached VAE checkpoint and the training library.
DEFAULT_WEIGHTS_PATH = "models/vae/selfies_vae.pt"
DEFAULT_LIBRARY_DIR = "data/library"

# Larger training corpus. The 601-molecule library gives the VAE a tiny SELFIES
# vocabulary, which chokes the search once the biological filters bite. If the raw
# ChEMBL pull is present we curate a much larger drug-like corpus from it (cached
# to CORPUS_DIR) to expand the latent chemical universe; otherwise we fall back to
# the small library. See LatentSpaceBridge._resolve_training_smiles.
CHEMBL_SOURCE = "data/chembl_v29.csv"      # raw ChEMBL_V29 pull: columns ID,smiles
CORPUS_DIR = "data/vae_train"              # curated drug-like corpus cache
DEFAULT_CORPUS_SIZE = 5000                 # target number of training molecules

# SELFIES padding / no-op symbol (decoder ignores it), reserved to index 0.
PAD_SYMBOL = "[nop]"

# If a decoded molecule fails RDKit sanitization (rare with SELFIES), fall back
# to this safe, trivially-valid baseline so the loop keeps a molecule per latent
# vector and Z/Y stay row-aligned.
SAFE_FALLBACK_SMILES = "c1ccccc1"  # benzene

# Fixed SELFIES sequence-length budget (symbols). This is a STRUCTURAL CONSTRAINT,
# not just a safety cap: holding it tight forces the VAE to learn to construct
# molecules within a small token budget, so decoded molecules stay smaller and
# more drug-like (fewer oversized structures that are slow to dock and out of the
# ADMET domain). Longer training molecules are truncated to it.
DEFAULT_MAX_LEN = 45


# ---------------------------------------------------------------------- #
# The GRU SELFIES-VAE
# ---------------------------------------------------------------------- #
class SelfiesVAE(nn.Module):
    """A minimal single-layer GRU variational autoencoder over SELFIES tokens.

    Encoder: embedding -> GRU -> (mu, logvar) at the fixed ``latent_dim``.
    Decoder: z -> initial GRU hidden state; autoregressive GRU emits one token
    distribution per step (teacher-forced at train time, greedy at inference).
    """

    def __init__(self, vocab_size, latent_dim, max_len,
                 emb_dim=64, hidden_dim=256, pad_idx=0):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.latent_dim = int(latent_dim)
        self.max_len = int(max_len)
        self.pad_idx = int(pad_idx)

        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.enc_gru = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.dec_gru = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, vocab_size)
        # A learned "start" input for the decoder's first step (avoids reserving a
        # start token in the output vocabulary).
        self.start_emb = nn.Parameter(torch.zeros(1, 1, emb_dim))

    def encode(self, x):
        """x: (B, L) long token ids -> (mu, logvar), each (B, latent_dim)."""
        emb = self.embed(x)
        _, h = self.enc_gru(emb)          # h: (1, B, hidden)
        h = h.squeeze(0)                  # (B, hidden)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode_train(self, z, target, word_dropout=0.0):
        """Teacher-forced decode. z: (B, latent), target: (B, L) -> logits (B,L,V).

        ``word_dropout`` randomly replaces teacher-forced input tokens with the
        pad symbol (Bowman et al. 2016). By starving the decoder of the ground-
        truth previous token, it is forced to rely on ``z``, which prevents the
        posterior collapse an autoregressive decoder otherwise falls into.
        """
        B, L = target.shape
        h = torch.tanh(self.latent_to_hidden(z)).unsqueeze(0)  # (1, B, hidden)
        start = self.start_emb.expand(B, 1, -1)                 # (B, 1, emb)
        inp_tokens = target[:, :-1]                             # (B, L-1)
        if word_dropout > 0.0:
            drop = torch.rand_like(inp_tokens, dtype=torch.float) < word_dropout
            inp_tokens = inp_tokens.masked_fill(drop, self.pad_idx)
        tgt_emb = self.embed(inp_tokens)                        # (B, L-1, emb)
        dec_in = torch.cat([start, tgt_emb], dim=1)             # (B, L, emb)
        out, _ = self.dec_gru(dec_in, h)                        # (B, L, hidden)
        return self.out(out)                                   # (B, L, vocab)

    @torch.no_grad()
    def decode_greedy(self, z):
        """Greedy (argmax) decode. z: (B, latent) -> token ids (B, max_len)."""
        B = z.shape[0]
        h = torch.tanh(self.latent_to_hidden(z)).unsqueeze(0)
        inp = self.start_emb.expand(B, 1, -1)
        tokens = []
        for _ in range(self.max_len):
            out, h = self.dec_gru(inp, h)          # out: (B, 1, hidden)
            logits = self.out(out[:, -1])          # (B, vocab)
            nxt = logits.argmax(dim=-1)            # (B,)
            tokens.append(nxt)
            inp = self.embed(nxt).unsqueeze(1)     # (B, 1, emb)
        return torch.stack(tokens, dim=1)          # (B, max_len)


# ---------------------------------------------------------------------- #
# The bridge
# ---------------------------------------------------------------------- #
class LatentSpaceBridge:
    """SELFIES-VAE encoder/decoder between latent vectors and SMILES.

    On construction, loads a cached checkpoint if one exists at ``weights_path``
    (and its latent dim matches); otherwise trains a fresh VAE on the library at
    ``library_dir`` and caches it. Training is quick (single-layer GRU, ~hundreds
    of small molecules) and runs on CPU.

    Args:
        latent_dim: Bottleneck dimensionality the BO loop searches over (50).
        low, high: Per-dimension bounds of the latent search box ([-1, 1]).
        weights_path: Where the trained checkpoint is cached / loaded.
        library_dir: Library dir (``smiles.csv``) used to train on first run.
        device: Torch device ("cpu").
        n_epochs, batch_size, lr, seed: Training hyperparameters (first run only).
        dtype: Torch dtype for emitted latent tensors (double, to match BoTorch).
    """

    def __init__(self, latent_dim=50, low=-1.0, high=1.0,
                 max_len=DEFAULT_MAX_LEN,
                 weights_path=DEFAULT_WEIGHTS_PATH,
                 library_dir=DEFAULT_LIBRARY_DIR,
                 corpus_size=DEFAULT_CORPUS_SIZE,
                 device="cpu", n_epochs=120, batch_size=64, lr=1e-3, seed=42,
                 dtype=torch.double):
        self.latent_dim = int(latent_dim)
        self.low = float(low)
        self.high = float(high)
        self.corpus_size = int(corpus_size)
        # Fixed structural budget on SELFIES length. A cached checkpoint trained
        # under a DIFFERENT max_len is considered stale (see _can_load), so changing
        # this value automatically triggers a one-time retrain.
        self.max_len = int(max_len)
        self.weights_path = weights_path
        self.library_dir = library_dir
        self.device = torch.device(device)
        self.dtype = dtype

        self.symbol_to_idx = None
        self.idx_to_symbol = None
        self.model = None

        if self._can_load(weights_path):
            self._load(weights_path)
        else:
            self._train_and_cache(
                library_dir, weights_path,
                n_epochs=n_epochs, batch_size=batch_size, lr=lr, seed=seed,
            )

    # ------------------------------------------------------------------ #
    # Latent-space geometry
    # ------------------------------------------------------------------ #
    @property
    def bounds(self):
        """The latent search box as a ``(2, latent_dim)`` tensor (row0 lo, row1 hi)."""
        lo = torch.full((self.latent_dim,), self.low, dtype=self.dtype)
        hi = torch.full((self.latent_dim,), self.high, dtype=self.dtype)
        return torch.stack([lo, hi], dim=0)

    # ------------------------------------------------------------------ #
    # Vocab / tokenization helpers
    # ------------------------------------------------------------------ #
    def _selfies_to_labels(self, selfies_str):
        """SELFIES string -> fixed-length (max_len) list of token ids (pad-filled).

        Unknown symbols (outside the training alphabet) and overflow past
        ``max_len`` are dropped; the remainder is right-padded with ``pad_idx``.
        """
        pad_idx = self.symbol_to_idx[PAD_SYMBOL]
        ids = []
        for sym in sf.split_selfies(selfies_str):
            if sym in self.symbol_to_idx:
                ids.append(self.symbol_to_idx[sym])
            if len(ids) >= self.max_len:
                break
        ids = ids[: self.max_len]
        ids += [pad_idx] * (self.max_len - len(ids))
        return ids

    def _labels_to_smiles(self, label_row):
        """Token-id row -> SMILES via SELFIES, then RDKit-sanitize/canonicalize.

        Returns a canonical SMILES string, or ``None`` if the (rare) decode is
        chemically invalid — the caller substitutes the safe fallback.
        """
        pad_idx = self.symbol_to_idx[PAD_SYMBOL]
        symbols = []
        for idx in label_row:
            idx = int(idx)
            if idx == pad_idx:
                continue  # [nop] is a no-op; skip to keep the string compact
            symbols.append(self.idx_to_symbol[idx])
        selfies_str = "".join(symbols)
        smiles = sf.decoder(selfies_str) if selfies_str else ""
        if not smiles:
            return None
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)

    # ------------------------------------------------------------------ #
    # encode / decode (public contract)
    # ------------------------------------------------------------------ #
    def encode(self, smiles_list):
        """Encode SMILES into latent vectors (posterior mean ``mu``).

        SMILES that cannot be converted to SELFIES are encoded as the all-pad
        sequence (a zero-information point) rather than dropped, so the returned
        tensor stays row-aligned with the input.

        Args:
            smiles_list: Iterable of SMILES strings.

        Returns:
            A ``(N, latent_dim)`` tensor of latent means.
        """
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]
        smiles_list = list(smiles_list)

        rows = []
        for smi in smiles_list:
            try:
                selfies_str = sf.encoder(smi)
            except Exception:
                selfies_str = None
            if selfies_str is None:
                labels = [self.symbol_to_idx[PAD_SYMBOL]] * self.max_len
            else:
                labels = self._selfies_to_labels(selfies_str)
            rows.append(labels)

        x = torch.tensor(rows, dtype=torch.long, device=self.device)
        self.model.eval()
        with torch.no_grad():
            mu, _ = self.model.encode(x)
        return mu.to(self.dtype).cpu()

    def decode(self, latent_vectors):
        """Decode latent vectors into RDKit-valid SMILES.

        Accepts ``(N, latent_dim)`` or ``(latent_dim,)``. Each row is greedily
        decoded to SELFIES -> SMILES and RDKit-sanitized; any (rare) invalid
        result is replaced by ``SAFE_FALLBACK_SMILES`` so exactly one valid SMILES
        is returned per input row (Z/Y stay aligned in the BO loop).

        Args:
            latent_vectors: Tensor/array of shape ``(N, latent_dim)``.

        Returns:
            A list of ``N`` canonical SMILES strings.
        """
        z = torch.as_tensor(latent_vectors, dtype=torch.float32, device=self.device)
        if z.ndim == 1:
            z = z.unsqueeze(0)

        self.model.eval()
        labels = self.model.decode_greedy(z).cpu().numpy()

        smiles = []
        n_fallback = 0
        for row in labels:
            smi = self._labels_to_smiles(row)
            if smi is None:
                smi = SAFE_FALLBACK_SMILES
                n_fallback += 1
            smiles.append(smi)
        if n_fallback:
            warnings.warn(
                f"decode: {n_fallback}/{len(smiles)} latent vectors decoded to an "
                f"invalid molecule; substituted the safe fallback "
                f"{SAFE_FALLBACK_SMILES!r}."
            )
        return smiles

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def _can_load(self, path):
        """True if a checkpoint exists matching this bridge's latent_dim AND max_len.

        Requiring ``max_len`` to match means changing the structural budget
        invalidates the cache and forces a one-time retrain under the new limit.
        """
        if not (path and os.path.exists(path)):
            return False
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            return False
        return (int(ckpt.get("latent_dim", -1)) == self.latent_dim
                and int(ckpt.get("max_len", -1)) == self.max_len)

    def _build_model_from(self, ckpt):
        """Instantiate the VAE from a checkpoint dict and load weights."""
        self.idx_to_symbol = list(ckpt["idx_to_symbol"])
        self.symbol_to_idx = {s: i for i, s in enumerate(self.idx_to_symbol)}
        self.max_len = int(ckpt["max_len"])
        model = SelfiesVAE(
            vocab_size=len(self.idx_to_symbol),
            latent_dim=self.latent_dim,
            max_len=self.max_len,
            emb_dim=int(ckpt["emb_dim"]),
            hidden_dim=int(ckpt["hidden_dim"]),
            pad_idx=self.symbol_to_idx[PAD_SYMBOL],
        ).to(self.device)
        model.load_state_dict(ckpt["state_dict"])
        self.model = model

    def _load(self, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self._build_model_from(ckpt)
        print(f"[VAE] Loaded cached SELFIES-VAE from {path} "
              f"(vocab={len(self.idx_to_symbol)}, max_len={self.max_len}, "
              f"latent_dim={self.latent_dim}).")

    # ------------------------------------------------------------------ #
    # Training (first run only)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _read_smiles_csv(path):
        """Read a SMILES column (``SMILES`` or first column) from a CSV."""
        import pandas as pd
        df = pd.read_csv(path)
        col = "SMILES" if "SMILES" in df.columns else df.columns[0]
        return [str(s) for s in df[col].tolist()]

    def _build_corpus_from_chembl(self, src, out_path, seed=42):
        """Curate a drug-like training corpus of ~``corpus_size`` SMILES from ChEMBL.

        The raw ChEMBL pull spans everything from small fragments to giant peptides.
        We sample a shuffled candidate pool and keep only molecules that are:
        RDKit-parseable, within the Lipinski MW/LogP envelope, not too large
        (heavy-atom bound), and — critically — SELFIES-encodable within the
        ``max_len`` token budget (so no training molecule is truncated). The result
        is canonicalized, de-duplicated, and cached to ``out_path``.
        """
        import pandas as pd
        from rdkit.Chem import Descriptors, Crippen

        print(f"[VAE] Curating a drug-like training corpus (target "
              f"{self.corpus_size}) from {src} ...")
        df = pd.read_csv(src)
        col = "smiles" if "smiles" in df.columns else (
            "SMILES" if "SMILES" in df.columns else df.columns[-1])
        series = df[col].dropna().astype(str)

        # Shuffle a candidate pool for chemical diversity; scan until the target is
        # met (a generous pool covers the ~drug-like acceptance rate).
        rng = np.random.default_rng(seed)
        pool_n = min(len(series), max(60000, self.corpus_size * 12))
        pool = series.sample(n=pool_n, random_state=seed).tolist()

        kept = []
        seen = set()
        for smi in pool:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            if mol.GetNumHeavyAtoms() > 45:
                continue
            if Descriptors.MolWt(mol) > 500.0 or Crippen.MolLogP(mol) > 5.0:
                continue
            canon = Chem.MolToSmiles(mol)
            if canon in seen:
                continue
            try:
                s = sf.encoder(canon)
            except Exception:
                s = None
            if s is None or sf.len_selfies(s) > self.max_len:
                continue
            seen.add(canon)
            kept.append(canon)
            if len(kept) >= self.corpus_size:
                break

        if not kept:
            raise RuntimeError(
                f"[VAE] ChEMBL corpus build from {src} yielded no drug-like, "
                "within-budget molecules; check the source file."
            )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        pd.DataFrame({"SMILES": kept}).to_csv(out_path, index=False)
        print(f"[VAE] Cached {len(kept)} drug-like training molecules to {out_path}.")
        return kept

    def _resolve_training_smiles(self, library_dir):
        """Return the training SMILES, preferring the largest available corpus.

        Priority: (1) a previously curated corpus at ``CORPUS_DIR/smiles.csv``;
        (2) freshly curated from ``CHEMBL_SOURCE`` if that raw pull exists;
        (3) the small ``<library_dir>/smiles.csv`` fallback.
        """
        corpus_path = os.path.join(CORPUS_DIR, "smiles.csv")
        if os.path.exists(corpus_path):
            print(f"[VAE] Using cached training corpus at {corpus_path}.")
            return self._read_smiles_csv(corpus_path)
        if os.path.exists(CHEMBL_SOURCE):
            return self._build_corpus_from_chembl(CHEMBL_SOURCE, corpus_path)
        fallback = os.path.join(library_dir, "smiles.csv")
        if not os.path.exists(fallback):
            raise FileNotFoundError(
                f"[VAE] Cannot train: no curated corpus, no {CHEMBL_SOURCE}, and no "
                f"library at {fallback}. Build one with `python data.py` first."
            )
        print(f"[VAE] ChEMBL source absent; falling back to small library {fallback}.")
        return self._read_smiles_csv(fallback)

    def _build_vocab(self, selfies_list):
        """Build symbol<->id maps from a list of SELFIES strings.

        ``self.max_len`` is a fixed structural budget set at construction (NOT
        derived from the data max), so this only builds the vocabulary and reports
        how many training molecules exceed the budget and will be truncated to it.
        """
        alphabet = sf.get_alphabet_from_selfies(selfies_list)
        alphabet.discard(PAD_SYMBOL)
        # Index 0 reserved for PAD_SYMBOL; the rest sorted for determinism.
        self.idx_to_symbol = [PAD_SYMBOL] + sorted(alphabet)
        self.symbol_to_idx = {s: i for i, s in enumerate(self.idx_to_symbol)}
        observed_max = max((sf.len_selfies(s) for s in selfies_list), default=1)
        n_truncated = sum(1 for s in selfies_list
                          if sf.len_selfies(s) > self.max_len)
        print(f"[VAE] max_len={self.max_len} (data max={observed_max}); "
              f"{n_truncated}/{len(selfies_list)} molecules exceed it and are "
              "truncated to the budget.")

    def _train_and_cache(self, library_dir, weights_path,
                         n_epochs, batch_size, lr, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)

        print(f"[VAE] No cached checkpoint at {weights_path}; training a fresh "
              f"SELFIES-VAE ...")
        raw_smiles = self._resolve_training_smiles(library_dir)

        # SMILES -> SELFIES (skip anything SELFIES can't represent).
        selfies_list = []
        n_skipped = 0
        for smi in raw_smiles:
            try:
                s = sf.encoder(smi)
            except Exception:
                s = None
            if s is None:
                n_skipped += 1
                continue
            selfies_list.append(s)
        if not selfies_list:
            raise RuntimeError("[VAE] No library molecules could be SELFIES-encoded.")
        print(f"[VAE] Training molecules: {len(selfies_list)} "
              f"({n_skipped} skipped as non-SELFIES-encodable).")

        self._build_vocab(selfies_list)
        print(f"[VAE] Vocab size={len(self.idx_to_symbol)}, max_len={self.max_len}.")

        X = torch.tensor(
            [self._selfies_to_labels(s) for s in selfies_list],
            dtype=torch.long, device=self.device,
        )

        emb_dim, hidden_dim = 64, 256
        model = SelfiesVAE(
            vocab_size=len(self.idx_to_symbol),
            latent_dim=self.latent_dim,
            max_len=self.max_len,
            emb_dim=emb_dim, hidden_dim=hidden_dim,
            pad_idx=self.symbol_to_idx[PAD_SYMBOL],
        ).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        recon_loss = nn.CrossEntropyLoss(ignore_index=self.symbol_to_idx[PAD_SYMBOL])
        n = X.shape[0]
        # Anti-posterior-collapse settings. An autoregressive decoder will happily
        # ignore z and reconstruct from teacher forcing alone (KL -> 0), leaving a
        # useless latent for BO. Two standard countermeasures:
        #   * word_dropout: starve the decoder of ground-truth previous tokens so
        #     it must use z (Bowman et al. 2016).
        #   * free_bits: floor the per-dimension KL so the encoder is never pushed
        #     to collapse mu -> 0; it keeps encoding real information.
        beta_max = 0.1
        anneal_epochs = max(1, n_epochs // 2)
        word_dropout = 0.3
        free_bits = 0.05  # nats per latent dimension

        model.train()
        for epoch in range(1, n_epochs + 1):
            perm = torch.randperm(n, device=self.device)
            beta = beta_max * min(1.0, epoch / anneal_epochs)
            tot_recon = tot_kl = 0.0
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                xb = X[idx]
                mu, logvar = model.encode(xb)
                z = model.reparameterize(mu, logvar)
                logits = model.decode_train(z, xb, word_dropout=word_dropout)
                rl = recon_loss(
                    logits.reshape(-1, logits.size(-1)), xb.reshape(-1)
                )
                # Per-dimension KL with a free-bits floor, then summed over dims.
                kl_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).mean(dim=0)
                kl = torch.clamp(kl_dim, min=free_bits).sum()
                loss = rl + beta * kl
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tot_recon += rl.item() * xb.shape[0]
                tot_kl += kl.item() * xb.shape[0]
            if epoch == 1 or epoch % 20 == 0 or epoch == n_epochs:
                print(f"[VAE] epoch {epoch:>4}/{n_epochs} "
                      f"recon={tot_recon / n:.4f} kl={tot_kl / n:.4f} beta={beta:.3f}")

        self.model = model
        self._cache(weights_path, emb_dim, hidden_dim)

        # Quick post-train sanity: reconstruct a handful of training molecules.
        self._report_reconstruction(selfies_list[: min(50, len(selfies_list))])

    def _cache(self, weights_path, emb_dim, hidden_dim):
        os.makedirs(os.path.dirname(weights_path) or ".", exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "idx_to_symbol": self.idx_to_symbol,
                "max_len": self.max_len,
                "latent_dim": self.latent_dim,
                "emb_dim": emb_dim,
                "hidden_dim": hidden_dim,
            },
            weights_path,
        )
        print(f"[VAE] Cached trained SELFIES-VAE to {weights_path}.")

    def _report_reconstruction(self, selfies_sample):
        """Print teacher-free reconstruction accuracy on a sample (diagnostic)."""
        smiles_in = [sf.decoder(s) for s in selfies_sample]
        x = torch.tensor(
            [self._selfies_to_labels(s) for s in selfies_sample],
            dtype=torch.long, device=self.device,
        )
        self.model.eval()
        with torch.no_grad():
            mu, _ = self.model.encode(x)
            labels = self.model.decode_greedy(mu).cpu().numpy()
        exact = 0
        for smi_in, row in zip(smiles_in, labels):
            smi_out = self._labels_to_smiles(row)
            mol_in = Chem.MolFromSmiles(smi_in) if smi_in else None
            if (smi_out is not None and mol_in is not None
                    and Chem.MolToSmiles(mol_in) == smi_out):
                exact += 1
        print(f"[VAE] Reconstruction (greedy, from mu) on {len(selfies_sample)} "
              f"training molecules: {exact}/{len(selfies_sample)} exact.")


if __name__ == "__main__":
    print("Constructing LatentSpaceBridge (trains on first run, else loads cache)...")
    bridge = LatentSpaceBridge(latent_dim=50)

    print(f"\nlatent_dim = {bridge.latent_dim}")
    assert bridge.bounds.shape == (2, 50)
    assert torch.all(bridge.bounds[0] == -1.0) and torch.all(bridge.bounds[1] == 1.0)

    # encode: real SMILES -> latent means, shape-correct and finite.
    test_smiles = ["CCO", "c1ccccc1", "CC(=O)O"]
    z = bridge.encode(test_smiles)
    print(f"encode({len(test_smiles)} smiles) -> {tuple(z.shape)} (expected (3, 50))")
    assert z.shape == (3, 50) and torch.isfinite(z).all()

    # decode: random latent points in the box -> valid SMILES (SELFIES guarantee).
    rng = np.random.default_rng(0)
    z_rand = torch.as_tensor(rng.uniform(-1.0, 1.0, size=(8, 50)))
    smi = bridge.decode(z_rand)
    print(f"decode(8 random z) -> {smi}")
    assert len(smi) == 8
    assert all(Chem.MolFromSmiles(s) is not None for s in smi), "decoded an invalid SMILES"

    # Determinism: same z -> same molecule (greedy decode).
    assert bridge.decode(z_rand) == smi
    # Single-vector convenience path.
    assert len(bridge.decode(z_rand[0])) == 1

    print("\nREAL SELFIES-VAE BRIDGE OK")
