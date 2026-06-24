"""
Reference implementations used ONLY by the verification harness for self-tests
and as a "golden" example of what a correct coregionalized model looks like.

You do NOT need to use these in BioMOBO. They exist so the test suite can:
  (a) prove its own logic works (positive control = ICM), and
  (b) prove it actually catches the common mistake (negative control =
      independent tasks dressed up as multitask).

Everything here is pure gpytorch (no botorch dependency).
"""
from __future__ import annotations

import torch
import gpytorch
from gpytorch.distributions import MultitaskMultivariateNormal


class TanimotoKernel(gpytorch.kernels.Kernel):
    """Minimal Tanimoto (Jaccard) kernel over count/binary fingerprint vectors.

    k(x, x') = <x, x'> / (||x||^2 + ||x'||^2 - <x, x'>)

    This mirrors the kernel used by the GP-MOBO repo. It is included so the
    reference model exercises the same "data covariance" path the real code uses.
    """

    has_lengthscale = False

    def forward(self, x1, x2, diag=False, **params):
        # x1: (..., n, d), x2: (..., m, d)
        dot = x1 @ x2.transpose(-1, -2)              # <x, x'>
        s1 = (x1 * x1).sum(-1, keepdim=True)         # ||x||^2  -> (..., n, 1)
        s2 = (x2 * x2).sum(-1, keepdim=True)         # ||x'||^2 -> (..., m, 1)
        denom = s1 + s2.transpose(-1, -2) - dot
        sim = dot / denom.clamp_min(1e-12)
        if diag:
            return sim.diagonal(dim1=-2, dim2=-1)
        return sim


class ReferenceICM(gpytorch.models.ExactGP):
    """CORRECT Intrinsic Coregionalization Model (positive control).

    Covariance = K_data (Tanimoto)  Kron  B  where B = WWᵀ + diag(v), rank>=1.
    Produces a MultitaskMultivariateNormal, i.e. learns cross-task covariance.
    """

    def __init__(self, train_x, train_y, num_tasks, rank=1, likelihood=None):
        if likelihood is None:
            likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
        super().__init__(train_x, train_y, likelihood)
        self.num_tasks = num_tasks
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            TanimotoKernel(), num_tasks=num_tasks, rank=rank  # rank>=1 => off-diagonal B
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultitaskMultivariateNormal(mean_x, covar_x)


class IndependentMultitaskBaseline(gpytorch.models.ExactGP):
    """WRONG model (negative control): multitask *shaped* output but tasks are
    independent (rank=0 coregionalization => diagonal B). This is exactly the
    failure mode the harness must catch: it LOOKS multitask but learns no
    cross-task correlation."""

    def __init__(self, train_x, train_y, num_tasks, likelihood=None):
        if likelihood is None:
            likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
        super().__init__(train_x, train_y, likelihood)
        self.num_tasks = num_tasks
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            TanimotoKernel(), num_tasks=num_tasks, rank=0  # rank=0 => diagonal B => independent
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultitaskMultivariateNormal(mean_x, covar_x)


def _toy_data(seed=0, n=24, d=16, num_tasks=4):
    """Binary fingerprints X; correlated tasks Y so a coregionalized model has
    something real to learn."""
    g = torch.Generator().manual_seed(seed)
    X = (torch.rand(n, d, generator=g) > 0.5).float()
    base = (X.sum(-1, keepdim=True) / d)            # a shared latent signal
    noise = 0.05 * torch.randn(n, num_tasks, generator=g)
    # task0 & task1 strongly correlated; task2 anti-correlated; task3 noisier
    Y = torch.cat([base, base, -base, 0.5 * base], dim=-1) + noise
    return X, Y


def fit(model, train_x, train_y, iters=60, lr=0.1):
    model.train()
    model.likelihood.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    for _ in range(iters):
        opt.zero_grad()
        out = model(train_x)
        loss = -mll(out, train_y)
        loss.backward()
        opt.step()
    model.eval()
    model.likelihood.eval()
    return model


def build_reference(kind="icm", seed=0, iters=60):
    """Return (model, train_x, train_y, num_tasks). kind in {'icm','independent'}."""
    X, Y = _toy_data(seed=seed)
    num_tasks = Y.shape[-1]
    cls = ReferenceICM if kind == "icm" else IndependentMultitaskBaseline
    model = cls(X, Y, num_tasks=num_tasks)
    fit(model, X, Y, iters=iters)
    return model, X, Y, num_tasks
