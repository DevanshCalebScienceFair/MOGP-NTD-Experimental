"""Multi-Output Gaussian Process for molecular property prediction.

This module builds a batch-independent multi-output GP on top of the project's
existing pieces:

  * Morgan fingerprints (``utils/featurize.py``) as the molecular representation.
  * A Tanimoto similarity kernel (``kernel.py``) as the GP covariance over those
    fingerprints.
  * A pretrained ADMET oracle (``train_admet_oracle.py`` -> ``models/pretrained_admet/``)
    that supplies the regression/classification targets to learn.

The MOGP places one independent scaled-Tanimoto GP on each objective and packs
them into a single ``MultitaskMultivariateNormal`` so predictions (mean and
per-task uncertainty) come out together. Target columns are always handled in
the fixed order ``[caco2, half_life, herg]``.

Run ``python mogp.py`` for a self-contained demo on a handful of known drugs.
"""

import os

import gpytorch
import joblib
import numpy as np
import torch

from utils.featurize import smiles_to_morgan, batch_smiles_to_morgan
from kernel import TanimotoKernel


# Directory holding the pretrained ADMET oracle models, and the fixed column
# order every Y matrix / prediction in this module adheres to.
ADMET_MODEL_DIR = os.path.join("models", "pretrained_admet")
TASK_NAMES = ["caco2", "half_life", "herg"]


class MOGPModel(gpytorch.models.ExactGP):
    """Batch-independent multi-output exact GP with a Tanimoto kernel.

    One scaled-Tanimoto GP is fitted per objective (via a batch dimension over
    tasks). The per-task GPs are combined into a single
    ``MultitaskMultivariateNormal`` so the model jointly predicts all objectives.

    Args:
        train_x: Fingerprint tensor of shape ``(N, 2048)``, float32.
        train_y: Target tensor of shape ``(N, num_tasks)``, float32. ``num_tasks``
            is inferred from ``train_y.shape[1]``.
        likelihood: A ``MultitaskGaussianLikelihood`` with matching ``num_tasks``.
    """

    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        num_tasks = train_y.shape[1]

        # One constant mean and one scaled Tanimoto kernel per task. The batch
        # dimension over tasks gives each objective its own outputscale (and the
        # likelihood gives each its own noise), i.e. independent GPs per output.
        self.mean_module = gpytorch.means.ConstantMean(
            batch_shape=torch.Size([num_tasks])
        )
        self.covar_module = gpytorch.kernels.ScaleKernel(
            TanimotoKernel(),
            batch_shape=torch.Size([num_tasks]),
        )

    def forward(self, x):
        # mean_x: (num_tasks, N), covar_x: (num_tasks, N, N) -> a batch of MVNs,
        # one per task, repacked into a single multitask distribution shaped
        # (N, num_tasks).
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )


def load_admet_scores(smiles_list):
    """Compute ADMET targets for a list of SMILES using the pretrained oracle.

    Loads the three pretrained models from ``models/pretrained_admet/`` and runs
    them on Morgan fingerprints of the input molecules. The half-life model is
    stored in log10(hours); this function reverses that transform so the returned
    value is in hours.

    Args:
        smiles_list: Iterable of SMILES strings.

    Returns:
        A tuple ``(Y, valid_smiles)`` where ``Y`` is a numpy array of shape
        ``(N_valid, 3)`` with columns ``[caco2, half_life, herg]`` and
        ``valid_smiles`` is the list of SMILES that featurized successfully (in
        input order).

    Raises:
        FileNotFoundError: If the pretrained model directory does not exist.
    """
    if not os.path.isdir(ADMET_MODEL_DIR):
        raise FileNotFoundError(
            f"ADMET oracle models not found at '{ADMET_MODEL_DIR}'. "
            "Run train_admet_oracle.py first to train and save them."
        )

    caco2 = joblib.load(os.path.join(ADMET_MODEL_DIR, "caco2.joblib"))["model"]
    half_life = joblib.load(os.path.join(ADMET_MODEL_DIR, "half_life.joblib"))["model"]
    herg = joblib.load(os.path.join(ADMET_MODEL_DIR, "herg.joblib"))["model"]

    fingerprints, valid_smiles = batch_smiles_to_morgan(smiles_list)
    if fingerprints.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32), valid_smiles

    caco2_pred = caco2.predict(fingerprints)
    # Stored in log10(hours); reverse the transform to get hours.
    half_life_pred = 10.0 ** half_life.predict(fingerprints)
    herg_pred = herg.predict_proba(fingerprints)[:, 1]

    Y = np.column_stack([caco2_pred, half_life_pred, herg_pred]).astype(np.float32)
    return Y, valid_smiles


def train_mogp(train_x, train_y, n_iterations=200, lr=0.1):
    """Train the MOGP on fingerprints and (per-column normalized) targets.

    Args:
        train_x: Fingerprint matrix of shape ``(N, 2048)``, int8.
        train_y: Target matrix of shape ``(N, 3)``, float32, columns
            ``[caco2, half_life, herg]``.
        n_iterations: Number of Adam steps.
        lr: Adam learning rate.

    Returns:
        A tuple ``(model, likelihood, y_mean, y_std)`` where ``y_mean`` and
        ``y_std`` are numpy arrays of shape ``(3,)`` used to reverse the target
        normalization at prediction time.
    """
    train_x_t = torch.from_numpy(np.asarray(train_x)).to(torch.float32)
    train_y = np.asarray(train_y, dtype=np.float32)

    # Per-column standardization. Guard against zero-variance columns so we never
    # divide by zero (a constant column normalizes to 0 and reverses to its mean).
    y_mean = train_y.mean(axis=0)
    y_std = train_y.std(axis=0)
    y_std[y_std == 0] = 1.0
    train_y_norm = (train_y - y_mean) / y_std
    train_y_t = torch.from_numpy(train_y_norm).to(torch.float32)

    num_tasks = train_y.shape[1]
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
    model = MOGPModel(train_x_t, train_y_t, likelihood)

    model.train()
    likelihood.train()

    # model.parameters() includes the likelihood's parameters because ExactGP
    # registers the likelihood as a submodule.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(n_iterations):
        optimizer.zero_grad()
        output = model(train_x_t)
        loss = -mll(output, train_y_t)
        loss.backward()
        optimizer.step()
        if (i + 1) % 20 == 0:
            print(f"Iter {i + 1:>4}/{n_iterations} - loss: {loss.item():.4f}")

    return model, likelihood, y_mean, y_std


def predict(model, likelihood, y_mean, y_std, X_new):
    """Predict ADMET targets and uncertainty for new molecules.

    Args:
        model: A trained ``MOGPModel``.
        likelihood: The matching ``MultitaskGaussianLikelihood``.
        y_mean: Normalization means, shape ``(3,)``.
        y_std: Normalization stds, shape ``(3,)``.
        X_new: Fingerprint matrix of shape ``(M, 2048)``, int8.

    Returns:
        A tuple ``(mean, variance)`` of numpy arrays, each shape ``(M, 3)``, with
        columns ``[caco2, half_life, herg]`` on the original (de-normalized)
        scale. ``variance`` is the per-objective predictive variance.
    """
    X_new_t = torch.from_numpy(np.asarray(X_new)).to(torch.float32)

    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        posterior = likelihood(model(X_new_t))
        mean = posterior.mean.cpu().numpy()
        variance = posterior.variance.cpu().numpy()

    # Reverse the per-column standardization applied during training.
    mean = mean * y_std + y_mean
    variance = variance * (y_std ** 2)
    return mean, variance


if __name__ == "__main__":
    train_smiles = {
        "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
        "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "Paracetamol": "CC(=O)Nc1ccc(O)cc1",
        "Chloroquine": "CCN(CC)CCCC(C)NC1=C2C=CC(=CC2=NC=C1)Cl",
        "Pyrimethamine": "C1=CC(=NC(=N1)N)CC2=CC=C(C=C2)Cl",
    }

    print("Loading ADMET scores for training molecules...")
    Y, valid_smiles = load_admet_scores(list(train_smiles.values()))

    print("\nADMET target matrix Y:")
    headers = ["caco2", "half_life_hours", "herg_prob"]
    print(f"{'molecule':>14}" + "".join(f"{h:>18}" for h in headers))
    name_by_smiles = {s: n for n, s in train_smiles.items()}
    for name, row in zip((name_by_smiles[s] for s in valid_smiles), Y):
        print(f"{name:>14}" + "".join(f"{v:18.4f}" for v in row))

    # Fingerprints for the valid training molecules.
    train_x = np.vstack([smiles_to_morgan(s) for s in valid_smiles])

    print("\nTraining MOGP for 100 iterations...")
    model, likelihood, y_mean, y_std = train_mogp(train_x, Y, n_iterations=100)

    new_smiles = {
        "Artemisinin": "C1CC2CC(=O)OC3OC4(C)CCC1C(C)(C2)C34",
        "Quinine": "COC1=CC2=C(C=CN=C2C=C1)C(O)C3CC4=CC=NC5=CC=C(C=C45)C3",
    }
    X_new, new_valid = batch_smiles_to_morgan(list(new_smiles.values()))
    new_name_by_smiles = {s: n for n, s in new_smiles.items()}

    mean, variance = predict(model, likelihood, y_mean, y_std, X_new)
    std = np.sqrt(variance)

    print("\nPredictions for new molecules:")
    for i, smiles in enumerate(new_valid):
        name = new_name_by_smiles[smiles]
        print(f"\n{name}:")
        print(f"  caco2           = {mean[i, 0]:.4f}  (std {std[i, 0]:.4f})")
        print(f"  half_life_hours = {mean[i, 1]:.4f}  (std {std[i, 1]:.4f})")
        print(f"  herg_prob       = {mean[i, 2]:.4f}  (std {std[i, 2]:.4f})")
