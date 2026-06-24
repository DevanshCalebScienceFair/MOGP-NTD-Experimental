"""
admet_oracle.py
===============

Inference wrapper for the Low-Fidelity ADMET Oracle.

Loads the three pretrained HistGradientBoosting models and exposes a single
`predict(smiles_list)` method returning a Pandas DataFrame that is the EXACT
same length (and order) as the input list. SMILES dropped by the featurizer
get NaN prediction rows — this positional alignment is required by the
downstream Bayesian Optimization loop.
"""

import os

import joblib
import numpy as np
import pandas as pd

from utils.featurize import batch_smiles_to_morgan


MODEL_DIR = os.path.join("models", "pretrained_admet")
N_BITS = 2048

# Maps output column -> (model filename, prediction kind).
#   "value" -> regressor .predict()
#   "proba" -> classifier .predict_proba()[:, 1]
MODEL_SPEC = {
    "Caco2_Permeability": ("caco2.joblib",     "value"),
    "Half_Life":          ("half_life.joblib", "value"),
    "hERG_Toxicity_Prob": ("herg.joblib",      "proba"),
}

OUTPUT_COLUMNS = ["SMILES", "Caco2_Permeability", "Half_Life", "hERG_Toxicity_Prob"]


class ADMETOracle:
    """Fast, in-memory ADMET property predictor.

    Parameters
    ----------
    model_dir : str
        Directory containing the serialized joblib models.
    """

    def __init__(self, model_dir=MODEL_DIR):
        self.model_dir = model_dir
        self.models = {}
        self.kinds = {}
        for column, (filename, kind) in MODEL_SPEC.items():
            path = os.path.join(model_dir, filename)
            self.models[column] = joblib.load(path)
            self.kinds[column] = kind

    @staticmethod
    def _valid_positions(smiles_list, valid_smiles):
        """Recover the original indices of the featurizer's accepted SMILES.

        `batch_smiles_to_morgan` returns `valid_smiles` (a subset of the input
        in original order) but not indices. An order-preserving two-pointer
        walk recovers the positions and stays correct when the input contains
        duplicate SMILES — critical for positional alignment.
        """
        positions = []
        j = 0  # pointer into valid_smiles
        for i, smiles in enumerate(smiles_list):
            if j < len(valid_smiles) and smiles == valid_smiles[j]:
                positions.append(i)
                j += 1
        if j != len(valid_smiles):
            raise ValueError(
                f"Alignment failed: matched {j} of {len(valid_smiles)} valid "
                "SMILES. Does batch_smiles_to_morgan preserve input order?"
            )
        return positions

    def predict(self, smiles_list):
        """Predict ADMET properties, preserving input length and order.

        Returns
        -------
        pandas.DataFrame
            One row per input SMILES (same order). Columns: SMILES,
            Caco2_Permeability, Half_Life, hERG_Toxicity_Prob. Rows for SMILES
            dropped by the featurizer contain NaN predictions.
        """
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]
        smiles_list = list(smiles_list)
        n = len(smiles_list)

        matrix, valid_smiles = batch_smiles_to_morgan(smiles_list, n_bits=N_BITS)
        X = np.asarray(matrix)
        positions = self._valid_positions(smiles_list, valid_smiles)

        result = {"SMILES": smiles_list}
        for column, model in self.models.items():
            # Full-length column pre-filled with NaN for featurizer drops.
            preds = np.full(n, np.nan, dtype=float)
            if X.shape[0] > 0:
                if self.kinds[column] == "proba":
                    vals = model.predict_proba(X)[:, 1]
                else:
                    vals = model.predict(X)
                preds[positions] = vals
            result[column] = preds

        return pd.DataFrame(result, columns=OUTPUT_COLUMNS)


if __name__ == "__main__":
    # Tiny smoke test (requires trained models + a working featurizer).
    oracle = ADMETOracle()
    demo = ["CC(=O)Oc1ccccc1C(=O)O", "CCO", "not_a_valid_smiles"]
    print(oracle.predict(demo))
