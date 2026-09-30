"""Load the shipped scorer bundle and rank candidates.

The bundle is an ensemble of two complementary families, both on descriptors only
(the VAE latent was ablated and did not earn its place -- see SCORER.md):

  * per-species `HistGradientBoostingRegressor`, exact cells only;
  * the multi-head Tobit MLP, which is the only member that sees the 1,749
    censored cells.

They are combined by RANK AVERAGING per head, not by averaging pMIC. Their
outputs are not on a comparable scale (one is a boosted tree's conditional mean,
the other a Tobit location parameter under a clamped sigma), and rank averaging
needs no calibration to be meaningful. Measured out-of-fold, the combination beats
both parents on rank correlation AND matches the better one on censored
concordance -- neither parent does both.

Everything downstream lives in normalized rank space [0, 1], which is what makes
the confidence bound and the breadth penalty commensurable across heads.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

from src.scorer.features import FeatureScaler, descriptor_frame
from src.scorer.model import PotencyScorer

DEFAULT_BUNDLE = "checkpoints/release/scorer_v1.pt"

# Fixed, not derived from len(sequences): a length-dependent batch would make the
# outputs depend on how many candidates are being scored, through reduction order.
PREDICT_BATCH = 256


class ScorerBundle:
    def __init__(self, payload):
        self.species = payload["species"]
        self.feature_names = payload["feature_names"]
        self.hgb = [pickle.loads(b) for b in payload["hgb"]]
        self.scalers = [FeatureScaler.from_state_dict(d) for d in payload["scalers"]]
        self.mlp = []
        for cfg, state in zip(payload["mlp_cfg"], payload["mlp_state"]):
            m = PotencyScorer(cfg["d_in"], cfg["n_heads"], width=cfg["width"])
            m.load_state_dict(state)
            m.eval()
            self.mlp.append(m)

    @classmethod
    def load(cls, path=DEFAULT_BUNDLE):
        return cls(torch.load(path, map_location="cpu", weights_only=False))

    def _members(self, x):
        """Per-member (N, J) predictions, HGB folds then MLP folds."""
        out = []
        for fold in self.hgb:
            out.append(np.column_stack([m.predict(x) for m in fold]))
        for model, scaler in zip(self.mlp, self.scalers):
            xs = torch.from_numpy(scaler.transform(x))
            preds = []
            with torch.no_grad():
                for i in range(0, len(xs), PREDICT_BATCH):
                    preds.append(model(xs[i : i + PREDICT_BATCH])[0])
            out.append(torch.cat(preds).numpy())
        return out

    @staticmethod
    def _fuse(ranks):
        """Fuse (M, N, J) normalized ranks into the shipped objective.

        Applied to every member for the ensemble, and to a family's slice for the
        per-family diagnostics. The arithmetic is meaningful over any subset
        because `rankdata` is applied per member independently, so a slice gives
        exactly what that family alone would have produced.
        """
        mu = ranks.mean(axis=0)
        sigma = ranks.std(axis=0)
        # Lower confidence bound: taking 100 from 50,000 is a maximum over many
        # noisy estimates, which selects hard for candidates whose error happened
        # to be positive. Demoting the ones the members disagree about is the
        # cheapest defense against that winner's curse.
        lcb = mu - sigma

        # Breadth means UNIFORMLY active, not high on average -- a peptide potent
        # against two organisms and dead against four must not win on the mean.
        return {"mu": mu, "sigma": sigma, "lcb": lcb,
                "breadth": lcb.mean(axis=1) - lcb.std(axis=1)}

    def score(self, sequences, detail=False):
        """Breadth score per sequence, higher is better, plus the diagnostics.

        `detail=True` additionally returns `families`, the same fusion applied to
        each family's folds alone, for ranking documentation. It costs nothing
        extra: the members are already computed, and the per-member ranks are
        already stacked, so a family is a slice rather than a second pass. That
        matters because `descriptor_frame` is an uncached per-sequence Python
        loop and is the dominant cost at 50,000 candidates.

        Read family breadth as an ordering, not as a magnitude comparable with
        the ensemble's: its sigma is a population std over that family's 5 folds
        where the ensemble's is over all 10, so family scores are systematically
        less conservative, and the ensemble breadth is NOT the mean of the two
        (std does not decompose that way).
        """
        x, names = descriptor_frame(sequences)
        if names != self.feature_names:
            raise SystemExit(
                "feature columns have changed since the scorer was trained.\n"
                f"  bundle: {len(self.feature_names)} columns\n"
                f"  now:    {len(names)} columns\n"
                "A silent column reordering preserves determinism and produces a "
                "completely wrong ranking, so this is fatal rather than a warning."
            )

        members = self._members(x)
        n, j = len(sequences), len(self.species)

        # Normalized ranks in [0, 1] per member per head.
        ranks = np.stack(
            [np.column_stack([rankdata(p[:, k]) / n for k in range(j)])
             for p in members]
        )                                                   # (M, N, J)

        out = self._fuse(ranks)
        out["species"] = self.species

        if detail:
            families, start = {}, 0
            for name, members_of in (("trees", self.hgb), ("tobit", self.mlp)):
                k = len(members_of)
                families[name] = self._fuse(ranks[start : start + k])
                families[name]["n_members"] = k
                start += k
            if start != len(members):
                # A third family added to _members but not here would silently
                # drop members from the per-family view while the ensemble stayed
                # correct -- i.e. documentation that quietly stops matching.
                raise SystemExit(
                    f"family split covers {start} of {len(members)} members"
                )
            out["families"] = families
        return out


_CACHE = {}


def load_scorer(path=DEFAULT_BUNDLE):
    path = str(path)
    if path not in _CACHE:
        _CACHE[path] = ScorerBundle.load(path)
    return _CACHE[path]
