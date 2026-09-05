"""Pointwise logistic (softplus) loss for DaBR."""

import torch

from models.losses.loss_utilities import compute_softplus_loss as _compute_softplus_loss


def compute_softplus_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Trouillon logistic loss for higher-is-better DaBR scores (labels +1 / -1).

    The encoder returns a plausibility score where **higher is better** (correct
    triples are close to 0, incorrect ones are much lower). Eq. 13 in the paper is
    ``softplus(-Y * phi)`` with the same higher-is-better convention, which is
    equivalent to ``softplus(-scores * labels)`` here.

    Positives (Y=+1): ``softplus(-s)`` - loss decreases as *s* increases.
    Negatives (Y=-1): ``softplus(+s)`` - loss decreases as *s* decreases.
    This matches filtered link prediction, which ranks candidates by descending score.
    """

    return _compute_softplus_loss(scores, labels, reduction=reduction)


def build_negsamp_loss_fn(args):
    """Factory for pointwise logistic negative-sampling training."""

    del args

    def loss_fn(
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
        weights=None,
        **_kwargs,
    ) -> torch.Tensor:
        return _compute_softplus_loss(
            pos_scores=pos_scores,
            neg_scores=neg_scores,
            weights=weights,
        )

    return loss_fn


build_loss_fn = build_negsamp_loss_fn


def compute_loss(args):
    return build_negsamp_loss_fn(args)
