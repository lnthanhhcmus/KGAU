"""Bayesian Personalized Ranking (BPR) loss for negative-sampling training."""

import torch

from models.losses.loss_utilities import compute_bpr_loss


def build_negsamp_loss_fn(args):
    """Factory for pairwise BPR on negative-sampling batches."""

    del args

    def loss_fn(
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
        weights: torch.Tensor | None = None,
        **_kwargs,
    ) -> torch.Tensor:
        return compute_bpr_loss(pos_scores, neg_scores, weights=weights)

    return loss_fn


build_bpr_loss_fn = build_negsamp_loss_fn
build_loss_fn = build_negsamp_loss_fn


def compute_loss(args):
    return build_negsamp_loss_fn(args)
