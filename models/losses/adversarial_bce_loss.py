"""Adversarial negative-sampling BCE loss for RotatE training."""

import torch

from models.losses.loss_utilities import compute_bce_loss


def compute_adversarial_bce_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    adversarial_temp: float,
    subsampling_weight: torch.Tensor,
) -> torch.Tensor:
    """Compute weighted RotatE-style adversarial BCE loss."""

    return compute_bce_loss(
        scores=pos_scores,
        pos_scores=pos_scores,
        neg_scores=neg_scores,
        weights=subsampling_weight,
        adversarial_temp=adversarial_temp,
    )


def build_negsamp_loss_fn(args):
    """Factory for adversarial negative-sampling training."""

    temp = float(
        getattr(args, 'adversarial_temp', None)
        or getattr(args, 'adversarial_temperature', 1.0)
        or 1.0
    )

    def loss_fn(pos_scores, neg_scores, weights=None, **_kwargs):
        return compute_adversarial_bce_loss(pos_scores, neg_scores, temp, weights)

    return loss_fn


build_loss_fn = build_negsamp_loss_fn


def compute_loss(args):
	return build_negsamp_loss_fn(args)
