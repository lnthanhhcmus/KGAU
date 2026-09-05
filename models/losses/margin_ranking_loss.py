"""Pairwise margin ranking loss for translational models (``margin_ranking`` loss)."""

import math

import torch

from models.losses.loss_utilities import compute_margin_broadcast_loss, compute_margin_loss


def _resolve_margin(args) -> float:
    margin = getattr(args, "margin", None)
    if margin is None:
        raw = getattr(args, "loss_arg", None)
        return 1.0 if raw is None or (isinstance(raw, float) and math.isnan(raw)) else float(raw)
    return float(margin)


def build_negsamp_loss_fn(args):
    """Factory for pairwise margin ranking on negative-sampling batches."""

    margin = _resolve_margin(args)

    def loss_fn(pos_scores: torch.Tensor, neg_scores: torch.Tensor, **_kwargs) -> torch.Tensor:
        return compute_margin_loss(pos_scores, neg_scores, margin=margin)

    return loss_fn


def build_1vsall_loss_fn(args):
    """Factory for 1-vs-all broadcast margin ranking (pos-vs-neg entity pairs)."""

    margin = _resolve_margin(args)

    def loss_fn(scores: torch.Tensor, targets: torch.Tensor, **_kwargs) -> torch.Tensor:
        return compute_margin_broadcast_loss(scores, targets, margin=margin, reduction="mean")

    return loss_fn


def build_kvsall_loss_fn(args):
    """Factory for KvsAll broadcast margin ranking (multi-positive pos-vs-neg pairs)."""

    margin = _resolve_margin(args)
    reduction = str(getattr(args, "margin_reduction", "sum"))

    def loss_fn(scores: torch.Tensor, targets: torch.Tensor, **_kwargs) -> torch.Tensor:
        return compute_margin_broadcast_loss(scores, targets, margin=margin, reduction=reduction)

    return loss_fn


build_margin_ranking_loss_fn = build_negsamp_loss_fn
build_loss_fn = build_negsamp_loss_fn


def compute_loss(args):
    return build_negsamp_loss_fn(args)
