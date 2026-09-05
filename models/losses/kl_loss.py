"""KL divergence loss for KvsAll multi-hot targets (``kl`` loss)."""

import torch

from models.losses.loss_utilities import compute_softmax_loss


def compute_kl_loss(
	scores: torch.Tensor,
	targets: torch.Tensor,
	*,
	reduction: str = 'sum',
) -> torch.Tensor:
	"""Compute KL divergence between softmax scores and a label distribution.

	For index targets (1vsAll), this reduces to cross-entropy with ``reduction``.

	For multi-hot / smoothed KvsAll labels, matches ``KLDivWithSoftmaxKgeLoss``.
	"""

	return compute_softmax_loss(scores, targets, reduction=reduction)


def build_kl_loss_fn(args):
	"""Factory for KvsAll KL training (sum reduction; divide by batch_size in strategy)."""

	reduction = str(getattr(args, 'kl_reduction', 'sum'))

	def loss_fn(scores: torch.Tensor, targets: torch.Tensor, **_kwargs) -> torch.Tensor:
		return compute_kl_loss(scores, targets, reduction=reduction)

	return loss_fn


build_kvsall_loss_fn = build_kl_loss_fn
build_loss_fn = build_kl_loss_fn


def compute_loss(args):
	return build_kl_loss_fn(args)
