"""Multi-class cross entropy loss for 1vsAll broadcasting (``ce`` loss)."""

import torch

from models.losses.loss_utilities import compute_softmax_loss


def compute_ce_loss(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
	"""Compute cross entropy for 1vsAll tail prediction.

	:param scores: ``[batch_size, num_entities]`` logits from ``score_hr_()``
	:param targets: ``[batch_size]`` true entity indices
	"""

	if not torch.is_tensor(targets):
		targets = torch.as_tensor(targets, device=scores.device, dtype=torch.long)
	else:
		targets = targets.to(device=scores.device, dtype=torch.long)
	return compute_softmax_loss(scores, targets, reduction='mean')


def build_ce_loss_fn(args):
	"""Factory for 1-vs-all cross-entropy training."""

	del args

	def loss_fn(scores: torch.Tensor, targets: torch.Tensor, **_kwargs) -> torch.Tensor:
		return compute_ce_loss(scores, targets)

	return loss_fn


build_1vsall_loss_fn = build_ce_loss_fn
build_loss_fn = build_ce_loss_fn


def compute_loss(args):
	return build_ce_loss_fn(args)
