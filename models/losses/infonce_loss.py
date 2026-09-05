"""InfoNCE loss computation for models: SimKGC."""

from dataclasses import dataclass
import torch

from models.losses.loss_utilities import compute_softmax_loss


@dataclass
class ModelOutput:
	"""Structured output from the model's forward pass, containing all necessary components for InfoNCE loss computation."""
	
	logits: torch.Tensor
	labels: torch.Tensor
	inv_t: torch.Tensor
	hr_vector: torch.Tensor
	tail_vector: torch.Tensor
	head_vector: torch.Tensor | None = None


def compute_listwise_loss(scores: torch.Tensor, truth_indices: torch.Tensor) -> torch.Tensor:
	"""Compute the listwise softmax loss for a batch of candidate scores."""

	if not torch.is_tensor(truth_indices):
		truth_indices = torch.as_tensor(truth_indices, device=scores.device, dtype=torch.long)
	else:
		truth_indices = truth_indices.to(device=scores.device, dtype=torch.long)

	return compute_softmax_loss(scores, truth_indices, reduction='mean')


def compute_infonce_logits(query_vec: torch.Tensor, candidate_vec: torch.Tensor, temp: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
	"""Compute the core InfoNCE logits shared across contrastive models."""

	logits = torch.mm(query_vec, candidate_vec.t())
	if margin > 0:
		logits.diagonal().sub_(margin)
	return logits * torch.exp(temp)


def compute_logits(model, output_dict: dict, batch_dict: dict) -> ModelOutput:
	"""Compute logits and labels for InfoNCE loss based on model outputs and batch information, applying necessary masking and adjustments."""
	
	hr_vector, tail_vector = output_dict['hr_vector'], output_dict['tail_vector']
	batch_size = hr_vector.size(0)
	labels = torch.arange(batch_size).to(hr_vector.device)

	logits = compute_infonce_logits(
		query_vec=hr_vector,
		candidate_vec=tail_vector,
		temp=model.log_inv_t,
		margin=model.add_margin if model.training else 0.0,
	)

	return ModelOutput(
		logits=logits,
		labels=labels,
		inv_t=model.log_inv_t.detach().exp(),
		hr_vector=hr_vector.detach(),
		tail_vector=tail_vector.detach(),
    )


def build_1vsall_loss_fn(args):
	"""Factory for 1-vs-all cross-entropy training."""

	del args

	def loss_fn(scores: torch.Tensor, targets: torch.Tensor, **_kwargs) -> torch.Tensor:
		return compute_softmax_loss(scores, targets.long(), reduction='mean')

	return loss_fn


def build_inbatch_loss_fn(args):
	"""Factory for symmetric InfoNCE CE used by in-batch contrastive training."""

	ce_loss_fn = build_1vsall_loss_fn(args)

	def loss_fn(logits: torch.Tensor, labels: torch.Tensor, **_kwargs) -> torch.Tensor:
		return ce_loss_fn(logits, labels)

	return loss_fn


build_loss_fn = build_1vsall_loss_fn


def compute_loss(args):
	return build_1vsall_loss_fn(args)
