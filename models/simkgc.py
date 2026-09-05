"""SimKGC cosine scorer and contrastive training state (``score_emb``)."""

from typing import Any

import torch
import torch.nn as nn

from base.model import KGEScorer, TextKGEModel


class ContrastiveTrainingState(nn.Module):
	"""Training-only InfoNCE parameters and pre-batch memory (not used at LP eval)."""

	def __init__(self, args: Any, hidden_size: int):
		super().__init__()
		self.args = args
		info_nce_t = getattr(args, 'infonce_t', None)
		if info_nce_t is None:
			info_nce_t = getattr(args, 't', None)
		if info_nce_t is None:
			info_nce_t = 0.05
		info_nce_t = float(info_nce_t)
		self.log_inv_t = nn.Parameter(
			torch.tensor(1.0 / info_nce_t).log(),
			requires_grad=bool(getattr(args, 'finetune_t', True)),
		)
		self.add_margin = float(getattr(args, 'additive_margin', 0.0))
		self.batch_size = int(getattr(args, 'batch_size', 512))
		pre_batch = getattr(args, 'pre_batch', None)
		self.pre_batch = int(pre_batch if pre_batch is not None else 0)
		num_pre_batch_vectors = max(1, self.pre_batch) * self.batch_size
		random_vector = torch.randn(num_pre_batch_vectors, hidden_size)
		self.register_buffer(
			'pre_batch_vectors',
			nn.functional.normalize(random_vector, dim=1),
			persistent=False,
		)
		self.offset = 0
		self.pre_batch_exs: list = [None for _ in range(num_pre_batch_vectors)]


def build_contrastive_state(args, hidden_size: int) -> ContrastiveTrainingState:
	return ContrastiveTrainingState(args, hidden_size)


class SimKGCScorer(KGEScorer):
	"""Cosine similarity via ``score_emb`` on L2-normalized query / entity vectors.

	``combine`` modes: ``hrt``, ``hr_`` (no head 1-vs-all or candidate paths).
	"""

	kgau_alignment_mode = 'cosine'

	def __init__(self, args=None):
		super().__init__()
		self.args = args

	def supports_candidate_scoring(self) -> bool:
		return False

	def score_emb(
		self,
		h_emb: torch.Tensor,
		r_emb: torch.Tensor,
		t_emb: torch.Tensor,
		combine: str,
		**kwargs,
	) -> torch.Tensor:
		"""``h_emb`` is the query vector; ``r_emb`` is unused."""

		del r_emb, kwargs
		n = h_emb.size(0)
		if combine == 'hrt':
			return torch.sum(h_emb * t_emb, dim=-1).view(n, -1)
		if combine == 'hr_':
			return torch.mm(h_emb, t_emb.t())
		raise ValueError(f'cannot handle combine="{combine}"')


def build_scorer(args) -> 'SimKGCScorer':
	return SimKGCScorer(args)


class SimKGCModel(TextKGEModel):
	"""Bind text embedders to ``SimKGCScorer`` (``scorers`` length 1 by default)."""

	def __init__(
		self,
		ent_embedder,
		query_embedder,
		scorers=None,
		args=None,
		contrastive_state=None,
	):
		if scorers is None:
			scorers = [SimKGCScorer(args)]
		super().__init__(
			ent_embedder,
			query_embedder,
			scorers=scorers,
			args=args,
			contrastive_state=contrastive_state,
		)

	def query_encoder(self, h: torch.Tensor, r: torch.Tensor, **kwargs) -> torch.Tensor:
		"""Joint text query from the HR encoder."""

		del kwargs
		return self.query_embedder.embed_hr(h, r)

	def inverse_query_encoder(self, r: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
		raise NotImplementedError('SimKGC does not define an inverse query encoder')
