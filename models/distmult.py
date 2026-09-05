"""DistMult scorer and model (``score_emb``)."""

import torch

from base.model import KGEScorer, KGEModel


def build_scorer(args) -> 'DistMultScorer':
	return DistMultScorer(args)


class DistMultScorer(KGEScorer):
	"""DistMult score function via a single ``score_emb``.

	``combine`` modes: ``hrt``, ``hr_``, ``_rt``, ``hr_c``, ``_rt_c``.
	"""

	bidirectional_score_batch = True

	def __init__(self, args=None):
		super().__init__()
		self.args = args

	def supports_candidate_scoring(self) -> bool:
		return True

	def score_emb(
		self,
		h_emb: torch.Tensor,
		r_emb: torch.Tensor,
		t_emb: torch.Tensor,
		combine: str,
		**kwargs,
	) -> torch.Tensor:
		del kwargs
		n = r_emb.size(0)

		if combine == 'hrt':
			return (h_emb * r_emb * t_emb).sum(dim=-1).view(n, -1)
		if combine == 'hr_':
			return torch.mm(h_emb * r_emb, t_emb.t())
		if combine == '_rt':
			return torch.mm(t_emb * r_emb, h_emb.t())
		if combine == 'hr_c':
			# t_emb is [B, C, D]
			return torch.bmm((h_emb * r_emb).unsqueeze(1), t_emb.transpose(1, 2)).squeeze(1)
		if combine == '_rt_c':
			# h_emb is [B, C, D]
			return torch.bmm((t_emb * r_emb).unsqueeze(1), h_emb.transpose(1, 2)).squeeze(1)
		raise ValueError(f'cannot handle combine="{combine}"')


class DistMultModel(KGEModel):
	"""Bind lookup embedders to ``DistMultScorer`` (``scorers`` length 1 by default)."""

	def __init__(
		self,
		ent_embedder,
		rel_embedder,
		scorers=None,
		args=None,
		aux_embedders=None,
	):
		if scorers is None:
			scorers = [DistMultScorer(args)]
		super().__init__(
			ent_embedder,
			rel_embedder,
			scorers=scorers,
			args=args,
			aux_embedders=aux_embedders,
		)

	def query_encoder(self, h: torch.Tensor, r: torch.Tensor, **kwargs) -> torch.Tensor:
		del kwargs
		return self.embed_h(h) * self.embed_r(r)

	def inverse_query_encoder(self, r: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
		del kwargs
		return self.embed_t(t) * self.embed_r(r)
