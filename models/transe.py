"""TransE scorer and model (``score_emb``)."""

import torch

from base.model import KGEScorer, KGEModel


def build_scorer(args) -> 'TransEScorer':
	return TransEScorer(args)


class TransEScorer(KGEScorer):
	"""TransE score function via a single ``score_emb``.

	Matches ``KnowledgeGraphEmbedding`` (Sun et al.) when ``transe_norm=1``:
	higher score is better, ``gamma - ||h + r - t||`` for ``hrt`` / ``hr_`` / ``hr_c``.
	Head combines use ``||h - (t - r)||`` (= ``||h + r - t||`` for 1-to-1).

	``combine`` modes: ``hrt``, ``hr_``, ``_rt``, ``hr_c``, ``_rt_c``.
	"""

	bidirectional_score_batch = True

	def __init__(self, args=None):
		super().__init__()
		self.args = args
		self.dim = int(getattr(args, 'dim', 0) or 0)
		margin_value = getattr(args, 'margin', None)
		if margin_value is None:
			margin_value = getattr(args, 'gamma', 6.0)
		self.gamma = float(margin_value)
		epsilon = float(getattr(args, 'epsilon', 2.0))
		self.embedding_range = float((self.gamma + epsilon) / max(self.dim, 1))
		norm_p = int(getattr(args, 'transe_norm', 1) or 1)
		if norm_p not in (1, 2):
			raise ValueError(f'transe_norm must be 1 or 2, got {norm_p}')
		self.norm_p = norm_p

	def supports_candidate_scoring(self) -> bool:
		return True

	def _entity_chunk_size(self, batch_size: int) -> int:
		"""Candidate chunk size for 1-vs-all scoring (controls peak GPU memory)."""

		configured = int(getattr(self.args, 'eval_entity_chunk_size', 256) or 256)
		bytes_budget = int(getattr(self.args, 'eval_entity_chunk_bytes', 128 * 1024 * 1024) or 128 * 1024 * 1024)
		per_candidate = max(1, batch_size * self.dim * 4 * 2)
		memory_limit = max(1, bytes_budget // per_candidate)
		return max(1, min(configured, memory_limit))

	def _score_distance(self, diff: torch.Tensor) -> torch.Tensor:
		return self.gamma - torch.norm(diff, p=self.norm_p, dim=-1)

	def _score_1vsall(self, query: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
		num_candidates = candidates.size(0)
		batch_size = query.size(0)
		chunk_size = self._entity_chunk_size(batch_size)
		scores = query.new_empty(batch_size, num_candidates)
		for start in range(0, num_candidates, chunk_size):
			end = min(start + chunk_size, num_candidates)
			diff = query.unsqueeze(1) - candidates[start:end].unsqueeze(0)
			scores[:, start:end] = self._score_distance(diff)
		return scores

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
			return self._score_distance((h_emb + r_emb) - t_emb).view(n, -1)
		if combine == 'hr_':
			return self._score_1vsall(h_emb + r_emb, t_emb)
		if combine == '_rt':
			return self._score_1vsall(-(r_emb - t_emb), h_emb)
		if combine == 'hr_c':
			# t_emb is [B, C, D]
			return self._score_distance((h_emb.unsqueeze(1) + r_emb.unsqueeze(1)) - t_emb)
		if combine == '_rt_c':
			# h_emb is [B, C, D]
			return self._score_distance(h_emb + (r_emb.unsqueeze(1) - t_emb.unsqueeze(1)))
		raise ValueError(f'cannot handle combine="{combine}"')


class TransEModel(KGEModel):
	"""Bind lookup embedders to ``TransEScorer`` (``scorers`` length 1 by default)."""

	def __init__(
		self,
		ent_embedder,
		rel_embedder,
		scorers=None,
		args=None,
		aux_embedders=None,
	):
		if scorers is None:
			scorers = [TransEScorer(args)]
		super().__init__(
			ent_embedder,
			rel_embedder,
			scorers=scorers,
			args=args,
			aux_embedders=aux_embedders,
		)

	def query_encoder(self, h: torch.Tensor, r: torch.Tensor, **kwargs) -> torch.Tensor:
		del kwargs
		return self.embed_h(h) + self.embed_r(r)

	def inverse_query_encoder(self, r: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
		del kwargs
		return self.embed_t(t) - self.embed_r(r)
