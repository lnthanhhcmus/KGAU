"""ComplEx scorer and model (Hermitian ``score_emb``, GB-Magic-aligned)."""

import torch

from base.model import KGEScorer, KGEModel


def build_scorer(args) -> 'ComplExScorer':
	return ComplExScorer(args)


class ComplExScorer(KGEScorer):
	"""ComplEx via complex product + Hermitian product (no 2×-dim expansion).

	``combine`` modes: ``hrt``, ``hr_``, ``_rt``, ``hr_c``, ``_rt_c``.
	Matches GB-Magic ``ComplEx.score`` / ``_hermitian_dot``.
	"""

	bidirectional_score_batch = True

	def __init__(self, args=None):
		super().__init__()
		self.args = args

	def supports_candidate_scoring(self) -> bool:
		return True

	@staticmethod
	def _complex_mult(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
		re_a, im_a = torch.chunk(a, 2, dim=-1)
		re_b, im_b = torch.chunk(b, 2, dim=-1)
		return torch.cat([re_a * re_b - im_a * im_b, re_a * im_b + im_a * re_b], dim=-1)

	@staticmethod
	def _complex_conj_mult(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
		"""``conj(a) ∘ b`` (head-batch query: ``conj(r) ∘ t``)."""

		re_a, im_a = torch.chunk(a, 2, dim=-1)
		re_b, im_b = torch.chunk(b, 2, dim=-1)
		return torch.cat([re_a * re_b + im_a * im_b, re_a * im_b - im_a * re_b], dim=-1)

	@staticmethod
	def _hermitian_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
		re_a, im_a = torch.chunk(a, 2, dim=-1)
		re_b, im_b = torch.chunk(b, 2, dim=-1)
		return (re_a * re_b + im_a * im_b).sum(dim=-1)

	@staticmethod
	def _hermitian_mm(query: torch.Tensor, entities: torch.Tensor) -> torch.Tensor:
		"""Hermitian products of queries ``[B, D]`` against entity table ``[E, D]`` → ``[B, E]``."""

		re_q, im_q = torch.chunk(query, 2, dim=-1)
		re_e, im_e = torch.chunk(entities, 2, dim=-1)
		return re_q.mm(re_e.transpose(0, 1)) + im_q.mm(im_e.transpose(0, 1))

	@staticmethod
	def _hermitian_bc(query: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
		"""Hermitian products of queries ``[B, D]`` against candidates ``[B, C, D]`` → ``[B, C]``."""

		re_q, im_q = torch.chunk(query, 2, dim=-1)
		re_c, im_c = torch.chunk(candidates, 2, dim=-1)
		return (re_q.unsqueeze(1) * re_c + im_q.unsqueeze(1) * im_c).sum(dim=-1)

	def score_emb(
		self,
		h_emb: torch.Tensor,
		r_emb: torch.Tensor,
		t_emb: torch.Tensor,
		combine: str,
		**kwargs,
	) -> torch.Tensor:
		"""ComplEx scores via ``query = h∘r`` / ``conj(r)∘t`` then Hermitian product."""

		del kwargs
		n = r_emb.size(0)

		if combine == 'hrt':
			query = self._complex_mult(h_emb, r_emb)
			return self._hermitian_dot(query, t_emb).view(n, -1)
		if combine == 'hr_':
			query = self._complex_mult(h_emb, r_emb)
			return self._hermitian_mm(query, t_emb)
		if combine == '_rt':
			query = self._complex_conj_mult(r_emb, t_emb)
			return self._hermitian_mm(query, h_emb)
		if combine == 'hr_c':
			# t_emb is [B, C, D]
			query = self._complex_mult(h_emb, r_emb)
			return self._hermitian_bc(query, t_emb)
		if combine == '_rt_c':
			# h_emb is [B, C, D]
			query = self._complex_conj_mult(r_emb, t_emb)
			return self._hermitian_bc(query, h_emb)
		raise ValueError(f'cannot handle combine="{combine}"')


class ComplExModel(KGEModel):
	"""Bind lookup embedders to ``ComplExScorer`` (``scorers`` length 1 by default).

	KGAU encoders (ComplEx does **not** fold the relation into the target):

	* ``query_encoder(h, r)`` → complex product ``h ∘ r`` (tail prediction)
	* ``inverse_query_encoder(r, t)`` → ``conj(r) ∘ t`` (head prediction)
	* ``target_encoder`` → raw head or tail entity embedding
	"""

	target_uses_relation = False

	def __init__(
		self,
		ent_embedder,
		rel_embedder,
		scorers=None,
		args=None,
		aux_embedders=None,
	):
		if scorers is None:
			scorers = [ComplExScorer(args)]
		super().__init__(
			ent_embedder,
			rel_embedder,
			scorers=scorers,
			args=args,
			aux_embedders=aux_embedders,
		)

	def query_encoder(self, h: torch.Tensor, r: torch.Tensor, **kwargs) -> torch.Tensor:
		"""Tail-prediction query: ``h ∘ r`` in the concatenated complex space."""

		del kwargs
		return ComplExScorer._complex_mult(self.embed_h(h), self.embed_r(r))

	def inverse_query_encoder(self, r: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
		"""Head-prediction query: ``conj(r) ∘ t``."""

		del kwargs
		return ComplExScorer._complex_conj_mult(self.embed_r(r), self.embed_t(t))

	def target_encoder(
		self,
		h: torch.Tensor,
		r: torch.Tensor,
		t: torch.Tensor,
		*,
		predict_head: bool = False,
		**kwargs,
	) -> torch.Tensor:
		"""Alignment target: tail entity (tail-batch) or head entity (head-batch)."""

		del r, kwargs
		if predict_head:
			return self.embed_h(h)
		return self.embed_t(t)
