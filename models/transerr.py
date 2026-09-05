"""TransERR scorer and model (``score_emb``)."""

import torch
import torch.nn.functional as F

from base.model import KGEScorer, KGEModel


def build_scorer(args) -> 'TransERRScorer':
	return TransERRScorer(args)


class TransERRScorer(KGEScorer):
	"""TransERR score function via a single ``score_emb``.

	Matches ``KnowledgeGraphEmbedding`` TransERR (Sun et al.) when
	``triple_relation_embedding`` is enabled: higher score is better,
	``gamma - ||h⊗wh + r - t⊗wt||_1`` with normalized quaternion relation parts.

	``combine`` modes: ``hrt``, ``hr_``, ``_rt``, ``hr_c``, ``_rt_c``.
	"""

	bidirectional_score_batch = True

	def __init__(self, args=None):
		super().__init__()
		self.args = args
		self.dim = int(getattr(args, 'dim', 0) or 0)
		margin_value = getattr(args, 'margin', None)
		if margin_value is None:
			margin_value = getattr(args, 'gamma', 12.0)
		self.gamma = float(margin_value)

	@staticmethod
	def _q_norm(relation: torch.Tensor) -> torch.Tensor:
		s_b, x_b, y_b, z_b = torch.chunk(relation, 4, dim=-1)
		denominator_b = torch.sqrt(s_b ** 2 + x_b ** 2 + y_b ** 2 + z_b ** 2)
		s_b = s_b / denominator_b
		x_b = x_b / denominator_b
		y_b = y_b / denominator_b
		z_b = z_b / denominator_b
		return torch.cat([s_b, x_b, y_b, z_b], dim=-1)

	@staticmethod
	def _calc(head: torch.Tensor, relation: torch.Tensor) -> torch.Tensor:
		s_a, x_a, y_a, z_a = torch.chunk(head, 4, dim=-1)
		s_b, x_b, y_b, z_b = torch.chunk(relation, 4, dim=-1)

		a = s_a * s_b - x_a * x_b - y_a * y_b - z_a * z_b
		b = s_a * x_b + s_b * x_a + y_a * z_b - y_b * z_a
		c = s_a * y_b + s_b * y_a + z_a * x_b - z_b * x_a
		d = s_a * z_b + s_b * z_a + x_a * y_b - x_b * y_a
		return torch.cat([a, b, c, d], dim=-1)

	def _relation_parts(self, r_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		if r_emb.size(-1) != self.dim * 3:
			raise ValueError(
				f'TransERR relation embeddings must have width {self.dim * 3}, got {r_emb.size(-1)}'
			)
		return torch.chunk(r_emb, 3, dim=-1)

	def _head_space(self, h_emb: torch.Tensor, r_emb: torch.Tensor) -> torch.Tensor:
		wh, _r_mid, _wt = self._relation_parts(r_emb)
		return self._calc(h_emb, self._q_norm(wh))

	def _tail_space(self, t_emb: torch.Tensor, r_emb: torch.Tensor) -> torch.Tensor:
		_wh, _r_mid, wt = self._relation_parts(r_emb)
		return self._calc(t_emb, self._q_norm(wt))

	def _compose_hr_query(self, h_emb: torch.Tensor, r_emb: torch.Tensor) -> torch.Tensor:
		_wh, r_mid, _wt = self._relation_parts(r_emb)
		return self._head_space(h_emb, r_emb) + r_mid

	def _compose_rt_query(self, r_emb: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
		_wh, r_mid, _wt = self._relation_parts(r_emb)
		return self._tail_space(t_emb, r_emb) - r_mid

	def _candidate_space(self, entity_embs: torch.Tensor, relation_part: torch.Tensor) -> torch.Tensor:
		s_a, x_a, y_a, z_a = torch.chunk(entity_embs, 4, dim=-1)
		s_b, x_b, y_b, z_b = torch.chunk(self._q_norm(relation_part), 4, dim=-1)

		s_a = s_a.unsqueeze(0)
		x_a = x_a.unsqueeze(0)
		y_a = y_a.unsqueeze(0)
		z_a = z_a.unsqueeze(0)
		s_b = s_b.unsqueeze(1)
		x_b = x_b.unsqueeze(1)
		y_b = y_b.unsqueeze(1)
		z_b = z_b.unsqueeze(1)

		a = s_a * s_b - x_a * x_b - y_a * y_b - z_a * z_b
		b = s_a * x_b + s_b * x_a + y_a * z_b - y_b * z_a
		c = s_a * y_b + s_b * y_a + z_a * x_b - z_b * x_a
		d = s_a * z_b + s_b * z_a + x_a * y_b - x_b * y_a
		return torch.cat([a, b, c, d], dim=-1)

	def _normalized_pair_score(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
		left = F.normalize(left, p=2, dim=-1)
		right = F.normalize(right, p=2, dim=-1)
		return torch.sum(left * right, dim=-1)

	def _normalized_1vsall_score(
		self,
		query: torch.Tensor,
		candidates: torch.Tensor,
	) -> torch.Tensor:
		query = F.normalize(query, p=2, dim=-1)
		candidates = F.normalize(candidates, p=2, dim=-1)
		return torch.sum(query.unsqueeze(1) * candidates, dim=-1)

	@staticmethod
	def _distance_1vsall_score(
		query: torch.Tensor,
		candidates: torch.Tensor,
		distance_degree: float,
	) -> torch.Tensor:
		return -torch.linalg.vector_norm(
			query.unsqueeze(1) - candidates,
			ord=distance_degree,
			dim=-1,
		)

	def _lp_distance_degree(self, kwargs: dict) -> float:
		degree = kwargs.pop('lp_distance_degree', None)
		if degree is None and self.args is not None:
			degree = getattr(self.args, 'lp_distance_degree', None)
		return float(degree if degree is not None else 2.0)

	def _align_for_candidates(
		self,
		h_emb: torch.Tensor,
		r_emb: torch.Tensor,
		t_emb: torch.Tensor,
	) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""Broadcast entity/relation tensors for optional candidate dimensions."""

		if h_emb.dim() == t_emb.dim():
			return h_emb, r_emb, t_emb

		if t_emb.dim() == h_emb.dim() + 1:
			return h_emb.unsqueeze(1), r_emb.unsqueeze(1), t_emb

		if h_emb.dim() == t_emb.dim() + 1:
			return h_emb, r_emb.unsqueeze(1), t_emb.unsqueeze(1)

		raise ValueError(
			f'Unsupported TransERR tensor ranks: h={h_emb.dim()}, r={r_emb.dim()}, t={t_emb.dim()}'
		)

	def _score_tensor(self, h_emb: torch.Tensor, r_emb: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
		if h_emb.dim() == 2 and r_emb.dim() == 2 and t_emb.dim() == 2:
			if h_emb.size(0) == r_emb.size(0) == t_emb.size(0):
				h_aligned, r_aligned, t_aligned = h_emb, r_emb, t_emb
			elif h_emb.size(0) == r_emb.size(0):
				batch_size = h_emb.size(0)
				num_candidates = t_emb.size(0)
				h_aligned = h_emb.unsqueeze(1).expand(batch_size, num_candidates, -1)
				r_aligned = r_emb.unsqueeze(1).expand(batch_size, num_candidates, -1)
				t_aligned = t_emb.unsqueeze(0).expand(batch_size, num_candidates, -1)
			elif t_emb.size(0) == r_emb.size(0):
				batch_size = r_emb.size(0)
				num_candidates = h_emb.size(0)
				h_aligned = h_emb.unsqueeze(0).expand(batch_size, num_candidates, -1)
				r_aligned = r_emb.unsqueeze(1).expand(batch_size, num_candidates, -1)
				t_aligned = t_emb.unsqueeze(1).expand(batch_size, num_candidates, -1)
			else:
				raise ValueError(
					f'Unsupported TransERR batch layout: h={tuple(h_emb.shape)}, '
					f'r={tuple(r_emb.shape)}, t={tuple(t_emb.shape)}'
				)
		else:
			h_aligned, r_aligned, t_aligned = self._align_for_candidates(h_emb, r_emb, t_emb)

		wh, r_mid, wt = self._relation_parts(r_aligned)
		wh = self._q_norm(wh)
		wt = self._q_norm(wt)
		diff = self._calc(h_aligned, wh) + r_mid - self._calc(t_aligned, wt)
		return self.gamma - torch.norm(diff, p=1, dim=-1)

	@staticmethod
	def embedding_regularization(model) -> torch.Tensor:
		"""Mean squared L2 norm over all entity and relation embeddings (official TransERR)."""

		ent_w = model.ent_embedder.embedding.weight
		rel_w = model.rel_embedder.embedding.weight
		ent_reg = torch.sum(ent_w ** 2, dim=-1)
		rel_reg = torch.sum(rel_w ** 2, dim=-1)
		return torch.cat([ent_reg, rel_reg]).mean()

	def _entity_chunk_size(self, batch_size: int) -> int:
		configured = int(getattr(self.args, 'eval_entity_chunk_size', 256) or 256)
		bytes_budget = int(getattr(self.args, 'eval_entity_chunk_bytes', 128 * 1024 * 1024) or 128 * 1024 * 1024)
		per_candidate = max(1, batch_size * self.dim * 4 * 4)
		memory_limit = max(1, bytes_budget // per_candidate)
		return max(1, min(configured, memory_limit))

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
		"""TransERR scores under ``combine``: ``hrt``, ``hr_``, ``_rt``, ``hr_c``, ``_rt_c``."""

		del kwargs
		n = r_emb.size(0)
		if combine == 'hrt':
			return self._score_tensor(h_emb, r_emb, t_emb).view(n, -1)
		if combine in ('hr_c', '_rt_c'):
			return self._score_tensor(h_emb, r_emb, t_emb)
		if combine == 'hr_':
			num_candidates = t_emb.size(0)
			batch_size = h_emb.size(0)
			chunk_size = self._entity_chunk_size(batch_size)
			scores = h_emb.new_empty(batch_size, num_candidates)
			for start in range(0, num_candidates, chunk_size):
				end = min(start + chunk_size, num_candidates)
				scores[:, start:end] = self._score_tensor(h_emb, r_emb, t_emb[start:end])
			return scores
		if combine == '_rt':
			num_candidates = h_emb.size(0)
			batch_size = t_emb.size(0)
			chunk_size = self._entity_chunk_size(batch_size)
			scores = t_emb.new_empty(batch_size, num_candidates)
			for start in range(0, num_candidates, chunk_size):
				end = min(start + chunk_size, num_candidates)
				scores[:, start:end] = self._score_tensor(h_emb[start:end], r_emb, t_emb)
			return scores
		raise ValueError(f'cannot handle combine="{combine}"')



	def normalized_score_hr(
		self,
		h_emb: torch.Tensor,
		r_emb: torch.Tensor,
		t_emb: torch.Tensor,
	) -> torch.Tensor:
		return self._normalized_pair_score(self._compose_hr_query(h_emb, r_emb), self._tail_space(t_emb, r_emb))

	def normalized_score_rt(
		self,
		h_emb: torch.Tensor,
		r_emb: torch.Tensor,
		t_emb: torch.Tensor,
	) -> torch.Tensor:
		return self._normalized_pair_score(self._head_space(h_emb, r_emb), self._compose_rt_query(r_emb, t_emb))

	def normalized_score_hr_(
		self,
		h_emb: torch.Tensor,
		r_emb: torch.Tensor,
		all_t_embs: torch.Tensor,
	) -> torch.Tensor:
		_wh, _r_mid, wt = self._relation_parts(r_emb)
		num_candidates = all_t_embs.size(0)
		batch_size = h_emb.size(0)
		chunk_size = self._entity_chunk_size(batch_size)
		query = self._compose_hr_query(h_emb, r_emb)
		scores = h_emb.new_empty(batch_size, num_candidates)
		for start in range(0, num_candidates, chunk_size):
			end = min(start + chunk_size, num_candidates)
			target = self._candidate_space(all_t_embs[start:end], wt)
			scores[:, start:end] = self._normalized_1vsall_score(query, target)
		return scores

	def normalized_score_rt_(
		self,
		all_h_embs: torch.Tensor,
		r_emb: torch.Tensor,
		t_emb: torch.Tensor,
	) -> torch.Tensor:
		wh, _r_mid, _wt = self._relation_parts(r_emb)
		num_candidates = all_h_embs.size(0)
		batch_size = t_emb.size(0)
		chunk_size = self._entity_chunk_size(batch_size)
		query = self._compose_rt_query(r_emb, t_emb)
		scores = t_emb.new_empty(batch_size, num_candidates)
		for start in range(0, num_candidates, chunk_size):
			end = min(start + chunk_size, num_candidates)
			target = self._candidate_space(all_h_embs[start:end], wh)
			scores[:, start:end] = self._normalized_1vsall_score(query, target)
		return scores

	def distance_score_hr_(
		self,
		h_emb: torch.Tensor,
		r_emb: torch.Tensor,
		all_t_embs: torch.Tensor,
		**kwargs,
	) -> torch.Tensor:
		"""Score tails by Lp distance in TransERR's relation-transformed space."""

		distance_degree = self._lp_distance_degree(kwargs)
		_wh, _r_mid, wt = self._relation_parts(r_emb)
		num_candidates = all_t_embs.size(0)
		batch_size = h_emb.size(0)
		chunk_size = self._entity_chunk_size(batch_size)
		
		query = self._compose_hr_query(h_emb, r_emb)
		scores = h_emb.new_empty(batch_size, num_candidates)
		for start in range(0, num_candidates, chunk_size):
			end = min(start + chunk_size, num_candidates)
			target = self._candidate_space(all_t_embs[start:end], wt)
			scores[:, start:end] = self._distance_1vsall_score(
				query,
				target,
				distance_degree,
			)
		return scores

	def distance_score_rt_(
		self,
		all_h_embs: torch.Tensor,
		r_emb: torch.Tensor,
		t_emb: torch.Tensor,
		**kwargs,
	) -> torch.Tensor:
		"""Score heads by Lp distance in TransERR's relation-transformed space."""

		distance_degree = self._lp_distance_degree(kwargs)
		wh, _r_mid, _wt = self._relation_parts(r_emb)
		num_candidates = all_h_embs.size(0)
		batch_size = t_emb.size(0)
		chunk_size = self._entity_chunk_size(batch_size)
		query = self._compose_rt_query(r_emb, t_emb)
		scores = t_emb.new_empty(batch_size, num_candidates)
		for start in range(0, num_candidates, chunk_size):
			end = min(start + chunk_size, num_candidates)
			target = self._candidate_space(all_h_embs[start:end], wh)
			scores[:, start:end] = self._distance_1vsall_score(
				query,
				target,
				distance_degree,
			)
		return scores


class TransERRModel(KGEModel):
	"""Bind lookup embedders to ``TransERRScorer`` (``scorers`` length 1 by default)."""

	target_uses_relation = True

	def __init__(
		self,
		ent_embedder,
		rel_embedder,
		scorers=None,
		args=None,
		aux_embedders=None,
	):
		if scorers is None:
			scorers = [TransERRScorer(args)]
		super().__init__(
			ent_embedder,
			rel_embedder,
			scorers=scorers,
			args=args,
			aux_embedders=aux_embedders,
		)

	target_uses_relation = True

	def query_encoder(self, h: torch.Tensor, r: torch.Tensor, **kwargs) -> torch.Tensor:
		del kwargs
		scorer = self.get_scorer()
		return scorer._compose_hr_query(self.embed_h(h), self.embed_r(r))

	def inverse_query_encoder(self, r: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
		del kwargs
		scorer = self.get_scorer()
		return scorer._compose_rt_query(self.embed_r(r), self.embed_t(t))

	def target_encoder(
		self,
		h: torch.Tensor,
		r: torch.Tensor,
		t: torch.Tensor,
		*,
		predict_head: bool = False,
		**kwargs,
	) -> torch.Tensor:
		del kwargs
		scorer = self.get_scorer()
		r_emb = self.embed_r(r)
		if predict_head:
			return scorer._head_space(self.embed_h(h), r_emb)
		return scorer._tail_space(self.embed_t(t), r_emb)

	def uniformity_head_encoder(self, h: torch.Tensor, r: torch.Tensor, **kwargs) -> torch.Tensor:
		del kwargs
		return self.get_scorer()._head_space(self.embed_h(h), self.embed_r(r))
