"""DaBR scorer and model (``score_emb``)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from base.model import KGEScorer, KGEModel


def build_scorer(args) -> 'DaBRScorer':
	return DaBRScorer(args)


class DaBRScorer(KGEScorer):
	"""DaBR score function via a single ``score_emb``.

	Quaternion feature ops use the last dimension (``dim=-1`` / ``size(-1)``) so the
	same kernels work for rank-2 ``[B, D]`` batches and rank-3 ``[B, C, D]`` broadcasts.
	That is equivalent to the original DaBR ``dim=1`` / ``size(1)`` convention on 2D tensors.

	``combine`` modes: ``hrt``, ``hr_``, ``_rt``, ``hr_c``, ``_rt_c``.
	Pass ``dr_emb`` / ``para`` via ``**kwargs``.
	"""

	bidirectional_score_batch = True
	kgau_alignment_mode = 'dabr_blocks'

	def __init__(self, args=None):
		super().__init__()
		self.args = args
		para_init = getattr(args, 'para', None)
		if para_init is None:
			para_init = getattr(args, 'lmbda', 0.1)
		self.para = nn.Parameter(torch.tensor([float(para_init)]))
		norm_p = int(getattr(args, 'dabr_distance_norm', 1) or 1)
		if norm_p not in (1, 2):
			raise ValueError(f'dabr_distance_norm must be 1 or 2, got {norm_p}')
		self.distance_norm = norm_p

	# ------------------------------------------------------------------
	# Core DaBR quaternion ops (llqy123/DaBR ``models/DaBR.py``)
	# ------------------------------------------------------------------

	@classmethod
	def _distance_score(
		self,
		h_emb: torch.Tensor,
		dr_emb: torch.Tensor,
		t_emb: torch.Tensor,
		norm_p: int = 1,
	) -> torch.Tensor:
		"""Geometric distance term from DaBR ``_calc``.

		``norm_p=1`` matches the paper's L1; ``norm_p=2`` uses L2 on the same
		quaternion-summed ``score_d`` vector.
		"""

		hrt = h_emb + dr_emb - t_emb
		s_d, x_d, y_d, z_d = torch.chunk(hrt, 4, dim=-1)
		return torch.norm(s_d + x_d + y_d + z_d, p=norm_p, dim=-1)

	@staticmethod
	def _normalize_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
		"""Unit-normalize each quaternion slot. DaBR: ``normalization``.

		Preserves leading batch dims (``[..., 4s] → [..., 4s]``). The original
		implementation flattened to 2D via ``reshape(-1, 4, size)``; that breaks
		``[B, C, D]`` broadcasting used by 1-vs-all scoring.
		"""

		size = quaternion.size(-1) // 4
		leading = quaternion.shape[:-1]

		reshaped = quaternion.reshape(*leading, 4, size)
		norm = torch.sqrt(torch.sum(reshaped ** 2, dim=-2, keepdim=True).clamp_min(1e-12))
		return (reshaped / norm).reshape(*leading, 4 * size)

	@staticmethod
	def _wise_quaternion(
		quaternion: torch.Tensor,
	) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
		"""Build the four Hamilton-product views. DaBR: ``make_wise_quaternion``."""

		if quaternion.dim() == 1:
			quaternion = quaternion.unsqueeze(0)
		size = quaternion.size(-1) // 4
		r, i, j, k = torch.split(quaternion, size, dim=-1)

		r2 = torch.cat([r, -i, -j, -k], dim=-1)
		i2 = torch.cat([i, r, -k, j], dim=-1)
		j2 = torch.cat([j, k, r, -i], dim=-1)
		k2 = torch.cat([k, -j, i, r], dim=-1)
		return r2, i2, j2, k2

	@staticmethod
	def _quaternion_wise_mul(quaternion: torch.Tensor) -> torch.Tensor:
		"""Sum the four component blocks after element-wise products.

		DaBR: ``get_quaternion_wise_mul``. Uses ``view(*shape[:-1], 4, size)`` and
		``sum(dim=-2)`` so leading dims are preserved (origin: ``view(-1, 4, size)``,
		``sum(..., 1)``).
		"""

		size = quaternion.size(-1) // 4
		reshaped = quaternion.view(*quaternion.shape[:-1], 4, size)
		return torch.sum(reshaped, dim=-2)

	@classmethod
	def _vec_vec_wise_multiplication(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
		"""Quaternion multiply ``left ⊗ right`` with unit-normalized ``right``.

		DaBR: ``vec_vec_wise_multiplication`` (always normalizes the right operand).
		"""

		normalized_right = self._normalize_quaternion(right)
		l_r, l_i, l_j, l_k = self._wise_quaternion(left)

		qp_r = self._quaternion_wise_mul(l_r * normalized_right)
		qp_i = self._quaternion_wise_mul(l_i * normalized_right)
		qp_j = self._quaternion_wise_mul(l_j * normalized_right)
		qp_k = self._quaternion_wise_mul(l_k * normalized_right)
		return torch.cat([qp_r, qp_i, qp_j, qp_k], dim=-1)

	@staticmethod
	def _quat_inv(embeddings: torch.Tensor) -> torch.Tensor:
		"""Multiplicative inverse ``q⁻¹ = conjugate(q) / |q|²``. DaBR: ``get_inv``."""

		r, i, j, k = torch.chunk(embeddings, 4, dim=-1)
		norm = (r ** 2 + i ** 2 + j ** 2 + k ** 2).clamp_min(1e-12)
		return torch.cat([r / norm, -i / norm, -j / norm, -k / norm], dim=-1)

	@classmethod
	def _semantic_score(
		self,
		h_emb: torch.Tensor,
		r_emb: torch.Tensor,
		t_emb: torch.Tensor,
	) -> torch.Tensor:
		"""Semantic matching term ``⟨h⊗r, t⊗r⁻¹⟩``. DaBR: semantic part of ``_calc``.

		``r⁻¹`` is passed raw; unit normalization happens inside
		``_vec_vec_wise_multiplication`` (same as the original).
		"""

		hr = self._vec_vec_wise_multiplication(h_emb, r_emb)
		tr = self._vec_vec_wise_multiplication(t_emb, self._quat_inv(r_emb))
		return torch.sum(hr * tr, dim=-1)

	@staticmethod
	def regularization(quaternion: torch.Tensor) -> torch.Tensor:
		"""Mean squared magnitude of the four quaternion components. DaBR: ``regularization``."""

		size = quaternion.size(-1) // 4
		r, i, j, k = torch.split(quaternion, size, dim=-1)
		return torch.mean(r ** 2) + torch.mean(i ** 2) + torch.mean(j ** 2) + torch.mean(k ** 2)

	@staticmethod
	def _coalesce_para(
		para: float | torch.Tensor | None,
		default: float | torch.Tensor,
	) -> float | torch.Tensor:
		"""Resolve optional ``para`` override; keep tensor defaults for autograd."""

		if para is None:
			return default
		if torch.is_tensor(para):
			return para
		return float(para)

	def supports_candidate_scoring(self) -> bool:
		return True

	def score_emb(
		self,
		h_emb: torch.Tensor,
		r_emb: torch.Tensor,
		t_emb: torch.Tensor,
		combine: str,
		dr_emb: torch.Tensor | None = None,
		para: float | torch.Tensor | None = None,
		**kwargs,
	) -> torch.Tensor:
		"""DaBR score under ``combine``.

		Default: ``⟨h⊗r, t⊗r⁻¹⟩ + λ‖(h+dr−t)_Σ‖``.
		With ``dabr_au_semantic_only``: semantic quaternion branch only.
		With ``dabr_au_distance_only``: TransE-style ``-‖h+dr−t‖`` (higher is better).
		"""

		del kwargs
		n = r_emb.size(0)
		mode = self._au_component_mode()
		para_value = self._coalesce_para(para, self.para)

		if combine == 'hrt':
			if mode == 'distance':
				if dr_emb is None:
					dr_emb = torch.zeros_like(h_emb)
				return (-self._distance_score(h_emb, dr_emb, t_emb, self.distance_norm)).view(n, -1)
			score_s = self._semantic_score(h_emb, r_emb, t_emb)
			if mode == 'semantic':
				return score_s.view(n, -1)
			if dr_emb is None:
				dr_emb = torch.zeros_like(h_emb)
			score_d = self._distance_score(h_emb, dr_emb, t_emb, self.distance_norm)
			return (score_s + para_value * score_d).view(n, -1)

		if combine == 'hr_c':
			# t_emb is [B, C, D]
			if mode == 'distance':
				if dr_emb is None:
					dr_emb = torch.zeros_like(h_emb)
				return -self._distance_score(
					h_emb.unsqueeze(1),
					dr_emb.unsqueeze(1),
					t_emb,
					self.distance_norm,
				)
			hr = self._vec_vec_wise_multiplication(h_emb, r_emb).unsqueeze(1)
			tr = self._vec_vec_wise_multiplication(
				t_emb,
				self._quat_inv(r_emb).unsqueeze(1),
			)
			score_s = torch.sum(hr * tr, dim=-1)
			if mode == 'semantic':
				return score_s
			if dr_emb is None:
				dr_emb = torch.zeros_like(h_emb)
			score_d = self._distance_score(
				h_emb.unsqueeze(1),
				dr_emb.unsqueeze(1),
				t_emb,
				self.distance_norm,
			)
			return score_s + para_value * score_d

		if combine == '_rt_c':
			# h_emb is [B, C, D]
			num_heads = h_emb.size(1)
			if mode == 'distance':
				if dr_emb is None:
					dr_emb = torch.zeros_like(t_emb)
				return -self._distance_score(
					h_emb,
					dr_emb.unsqueeze(1).expand(-1, num_heads, -1),
					t_emb.unsqueeze(1).expand(-1, num_heads, -1),
					self.distance_norm,
				)
			hr = self._vec_vec_wise_multiplication(h_emb, r_emb.unsqueeze(1))
			tr = self._vec_vec_wise_multiplication(
				t_emb.unsqueeze(1),
				self._quat_inv(r_emb).unsqueeze(1),
			)
			score_s = torch.sum(hr * tr, dim=-1)
			if mode == 'semantic':
				return score_s
			if dr_emb is None:
				dr_emb = torch.zeros_like(t_emb)
			score_d = self._distance_score(
				h_emb,
				dr_emb.unsqueeze(1).expand(-1, num_heads, -1),
				t_emb.unsqueeze(1).expand(-1, num_heads, -1),
				self.distance_norm,
			)
			return score_s + para_value * score_d

		if combine == 'hr_':
			if dr_emb is None:
				dr_emb = torch.zeros_like(h_emb)
			num_candidates = t_emb.size(0)
			batch_size = h_emb.size(0)
			embed_dim = t_emb.size(-1)
			chunk_size = self._entity_chunk_size(batch_size, embed_dim)
			if num_candidates <= chunk_size:
				return self._score_hr_candidate_chunk(h_emb, r_emb, t_emb, dr_emb, para_value)
			scores = h_emb.new_empty(batch_size, num_candidates)
			for start in range(0, num_candidates, chunk_size):
				end = min(start + chunk_size, num_candidates)
				scores[:, start:end] = self._score_hr_candidate_chunk(
					h_emb, r_emb, t_emb[start:end], dr_emb, para_value,
				)
			return scores

		if combine == '_rt':
			if dr_emb is None:
				dr_emb = torch.zeros_like(t_emb)
			num_heads = h_emb.size(0)
			batch_size = r_emb.size(0)
			embed_dim = h_emb.size(-1)
			chunk_size = self._entity_chunk_size(batch_size, embed_dim)
			if num_heads <= chunk_size:
				return self._score_rt_candidate_chunk(h_emb, r_emb, t_emb, dr_emb, para_value)
			scores = r_emb.new_empty(batch_size, num_heads)
			for start in range(0, num_heads, chunk_size):
				end = min(start + chunk_size, num_heads)
				scores[:, start:end] = self._score_rt_candidate_chunk(
					h_emb[start:end], r_emb, t_emb, dr_emb, para_value,
				)
			return scores

		raise ValueError(f'cannot handle combine="{combine}"')

	# ------------------------------------------------------------------
	# AU / DirectAU block vectors
	# ------------------------------------------------------------------

	def _coalesce_dr(
		self,
		h_emb: torch.Tensor,
		dr_emb: torch.Tensor | None,
	) -> torch.Tensor:
		"""Return ``dr_emb`` or zeros shaped like ``h_emb``."""

		if dr_emb is None:
			return torch.zeros_like(h_emb)
		return dr_emb

	def _au_head_vector(
		self,
		h_emb: torch.Tensor,
		r_emb: torch.Tensor,
		dr_emb: torch.Tensor | None = None,
	) -> torch.Tensor:
		"""Head-side AU vector ``cat(h⊗r, h+dr)``."""

		dr_emb = self._coalesce_dr(h_emb, dr_emb)
		q_mult = self._vec_vec_wise_multiplication(h_emb, r_emb)
		return torch.cat([q_mult, h_emb + dr_emb], dim=-1)

	def _au_tail_vector(
		self,
		t_emb: torch.Tensor,
		r_emb: torch.Tensor,
	) -> torch.Tensor:
		"""Tail-side AU vector ``cat(t⊗r⁻¹, t)`` (matches ``⟨h⊗r, t⊗r⁻¹⟩``)."""

		t_mult = self._vec_vec_wise_multiplication(t_emb, self._quat_inv(r_emb))
		return torch.cat([t_mult, t_emb], dim=-1)



	def au_entity_embeddings(self, entity_emb: torch.Tensor) -> torch.Tensor:
		"""Map entity rows into the LP / uniformity vector space.

		Distance-only (TransE-style) keeps raw entities. Hybrid / semantic AU widens
		via ``cat(e, e)`` for two-block concat vectors.
		"""

		if self._distance_only():
			return entity_emb
		return torch.cat([entity_emb, entity_emb], dim=-1)

	# ------------------------------------------------------------------
	# Native 1-vs-all DaBR scoring
	# ------------------------------------------------------------------

	def _entity_chunk_size(self, batch_size: int, embed_dim: int) -> int:
		"""Candidate chunk size for 1-vs-all scoring (peak GPU memory control)."""

		configured = int(getattr(self.args, 'eval_entity_chunk_size', 256) or 256)
		bytes_budget = int(
			getattr(self.args, 'eval_entity_chunk_bytes', 128 * 1024 * 1024) or 128 * 1024 * 1024
		)
		# Several [B, C, D] quaternion intermediates are materialized per chunk.
		per_candidate = max(1, batch_size * embed_dim * 4 * 12)
		memory_limit = max(1, bytes_budget // per_candidate)
		return max(1, min(configured, memory_limit))

	def _score_hr_candidate_chunk(
		self,
		h_emb: torch.Tensor,
		r_emb: torch.Tensor,
		t_emb_chunk: torch.Tensor,
		dr_emb: torch.Tensor,
		para_value: float | torch.Tensor,
	) -> torch.Tensor:
		"""1-vs-all tail scores for one entity chunk (broadcast over candidates)."""

		hr = self._vec_vec_wise_multiplication(h_emb, r_emb).unsqueeze(1)
		tr = self._vec_vec_wise_multiplication(
			t_emb_chunk.unsqueeze(0),
			self._quat_inv(r_emb).unsqueeze(1)
		)

		score_s = torch.sum(hr * tr, dim=-1)
		mode = self._au_component_mode()
		if mode == 'semantic':
			return score_s
		score_d = self._distance_score(
			h_emb.unsqueeze(1),
			dr_emb.unsqueeze(1),
			t_emb_chunk.unsqueeze(0),
			self.distance_norm
		)
		if mode == 'distance':
			return -score_d
		return score_s + para_value * score_d

	def _score_rt_candidate_chunk(
		self,
		h_emb_chunk: torch.Tensor,
		r_emb: torch.Tensor,
		t_emb: torch.Tensor,
		dr_emb: torch.Tensor,
		para_value: float | torch.Tensor,
	) -> torch.Tensor:
		"""1-vs-all head scores for one entity chunk (broadcast over candidates)."""

		num_heads = h_emb_chunk.size(0)
		hr = self._vec_vec_wise_multiplication(
			h_emb_chunk.unsqueeze(0),
			r_emb.unsqueeze(1),
		)
		tr = self._vec_vec_wise_multiplication(
			t_emb.unsqueeze(1),
			self._quat_inv(r_emb).unsqueeze(1),
		)

		score_s = torch.sum(hr * tr, dim=-1)
		mode = self._au_component_mode()
		if mode == 'semantic':
			return score_s
		score_d = self._distance_score(
			h_emb_chunk.unsqueeze(0).expand(r_emb.size(0), num_heads, -1),
			dr_emb.unsqueeze(1).expand(-1, num_heads, -1),
			t_emb.unsqueeze(1).expand(-1, num_heads, -1),
			self.distance_norm,
		)
		if mode == 'distance':
			return -score_d
		return score_s + para_value * score_d

	# ------------------------------------------------------------------
	# Normalized (cosine) and distance LP over AU blocks
	# ------------------------------------------------------------------

	@staticmethod
	def _split_au_blocks(vectors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		"""Split ``cat(semantic, additive)`` AU vectors at the midpoint."""

		mid = vectors.size(-1) // 2
		return vectors[..., :mid], vectors[..., mid:]

	def _lp_combine_weight(self) -> torch.Tensor:
		"""Learnable λ (``para``) fusing the additive branch, mirroring DaBR ``φ = s + λ d``.

		Cosine LP must reuse the same λ as training (``L = L_sem + λ L_dist``) so the
		ranked geometry matches the objective instead of weighting both blocks equally.
		"""

		return self.para.reshape(())

	def _semantic_only(self) -> bool:
		"""DaBR-AU semantic-only mode: rank on the quaternion branch alone (no distance)."""

		return self._au_component_mode() == 'semantic'

	def _distance_only(self) -> bool:
		"""DaBR-AU distance-only mode: TransE-style AU on ``h+dr ↔ t`` (no semantic)."""

		return self._au_component_mode() == 'distance'

	def _au_component_mode(self) -> str:
		"""Resolve DaBR-AU component mode: ``semantic``, ``distance``, or ``both``."""

		semantic_only = bool(getattr(self.args, 'dabr_au_semantic_only', False))
		distance_only = bool(getattr(self.args, 'dabr_au_distance_only', False))
		if semantic_only and distance_only:
			raise ValueError(
				'dabr_au_semantic_only and dabr_au_distance_only are mutually exclusive',
			)
		if semantic_only:
			return 'semantic'
		if distance_only:
			return 'distance'
		return 'both'

	@classmethod
	def _normalized_pair_score(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
		"""Cosine similarity along the last dimension."""

		left = F.normalize(left, p=2, dim=-1)
		right = F.normalize(right, p=2, dim=-1)
		return torch.sum(left * right, dim=-1)

	def _normalized_block_pair_score(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
		"""DaBR-AU cosine LP over AU blocks.

		- Semantic-only: cosine on the quaternion branch.
		- Distance-only: TransE-style cosine on ``h+dr`` / ``t``.
		- Hybrid: original semantic dot + λ·cosine(additive).
		"""

		l_sem, l_add = self._split_au_blocks(left)
		r_sem, r_add = self._split_au_blocks(right)
		mode = self._au_component_mode()
		if mode == 'semantic':
			return self._normalized_pair_score(l_sem, r_sem)
		if mode == 'distance':
			return self._normalized_pair_score(l_add, r_add)
		semantic = torch.sum(l_sem * r_sem, dim=-1)
		lam = self._lp_combine_weight()
		return semantic + lam * self._normalized_pair_score(l_add, r_add)

	def normalized_score_hr(
		self,
		h_emb: torch.Tensor,
		r_emb: torch.Tensor,
		t_emb: torch.Tensor,
		dr_emb: torch.Tensor | None = None,
		**kwargs,
	) -> torch.Tensor:
		"""1-to-1 cosine score for tail prediction."""

		del kwargs
		dr_emb = self._coalesce_dr(h_emb, dr_emb)
		if self._distance_only():
			return self._normalized_pair_score(h_emb + dr_emb, t_emb)
		return self._normalized_block_pair_score(
			self._au_head_vector(h_emb, r_emb, dr_emb),
			self._au_tail_vector(t_emb, r_emb),
		)

	def normalized_score_rt(
		self,
		h_emb: torch.Tensor,
		r_emb: torch.Tensor,
		t_emb: torch.Tensor,
		dr_emb: torch.Tensor | None = None,
		**kwargs,
	) -> torch.Tensor:
		"""1-to-1 cosine score for head prediction."""

		del kwargs
		dr_emb = self._coalesce_dr(h_emb, dr_emb)
		if self._distance_only():
			return self._normalized_pair_score(t_emb - dr_emb, h_emb)
		return self._normalized_block_pair_score(
			self._au_head_vector(h_emb, r_emb, dr_emb),
			self._au_tail_vector(t_emb, r_emb),
		)

	def _raw_from_entity_au(self, entity_au: torch.Tensor) -> torch.Tensor:
		"""Recover raw entity rows from LP entity table vectors.

		Hybrid/semantic widen via ``cat(e, e)``; distance-only keeps raw ``e``.
		"""

		if self._distance_only():
			return entity_au
		return entity_au[..., : entity_au.size(-1) // 2]

	def _group_batch_by_relation(
		self,
		r_emb: torch.Tensor,
		*tensors: torch.Tensor,
	) -> list[tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]]:
		"""Group batch rows that share the same relation embedding."""

		batch_size = r_emb.size(0)
		if batch_size <= 1:
			return [(torch.arange(batch_size, device=r_emb.device), r_emb, list(tensors))]

		unique_r, inverse = torch.unique(r_emb, dim=0, return_inverse=True)
		groups: list[tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]] = []
		for rel_idx in range(unique_r.size(0)):
			row_mask = inverse == rel_idx
			row_indices = row_mask.nonzero(as_tuple=True)[0]
			grouped = [tensor[row_mask] for tensor in tensors]
			groups.append((row_indices, unique_r[rel_idx:rel_idx + 1], grouped))
		return groups

	def _au_head_vectors_batch(
		self,
		entity_emb: torch.Tensor,
		r_emb: torch.Tensor,
		dr_emb: torch.Tensor | None = None,
	) -> torch.Tensor:
		"""Relation-aware head AU candidates ``[B, C, 2D]`` for 1-vs-all LP."""

		num_ent = entity_emb.size(0)
		if r_emb.dim() == 1:
			r_emb = r_emb.unsqueeze(0)
		batch_size = r_emb.size(0)
		ent_exp = entity_emb.unsqueeze(0).expand(batch_size, num_ent, -1)
		dr_emb = self._coalesce_dr(ent_exp[:, 0], dr_emb)
		r_exp = r_emb.unsqueeze(1).expand(batch_size, num_ent, -1)
		dr_exp = dr_emb.unsqueeze(1).expand(batch_size, num_ent, -1)
		flat_ent = ent_exp.reshape(batch_size * num_ent, -1)
		flat_r = r_exp.reshape(batch_size * num_ent, -1)
		q_mult = self._vec_vec_wise_multiplication(flat_ent, flat_r).view(batch_size, num_ent, -1)
		return torch.cat([q_mult, ent_exp + dr_exp], dim=-1)

	def _au_tail_vectors_batch(
		self,
		entity_emb: torch.Tensor,
		r_emb: torch.Tensor,
	) -> torch.Tensor:
		"""Relation-aware tail AU candidates ``[B, C, 2D]``: ``cat(t⊗r⁻¹, t)``."""

		num_ent = entity_emb.size(0)
		if r_emb.dim() == 1:
			r_emb = r_emb.unsqueeze(0)
		batch_size = r_emb.size(0)
		ent_exp = entity_emb.unsqueeze(0).expand(batch_size, num_ent, -1)
		r_inv_exp = self._quat_inv(r_emb).unsqueeze(1).expand(batch_size, num_ent, -1)
		flat_ent = ent_exp.reshape(batch_size * num_ent, -1)
		flat_r_inv = r_inv_exp.reshape(batch_size * num_ent, -1)
		t_mult = self._vec_vec_wise_multiplication(flat_ent, flat_r_inv).view(batch_size, num_ent, -1)
		return torch.cat([t_mult, ent_exp], dim=-1)

	@classmethod
	def _distance_1vsall_score(
		self,
		query: torch.Tensor,
		candidates: torch.Tensor,
		degree: float,
		*,
		mode: str = 'both',
	) -> torch.Tensor:
		"""Negative block-wise Lp distance (higher is better)."""

		q_sem, q_add = self._split_au_blocks(query)
		c_sem, c_add = self._split_au_blocks(candidates)
		if mode == 'semantic':
			return -torch.norm(q_sem.unsqueeze(1) - c_sem, p=degree, dim=-1)
		if mode == 'distance':
			return -torch.norm(q_add.unsqueeze(1) - c_add, p=degree, dim=-1)
		dist_sem = torch.norm(q_sem.unsqueeze(1) - c_sem, p=degree, dim=-1)
		dist_add = torch.norm(q_add.unsqueeze(1) - c_add, p=degree, dim=-1)
		return -(dist_sem + dist_add)

	def _normalized_1vsall_score(self, query: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
		"""1-vs-all DaBR-AU cosine LP over AU blocks.

		Semantic-only / distance-only use cosine on one branch; hybrid uses original
		semantic dot + λ·cosine(additive).
		"""

		q_sem, q_add = self._split_au_blocks(query)
		c_sem, c_add = self._split_au_blocks(candidates)
		mode = self._au_component_mode()
		if mode == 'semantic':
			q_sem = F.normalize(q_sem, p=2, dim=-1)
			c_sem = F.normalize(c_sem, p=2, dim=-1)
			return (q_sem.unsqueeze(1) * c_sem).sum(dim=-1)
		if mode == 'distance':
			q_add = F.normalize(q_add, p=2, dim=-1)
			c_add = F.normalize(c_add, p=2, dim=-1)
			return (q_add.unsqueeze(1) * c_add).sum(dim=-1)
		semantic = (q_sem.unsqueeze(1) * c_sem).sum(dim=-1)
		q_add = F.normalize(q_add, p=2, dim=-1)
		c_add = F.normalize(c_add, p=2, dim=-1)
		lam = self._lp_combine_weight()
		return semantic + lam * (q_add.unsqueeze(1) * c_add).sum(dim=-1)

	def _au_hr_scores_chunked(
		self,
		query: torch.Tensor,
		all_entity_embs: torch.Tensor,
		r_emb: torch.Tensor,
		predict_head: bool,
		dr_emb: torch.Tensor | None = None,
		use_distance: bool = False,
		distance_degree: float = 2.0,
	) -> torch.Tensor:
		"""Chunked 1-vs-all AU scores with relation-aware candidates."""

		num_candidates = all_entity_embs.size(0)
		batch_size = query.size(0)
		embed_dim = all_entity_embs.size(-1)
		chunk_size = self._entity_chunk_size(max(batch_size, 1), embed_dim * 2)

		scores = query.new_empty(batch_size, num_candidates)
		for start in range(0, num_candidates, chunk_size):
			end = min(start + chunk_size, num_candidates)
			entity_chunk = all_entity_embs[start:end]

			if predict_head:
				targets = self._au_head_vectors_batch(entity_chunk, r_emb, dr_emb)
			else:
				targets = self._au_tail_vectors_batch(entity_chunk, r_emb)

			if targets.size(0) == 1 and batch_size > 1:
				targets = targets.expand(batch_size, -1, -1)

			if use_distance:
				scores[:, start:end] = self._distance_1vsall_score(
					query, targets, degree=distance_degree, mode=self._au_component_mode(),
				)
			else:
				scores[:, start:end] = self._normalized_1vsall_score(query, targets)
		return scores

	def normalized_score_hr_(
		self,
		h_emb: torch.Tensor,
		r_emb: torch.Tensor,
		all_t_embs: torch.Tensor,
		dr_emb: torch.Tensor | None = None,
		**kwargs,
	) -> torch.Tensor:
		"""1-vs-all cosine tail scores.

		Distance-only: TransE-style ``cos(h+dr, e)`` against the raw entity table.
		"""

		del kwargs
		all_t_embs = self._raw_from_entity_au(all_t_embs)
		dr_emb = self._coalesce_dr(h_emb, dr_emb)
		if self._distance_only():
			query = F.normalize(h_emb + dr_emb, p=2, dim=-1)
			cand = F.normalize(all_t_embs, p=2, dim=-1)
			return torch.mm(query, cand.t())

		scores = h_emb.new_empty(h_emb.size(0), all_t_embs.size(0))
		for row_indices, r_row, (h_sub, dr_sub) in self._group_batch_by_relation(r_emb, h_emb, dr_emb):
			r_sub = r_row.expand(h_sub.size(0), -1)
			query = self._au_head_vector(h_sub, r_sub, dr_sub)
			group_scores = self._au_hr_scores_chunked(
				query, all_t_embs, r_row, predict_head=False,
			)
			scores.index_copy_(0, row_indices, group_scores)
		return scores

	def normalized_score_rt_(
		self,
		all_h_embs: torch.Tensor,
		r_emb: torch.Tensor,
		t_emb: torch.Tensor,
		dr_emb: torch.Tensor | None = None,
		**kwargs,
	) -> torch.Tensor:
		"""1-vs-all cosine head scores.

		Distance-only: TransE-style ``cos(t-dr, e)`` against the raw entity table.
		"""

		del kwargs
		all_h_embs = self._raw_from_entity_au(all_h_embs)
		dr_emb = self._coalesce_dr(t_emb, dr_emb)
		if self._distance_only():
			query = F.normalize(t_emb - dr_emb, p=2, dim=-1)
			cand = F.normalize(all_h_embs, p=2, dim=-1)
			return torch.mm(query, cand.t())

		scores = t_emb.new_empty(t_emb.size(0), all_h_embs.size(0))
		for row_indices, r_row, (t_sub, dr_sub) in self._group_batch_by_relation(r_emb, t_emb, dr_emb):
			r_sub = r_row.expand(t_sub.size(0), -1)
			query = self._au_tail_vector(t_sub, r_sub)
			group_scores = self._au_hr_scores_chunked(
				query,
				all_h_embs,
				r_row,
				predict_head=True,
				dr_emb=dr_sub[:1] if dr_sub is not None else None,
			)
			scores.index_copy_(0, row_indices, group_scores)
		return scores

	def _coalesce_lp_distance_degree(self, kwargs: dict) -> float:
		"""Resolve Lp degree for distance-based AU link prediction."""

		degree = kwargs.pop('lp_distance_degree', None)
		if degree is None and self.args is not None:
			degree = getattr(self.args, 'lp_distance_degree', None)
		return float(degree if degree is not None else 2.0)

	def distance_score_hr_(
		self,
		h_emb: torch.Tensor,
		r_emb: torch.Tensor,
		all_t_embs: torch.Tensor,
		dr_emb: torch.Tensor | None = None,
		**kwargs,
	) -> torch.Tensor:
		"""1-vs-all negative block-Lp tail scores."""

		distance_degree = self._coalesce_lp_distance_degree(kwargs)
		dr_emb = self._coalesce_dr(h_emb, dr_emb)
		all_t_embs = self._raw_from_entity_au(all_t_embs)
		batch_size = h_emb.size(0)
		num_candidates = all_t_embs.size(0)

		scores = h_emb.new_empty(batch_size, num_candidates)
		for row_indices, r_row, (h_sub, dr_sub) in self._group_batch_by_relation(r_emb, h_emb, dr_emb):
			r_sub = r_row.expand(h_sub.size(0), -1)
			query = self._au_head_vector(h_sub, r_sub, dr_sub)
			group_scores = self._au_hr_scores_chunked(
				query,
				all_t_embs,
				r_row,
				predict_head=False,
				use_distance=True,
				distance_degree=distance_degree,
			)
			scores.index_copy_(0, row_indices, group_scores)
		return scores

	def distance_score_rt_(
		self,
		all_h_embs: torch.Tensor,
		r_emb: torch.Tensor,
		t_emb: torch.Tensor,
		dr_emb: torch.Tensor | None = None,
		**kwargs,
	) -> torch.Tensor:
		"""1-vs-all negative block-Lp head scores."""

		distance_degree = self._coalesce_lp_distance_degree(kwargs)
		dr_emb = self._coalesce_dr(t_emb, dr_emb)
		all_h_embs = self._raw_from_entity_au(all_h_embs)
		batch_size = t_emb.size(0)
		num_candidates = all_h_embs.size(0)

		scores = t_emb.new_empty(batch_size, num_candidates)
		for row_indices, r_row, (t_sub, dr_sub) in self._group_batch_by_relation(r_emb, t_emb, dr_emb):
			r_sub = r_row.expand(t_sub.size(0), -1)
			query = self._au_tail_vector(t_sub, r_sub)
			group_scores = self._au_hr_scores_chunked(
				query,
				all_h_embs,
				r_row,
				predict_head=True,
				dr_emb=dr_sub[:1] if dr_sub is not None else None,
				use_distance=True,
				distance_degree=distance_degree,
			)
			scores.index_copy_(0, row_indices, group_scores)
		return scores


class DaBRSemanticScorer(KGEScorer):
	"""Semantic matching component ``⟨h⊗r, t⊗r⁻¹⟩`` (hybrid DaBR building block)."""

	bidirectional_score_batch = True
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
		del kwargs
		if combine != 'hrt':
			raise ValueError(f'DaBRSemanticScorer only supports combine="hrt", got "{combine}"')
		return DaBRScorer._semantic_score(h_emb, r_emb, t_emb).view(r_emb.size(0), -1)

	@staticmethod
	def au_query_target(
		h_emb: torch.Tensor,
		r_emb: torch.Tensor,
		t_emb: torch.Tensor,
		*,
		predict_head: bool = False,
	) -> tuple[torch.Tensor, torch.Tensor]:
		"""AU pair for the semantic branch: ``(h⊗r, t⊗r⁻¹)`` (swapped for head pred)."""

		hr = DaBRScorer._vec_vec_wise_multiplication(h_emb, r_emb)
		tr = DaBRScorer._vec_vec_wise_multiplication(t_emb, DaBRScorer._quat_inv(r_emb))
		if predict_head:
			return tr, hr
		return hr, tr


class DaBRDistanceScorer(KGEScorer):
	"""Geometric distance component (hybrid DaBR building block; needs ``dr_emb``)."""

	bidirectional_score_batch = True
	kgau_alignment_mode = 'cosine'

	def __init__(self, args=None):
		super().__init__()
		self.args = args
		norm_p = int(getattr(args, 'dabr_distance_norm', 1) or 1) if args is not None else 1
		if norm_p not in (1, 2):
			raise ValueError(f'dabr_distance_norm must be 1 or 2, got {norm_p}')
		self.distance_norm = norm_p

	def supports_candidate_scoring(self) -> bool:
		return False

	def score_emb(
		self,
		h_emb: torch.Tensor,
		r_emb: torch.Tensor,
		t_emb: torch.Tensor,
		combine: str,
		dr_emb: torch.Tensor | None = None,
		**kwargs,
	) -> torch.Tensor:
		del r_emb, kwargs  # distance term uses relation drift ``dr``, not ``r``
		if combine != 'hrt':
			raise ValueError(f'DaBRDistanceScorer only supports combine="hrt", got "{combine}"')
		if dr_emb is None:
			raise ValueError('DaBRDistanceScorer.score_emb requires dr_emb')
		return DaBRScorer._distance_score(h_emb, dr_emb, t_emb, self.distance_norm).view(
			h_emb.size(0), -1
		)

	@staticmethod
	def au_query_target(
		h_emb: torch.Tensor,
		t_emb: torch.Tensor,
		dr_emb: torch.Tensor,
		*,
		predict_head: bool = False,
	) -> tuple[torch.Tensor, torch.Tensor]:
		"""AU pair for the distance branch (TransE-style).

		Tail prediction: ``(h+dr, t)``.
		Head prediction: ``(t-dr, h)`` — same geometry as TransE ``inverse_query``,
		not ``(t, h+dr)``. Cosine(t, h+dr) ≠ cosine(t-dr, h) under unit-norm AU,
		so the TransE form is required for head-batch training and rt_forward eval.
		"""

		if predict_head:
			return t_emb - dr_emb, h_emb
		return h_emb + dr_emb, t_emb


class DaBRModel(KGEModel):
	"""Hybrid DaBR binder: primary ``DaBRScorer`` plus semantic/distance components.

	``scorers[0]`` is the full combining scorer used by default LP paths
	(``φ = s + λ d`` in this codebase's higher-is-better form; official OpenKE
	energy is ``-(s + λ d)``).
	``scorers[1]`` / ``scorers[2]`` are the semantic and distance components used
	by KGAU: separate AU losses combined as ``L = L_s + λ L_d`` with the same
	learnable ``para`` (λ) as the primary scorer.

	With ``dabr_au_independent_spheres``, the distance branch uses a second entity
	table (``aux_embedders['ent_dist']``) so the two AU hyperspheres have no
	shared entity parameters; LP scores are fused as ``⟨h⊗r, t⊗r⁻¹⟩ + λ·cos_dist``.
	"""

	def __init__(
		self,
		ent_embedder,
		rel_embedder,
		scorers=None,
		args=None,
		aux_embedders=None,
	):
		if scorers is None:
			scorers = [
				DaBRScorer(args),
				DaBRSemanticScorer(args),
				DaBRDistanceScorer(args),
			]
		super().__init__(
			ent_embedder,
			rel_embedder,
			scorers=scorers,
			args=args,
			aux_embedders=aux_embedders,
		)
		# Distance-only is TransE-style: targets are raw entities (not relation-composed).
		self.target_uses_relation = not bool(
			getattr(args, 'dabr_au_distance_only', False) if args is not None else False
		)
		if self._independent_spheres() and 'ent_dist' not in self.aux_embedders:
			raise ValueError(
				'dabr_au_independent_spheres requires aux_embedders[\"ent_dist\"] '
				'(second entity table for the distance hypersphere)',
			)

	target_uses_relation = True

	def _distance_only_mode(self) -> bool:
		return bool(getattr(self.args, 'dabr_au_distance_only', False))

	def _independent_spheres(self) -> bool:
		return bool(getattr(self.args, 'dabr_au_independent_spheres', False))

	def _scorer_kwargs(self, r: torch.Tensor | None = None, **extra):
		"""Forward only relation-side aux embeddings (``dr``), never ``ent_dist``."""

		kwargs = dict(extra)
		if r is not None and 'dr' in self.aux_embedders:
			kwargs['dr_emb'] = self._embed(self.aux_embedders['dr'], r)
		return kwargs

	def _embed_dist_entity(self, indices: torch.Tensor) -> torch.Tensor:
		"""Entity rows for the distance hypersphere (independent table or shared)."""

		if self._independent_spheres():
			return self._embed(self.aux_embedders['ent_dist'], indices)
		return self.embed_h(indices)

	def _embed_all_dist_entities(self) -> torch.Tensor:
		if self._independent_spheres():
			return self._embed_all(self.aux_embedders['ent_dist'])
		return self.embed_all_entities()

	def query_encoder(self, h: torch.Tensor, r: torch.Tensor, **kwargs) -> torch.Tensor:
		"""Tail-prediction query.

		Distance-only: TransE-style ``h+dr``. Otherwise legacy ``cat(h⊗r, h+dr)``.
		"""

		scorer_kwargs = {**self._scorer_kwargs(r), **kwargs}
		h_emb = self.embed_h(h)
		dr_emb = scorer_kwargs.get('dr_emb')
		if self._distance_only_mode():
			if dr_emb is None:
				dr_emb = torch.zeros_like(h_emb)
			return h_emb + dr_emb
		return self.get_scorer()._au_head_vector(h_emb, self.embed_r(r), dr_emb)

	def inverse_query_encoder(self, r: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
		"""Head-prediction query.

		Distance-only: TransE-style ``t-dr``. Otherwise legacy ``cat(t⊗r⁻¹, t)``.
		"""

		scorer_kwargs = {**self._scorer_kwargs(r), **kwargs}
		t_emb = self.embed_t(t)
		dr_emb = scorer_kwargs.get('dr_emb')
		if self._distance_only_mode():
			if dr_emb is None:
				dr_emb = torch.zeros_like(t_emb)
			return t_emb - dr_emb
		del kwargs
		return self.get_scorer()._au_tail_vector(t_emb, self.embed_r(r))

	def target_encoder(
		self,
		h: torch.Tensor,
		r: torch.Tensor,
		t: torch.Tensor,
		*,
		predict_head: bool = False,
		**kwargs,
	) -> torch.Tensor:
		scorer = self.get_scorer()
		r_emb = self.embed_r(r)
		scorer_kwargs = {**self._scorer_kwargs(r), **kwargs}
		if predict_head:
			return scorer._au_head_vector(self.embed_h(h), r_emb, scorer_kwargs.get('dr_emb'))
		return scorer._au_tail_vector(self.embed_t(t), r_emb)

	def uniformity_head_encoder(self, h: torch.Tensor, r: torch.Tensor, **kwargs) -> torch.Tensor:
		del r, kwargs
		return self.embed_h(h)

	def dabr_combine_weight(self) -> torch.Tensor:
		"""Learnable λ (``para``) used by original DaBR to fuse semantic + distance."""

		return self.get_scorer(0).para.view(())

	def get_component_queries_targets(
		self,
		h: torch.Tensor,
		r: torch.Tensor,
		t: torch.Tensor,
		*,
		predict_head: bool = False,
	) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
		"""Per-component AU ``(query, target, head)`` for semantic and/or distance.

		Semantic aligns ``h⊗r`` with ``t⊗r⁻¹``. Distance is TransE-style:
		``(h+dr, t)`` for tail batches and ``(t-dr, h)`` for head batches.

		With ``dabr_au_independent_spheres``, semantic uses the primary entity table
		and distance uses ``aux_embedders['ent_dist']`` (no shared entity params).

		- ``dabr_au_semantic_only``: semantic sphere only
		- ``dabr_au_distance_only``: distance / translation sphere only
		- default / independent: both, fused later as ``L_sem + L_dist`` (λ at eval)
		"""

		mode = 'both'
		scorer = self.get_scorer(0)
		if hasattr(scorer, '_au_component_mode'):
			mode = scorer._au_component_mode()
		elif bool(getattr(self.args, 'dabr_au_semantic_only', False)):
			mode = 'semantic'
		elif bool(getattr(self.args, 'dabr_au_distance_only', False)):
			mode = 'distance'

		parts: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
		if mode in ('semantic', 'both'):
			h_sem = self.embed_h(h)
			t_sem = self.embed_t(t)
			r_emb = self.embed_r(r)
			q_s, t_s = DaBRSemanticScorer.au_query_target(
				h_sem, r_emb, t_sem, predict_head=predict_head,
			)
			parts.append((q_s, t_s, h_sem))
		if mode in ('distance', 'both'):
			h_dist = self._embed_dist_entity(h)
			t_dist = self._embed_dist_entity(t)
			dr_emb = self._scorer_kwargs(r).get('dr_emb')
			if dr_emb is None:
				dr_emb = torch.zeros_like(h_dist)
			q_d, t_d = DaBRDistanceScorer.au_query_target(
				h_dist, t_dist, dr_emb, predict_head=predict_head,
			)
			parts.append((q_d, t_d, h_dist))
		if self.normalize_au_vectors:
			parts = [
				(
					self._normalize_au_vector(q),
					self._normalize_au_vector(tgt),
					self._normalize_au_vector(hd),
				)
				for q, tgt, hd in parts
			]
		return parts

	def score_hr_(
		self,
		h: torch.Tensor,
		r: torch.Tensor,
		all_t_embs: torch.Tensor | None = None,
		**kwargs,
	) -> torch.Tensor:
		"""1-vs-all tail scores; independent spheres use ``⟨h⊗r,e⊗r⁻¹⟩ + λ·cos_dist``."""

		if not self._independent_spheres():
			return super().score_hr_(h, r, all_t_embs, **kwargs)
		del all_t_embs, kwargs
		return self._score_independent_hr_(h, r)

	def score_rt_(
		self,
		r: torch.Tensor,
		t: torch.Tensor,
		all_h_embs: torch.Tensor | None = None,
		**kwargs,
	) -> torch.Tensor:
		"""1-vs-all head scores; independent spheres use ``⟨t⊗r⁻¹,e⊗r⟩ + λ·cos_dist``."""

		if not self._independent_spheres():
			return super().score_rt_(r, t, all_h_embs, **kwargs)
		del all_h_embs, kwargs
		return self._score_independent_rt_(r, t)

	def score_hrt(
		self,
		h: torch.Tensor,
		r: torch.Tensor,
		t: torch.Tensor,
		**kwargs,
	) -> torch.Tensor:
		if not self._independent_spheres():
			return super().score_hrt(h, r, t, **kwargs)
		del kwargs
		scorer = self.get_scorer(0)
		lam = self.dabr_combine_weight()
		h_sem, t_sem, r_emb = self.embed_h(h), self.embed_t(t), self.embed_r(r)
		# Original semantic scorer (unnormalized quaternion inner product).
		sem = DaBRScorer._semantic_score(h_sem, r_emb, t_sem)
		h_dist, t_dist = self._embed_dist_entity(h), self._embed_dist_entity(t)
		dr_emb = self._scorer_kwargs(r).get('dr_emb')
		if dr_emb is None:
			dr_emb = torch.zeros_like(h_dist)
		q_d, t_d = DaBRDistanceScorer.au_query_target(h_dist, t_dist, dr_emb, predict_head=False)
		dist = scorer._normalized_pair_score(q_d, t_d)
		return sem + lam * dist

	def _score_independent_hr_(self, h: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
		"""Tail prediction: ``⟨h⊗r, e⊗r⁻¹⟩ + λ·cos(h_d+dr, e_d)`` over all entities."""

		scorer = self.get_scorer(0)
		lam = self.dabr_combine_weight()
		all_sem = self.embed_all_entities()
		all_dist = self._embed_all_dist_entities()
		h_sem = self.embed_h(h)
		r_emb = self.embed_r(r)
		dr_emb = self._scorer_kwargs(r).get('dr_emb')
		if dr_emb is None:
			dr_emb = torch.zeros_like(h_sem)
		h_dist = self._embed_dist_entity(h)
		batch_size = h_sem.size(0)
		num_ent = all_sem.size(0)
		scores = h_sem.new_empty(batch_size, num_ent)

		for row_indices, r_row, (h_sem_sub, h_dist_sub, dr_sub) in scorer._group_batch_by_relation(
			r_emb, h_sem, h_dist, dr_emb,
		):
			r_sub = r_row.expand(h_sem_sub.size(0), -1)
			q_sem = DaBRScorer._vec_vec_wise_multiplication(h_sem_sub, r_sub)
			r_inv = DaBRScorer._quat_inv(r_row).expand(num_ent, -1)
			t_rot = DaBRScorer._vec_vec_wise_multiplication(all_sem, r_inv)
			# Original semantic: unnormalized inner product (not cosine).
			sem = torch.mm(q_sem, t_rot.t())

			q_dist = F.normalize(h_dist_sub + dr_sub, p=2, dim=-1)
			c_dist = F.normalize(all_dist, p=2, dim=-1)
			dist = torch.mm(q_dist, c_dist.t())
			scores.index_copy_(0, row_indices, sem + lam * dist)
		return scores

	def _score_independent_rt_(self, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
		"""Head prediction: ``⟨t⊗r⁻¹, e⊗r⟩ + λ·cos(t_d−dr, e_d)`` over all entities."""

		scorer = self.get_scorer(0)
		lam = self.dabr_combine_weight()
		all_sem = self.embed_all_entities()
		all_dist = self._embed_all_dist_entities()
		t_sem = self.embed_t(t)
		r_emb = self.embed_r(r)
		dr_emb = self._scorer_kwargs(r).get('dr_emb')
		if dr_emb is None:
			dr_emb = torch.zeros_like(t_sem)
		t_dist = self._embed_dist_entity(t)
		batch_size = t_sem.size(0)
		num_ent = all_sem.size(0)
		scores = t_sem.new_empty(batch_size, num_ent)

		for row_indices, r_row, (t_sem_sub, t_dist_sub, dr_sub) in scorer._group_batch_by_relation(
			r_emb, t_sem, t_dist, dr_emb,
		):
			r_sub = r_row.expand(t_sem_sub.size(0), -1)
			q_sem = DaBRScorer._vec_vec_wise_multiplication(t_sem_sub, DaBRScorer._quat_inv(r_sub))
			r_exp = r_row.expand(num_ent, -1)
			h_rot = DaBRScorer._vec_vec_wise_multiplication(all_sem, r_exp)
			# Original semantic: unnormalized inner product (not cosine).
			sem = torch.mm(q_sem, h_rot.t())

			q_dist = F.normalize(t_dist_sub - dr_sub, p=2, dim=-1)
			c_dist = F.normalize(all_dist, p=2, dim=-1)
			dist = torch.mm(q_dist, c_dist.t())
			scores.index_copy_(0, row_indices, sem + lam * dist)
		return scores


def build_scorers(args) -> list:
	"""Return hybrid DaBR scorers: primary + semantic + distance components."""

	return [
		DaBRScorer(args),
		DaBRSemanticScorer(args),
		DaBRDistanceScorer(args),
	]
