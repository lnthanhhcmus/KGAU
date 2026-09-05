"""RotatE scorer and model (``score_emb``)."""

import math

import torch

from base.model import KGEScorer, KGEModel


def build_scorer(args) -> 'RotatEScorer':
	return RotatEScorer(args)


def _is_libkge_rotate(args) -> bool:
	"""Return True for plain RotatE (``rotate``), not RotatE-AU or KGE adversarial runs."""

	if bool(getattr(args, 'adversarial_training', False)):
		return False
	return str(getattr(args, 'model', '') or '').lower() == 'rotate'


@torch.no_grad()
def normalize_rotate_phases(model) -> None:
	"""Keep relation phases in [-pi, pi] (``RotatE.normalize_phases``)."""

	from utils.device import get_model_obj

	model_obj = get_model_obj(model)
	rel_embedder = getattr(model_obj, 'rel_embedder', None)
	if rel_embedder is None or not hasattr(rel_embedder, 'weight'):
		return
	phases = rel_embedder.weight.data
	phases = phases + math.pi
	phases = torch.remainder(phases, 2.0 * math.pi)
	phases = phases - math.pi
	rel_embedder.weight.data[:] = phases[:]


class RotatEScorer(KGEScorer):
	"""RotatE score function via a single ``score_emb``.

	``combine`` modes: ``hrt``, ``hr_``, ``_rt``, ``hr_c``, ``_rt_c``.
	"""

	bidirectional_score_batch = True

	def __init__(self, args=None):
		super().__init__()
		self.args = args
		self._libkge = _is_libkge_rotate(args)
		if self._libkge:
			self.l_norm = float(getattr(args, 'l_norm', 1.0))
		else:
			self.dim = int(getattr(args, 'dim', 0) or 0)
			margin_value = getattr(args, 'margin', None)
			self.margin = float(6.0 if margin_value is None else margin_value)
			epsilon = float(getattr(args, 'epsilon', 2.0))
			self.embedding_range = float((self.margin + epsilon) / max(self.dim, 1))

	def supports_candidate_scoring(self) -> bool:
		return True

	def _phase(self, relation_emb: torch.Tensor) -> torch.Tensor:
		"""Map raw relation tensors to the RotatE phase space."""

		if self._libkge:
			return relation_emb
		return relation_emb / (self.embedding_range / math.pi)

	@staticmethod
	def _split_complex(embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		"""Split concatenated real and imaginary parts."""

		return torch.chunk(embeddings, 2, dim=-1)

	@staticmethod
	def _rotation(relation_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		"""Convert relation radians to points on the complex unit circle."""

		return torch.cos(relation_emb), torch.sin(relation_emb)

	@staticmethod
	def _hadamard_complex(
		x_re: torch.Tensor,
		x_im: torch.Tensor,
		y_re: torch.Tensor,
		y_im: torch.Tensor,
	) -> tuple[torch.Tensor, torch.Tensor]:
		return x_re * y_re - x_im * y_im, x_re * y_im + x_im * y_re

	@staticmethod
	def _abs_complex(x_re: torch.Tensor, x_im: torch.Tensor) -> torch.Tensor:
		# Complex modulus via norm: avoids NaN grads at zero vs sqrt(x^2+y^2).
		x_re_im = torch.stack((x_re, x_im), dim=0)
		return torch.norm(x_re_im, dim=0)

	def _norm_nonnegative(self, values: torch.Tensor, dim: int) -> torch.Tensor:
		"""Lp norm along ``dim`` for non-negative inputs (``norm_nonnegative``)."""

		if self.l_norm == 1.0:
			return values.sum(dim=dim)
		return torch.norm(values, dim=dim, p=self.l_norm)

	def _libkge_distance(
		self,
		q_re: torch.Tensor,
		q_im: torch.Tensor,
		cand_re: torch.Tensor,
		cand_im: torch.Tensor,
		*,
		pairwise: bool,
	) -> torch.Tensor:
		"""Return negative RotatE distance scores (higher is better)."""

		if pairwise:
			diff_re = q_re.unsqueeze(1) - cand_re.unsqueeze(0)
			diff_im = q_im.unsqueeze(1) - cand_im.unsqueeze(0)
			norm_dim = 2
		else:
			diff_re = q_re - cand_re
			diff_im = q_im - cand_im
			norm_dim = 1
		diff_abs = self._abs_complex(diff_re, diff_im)
		return -self._norm_nonnegative(diff_abs, dim=norm_dim)

	def _entity_chunk_size(self, batch_size: int, half_dim: int) -> int:
		"""Candidate chunk size for 1-vs-all scoring (controls peak GPU memory)."""

		configured = int(getattr(self.args, 'eval_entity_chunk_size', 256) or 256)
		bytes_budget = int(getattr(self.args, 'eval_entity_chunk_bytes', 128 * 1024 * 1024) or 128 * 1024 * 1024)
		per_candidate = max(1, batch_size * half_dim * 4 * 3)
		memory_limit = max(1, bytes_budget // per_candidate)
		return max(1, min(configured, memory_limit))

	def _libkge_distance_1vsall(
		self,
		q_re: torch.Tensor,
		q_im: torch.Tensor,
		cand_re: torch.Tensor,
		cand_im: torch.Tensor,
	) -> torch.Tensor:
		num_candidates = cand_re.size(0)
		batch_size = q_re.size(0)
		half_dim = q_re.size(-1)
		chunk_size = self._entity_chunk_size(batch_size, half_dim)
		scores = q_re.new_empty(batch_size, num_candidates)
		for start in range(0, num_candidates, chunk_size):
			end = min(start + chunk_size, num_candidates)
			scores[:, start:end] = self._libkge_distance(
				q_re,
				q_im,
				cand_re[start:end],
				cand_im[start:end],
				pairwise=True,
			)
		return scores

	def _margin_distance_1vsall(
		self,
		q_re: torch.Tensor,
		q_im: torch.Tensor,
		cand_re: torch.Tensor,
		cand_im: torch.Tensor,
	) -> torch.Tensor:
		num_candidates = cand_re.size(0)
		batch_size = q_re.size(0)
		half_dim = q_re.size(-1)
		chunk_size = self._entity_chunk_size(batch_size, half_dim)
		scores = q_re.new_empty(batch_size, num_candidates)
		for start in range(0, num_candidates, chunk_size):
			end = min(start + chunk_size, num_candidates)
			re_score = q_re.unsqueeze(1) - cand_re[start:end].unsqueeze(0)
			im_score = q_im.unsqueeze(1) - cand_im[start:end].unsqueeze(0)
			diff_abs = self._abs_complex(re_score, im_score)
			scores[:, start:end] = self.margin - diff_abs.sum(dim=-1)
		return scores

	def _rotate_query(
		self,
		h_re: torch.Tensor,
		h_im: torch.Tensor,
		r_emb: torch.Tensor,
		*,
		inverse: bool,
	) -> tuple[torch.Tensor, torch.Tensor]:
		r_re, r_im = self._rotation(self._phase(r_emb))
		if inverse:
			r_im = -r_im
			return self._hadamard_complex(r_re, r_im, h_re, h_im)
		return self._hadamard_complex(h_re, h_im, r_re, r_im)

	def _distance_score(
		self,
		h_emb: torch.Tensor,
		r_emb: torch.Tensor,
		t_emb: torch.Tensor,
		*,
		predict_head: bool,
	) -> torch.Tensor:
		"""Shared RotatE distance for tail- or head-prediction."""

		h_re, h_im = self._split_complex(h_emb)
		t_re, t_im = self._split_complex(t_emb)
		if self._libkge:
			if predict_head:
				q_re, q_im = self._rotate_query(t_re, t_im, r_emb, inverse=True)
				return self._libkge_distance(q_re, q_im, h_re, h_im, pairwise=False)
			q_re, q_im = self._rotate_query(h_re, h_im, r_emb, inverse=False)
			return self._libkge_distance(q_re, q_im, t_re, t_im, pairwise=False)

		return self._margin_distance_pairwise(h_re, h_im, r_emb, t_re, t_im, predict_head=predict_head)

	def _candidate_distance_score(
		self,
		h_emb: torch.Tensor,
		r_emb: torch.Tensor,
		t_emb: torch.Tensor,
		*,
		predict_head: bool,
	) -> torch.Tensor:
		"""RotatE distance for [B, C, D] candidate tensors (negative sampling)."""

		h_re, h_im = self._split_complex(h_emb)
		t_re, t_im = self._split_complex(t_emb)
		if self._libkge:
			if predict_head:
				q_re, q_im = self._rotate_query(t_re, t_im, r_emb, inverse=True)
				if h_re.dim() == 3:
					diff_re = q_re.unsqueeze(1) - h_re
					diff_im = q_im.unsqueeze(1) - h_im
					norm_dim = 2
				else:
					diff_re = q_re - h_re
					diff_im = q_im - h_im
					norm_dim = 1
			else:
				q_re, q_im = self._rotate_query(h_re, h_im, r_emb, inverse=False)
				if t_re.dim() == 3:
					diff_re = q_re.unsqueeze(1) - t_re
					diff_im = q_im.unsqueeze(1) - t_im
					norm_dim = 2
				else:
					diff_re = q_re - t_re
					diff_im = q_im - t_im
					norm_dim = 1
			diff_abs = self._abs_complex(diff_re, diff_im)
			return -self._norm_nonnegative(diff_abs, dim=norm_dim)

		return self._margin_distance_pairwise(h_re, h_im, r_emb, t_re, t_im, predict_head=predict_head)

	def _margin_distance_pairwise(
		self,
		h_re: torch.Tensor,
		h_im: torch.Tensor,
		r_emb: torch.Tensor,
		t_re: torch.Tensor,
		t_im: torch.Tensor,
		*,
		predict_head: bool,
	) -> torch.Tensor:
		"""Margin RotatE distance with optional per-row candidate broadcasting."""

		phase = self._phase(r_emb)
		r_re = torch.cos(phase)
		r_im = torch.sin(phase)
		candidate_dim = h_re.dim() == 3 if predict_head else t_re.dim() == 3
		if predict_head:
			if candidate_dim:
				re_score = r_re.unsqueeze(1) * t_re.unsqueeze(1) + r_im.unsqueeze(1) * t_im.unsqueeze(1) - h_re
				im_score = r_re.unsqueeze(1) * t_im.unsqueeze(1) - r_im.unsqueeze(1) * t_re.unsqueeze(1) - h_im
			else:
				re_score = r_re * t_re + r_im * t_im - h_re
				im_score = r_re * t_im - r_im * t_re - h_im
		elif candidate_dim:
			re_score = h_re.unsqueeze(1) * r_re.unsqueeze(1) - h_im.unsqueeze(1) * r_im.unsqueeze(1) - t_re
			im_score = h_re.unsqueeze(1) * r_im.unsqueeze(1) + h_im.unsqueeze(1) * r_re.unsqueeze(1) - t_im
		else:
			re_score = h_re * r_re - h_im * r_im - t_re
			im_score = h_re * r_im + h_im * r_re - t_im
		diff_abs = self._abs_complex(re_score, im_score)
		return self.margin - diff_abs.sum(dim=-1)

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
			return self._distance_score(h_emb, r_emb, t_emb, predict_head=False).view(n, -1)
		if combine == 'hr_':
			h_re, h_im = self._split_complex(h_emb)
			t_re, t_im = self._split_complex(t_emb)
			q_re, q_im = self._rotate_query(h_re, h_im, r_emb, inverse=False)
			if self._libkge:
				return self._libkge_distance_1vsall(q_re, q_im, t_re, t_im)
			return self._margin_distance_1vsall(q_re, q_im, t_re, t_im)
		if combine == '_rt':
			h_re, h_im = self._split_complex(h_emb)
			t_re, t_im = self._split_complex(t_emb)
			q_re, q_im = self._rotate_query(t_re, t_im, r_emb, inverse=True)
			if self._libkge:
				return self._libkge_distance_1vsall(q_re, q_im, h_re, h_im)
			return self._margin_distance_1vsall(q_re, q_im, h_re, h_im)
		if combine == 'hr_c':
			return self._candidate_distance_score(h_emb, r_emb, t_emb, predict_head=False)
		if combine == '_rt_c':
			return self._candidate_distance_score(h_emb, r_emb, t_emb, predict_head=True)
		raise ValueError(f'cannot handle combine="{combine}"')


class RotatEModel(KGEModel):
	"""Bind lookup embedders to ``RotatEScorer`` (``scorers`` length 1 by default)."""

	def __init__(
		self,
		ent_embedder,
		rel_embedder,
		scorers=None,
		args=None,
		aux_embedders=None,
	):
		if scorers is None:
			scorers = [RotatEScorer(args)]
		super().__init__(
			ent_embedder,
			rel_embedder,
			scorers=scorers,
			args=args,
			aux_embedders=aux_embedders,
		)

	def query_encoder(self, h: torch.Tensor, r: torch.Tensor, **kwargs) -> torch.Tensor:
		"""Tail-prediction query: rotate ``h`` by ``r`` in complex space."""

		del kwargs
		scorer = self.get_scorer()
		h_re, h_im = scorer._split_complex(self.embed_h(h))
		q_re, q_im = scorer._rotate_query(h_re, h_im, self.embed_r(r), inverse=False)
		return torch.cat([q_re, q_im], dim=-1)

	def inverse_query_encoder(self, r: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
		"""Head-prediction query: rotate ``t`` by ``r⁻¹``."""

		del kwargs
		scorer = self.get_scorer()
		t_re, t_im = scorer._split_complex(self.embed_t(t))
		q_re, q_im = scorer._rotate_query(t_re, t_im, self.embed_r(r), inverse=True)
		return torch.cat([q_re, q_im], dim=-1)
