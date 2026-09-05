"""pRotatE scorer and model (``score_emb``)."""

import math

import torch
import torch.nn as nn

from base.model import KGEScorer, KGEModel


def build_scorer(args) -> 'pRotatEScorer':
	return pRotatEScorer(args)


@torch.no_grad()
def normalize_protate_phases(model) -> None:
	"""Wrap entity/relation embeddings so pRotatE phases stay in [-pi, pi].

	Matches the original RotatE/pRotatE reference implementation, which maps
	raw tables through ``embedding / (embedding_range / pi)`` before scoring.
	Without wrapping, Adagrad can drift embeddings to large values; ``sin(phase)``
	then oscillates and 1-vs-all ranks stay near random even while negsamp loss falls.
	"""

	from utils.device import get_model_obj

	model_obj = get_model_obj(model)
	scorer = model_obj.get_scorer()
	embedding_range = float(getattr(scorer, 'embedding_range', 0.0) or 0.0)
	if embedding_range <= 0.0:
		return
	phase_scale = embedding_range / math.pi

	for attr in ('ent_embedder', 'rel_embedder'):
		embedder = getattr(model_obj, attr, None)
		if embedder is None or not hasattr(embedder, 'weight'):
			continue
		embeddings = embedder.weight.data
		phases = embeddings / phase_scale
		phases = phases + math.pi
		phases = torch.remainder(phases, 2.0 * math.pi)
		phases = phases - math.pi
		embedder.weight.data[:] = phases * phase_scale


class pRotatEScorer(KGEScorer):
	"""pRotatE score function via a single ``score_emb``.

	``combine`` modes: ``hrt``, ``hr_``, ``_rt``, ``hr_c``, ``_rt_c``.
	"""

	bidirectional_score_batch = True
	kgau_alignment_mode = 'sin_phase'

	def __init__(self, args=None):
		super().__init__()
		self.args = args
		self.dim = int(getattr(args, 'dim', 0) or 0)
		margin_value = getattr(args, 'margin', None)
		self.margin = float(6.0 if margin_value is None else margin_value)
		epsilon = float(getattr(args, 'epsilon', 2.0))
		self.embedding_range = float((self.margin + epsilon) / max(self.dim, 1))
		# Matches KnowledgeGraphEmbedding pRotatE: modulus = 0.5 * embedding_range.
		self.modulus = nn.Parameter(torch.tensor([[0.5 * self.embedding_range]]))

	def supports_candidate_scoring(self) -> bool:
		return True

	def _phase(self, embeddings: torch.Tensor) -> torch.Tensor:
		"""Map raw tensors into the phase space used by pRotatE."""

		return embeddings / (self.embedding_range / math.pi)

	@staticmethod
	def _wrap_phase(phase: torch.Tensor) -> torch.Tensor:
		"""Map radians to [-pi, pi] (matches ``normalize_protate_phases``)."""

		return torch.remainder(phase + math.pi, 2.0 * math.pi) - math.pi

	def _cosine_phase_vector(self, phase: torch.Tensor) -> torch.Tensor:
		"""RotatE-style AU/LP coords: per-dim unit circle ``[cos, sin]`` (length ``2 * dim``)."""

		phase = self._wrap_phase(phase)
		return torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)

	def _score_phase(self, phase: torch.Tensor) -> torch.Tensor:
		# squeeze: KGE stores modulus as [[m]]; keep broadcast-safe for [B] and [B, N].
		return self.margin - torch.abs(torch.sin(phase)).sum(dim=-1) * self.modulus.squeeze()

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
		h_ph = self._phase(h_emb)
		r_ph = self._phase(r_emb)
		t_ph = self._phase(t_emb)

		if combine == 'hrt':
			return self._score_phase(h_ph + r_ph - t_ph).view(n, -1)
		if combine == 'hr_':
			phase = h_ph.unsqueeze(1) + r_ph.unsqueeze(1) - t_ph.unsqueeze(0)
			return self._score_phase(phase)
		if combine == '_rt':
			phase = h_ph.unsqueeze(0) + (r_ph.unsqueeze(1) - t_ph.unsqueeze(1))
			return self._score_phase(phase)
		if combine == 'hr_c':
			# t_emb is [B, C, D]
			phase = h_ph.unsqueeze(1) + r_ph.unsqueeze(1) - t_ph
			return self._score_phase(phase)
		if combine == '_rt_c':
			# h_emb is [B, C, D]
			phase = h_ph + (r_ph.unsqueeze(1) - t_ph.unsqueeze(1))
			return self._score_phase(phase)
		raise ValueError(f'cannot handle combine="{combine}"')

	def au_entity_embeddings(self, entity_emb: torch.Tensor) -> torch.Tensor:
		return self._cosine_phase_vector(self._phase(entity_emb))


class pRotatEModel(KGEModel):
	"""Bind lookup embedders to ``pRotatEScorer`` (``scorers`` length 1 by default)."""

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
			scorers = [pRotatEScorer(args)]
		super().__init__(
			ent_embedder,
			rel_embedder,
			scorers=scorers,
			args=args,
			aux_embedders=aux_embedders,
		)

	def query_encoder(self, h: torch.Tensor, r: torch.Tensor, **kwargs) -> torch.Tensor:
		del kwargs
		scorer = self.get_scorer()
		composed = scorer._wrap_phase(scorer._phase(self.embed_h(h)) + scorer._phase(self.embed_r(r)))
		return scorer._cosine_phase_vector(composed)

	def inverse_query_encoder(self, r: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
		del kwargs
		scorer = self.get_scorer()
		composed = scorer._wrap_phase(scorer._phase(self.embed_r(r)) - scorer._phase(self.embed_t(t)))
		return scorer._cosine_phase_vector(composed)

	def target_encoder(
		self,
		h: torch.Tensor,
		r: torch.Tensor,
		t: torch.Tensor,
		*,
		predict_head: bool = False,
		**kwargs,
	) -> torch.Tensor:
		del r, kwargs
		scorer = self.get_scorer()
		entity = self.embed_h(h) if predict_head else self.embed_t(t)
		return scorer.au_entity_embeddings(entity)

	def uniformity_head_encoder(self, h: torch.Tensor, r: torch.Tensor, **kwargs) -> torch.Tensor:
		del r, kwargs
		return self.get_scorer().au_entity_embeddings(self.embed_h(h))
