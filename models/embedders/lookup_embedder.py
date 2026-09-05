"""Lookup-table embedders and factories for all index-based KGE models."""

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from base.model import KGEEmbedder, ParameterEmbedder
from data.dict_hub import get_entity_dict
from utils.relations import load_relation_to_idx


class LookupEmbedder(KGEEmbedder):
	"""Lightweight embedding table with configurable init and optional dropout."""

	input_mode = 'indices'

	def __init__(
		self,
		num_items: int,
		dim: int,
		args: Any | None = None,
		*,
		role: str = 'entity',
	):
		super().__init__(args, dim=int(dim))
		self.num_items = int(num_items)
		self.role = role
		self.dropout_rate = self.dropout_rate_from_args(args, role)
		sparse = bool(getattr(args, 'sparse_embeddings', False))
		self.embedding = nn.Embedding(self.num_items, self.dim, sparse=sparse)
		self._reset_parameters()

	@staticmethod
	def dropout_rate_from_args(args: Any | None, role: str) -> float:
		if role == 'entity':
			raw = getattr(args, 'entity_dropout', None) if args is not None else None
			if raw is None:
				raw = getattr(args, 'dropout', 0.0) if args is not None else 0.0
		else:
			raw = getattr(args, 'relation_dropout', None) if args is not None else None
			if raw is None:
				raw = getattr(args, 'dropout', 0.0) if args is not None else 0.0
		return float(raw or 0.0)

	@staticmethod
	def scaled_init(module: nn.Module, dim: int, sigma: float = 0.2) -> None:
		"""DistMult/ComplEx-style scale: ``(dim / σ²)^(1/6)``."""

		scale = (dim / sigma ** 2) ** (1 / 6)
		for param in module.parameters():
			param.data.div_(scale)

	@staticmethod
	def initialize(module: nn.Module, args: Any | None, dim: int, role: str = 'entity') -> None:
		"""Initialize a lookup embedding table (``lookup_embedder.initialize``)."""

		del role  # reserved for role-specific defaults
		init_method = str(getattr(args, 'init_method', '') or '').lower() if args is not None else ''
		if not init_method:
			model_name = str(getattr(args, 'model', '') or '').lower() if args is not None else ''
			if any(name in model_name for name in ('rotate', 'protate', 'transe')):
				margin_raw = getattr(args, 'margin', None)
				margin = 6.0 if margin_raw is None else float(margin_raw)
				epsilon_raw = getattr(args, 'epsilon', None)
				epsilon = 2.0 if epsilon_raw is None else float(epsilon_raw)
				bound = (margin + epsilon) / max(1, dim)
				nn.init.uniform_(module.embedding.weight, a=-bound, b=bound)
				return
			if model_name in {'distmult', 'distmult-au', 'complex', 'complex-au'}:
				LookupEmbedder.scaled_init(module, dim)
				return
			nn.init.xavier_uniform_(module.embedding.weight)
			return

		weight = module.embedding.weight
		if init_method == 'uniform_':
			low = float(getattr(args, 'init_uniform_a', -0.05))
			high = getattr(args, 'init_uniform_b', None)
			high = float(-low if high is None else high)
			if low > high:
				low, high = high, low
			nn.init.uniform_(weight, a=low, b=high)
		elif init_method in {'xavier_uniform_', 'xavier_uniform'}:
			gain = float(getattr(args, 'init_xavier_gain', 1.0))
			nn.init.xavier_uniform_(weight, gain=gain)
		elif init_method in {'xavier_normal_', 'xavier_normal'}:
			gain = float(getattr(args, 'init_xavier_gain', 1.0))
			nn.init.xavier_normal_(weight, gain=gain)
		elif init_method == 'normal_':
			mean = float(getattr(args, 'init_normal_mean', 0.0))
			std = float(getattr(args, 'init_normal_std', 1.0))
			nn.init.normal_(weight, mean=mean, std=std)
		elif init_method == 'scaled':
			LookupEmbedder.scaled_init(module, dim)
		elif init_method == 'kbc':
			scale = float(getattr(args, 'init_scale', 1e-3))
			weight.data.mul_(scale)
		else:
			nn.init.xavier_uniform_(weight)

	def _reset_parameters(self) -> None:
		model_name = str(getattr(self.args, 'model', '')).lower()
		if getattr(self.args, 'init_method', None):
			self.initialize(self, self.args, self.dim, role=self.role)
			return
		if any(name in model_name for name in ('rotate', 'protate', 'transe', 'transerr')):
			margin = _float_arg(self.args, 'margin', 6.0)
			epsilon = _float_arg(self.args, 'epsilon', 2.0)
			init_dim = self.dim
			if 'transerr' in model_name and self.role == 'relation':
				init_dim = int(getattr(self.args, 'dim', self.dim) or self.dim)
			bound = (margin + epsilon) / max(1, init_dim)
			nn.init.uniform_(self.embedding.weight, a=-bound, b=bound)
		else:
			nn.init.xavier_uniform_(self.embedding.weight)

	def forward(self, indices: torch.Tensor) -> torch.Tensor:
		vectors = self.embedding(indices.long())
		if self.training and self.dropout_rate > 0.0:
			vectors = F.dropout(vectors, p=self.dropout_rate, training=True)
		return vectors

	def get_all(self) -> torch.Tensor:
		vectors = self.embedding.weight
		if self.training and self.dropout_rate > 0.0:
			vectors = F.dropout(vectors, p=self.dropout_rate, training=True)
		return vectors

	def embed(self, indices: torch.Tensor) -> torch.Tensor:
		return self.forward(indices)

	def embed_all(self) -> torch.Tensor:
		return self.get_all()


class ComplExEntityEmbedder(nn.Module):
	"""Concatenate real/imaginary entity lookup tables for ComplEx."""

	def __init__(self, ent_re: LookupEmbedder, ent_im: LookupEmbedder):
		super().__init__()
		self.ent_re = ent_re
		self.ent_im = ent_im

	def forward(self, indices: torch.Tensor) -> torch.Tensor:
		return torch.cat([self.ent_re(indices), self.ent_im(indices)], dim=-1)

	def embed(self, indices: torch.Tensor) -> torch.Tensor:
		return self.forward(indices)

	def embed_all(self) -> torch.Tensor:
		device = self.ent_re.embedding.weight.device
		return self.forward(torch.arange(self.ent_re.num_items, device=device))


class ComplExRelationEmbedder(nn.Module):
	"""Concatenate real/imaginary relation lookup tables for ComplEx."""

	def __init__(self, rel_re: LookupEmbedder, rel_im: LookupEmbedder):
		super().__init__()
		self.rel_re = rel_re
		self.rel_im = rel_im

	def forward(self, indices: torch.Tensor) -> torch.Tensor:
		return torch.cat([self.rel_re(indices), self.rel_im(indices)], dim=-1)

	def embed(self, indices: torch.Tensor) -> torch.Tensor:
		return self.forward(indices)


def _counts(args) -> tuple[int, int]:
	entity_dict = get_entity_dict()
	rel_to_idx = load_relation_to_idx(args)
	return len(entity_dict), max(len(rel_to_idx), 1)


def _model_name(args) -> str:
	return str(getattr(args, 'model', '') or '').lower()


def _float_arg(args, name: str, default: float) -> float:
	"""Read a float hyperparameter; treat JSON/CLI ``null`` like unset."""

	raw = getattr(args, name, None)
	return default if raw is None else float(raw)


def _is_complex(args) -> bool:
	return 'complex' in _model_name(args)


def _is_rotate(args) -> bool:
	name = _model_name(args)
	return 'rotate' in name and 'protate' not in name


def _is_protate(args) -> bool:
	return 'protate' in _model_name(args)


def _is_dabr(args) -> bool:
	return 'dabr' in _model_name(args)


def _is_transerr(args) -> bool:
	return 'transerr' in _model_name(args)


def _embedding_range(args, dim: int) -> float:
	margin = _float_arg(args, 'margin', 6.0)
	epsilon = _float_arg(args, 'epsilon', 2.0)
	return (margin + epsilon) / max(1, dim)


def _adversarial_gamma(args) -> float:
	"""RotatE-style gamma/margin used for adversarial embedding init (default 200)."""

	raw = getattr(args, 'margin', None)
	if raw is None:
		raw = getattr(args, 'gamma', 200.0)
	return float(raw)


def _adversarial_uniform_init(embedder: LookupEmbedder, args, dim: int) -> None:
	"""Uniform init with range ``(gamma + 2) / dim`` (RotatE adversarial training)."""

	epsilon = _float_arg(args, 'epsilon', 2.0)
	embedding_range = (_adversarial_gamma(args) + epsilon) / max(dim, 1)
	nn.init.uniform_(embedder.embedding.weight, a=-embedding_range, b=embedding_range)


def _complex_part_dim(args, dim: int, *, role: str) -> int:
	"""Per-component ComplEx table width (RotatE ``-de`` / ``-dr``); default ON preserves existing ComplEx."""

	flag = 'double_entity_embedding' if role == 'entity' else 'double_relation_embedding'
	if bool(getattr(args, flag, True)):
		return dim
	return max(dim // 2, 1)


def _init_lookup_table(embedder: LookupEmbedder, args, dim: int, role: str) -> None:
	if bool(getattr(args, 'adversarial_training', False)):
		_adversarial_uniform_init(embedder, args, dim)
	elif getattr(args, 'init_method', None):
		LookupEmbedder.initialize(embedder, args, dim, role=role)
	elif _model_name(args) in {'distmult', 'distmult-au', 'complex', 'complex-au'}:
		LookupEmbedder.scaled_init(embedder, dim)
	else:
		embedder._reset_parameters()


def _init_rotate_weight(weight: nn.Parameter, args, *, role: str, hidden_dim: int) -> None:
	"""Initialize RotatE/pRotatE parameter tables (``lookup_embedder.initialize``)."""

	full_dim = hidden_dim * 2 if role == 'entity' and _is_rotate(args) else hidden_dim
	if _is_protate(args):
		bound = _embedding_range(args, hidden_dim)
		nn.init.uniform_(weight, a=-bound, b=bound)
		return
	if role == 'relation' and _is_rotate(args):
		if (
			str(getattr(args, 'model', '') or '').lower() == 'rotate'
			and not bool(getattr(args, 'adversarial_training', False))
		):
			low = float(getattr(args, 'init_relation_uniform_a', -math.pi))
			high = float(getattr(args, 'init_relation_uniform_b', math.pi))
			nn.init.uniform_(weight, a=low, b=high)
			return
	if getattr(args, 'init_method', None):
		temp = LookupEmbedder(weight.size(0), full_dim, args, role=role)
		LookupEmbedder.initialize(temp, args, full_dim, role=role)
		weight.data.copy_(temp.embedding.weight.data)
		return
	bound = _embedding_range(args, hidden_dim)
	nn.init.uniform_(weight, a=-bound, b=bound)


def build_entity_embedder(args) -> nn.Module:
	n_ent, _n_rel = _counts(args)
	dim = int(getattr(args, 'dim', 200))

	if _is_complex(args):
		entity_dim = _complex_part_dim(args, dim, role='entity')
		ent_re = LookupEmbedder(n_ent, entity_dim, args, role='entity')
		ent_im = LookupEmbedder(n_ent, entity_dim, args, role='entity')
		if bool(getattr(args, 'adversarial_training', False)):
			for module in (ent_re, ent_im):
				_adversarial_uniform_init(module, args, dim)
		elif getattr(args, 'init_scaled', True) and not getattr(args, 'init_method', None):
			for module in (ent_re, ent_im):
				LookupEmbedder.scaled_init(module, entity_dim)
		return ComplExEntityEmbedder(ent_re, ent_im)

	if _is_rotate(args):
		hidden_dim = dim
		weight = nn.Parameter(torch.zeros(n_ent, hidden_dim * 2))
		_init_rotate_weight(weight, args, role='entity', hidden_dim=hidden_dim)
		return ParameterEmbedder(weight)

	if _is_protate(args):
		hidden_dim = dim
		weight = nn.Parameter(torch.zeros(n_ent, hidden_dim))
		_init_rotate_weight(weight, args, role='entity', hidden_dim=hidden_dim)
		return ParameterEmbedder(weight)

	if _is_dabr(args):
		emb_dim = 4 * dim
		embedder = LookupEmbedder(n_ent, emb_dim, args, role='entity')
		nn.init.xavier_uniform_(embedder.embedding.weight)
		return embedder

	if _is_transerr(args):
		embedder = LookupEmbedder(n_ent, dim, args, role='entity')
		_init_lookup_table(embedder, args, dim, role='entity')
		return embedder

	embedder = LookupEmbedder(n_ent, dim, args, role='entity')
	_init_lookup_table(embedder, args, dim, role='entity')
	return embedder


def build_relation_embedder(args) -> nn.Module:
	_, n_rel = _counts(args)
	dim = int(getattr(args, 'dim', 200))

	if _is_complex(args):
		relation_dim = _complex_part_dim(args, dim, role='relation')
		rel_re = LookupEmbedder(n_rel, relation_dim, args, role='relation')
		rel_im = LookupEmbedder(n_rel, relation_dim, args, role='relation')
		if bool(getattr(args, 'adversarial_training', False)):
			for module in (rel_re, rel_im):
				_adversarial_uniform_init(module, args, dim)
		elif getattr(args, 'init_scaled', True) and not getattr(args, 'init_method', None):
			for module in (rel_re, rel_im):
				LookupEmbedder.scaled_init(module, relation_dim)
		return ComplExRelationEmbedder(rel_re, rel_im)

	if _is_rotate(args) or _is_protate(args):
		hidden_dim = dim
		weight = nn.Parameter(torch.zeros(n_rel, hidden_dim))
		_init_rotate_weight(weight, args, role='relation', hidden_dim=hidden_dim)
		return ParameterEmbedder(weight)

	if _is_dabr(args):
		emb_dim = 4 * int(getattr(args, 'dim', getattr(args, 'hidden_size', 100)))
		embedder = LookupEmbedder(n_rel, emb_dim, args, role='relation')
		nn.init.xavier_uniform_(embedder.embedding.weight)
		return embedder

	if _is_transerr(args):
		if not bool(getattr(args, 'triple_relation_embedding', True)):
			raise ValueError('TransERR requires triple_relation_embedding')
		embedder = LookupEmbedder(n_rel, dim * 3, args, role='relation')
		_init_lookup_table(embedder, args, dim, role='relation')
		return embedder

	embedder = LookupEmbedder(n_rel, dim, args, role='relation')
	_init_lookup_table(embedder, args, dim, role='relation')
	return embedder


def build_dr_embedder(args) -> LookupEmbedder:
	"""DaBR-specific relation drift table (bound as ``aux_embedders['dr']`` on ``KGEModel``)."""

	_, n_rel = _counts(args)
	emb_dim = 4 * int(getattr(args, 'dim', getattr(args, 'hidden_size', 100)))
	embedder = LookupEmbedder(n_rel, emb_dim, args, role='relation')
	nn.init.xavier_uniform_(embedder.embedding.weight)
	return embedder
