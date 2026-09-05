"""Alignment and uniformity loss for KGAU."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Shared with NegSamp / AllNeg chunk heuristics: cap peak pair-block memory.
_UNIFORM_PAIR_CHUNK_BYTES_BUDGET = 512 * 1024 * 1024


def resolve_uniform_pair_chunk_size(n: int, dim: int, explicit: int = 0) -> int:
	"""Pair-block width C for chunked uniformity.

	explicit > 0: use min(explicit, n).
	Else (0 = auto): choose C so a [C, C] float32 block stays near the shared
	~512MiB budget, with a soft cap of 256 (reference: B=512 entity term n=1024).
	"""

	if n <= 1:
		return max(n, 1)
	explicit = int(explicit or 0)
	if explicit > 0:
		return min(explicit, n)
	max_by_pair = int((_UNIFORM_PAIR_CHUNK_BYTES_BUDGET / 8.0) ** 0.5)
	max_by_dim = max(_UNIFORM_PAIR_CHUNK_BYTES_BUDGET // max(dim * 4 * 4, 1), 32)
	auto = max(32, min(max_by_pair, max_by_dim, n, 256))
	return int(auto)


def _normalized_sqdist_block(xi: torch.Tensor, xj: torch.Tensor) -> torch.Tensor:
	"""||a-b||^2 for L2-normalized rows via 2 - 2 a·b. Shape [Ci, Cj]."""

	return (2.0 - 2.0 * (xi @ xj.transpose(0, 1))).clamp_min(0)


def chunked_pairwise_uniformity(
	x: torch.Tensor,
	uniform_t: float | torch.Tensor = 4,
	pair_chunk_size: int = 0,
	*,
	already_normalized: bool = False,
) -> torch.Tensor:
	"""Exact Wang-Isola uniformity: log(mean_{i<j} exp(-t ||x_i-x_j||^2)).

	Same pairs as ``torch.pdist``, but accumulates over [C,C] blocks so peak
	memory stays O(C^2 + n·D) instead of pdist's backward spike ~O(n^2·D).
	"""

	if not already_normalized:
		x = F.normalize(x, dim=-1)
	n = x.size(0)
	if n < 2:
		return (x * 0).sum()

	chunk = resolve_uniform_pair_chunk_size(n, x.size(1), pair_chunk_size)
	sum_exp = x.new_zeros(())
	count = 0

	for i0 in range(0, n, chunk):
		i1 = min(i0 + chunk, n)
		xi = x[i0:i1]
		for j0 in range(i0, n, chunk):
			j1 = min(j0 + chunk, n)
			xj = x[j0:j1]
			sq = _normalized_sqdist_block(xi, xj)
			if i0 == j0:
				tri = torch.triu(
					torch.ones(i1 - i0, j1 - j0, device=x.device, dtype=torch.bool),
					diagonal=1,
				)
				vals = sq.masked_select(tri)
			else:
				vals = sq.reshape(-1)
			sum_exp = sum_exp + (-uniform_t * vals).exp().sum()
			count += vals.numel()

	return (sum_exp / max(count, 1)).log()


def chunked_pairwise_margin_uniformity(
	x: torch.Tensor,
	uniform_margin: float = 2.0,
	uniform_t: float | torch.Tensor = 4,
	pair_chunk_size: int = 0,
	*,
	already_normalized: bool = False,
) -> torch.Tensor:
	"""Exact soft-margin uniformity: log(mean_{i<j} exp(t * ReLU(m - ||x_i-x_j||^2)))."""

	if not already_normalized:
		x = F.normalize(x, dim=-1)
	n = x.size(0)
	if n < 2:
		return (x * 0).sum()

	chunk = resolve_uniform_pair_chunk_size(n, x.size(1), pair_chunk_size)
	sum_exp = x.new_zeros(())
	count = 0

	for i0 in range(0, n, chunk):
		i1 = min(i0 + chunk, n)
		xi = x[i0:i1]
		for j0 in range(i0, n, chunk):
			j1 = min(j0 + chunk, n)
			xj = x[j0:j1]
			sq = _normalized_sqdist_block(xi, xj)
			if i0 == j0:
				tri = torch.triu(
					torch.ones(i1 - i0, j1 - j0, device=x.device, dtype=torch.bool),
					diagonal=1,
				)
				vals = sq.masked_select(tri)
			else:
				vals = sq.reshape(-1)
			sum_exp = sum_exp + F.relu(uniform_margin - vals).mul(uniform_t).exp().sum()
			count += vals.numel()

	return (sum_exp / max(count, 1)).log()


def chunked_pairwise_sqdist(
	x: torch.Tensor,
	pair_chunk_size: int = 0,
	*,
	already_normalized: bool = False,
) -> torch.Tensor:
	"""Collect exact i<j squared distances via [C,C] blocks (no ``torch.pdist``)."""

	if not already_normalized:
		x = F.normalize(x, dim=-1)
	n = x.size(0)
	if n < 2:
		return x.new_empty(0)

	chunk = resolve_uniform_pair_chunk_size(n, x.size(1), pair_chunk_size)
	parts: list[torch.Tensor] = []
	for i0 in range(0, n, chunk):
		i1 = min(i0 + chunk, n)
		xi = x[i0:i1]
		for j0 in range(i0, n, chunk):
			j1 = min(j0 + chunk, n)
			xj = x[j0:j1]
			sq = _normalized_sqdist_block(xi, xj)
			if i0 == j0:
				tri = torch.triu(
					torch.ones(i1 - i0, j1 - j0, device=x.device, dtype=torch.bool),
					diagonal=1,
				)
				parts.append(sq.masked_select(tri))
			else:
				parts.append(sq.reshape(-1))
	return torch.cat(parts, dim=0) if parts else x.new_empty(0)


def distinct_first_indices(keys: torch.Tensor) -> torch.Tensor:
	"""Return row indices of the first occurrence of each unique key in the batch."""

	if keys.numel() == 0:
		return keys.new_empty(0, dtype=torch.long)
	if keys.dim() == 1:
		_, inverse = torch.unique(keys, sorted=False, return_inverse=True)
	else:
		_, inverse = torch.unique(keys, dim=0, sorted=False, return_inverse=True)
	num_unique = int(inverse.max().item()) + 1
	indices = torch.full((num_unique,), keys.size(0), dtype=torch.long, device=keys.device)
	positions = torch.arange(keys.size(0), device=keys.device)
	indices.scatter_reduce_(0, inverse, positions, reduce='amin')
	return indices


def _coalesce_float(value, default: float) -> float:
	"""Treat missing or JSON-null hyperparameters as the default."""

	return default if value is None else float(value)


def select_distinct_rows(vectors: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
	"""Keep one embedding row per unique key (first occurrence in the batch)."""

	if vectors.size(0) == 0:
		return vectors
	keys = keys.to(device=vectors.device)
	indices = distinct_first_indices(keys)
	return vectors.index_select(0, indices)


_GAMMA_NAMES = ('q', 't', 'h', 'ent', 'cross')


class KGAULoss(nn.Module):
	"""Alignment and uniformity loss for knowledge graph embeddings."""

	def __init__(
		self,
		gamma_q=1.0,
		gamma_t=1.0,
		gamma_h=0.0,
		gamma_ent=0.0,
		gamma_cross=0.0,
		alpha: float = 1.0,
		align_alpha: float = 2.0,
		tuni=2.0,
		learnable_tuni: bool = False,
		learnable_au_alpha: bool = False,
		learnable_au_gammas: bool = False,
		tuni_as_alpha: bool = False,
		max_uniformity_samples: int = 1024,
		additive_margin: float = 0.0,
		alignment_mode: str = 'cosine',
		normalize_uniformity: bool = True,
		assume_unit_norm: bool = False,
		average_uniformity_terms: bool = False,
		uniformity_full_pdist: bool = False,
		uniformity_pdist_gb: float | None = None,
		uniform_pair_chunk_size: int = 0,
	):
		super().__init__()
		self.average_uniformity_terms = bool(average_uniformity_terms)
		self.uniformity_full_pdist = bool(uniformity_full_pdist)
		self.uniformity_pdist_gb = uniformity_pdist_gb
		# 0 = auto (~512MiB / soft-cap 256); >0 forces pair-block width. Always used
		# instead of torch.pdist when computing full i<j uniformity.
		self.uniform_pair_chunk_size = int(uniform_pair_chunk_size or 0)
		self.tuni_as_alpha = bool(tuni_as_alpha)
		self.learnable_au_alpha = bool(learnable_au_alpha)
		self.learnable_au_gammas = bool(learnable_au_gammas)
		alpha_init = _coalesce_float(alpha, 1.0)
		# Exponent on element-wise (q-t) in alignment; 2.0 recovers squared L2.
		self.align_alpha = _coalesce_float(align_alpha, 2.0)
		self.register_buffer('alpha_init', torch.tensor(alpha_init))
		if not self.learnable_au_alpha:
			self._alpha = alpha_init
		else:
			# Bounded upward adjustment only: exp(adj) in [1, inf), init at 0.
			# Unconstrained log-scale alpha falls under alignment loss minimization.
			self.log_alpha_adj = nn.Parameter(torch.zeros(()))
		self.register_buffer('alpha_schedule_mult', torch.tensor(1.0))
		gamma_inits = {
			'q': _coalesce_float(gamma_q, 1.0),
			't': _coalesce_float(gamma_t, 1.0),
			'h': _coalesce_float(gamma_h, 0.0),
			'ent': _coalesce_float(gamma_ent, 0.0),
			'cross': _coalesce_float(gamma_cross, 0.0),
		}
		for name, value in gamma_inits.items():
			init = float(value)
			# init <= 0 disables a term (fixed or learnable); only positive inits are scheduled/learned.
			self.register_buffer(f'gamma_init_{name}', torch.tensor(init))
			if not self.learnable_au_gammas:
				setattr(self, f'_gamma_{name}', init)
			elif init > 0.0:
				# Bounded downward adjustment only: exp(adj) in (0, 1], init at 0.
				# Unconstrained log-scale gammas rise under AU loss because uniformity is negative.
				setattr(self, f'log_gamma_adj_{name}', nn.Parameter(torch.zeros(())))
		self.register_buffer('gamma_schedule_mult', torch.tensor(1.0))
		# `tuni` is the uniformity temperature; with ``tuni_as_alpha`` it also replaces alpha.
		tuni_val = _coalesce_float(tuni, 2.0)
		self.register_buffer('tuni_init_log', torch.tensor(math.log(tuni_val)))
		if learnable_tuni:
			self.log_tuni = nn.Parameter(torch.tensor(math.log(tuni_val)))
		else:
			self._tuni = tuni_val
		self.max_uniformity_samples = max_uniformity_samples
		# InfoNCE additive margin gamma; geometric threshold m = 2 * gamma on squared L2.
		self.additive_margin = _coalesce_float(additive_margin, 0.0)
		# `cosine`: L2-normalize paired vectors (DistMult/ComplEx/SimKGC).
		# `phase_residual`: element-wise squared phase residual without global normalization.
		# `sin_phase`: pRotatE link-pred term sum_i |sin(theta_q,i - theta_t,i)| (no global normalize).
		self.alignment_mode = alignment_mode or 'cosine'
		self.normalize_uniformity = normalize_uniformity
		# When True, q/t/h/ent inputs are already L2-normalized (``normalize_au_vectors`` in the model).
		self.assume_unit_norm = bool(assume_unit_norm)


def compute_loss(args):
	"""KGAU builds ``KGAULoss`` inside the strategy; loss pillar is optional."""
	del args
	return None
