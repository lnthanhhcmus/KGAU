"""Relation vocabulary helpers.

These are dataset/config concerns, not ``KGEModel`` methods: they run before a model
exists (builder, dict hub, samplers) and mutate relation→index maps.
"""

from __future__ import annotations

import json
import os
from typing import Any

import torch

from data.dataset import load_data


def _relation_path_candidates(args) -> list[str]:
	paths = []
	for source_path in [getattr(args, 'train_path', ''), getattr(args, 'valid_path', ''), getattr(args, 'test_path', '')]:
		if not source_path:
			continue
		paths.append(os.path.join(os.path.dirname(source_path), 'relation2id.json'))
		paths.append(os.path.join(os.path.dirname(source_path), 'relations.json'))
		paths.append(os.path.join(os.path.dirname(source_path), 'relation2idx.json'))
	paths.append(os.path.join('data', getattr(args, 'dataset', ''), 'relation2id.json'))
	paths.append(os.path.join('data', getattr(args, 'dataset', ''), 'preprocessed', 'relation2id.json'))
	return paths


def use_kbc_reciprocal_relations(args: Any | None) -> bool:
	"""KGDirectAU / kbc-style reciprocal layout (inverse id = fwd + n_forward)."""

	return bool(getattr(args, 'kbc_reciprocal_relations', False))


def use_reciprocal_relations(args: Any | None) -> bool:
	"""True when reciprocal relations (``add_reciprocal_relations`` or kbc layout) are enabled."""

	return bool(getattr(args, 'add_reciprocal_relations', False)) or use_kbc_reciprocal_relations(args)


def kbc_forward_relation_count(relation_to_idx: dict[str, int]) -> int:
	"""Number of forward relation slots (kbc ``n_predicates // 2`` after doubling)."""

	forward_values = [
		int(value)
		for key, value in relation_to_idx.items()
		if not str(key).startswith('inverse ')
	]
	if not forward_values:
		return 0
	return max(forward_values) + 1


def add_kbc_inverse_relations(relation_to_idx: dict[str, int]) -> dict[str, int]:
	"""Assign inverse relation id = forward_id + n_forward (kbc reciprocal layout)."""

	updated = dict(relation_to_idx)
	n_forward = kbc_forward_relation_count(updated)
	for relation, idx in list(updated.items()):
		if str(relation).startswith('inverse '):
			continue
		inverse_relation = f'inverse {relation}'
		if inverse_relation not in updated:
			updated[inverse_relation] = int(idx) + n_forward
	return updated


def add_inverse_relations(relation_to_idx: dict[str, int]) -> dict[str, int]:
	"""Assign each forward relation a distinct inverse-relation ID (next free index)."""

	updated = dict(relation_to_idx)
	next_idx = max(updated.values(), default=-1) + 1
	for relation in list(updated.keys()):
		if relation.startswith('inverse '):
			continue
		inverse_relation = f'inverse {relation}'
		if inverse_relation not in updated:
			updated[inverse_relation] = next_idx
			next_idx += 1
	return updated


def build_forward_to_inverse_index_tensor(relation_to_idx: dict[str, int]) -> torch.Tensor | None:
	"""Map forward relation indices to inverse indices (KvsAll ``_rt`` / kbc CE head eval)."""

	if not relation_to_idx:
		return None
	max_idx = max(int(value) for value in relation_to_idx.values())
	mapping = torch.full((max_idx + 1,), -1, dtype=torch.long)
	for relation, fwd_idx in relation_to_idx.items():
		if str(relation).startswith('inverse '):
			continue
		inv_idx = relation_to_idx.get(f'inverse {relation}')
		if inv_idx is not None:
			mapping[int(fwd_idx)] = int(inv_idx)
	n_forward = kbc_forward_relation_count(relation_to_idx)
	if n_forward > 0:
		for fwd_idx in range(n_forward):
			if int(mapping[fwd_idx]) < 0:
				inv_idx = int(fwd_idx) + n_forward
				if inv_idx <= max_idx:
					mapping[fwd_idx] = inv_idx
	if int(mapping.ge(0).sum()) == 0:
		return None
	return mapping


def _apply_relation_display_aliases(relation_to_idx: dict[str, int], args) -> dict[str, int]:
	"""Add human-readable relation aliases (FB15k-237 path IDs -> display strings)."""

	dataset = str(getattr(args, 'dataset', '') or '').lower()
	if dataset != 'fb15k237':
		return relation_to_idx

	from data.preprocess import _normalize_fb15k237_relation

	updated = dict(relation_to_idx)
	for relation, idx in relation_to_idx.items():
		relation_str = str(relation)
		if relation_str.startswith('inverse '):
			base_id = relation_str[len('inverse '):]
			if not base_id.startswith('/'):
				continue
			display = _normalize_fb15k237_relation(base_id)
			updated[f'inverse {display}'] = idx
			continue
		if not relation_str.startswith('/'):
			continue
		display = _normalize_fb15k237_relation(relation_str)
		updated[display] = idx
		normalized = ' '.join(display.split())
		if normalized != display:
			updated[normalized] = idx
	return updated


def load_relation_to_idx(args) -> dict[str, int]:
	"""Load relation→index map, optionally doubling for reciprocal training."""

	for path in _relation_path_candidates(args):
		if not path or not os.path.exists(path):
			continue
		with open(path, 'r', encoding='utf-8') as handle:
			mapping = json.load(handle)
		if isinstance(mapping, dict):
			relation_to_idx = {str(key): int(value) for key, value in mapping.items()}
			if use_kbc_reciprocal_relations(args):
				relation_to_idx = add_kbc_inverse_relations(relation_to_idx)
			elif use_reciprocal_relations(args):
				relation_to_idx = add_inverse_relations(relation_to_idx)
			return _apply_relation_display_aliases(relation_to_idx, args)

	relations: list[str] = []
	seen: set[str] = set()
	for example in load_data(getattr(args, 'train_path', ''), add_forward_triplet=False, add_backward_triplet=False):
		if example.relation not in seen:
			seen.add(example.relation)
			relations.append(example.relation)
	relation_to_idx = {relation: idx for idx, relation in enumerate(relations)}
	if use_kbc_reciprocal_relations(args):
		relation_to_idx = add_kbc_inverse_relations(relation_to_idx)
	elif use_reciprocal_relations(args):
		relation_to_idx = add_inverse_relations(relation_to_idx)
	return _apply_relation_display_aliases(relation_to_idx, args)


def resolve_relation_index(relation: str, relation_to_idx: dict[str, int]) -> int:
	"""Resolve a relation string to an index (normalized / inverse fallbacks)."""

	if relation in relation_to_idx:
		return relation_to_idx[relation]
	normalized = ' '.join(relation.split())
	if normalized in relation_to_idx:
		return relation_to_idx[normalized]
	if relation.startswith('inverse '):
		base_relation = relation[len('inverse '):]
		inverse_relation = f'inverse {base_relation}'
		if inverse_relation in relation_to_idx:
			return relation_to_idx[inverse_relation]
		if base_relation in relation_to_idx:
			return relation_to_idx[base_relation]
	raise KeyError(relation)


def as_index_tensor(values, lookup, device: torch.device) -> torch.Tensor:
	"""Convert string IDs (or an existing tensor) to a long index tensor."""

	if torch.is_tensor(values):
		return values.to(device=device, dtype=torch.long)
	return torch.tensor([lookup(value) for value in values], dtype=torch.long, device=device)
