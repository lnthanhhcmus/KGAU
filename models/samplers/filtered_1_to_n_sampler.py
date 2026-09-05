"""Filtered 1-N negative sampler adapted from RotatE / GB-Magic TrainDataset.

Provides:
- ``FilteredSubsampler``: in-process batch sampling (hybrid / fallback)
- ``FilteredNegSampDataset`` + workerized DataLoaders so CPU filtering overlaps GPU
"""

from __future__ import annotations

import os
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class FilteredSubsampler:
	"""Filtered 1-N negative sampler for RotatE-style training."""

	def __init__(self, triples, nentity: int, num_negatives: int, num_negatives_h: int | None = None):
		self.nentity = int(nentity)
		self.num_negatives_tail = int(num_negatives)
		self.num_negatives_head = int(num_negatives if num_negatives_h is None else num_negatives_h)
		self.count = self._count_frequency(triples)
		self.true_head, self.true_tail = self._build_filter_dicts(triples)

	@staticmethod
	def _normalize_triple(triple) -> tuple[int, int, int]:
		"""Normalize a triple to (head, relation, tail) format and convert to integers."""

		if torch.is_tensor(triple):
			triple = triple.detach().cpu().tolist()
		return int(triple[0]), int(triple[1]), int(triple[2])

	@staticmethod
	def _count_frequency(triples, start: int = 4) -> dict[tuple[int, int], int]:
		"""Count the frequency of (head, relation) and (tail, -relation-1) pairs in the training triples."""

		count = {}
		for triple in triples:
			head, relation, tail = FilteredSubsampler._normalize_triple(triple)
			if (head, relation) not in count:
				count[(head, relation)] = start
			else:
				count[(head, relation)] += 1

			if (tail, -relation - 1) not in count:
				count[(tail, -relation - 1)] = start
			else:
				count[(tail, -relation - 1)] += 1
		return count

	@staticmethod
	def _build_filter_dicts(triples) -> tuple[dict[tuple[int, int], np.ndarray], dict[tuple[int, int], np.ndarray]]:
		"""Build dictionaries mapping (head, relation) to true tails and (relation, tail) to true heads for filtering."""

		true_head = {}
		true_tail = {}

		for triple in triples:
			head, relation, tail = FilteredSubsampler._normalize_triple(triple)
			if (head, relation) not in true_tail:
				true_tail[(head, relation)] = []
			true_tail[(head, relation)].append(tail)

			if (relation, tail) not in true_head:
				true_head[(relation, tail)] = []
			true_head[(relation, tail)].append(head)

		for relation, tail in true_head:
			true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
		for head, relation in true_tail:
			true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))

		return true_head, true_tail

	def _ensure_tensor_triples(self, batch_triples) -> torch.Tensor:
		"""Convert batch triples to a tensor of shape [B, 3] if they are not already tensors."""

		if torch.is_tensor(batch_triples):
			return batch_triples.long()
		if isinstance(batch_triples, dict):
			if {"head_id", "relation", "tail_id"}.issubset(batch_triples.keys()):
				return torch.stack(
					[
						batch_triples["head_id"].long(),
						batch_triples["relation"].long(),
						batch_triples["tail_id"].long(),
					],
					dim=-1,
				)
		return torch.tensor([self._normalize_triple(t) for t in batch_triples], dtype=torch.long)

	def _subsampling_weights(self, head: np.ndarray, relation: np.ndarray, tail: np.ndarray) -> torch.Tensor:
		"""Compute sqrt-inverse-frequency subsampling weights for a batch."""

		weights = np.empty(head.shape[0], dtype=np.float64)
		for idx, (h, r, t) in enumerate(zip(head, relation, tail)):
			weights[idx] = self.count.get((int(h), int(r)), 4) + self.count.get((int(t), -int(r) - 1), 4)
		return torch.from_numpy(np.sqrt(1.0 / weights)).float()

	def _sample_filtered_negatives_row(
		self,
		key: tuple[int, int],
		filter_dict: dict,
		oversample: int,
		*,
		num_negatives: int,
	) -> np.ndarray:
		"""Draw filtered negatives for one row, resampling only when the first pass is short."""

		blocked = filter_dict.get(key)
		if blocked is None or blocked.size == 0:
			return np.random.randint(self.nentity, size=num_negatives, dtype=np.int64)

		pool_size = max(oversample, num_negatives + blocked.size, num_negatives * 2)
		candidates = np.random.randint(self.nentity, size=pool_size, dtype=np.int64)
		valid = candidates[~np.isin(candidates, blocked, assume_unique=True)]
		if valid.size >= num_negatives:
			return valid[:num_negatives]

		collected = [valid] if valid.size else []
		remaining = num_negatives - sum(part.size for part in collected)
		attempts = 0
		while remaining > 0 and attempts < 3:
			candidate = np.random.randint(self.nentity, size=max(oversample, remaining * 2), dtype=np.int64)
			extra = candidate[~np.isin(candidate, blocked, assume_unique=True)]
			if extra.size:
				collected.append(extra)
				remaining -= extra.size
			attempts += 1
		if not collected:
			return np.random.randint(self.nentity, size=num_negatives, dtype=np.int64)
		return np.concatenate(collected)[:num_negatives]

	def sample(self, batch_triples, mode: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
		"""Sample filtered negatives and subsampling weights for a batch.

		Returns: positive_sample [B,3], negative_sample [B,num_neg], subsampling_weight [B]
		"""

		positive_sample = self._ensure_tensor_triples(batch_triples)
		batch_size = positive_sample.size(0)

		if positive_sample.is_cuda:
			head = positive_sample[:, 0].detach().cpu().numpy()
			relation = positive_sample[:, 1].detach().cpu().numpy()
			tail = positive_sample[:, 2].detach().cpu().numpy()
		else:
			head = positive_sample[:, 0].numpy()
			relation = positive_sample[:, 1].numpy()
			tail = positive_sample[:, 2].numpy()

		subsampling_weight = self._subsampling_weights(head, relation, tail)
		if mode == "head-batch":
			num_negatives = self.num_negatives_head
		elif mode == "tail-batch":
			num_negatives = self.num_negatives_tail
		else:
			raise ValueError(f"Training batch mode {mode} not supported")
		oversample = max(num_negatives * 2, 64)
		negative_sample = np.empty((batch_size, num_negatives), dtype=np.int64)
		if mode == "head-batch":
			for i, (r, t) in enumerate(zip(relation, tail)):
				negative_sample[i] = self._sample_filtered_negatives_row(
					(int(r), int(t)), self.true_head, oversample, num_negatives=num_negatives
				)
		elif mode == "tail-batch":
			for i, (h, r) in enumerate(zip(head, relation)):
				negative_sample[i] = self._sample_filtered_negatives_row(
					(int(h), int(r)), self.true_tail, oversample, num_negatives=num_negatives
				)
		else:
			raise ValueError(f"Training batch mode {mode} not supported")

		return positive_sample, torch.from_numpy(negative_sample).long(), subsampling_weight, mode


class FilteredNegSampDataset(Dataset):
	"""Per-triple filtered NegSamp dataset (GB-Magic ``TrainDataset``); worker-safe."""

	def __init__(
		self,
		triples,
		nentity: int,
		negative_sample_size: int,
		mode: str,
		*,
		count: dict[tuple[int, int], int] | None = None,
		true_head: dict | None = None,
		true_tail: dict | None = None,
	):
		if mode not in ("head-batch", "tail-batch"):
			raise ValueError(f"Training batch mode {mode} not supported")
		if torch.is_tensor(triples):
			triples = triples.detach().cpu().tolist()
		self.triples = [FilteredSubsampler._normalize_triple(t) for t in triples]
		self.nentity = int(nentity)
		self.negative_sample_size = int(negative_sample_size)
		self.mode = mode
		if count is None or true_head is None or true_tail is None:
			self.count = FilteredSubsampler._count_frequency(self.triples)
			self.true_head, self.true_tail = FilteredSubsampler._build_filter_dicts(self.triples)
		else:
			self.count = count
			self.true_head = true_head
			self.true_tail = true_tail

	def __len__(self) -> int:
		return len(self.triples)

	def __getitem__(self, idx: int):
		head, relation, tail = self.triples[idx]
		weight = self.count.get((head, relation), 4) + self.count.get((tail, -relation - 1), 4)
		subsampling_weight = torch.sqrt(torch.tensor([1.0 / weight], dtype=torch.float32))

		num_neg = self.negative_sample_size
		negative_sample_list = []
		negative_sample_size = 0
		blocked = (
			self.true_head.get((relation, tail))
			if self.mode == "head-batch"
			else self.true_tail.get((head, relation))
		)
		if blocked is None:
			blocked = np.empty(0, dtype=np.int64)

		while negative_sample_size < num_neg:
			candidates = np.random.randint(self.nentity, size=num_neg * 2)
			mask = np.isin(candidates, blocked, invert=True)
			valid = candidates[mask]
			negative_sample_list.append(valid)
			negative_sample_size += valid.size

		negative_sample = np.concatenate(negative_sample_list)[:num_neg]
		return (
			torch.tensor([head, relation, tail], dtype=torch.long),
			torch.from_numpy(np.asarray(negative_sample, dtype=np.int64)).long(),
			subsampling_weight,
			self.mode,
		)

	@staticmethod
	def collate_fn(data):
		positive_sample = torch.stack([row[0] for row in data], dim=0)
		negative_sample = torch.stack([row[1] for row in data], dim=0)
		subsampling_weight = torch.cat([row[2] for row in data], dim=0)
		mode = data[0][3]
		return positive_sample, negative_sample, subsampling_weight, mode


def _suggested_max_workers() -> int:
	try:
		return len(os.sched_getaffinity(0))
	except (AttributeError, OSError):
		return os.cpu_count() or 1


def resolve_negsamp_num_workers(args, num_loaders: int = 2) -> int:
	"""Clamp DataLoader workers (GB-Magic ``resolve_num_workers``). Default request = 4."""

	raw = getattr(args, "workers", None)
	requested = 4 if raw is None else int(raw)
	suggested = _suggested_max_workers()
	budget = max(0, suggested // max(int(num_loaders), 1))
	if requested <= 0:
		return 0
	return min(requested, budget if budget > 0 else requested)


def build_filtered_negsamp_dataloaders(args, train_triples, nentity: int) -> dict[str, DataLoader]:
	"""Build head/tail DataLoaders that sample filtered negatives in worker processes."""

	n_sample_t = getattr(args, "n_sample_t", None)
	n_sample_h = getattr(args, "n_sample_h", None)
	n_sample = getattr(args, "n_sample", None)
	num_neg_t = int(n_sample_t if n_sample_t is not None else (n_sample or 1))
	num_neg_h = int(n_sample_h if n_sample_h is not None else (n_sample or 1))

	if torch.is_tensor(train_triples):
		triple_list = train_triples.detach().cpu().tolist()
	else:
		triple_list = train_triples
	count = FilteredSubsampler._count_frequency(triple_list)
	true_head, true_tail = FilteredSubsampler._build_filter_dicts(triple_list)

	batch_size = max(int(getattr(args, "batch_size", 1024) or 1024), 1)
	num_workers = resolve_negsamp_num_workers(args, num_loaders=2)
	loader_kwargs = {
		"batch_size": batch_size,
		"shuffle": bool(getattr(args, "shuffle_train", True)),
		"num_workers": num_workers,
		"pin_memory": torch.cuda.is_available(),
		"collate_fn": FilteredNegSampDataset.collate_fn,
		"drop_last": False,
	}
	if num_workers > 0:
		loader_kwargs["persistent_workers"] = True
		loader_kwargs["prefetch_factor"] = 2

	head_ds = FilteredNegSampDataset(
		triple_list,
		int(nentity),
		num_neg_h,
		"head-batch",
		count=count,
		true_head=true_head,
		true_tail=true_tail,
	)
	tail_ds = FilteredNegSampDataset(
		triple_list,
		int(nentity),
		num_neg_t,
		"tail-batch",
		count=count,
		true_head=true_head,
		true_tail=true_tail,
	)
	return {
		"head": DataLoader(head_ds, **loader_kwargs),
		"tail": DataLoader(tail_ds, **loader_kwargs),
	}


def iter_bidirectional_presampled_batches(
	head_loader: DataLoader,
	tail_loader: DataLoader,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]]:
	"""Alternate tail/head batches for one finite epoch (same order as NegSampStrategy)."""

	head_batches = iter(head_loader)
	tail_batches = iter(tail_loader)
	head_done = False
	tail_done = False
	step = 0
	while not (head_done and tail_done):
		step += 1
		if step % 2 == 0:
			if not head_done:
				try:
					yield next(head_batches)
					continue
				except StopIteration:
					head_done = True
			if not tail_done:
				try:
					yield next(tail_batches)
				except StopIteration:
					tail_done = True
		else:
			if not tail_done:
				try:
					yield next(tail_batches)
					continue
				except StopIteration:
					tail_done = True
			if not head_done:
				try:
					yield next(head_batches)
				except StopIteration:
					head_done = True


def build_sampler(args, train_triples, model):
	"""Construct a filtered 1-N subsampler for adversarial RotatE-style training."""

	from models.builder import _resolve_nentity

	nentity = _resolve_nentity(args, model)
	n_sample_t = getattr(args, "n_sample_t", None)
	n_sample_h = getattr(args, "n_sample_h", None)
	n_sample = getattr(args, "n_sample", None)
	if n_sample_t is not None or n_sample_h is not None:
		num_neg_t = int(n_sample_t if n_sample_t is not None else (n_sample or 1))
		num_neg_h = int(n_sample_h if n_sample_h is not None else (n_sample or 1))
		return FilteredSubsampler(train_triples, int(nentity), num_neg_t, num_negatives_h=num_neg_h)
	num_neg = int(n_sample or 1)
	return FilteredSubsampler(train_triples, int(nentity), num_neg)
