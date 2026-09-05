"""OpenKE / DaBR-style with-replacement positive batch sampling.

Matches ``getBatch`` in OpenKE ``Base.cpp``: each positive slot is drawn as
``i = rand_max(id, trainTotal)`` independently (with replacement).
"""

from __future__ import annotations

from typing import Any, Iterator

import torch


def is_dabr_model(args: Any) -> bool:
	"""True when the configured model is DaBR or DaBR-AU."""

	model = str(getattr(args, 'model', '') or '').lower()
	scorer_path = str(
		getattr(args, 'model_scorer_path', '')
		or getattr(args, 'model_encoder_path', '')
		or ''
	).lower()
	return 'dabr' in model or 'dabr' in scorer_path


def uses_openke_batch_sampling(args: Any) -> bool:
	"""Whether training should sample positives with replacement (OpenKE DaBR).

	Explicit ``openke_batch_sampling`` wins; otherwise defaults on for DaBR models.
	"""

	value = getattr(args, 'openke_batch_sampling', None)
	if value is not None:
		return bool(value)
	return is_dabr_model(args)


def resolve_openke_batch_size(num_examples: int, args: Any) -> int:
	"""Resolve batch size; if ``n_batches`` is set, use OpenKE ``N // nbatches``."""

	n_batches = getattr(args, 'n_batches', None)
	if n_batches:
		return max(int(num_examples) // int(n_batches), 1)
	return max(int(getattr(args, 'batch_size', 1) or 1), 1)


def resolve_openke_n_batches(num_examples: int, batch_size: int, args: Any) -> int:
	"""Batches per epoch: explicit ``n_batches``, else ``max(N // batch_size, 1)``."""

	n_batches = getattr(args, 'n_batches', None)
	if n_batches:
		return max(int(n_batches), 1)
	return max(int(num_examples) // max(int(batch_size), 1), 1)


def sample_openke_indices(
	num_examples: int,
	batch_size: int,
	*,
	device: torch.device | None = None,
	generator: torch.Generator | None = None,
) -> torch.Tensor:
	"""Draw ``batch_size`` train indices uniformly with replacement."""

	if num_examples <= 0:
		raise ValueError('Cannot sample OpenKE batches from an empty training set')
	kwargs: dict[str, Any] = {}
	if generator is not None:
		kwargs['generator'] = generator
	indices = torch.randint(0, int(num_examples), (int(batch_size),), **kwargs)
	if device is not None:
		indices = indices.to(device)
	return indices


def iter_openke_index_batches(
	num_examples: int,
	batch_size: int,
	n_batches: int,
	*,
	device: torch.device | None = None,
	generator: torch.Generator | None = None,
) -> Iterator[torch.Tensor]:
	"""Yield ``n_batches`` index tensors of length ``batch_size`` (with replacement)."""

	for _ in range(int(n_batches)):
		yield sample_openke_indices(
			num_examples,
			batch_size,
			device=device,
			generator=generator,
		)


def iter_openke_triple_batches(
	src: torch.Tensor,
	rel: torch.Tensor,
	dst: torch.Tensor,
	batch_size: int,
	n_batches: int,
	*,
	generator: torch.Generator | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
	"""Yield OpenKE-style (src, rel, dst) mini-batches with replacement."""

	num_examples = int(src.size(0))
	for indices in iter_openke_index_batches(
		num_examples,
		batch_size,
		n_batches,
		device=src.device,
		generator=generator,
	):
		yield src.index_select(0, indices), rel.index_select(0, indices), dst.index_select(0, indices)
