"""Capability markers for embedder implementations."""

from typing import Literal

InputMode = Literal['indices', 'tokens']


def embedder_input_mode(embedder) -> InputMode:
	"""Return how training batches are formed for this embedder."""

	return getattr(embedder, 'input_mode', 'indices')
