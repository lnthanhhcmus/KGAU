"""Head / backward link-prediction eval mode routing (KGDirectAU-specific).
"""

from __future__ import annotations

from typing import Any

from utils.relations import use_kbc_reciprocal_relations, use_reciprocal_relations

HEAD_EVAL_MODES = frozenset({'rt_forward', 'rt_inverse', 'hr_inverse'})


def resolve_head_eval_mode(args: Any | None, *, eval_forward: bool) -> str:
	"""Choose backward link-prediction scoring to match the training recipe.

	Returns one of ``tail``, ``rt_forward``, ``rt_inverse``, or ``hr_inverse``.

	When ``args.head_eval_mode`` is set in JSON/CLI, it overrides strategy-based
	inference for the backward (head) pass. Use ``rt_forward`` for adversarial
	BCE / direct head prediction with the forward relation (no inverse relation).
	KGAU with reciprocal relations trains inverse triplets ``(t, r^{-1}, h)`` as
	tail prediction, so backward eval defaults to ``hr_inverse``.
	"""

	if eval_forward:
		return 'tail'
	if args is None:
		return 'rt_forward'

	explicit = getattr(args, 'head_eval_mode', None)
	if explicit is not None:
		mode = str(explicit).strip().lower()
		if mode in {'auto', 'infer', 'default', ''}:
			pass
		elif mode in HEAD_EVAL_MODES:
			return mode
		else:
			raise ValueError(
				f'Unsupported head_eval_mode={explicit!r}; '
				f'expected one of {sorted(HEAD_EVAL_MODES)} or auto'
			)

	strategy = (getattr(args, 'model_strategy_path', '') or '').replace('\\', '/').lower()
	loss_path = (getattr(args, 'model_loss_path', '') or '').replace('\\', '/').lower()

	if 'negsamp' in strategy or 'adversarial_bce' in loss_path:
		return 'rt_forward'

	if 'kvsall' in strategy:
		from models.strategies.kvsall_strategy import kvsall_uses_rt_training

		if kvsall_uses_rt_training(args) and use_reciprocal_relations(args):
			return 'rt_inverse'
		return 'rt_forward'

	if '1vsall' in strategy:
		if use_kbc_reciprocal_relations(args) and not bool(getattr(args, 'bidirectional_1vsall', True)):
			return 'hr_inverse'
		return 'rt_forward'

	if 'kgau' in strategy:
		if bool(getattr(args, 'kgau_bidirectional', False)):
			return 'rt_forward'
		if use_reciprocal_relations(args):
			return 'hr_inverse'
		return 'rt_forward'

	return 'rt_forward'


def uses_forward_examples_for_backward_eval(args: Any | None) -> bool:
	"""Backward eval always starts from forward (h, r, t) test triples for index KGE."""

	return resolve_head_eval_mode(args, eval_forward=False) != 'tail'
