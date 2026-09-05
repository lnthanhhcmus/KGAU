"""Universal step-based vs epoch-based training cadence helpers."""

from typing import Any


def config_int(args: Any, name: str, default: int | None = None) -> int | None:
	value = getattr(args, name, None)
	if value is None:
		return default
	return int(value)


def config_float(args: Any, name: str, default: float) -> float:
	value = getattr(args, name, None)
	return default if value is None else float(value)


def uses_step_cadence(args: Any) -> bool:
	"""Return True when training should follow optimizer steps instead of epoch boundaries."""

	cadence = getattr(args, 'training_cadence', None)
	if cadence is not None:
		return str(cadence).lower() == 'step'
	return config_int(args, 'max_steps', None) is not None


def resolve_max_steps(args: Any) -> int | None:
	return config_int(args, 'max_steps', None)


def resolve_valid_steps(args: Any) -> int | None:
	for key in ('valid_steps', 'eval_every_n_step'):
		value = config_int(args, key, None)
		if value is not None and value > 0:
			return value
	return None


def resolve_save_checkpoint_steps(args: Any) -> int | None:
	value = config_int(args, 'save_checkpoint_steps', None)
	if value is not None and value > 0:
		return value
	return None


def resolve_warm_up_steps(args: Any) -> int | None:
	explicit = config_int(args, 'warm_up_steps', None)
	if explicit is not None:
		return explicit
	max_steps = resolve_max_steps(args)
	if max_steps is not None:
		return max_steps // 2
	return None


def _resolve_warm_up_ratio_values(args: Any) -> list[float]:
	"""Return sorted warm-up ratios in (0, 1), or an empty list when unset/invalid."""

	raw = getattr(args, 'warm_up_ratio', None)
	if raw is None:
		return []
	if isinstance(raw, (float, int, str)):
		raw = [raw]

	ratios: list[float] = []
	for value in raw:
		try:
			ratio = float(value)
		except (TypeError, ValueError):
			continue
		if 0.0 < ratio < 1.0:
			ratios.append(ratio)

	if not ratios:
		return []

	# Keep deterministic order and remove duplicates after float parsing.
	return sorted(set(ratios))


def _ceil_batches(num_items: int, batch_size: int) -> int:
	return max(1, (int(num_items) + int(batch_size) - 1) // int(batch_size))


def estimate_steps_per_epoch(trainer: Any) -> int | None:
	"""Estimate optimizer steps in one training epoch from trainer state."""

	args = getattr(trainer, 'args', None)
	if args is None:
		return None

	batch_size = max(config_int(args, 'batch_size', 1024) or 1024, 1)

	# Prefer DataLoader length when present (covers OpenKE RandomSampler+BatchSampler).
	for loader_attr in ('train_dataloader', 'train_loader'):
		loader = getattr(trainer, loader_attr, None)
		if loader is not None:
			try:
				return max(len(loader), 1)
			except TypeError:
				pass

	from utils.openke_batch_sampling import (
		resolve_openke_batch_size,
		resolve_openke_n_batches,
		uses_openke_batch_sampling,
	)

	if uses_openke_batch_sampling(args):
		train_src = getattr(trainer, 'train_src', None)
		if train_src is not None:
			num_triples = int(train_src.size(0))
		else:
			train_triples = getattr(trainer, 'train_triples', None)
			if train_triples is not None:
				num_triples = int(train_triples.size(0))
			else:
				train_examples = getattr(trainer, 'train_examples', None)
				num_triples = len(train_examples) if train_examples is not None else 0
		if num_triples > 0:
			openke_bs = resolve_openke_batch_size(num_triples, args)
			n_batches = resolve_openke_n_batches(num_triples, openke_bs, args)
			bidirectional = bool(getattr(trainer, 'kgau_bidirectional', False))
			return n_batches * (2 if bidirectional else 1)

	train_triples = getattr(trainer, 'train_triples', None)
	if train_triples is not None:
		num_triples = int(train_triples.size(0))
		bidirectional = not bool(getattr(trainer, '_pointwise_mode', False))
		return _ceil_batches(num_triples, batch_size) * (2 if bidirectional else 1)

	train_src = getattr(trainer, 'train_src', None)
	if train_src is not None:
		num_triples = int(train_src.size(0))
		bidirectional = bool(getattr(trainer, 'kgau_bidirectional', False))
		return _ceil_batches(num_triples, batch_size) * (2 if bidirectional else 1)

	train_examples = getattr(trainer, 'train_examples', None)
	if train_examples is not None:
		bidirectional = bool(getattr(trainer, 'kgau_bidirectional', False))
		return _ceil_batches(len(train_examples), batch_size) * (2 if bidirectional else 1)

	return None


def resolve_training_budget_steps(
	args: Any,
	*,
	steps_per_epoch: int | None = None,
) -> int | None:
	"""Resolve total optimizer-step budget for ratio-based LR decay.

	Step cadence uses ``max_steps``. Epoch cadence uses ``epochs * steps_per_epoch``
	when the per-epoch step count is known.
	"""

	max_steps = resolve_max_steps(args)
	if max_steps is not None and max_steps > 0:
		return max_steps

	epochs = config_int(args, 'epochs', None)
	if (
		epochs is not None
		and epochs > 0
		and steps_per_epoch is not None
		and steps_per_epoch > 0
	):
		return epochs * steps_per_epoch

	return None


def resolve_warm_up_ratio_steps(
	args: Any,
	*,
	budget_steps: int | None = None,
	steps_per_epoch: int | None = None,
) -> list[int]:
	"""Resolve ratio-based LR decay steps from ``warm_up_ratio`` and training budget.

	Works for both cadences:
	- step: ``warm_up_ratio`` × ``max_steps``
	- epoch: ``warm_up_ratio`` × ``epochs * steps_per_epoch``

	Example: ``warm_up_ratio=[0.2, 0.5, 0.8]`` with ``max_steps=100000``
	produces ``[20000, 50000, 80000]``.
	"""

	if budget_steps is None:
		budget_steps = resolve_training_budget_steps(args, steps_per_epoch=steps_per_epoch)
	if budget_steps is None or budget_steps <= 0:
		return []

	ratios = _resolve_warm_up_ratio_values(args)
	if not ratios:
		return []

	steps: list[int] = []
	for ratio in ratios:
		step = max(1, int(budget_steps * ratio))
		if step >= budget_steps:
			continue
		if not steps or step > steps[-1]:
			steps.append(step)

	return steps


def resolve_lr_decay_factor(args: Any) -> float:
	return config_float(args, 'lr_decay_factor', 0.1)


def should_validate_at_step(step: int, args: Any) -> bool:
	interval = resolve_valid_steps(args)
	return interval is not None and step > 0 and step % interval == 0


def should_save_checkpoint_at_step(step: int, args: Any) -> bool:
	interval = resolve_save_checkpoint_steps(args)
	return interval is not None and step > 0 and step % interval == 0


def trainer_supports_step_batches(trainer: Any) -> bool:
	return (
		callable(getattr(trainer, 'iter_training_batches', None))
		and callable(getattr(trainer, 'train_batch', None))
	)


def get_trainer_global_step(trainer: Any) -> int:
	return int(getattr(trainer, 'global_step', 0) or 0)


def init_step_cadence_state(trainer: Any) -> None:
	"""Attach step-cadence fields to any strategy trainer before the step loop runs."""

	if not hasattr(trainer, 'global_step'):
		trainer.global_step = 0

	steps_per_epoch = estimate_steps_per_epoch(trainer)
	trainer.steps_per_epoch = steps_per_epoch
	budget_steps = resolve_training_budget_steps(
		trainer.args,
		steps_per_epoch=steps_per_epoch,
	)
	trainer.lr_decay_steps = resolve_warm_up_ratio_steps(
		trainer.args,
		budget_steps=budget_steps,
		steps_per_epoch=steps_per_epoch,
	)
	trainer.lr_decay_step_index = 0
	if trainer.lr_decay_steps:
		trainer.next_lr_decay_step = trainer.lr_decay_steps[0]
	else:
		trainer.next_lr_decay_step = resolve_warm_up_steps(trainer.args)
	trainer.lr_decay_factor = resolve_lr_decay_factor(trainer.args)
	trainer._stop_training = False

	if trainer.lr_decay_steps:
		from utils.logger import logger

		cadence = str(getattr(trainer.args, 'training_cadence', '') or '').lower()
		if cadence == 'step' or resolve_max_steps(trainer.args) is not None:
			budget_label = f'max_steps={budget_steps}'
		elif steps_per_epoch is not None and budget_steps is not None:
			epochs = config_int(trainer.args, 'epochs', None)
			budget_label = (
				f'epochs={epochs} x steps_per_epoch={steps_per_epoch} '
				f'= budget_steps={budget_steps}'
			)
		else:
			budget_label = f'budget_steps={budget_steps}'
		logger.info(
			'LR decay schedule from warm_up_ratio %s at steps %s (%s)',
			getattr(trainer.args, 'warm_up_ratio', None),
			trainer.lr_decay_steps,
			budget_label,
		)


def increment_trainer_global_step(trainer: Any) -> int:
	trainer.global_step = get_trainer_global_step(trainer) + 1
	return trainer.global_step


def _rebuild_trainer_optimizer(trainer: Any, new_lr: float) -> None:
	"""Recreate the optimizer at a new LR (legacy KnowledgeGraphEmbedding warm-up decay)."""

	model = getattr(trainer, 'model', None)
	args = getattr(trainer, 'args', None)
	if model is None or args is None:
		return

	from models.builder import build_optimizer

	weight_decay = config_float(args, 'weight_decay', 0.0)
	params = filter(lambda p: p.requires_grad, model.parameters())
	trainer.optimizer = build_optimizer(args, params, weight_decay)
	for param_group in trainer.optimizer.param_groups:
		param_group['lr'] = new_lr


def _scale_optimizer_lrs(optimizer, scale: float) -> None:
	"""Multiply every param-group LR by ``scale`` (GB-Magic / RotatE-style decay)."""

	for group in optimizer.param_groups:
		group['lr'] = float(group['lr']) * scale


def maybe_decay_lr_at_step(trainer: Any, step: int) -> None:
	"""Apply LR decays at configured milestone steps.

	Priority:
	1) ratio milestones from ``warm_up_ratio`` + training budget (``max_steps`` or
	   ``epochs * steps_per_epoch``)
	2) legacy single ``warm_up_steps`` then geometric ``*3`` schedule
	"""

	next_step = getattr(trainer, 'next_lr_decay_step', None)
	optimizer = getattr(trainer, 'optimizer', None)
	if next_step is None or optimizer is None or step < int(next_step):
		return

	decay_factor = max(float(getattr(trainer, 'lr_decay_factor', 0.1)), 0.0)
	args = getattr(trainer, 'args', None)
	preserve = bool(getattr(args, 'lr_decay_preserve_optimizer', False)) if args is not None else False
	if preserve:
		_scale_optimizer_lrs(optimizer, decay_factor)
		new_lr = float(optimizer.param_groups[0]['lr'])
	else:
		new_lr = float(optimizer.param_groups[0]['lr']) * decay_factor
		_rebuild_trainer_optimizer(trainer, new_lr)

	from utils.logger import logger

	logger.info('Change learning rate to %.8f at step %d', new_lr, step)

	decay_steps = getattr(trainer, 'lr_decay_steps', None) or []
	if decay_steps:
		decay_idx = int(getattr(trainer, 'lr_decay_step_index', 0) or 0) + 1
		trainer.lr_decay_step_index = decay_idx
		trainer.next_lr_decay_step = decay_steps[decay_idx] if decay_idx < len(decay_steps) else None
		return

	trainer.next_lr_decay_step = int(next_step * 3)


def should_stop_at_step(trainer: Any, step: int) -> bool:
	max_steps = resolve_max_steps(trainer.args)
	if max_steps is not None and step >= max_steps:
		return True
	return bool(getattr(trainer, '_stop_training', False))
