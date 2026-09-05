"""Five-pillar component factory: embedder, scorer, loss, sampler, strategy."""

from importlib import import_module
import importlib.util
import inspect
import os
import sys
from types import ModuleType
from typing import Any

import torch
import torch.nn as nn

from utils.relations import (
	kbc_forward_relation_count,
	use_kbc_reciprocal_relations,
	use_reciprocal_relations,
)
from data.dataset import PointwiseDataset, load_data
from data.dict_hub import get_entity_dict, get_relation_id_map
from models.embedders.base import embedder_input_mode
from utils.device import get_model_obj


def _normalize_path(path: str) -> str:
	if path.endswith('.py'):
		path = path[:-3]
		path = path.replace('/', '.').replace('\\', '.')
	if path.startswith('./') or path.startswith('.\\'):
		path = path[2:]
	return path


def import_module_from_path(path: str) -> ModuleType:
	if not path:
		raise ValueError('Empty module path')
	module_name = _normalize_path(path)
	if module_name in sys.modules:
		return sys.modules[module_name]
	if os.path.exists(path) and path.endswith('.py'):
		try:
			return import_module(module_name)
		except ModuleNotFoundError:
			abs_path = os.path.abspath(path)
			spec = importlib.util.spec_from_file_location(module_name, abs_path)
			module = importlib.util.module_from_spec(spec)
			sys.modules[module_name] = module
			spec.loader.exec_module(module)
			return module
	return import_module(module_name)


def load_attr_from_path(path: str, attr: str) -> Any:
	module = import_module_from_path(path)
	if not hasattr(module, attr):
		raise AttributeError(f'Module {path} has no attribute {attr}')
	return getattr(module, attr)


def _config_path(args, name: str, legacy_name: str = '') -> str:
	value = getattr(args, name, '') or ''
	if value:
		return value
	if legacy_name:
		return getattr(args, legacy_name, '') or ''
	return ''


_INDEX_KGE_MODELS = frozenset({
	'distmult', 'distmult-au', 'distmult-adversarial', 'distmult-adversarial-au',
	'complex', 'complex-au', 'dabr', 'dabr-au', 'rotate', 'rotate-au', 'protate', 'protate-au',
	'transe', 'transe-au', 'transerr', 'transerr-au',
})


def is_index_kge_model(args) -> bool:
	"""Return True for lookup-table KGE models (not token-input KGE)."""

	return str(getattr(args, 'model', '') or '').lower() in _INDEX_KGE_MODELS


def _strategy_paradigm(strategy_path: str) -> str:
	path = strategy_path.replace('\\', '/').lower()
	if 'negsamp' in path or 'adversarial' in path or 'pointwise' in path:
		return 'negsamp'
	if 'kvsall' in path:
		return 'kvsall'
	if '1vsall' in path or 'softmax' in path:
		return '1vsall'
	if 'kgau' in path:
		return 'kgau'
	if 'inbatch' in path or 'contrastive' in path or 'simkgc' in path:
		return 'inbatch'
	return 'generic'


def build_entity_embedder(args) -> nn.Module:
	embedder_path = _config_path(args, 'model_embedder_path')
	if not embedder_path:
		raise ValueError('model_embedder_path is required')
	return load_attr_from_path(embedder_path, 'build_entity_embedder')(args)


def build_relation_embedder(args) -> nn.Module:
	embedder_path = _config_path(args, 'model_embedder_path')
	if not embedder_path:
		raise ValueError('model_embedder_path is required')
	return load_attr_from_path(embedder_path, 'build_relation_embedder')(args)


def build_scorer_module(args) -> nn.Module | list[nn.Module]:
	scorer_path = _config_path(args, 'model_scorer_path', 'model_encoder_path')
	if not scorer_path:
		raise ValueError('model_scorer_path is required')
	module = import_module_from_path(scorer_path)
	if hasattr(module, 'build_scorers'):
		scorers = module.build_scorers(args)
		if not isinstance(scorers, (list, tuple)):
			raise TypeError(f'{scorer_path} build_scorers must return a sequence')
		return list(scorers)
	return load_attr_from_path(scorer_path, 'build_scorer')(args)


def _resolve_model_class(scorer_path: str):
	"""Return the concrete ``*Model`` class exported by a model module."""

	from base.model import KGEModel, TextKGEModel

	module = import_module_from_path(scorer_path)
	explicit = getattr(module, 'MODEL_CLASS', None)
	if explicit is not None:
		return explicit

	candidates = []
	for name, obj in inspect.getmembers(module, inspect.isclass):
		if obj.__module__ != module.__name__:
			continue
		if not name.endswith('Model'):
			continue
		if obj in (KGEModel, TextKGEModel):
			continue
		if issubclass(obj, KGEModel):
			candidates.append(obj)
	if len(candidates) == 1:
		return candidates[0]
	if not candidates:
		raise AttributeError(f'{scorer_path} has no *Model class subclassing KGEModel')
	names = ', '.join(sorted(c.__name__ for c in candidates))
	raise AttributeError(f'{scorer_path} has multiple Model classes ({names}); set MODEL_CLASS')


def _is_token_embedder_model(args, ent_embedder: nn.Module | None = None) -> bool:
	if ent_embedder is not None:
		return embedder_input_mode(ent_embedder) == 'tokens'
	embedder_path = _config_path(args, 'model_embedder_path')
	return embedder_path.replace('\\', '/').endswith('text_embedder.py')


def bind_model(
	args,
	ent_embedder: nn.Module,
	rel_embedder: nn.Module,
	scorer: nn.Module | list[nn.Module],
	aux_embedders: dict[str, nn.Module] | None = None,
) -> nn.Module:
	"""Bind embedders and scorers into the concrete ``*Model`` from ``model_scorer_path``."""

	from base.model import TextKGEModel

	scorer_path = _config_path(args, 'model_scorer_path', 'model_encoder_path')
	model_cls = _resolve_model_class(scorer_path)
	scorers = scorer if isinstance(scorer, list) else [scorer]

	if issubclass(model_cls, TextKGEModel):
		from copy import deepcopy

		from models.embedders.text_embedder import TextQueryEmbedder

		if getattr(args, 'shared_encoder', False):
			query_embedder = TextQueryEmbedder(args, shared_encoder=ent_embedder.encoder)
		else:
			query_embedder = TextQueryEmbedder(args, shared_encoder=deepcopy(ent_embedder.encoder))
		return model_cls(ent_embedder, query_embedder, scorers=scorers, args=args)

	return model_cls(
		ent_embedder,
		rel_embedder,
		scorers=scorers,
		args=args,
		aux_embedders=aux_embedders,
	)


def _build_aux_embedders(args) -> dict[str, nn.Module] | None:
	model_name = str(getattr(args, 'model', '') or '').lower()
	if 'dabr' not in model_name:
		return None
	# Semantic-only DaBR-AU does not use the relation-drift table.
	if config_bool(args, 'dabr_au_semantic_only', False):
		return None
	if (
		config_bool(args, 'dabr_au_independent_spheres', False)
		and (
			config_bool(args, 'dabr_au_semantic_only', False)
			or config_bool(args, 'dabr_au_distance_only', False)
		)
	):
		raise ValueError(
			'dabr_au_independent_spheres cannot be combined with '
			'dabr_au_semantic_only or dabr_au_distance_only',
		)
	embedder_path = _config_path(args, 'model_embedder_path')
	aux: dict[str, nn.Module] = {
		'dr': load_attr_from_path(embedder_path, 'build_dr_embedder')(args),
	}
	# Second entity table for the distance hypersphere (semantic keeps primary ent_embedder).
	if config_bool(args, 'dabr_au_independent_spheres', False):
		aux['ent_dist'] = load_attr_from_path(embedder_path, 'build_entity_embedder')(args)
	return aux


def build_model(args) -> nn.Module:
	"""Assemble the model pillar (embedder + scorer bound by ``KGEModel`` / ``TextKGEModel``)."""

	ent_embedder = build_entity_embedder(args)
	scorer = build_scorer_module(args)
	if embedder_input_mode(ent_embedder) == 'tokens':
		return bind_model(args, ent_embedder, None, scorer, aux_embedders=_build_aux_embedders(args))
	rel_embedder = build_relation_embedder(args)
	return bind_model(args, ent_embedder, rel_embedder, scorer, aux_embedders=_build_aux_embedders(args))


def load_loss_fn(args):
	"""Load the loss pillar from ``model_loss_path`` via ``compute_loss``."""

	loss_path = getattr(args, 'model_loss_path', '') or ''
	if not loss_path:
		return None

	compute_loss = load_attr_from_path(loss_path, 'compute_loss')
	if not callable(compute_loss):
		raise TypeError(f'{loss_path} compute_loss must be callable')

	if len(inspect.signature(compute_loss).parameters) == 1:
		return compute_loss(args)
	return compute_loss


def load_sampler(args, model: nn.Module | None = None, train_triples: torch.Tensor | None = None):
	"""Load the sampler pillar when configured (1vsAll/KGAU/in-batch skip it)."""

	sampler_path = getattr(args, 'model_sampler_path', '') or ''
	if not sampler_path:
		return None

	module = import_module_from_path(sampler_path)
	build_sampler = getattr(module, 'build_sampler', None)
	if callable(build_sampler):
		try:
			return build_sampler(args, train_triples, model)
		except TypeError:
			return build_sampler(args)

	sampler_cls = getattr(module, 'FilteredSubsampler', None)
	if sampler_cls is not None and train_triples is not None:
		nentity = _resolve_nentity(args, model)
		num_neg_t = int(getattr(args, 'n_sample_t', None) or getattr(args, 'n_sample', 1))
		num_neg_h = int(getattr(args, 'n_sample_h', None) or getattr(args, 'n_sample', 1))
		if getattr(args, 'n_sample_t', None) is not None or getattr(args, 'n_sample_h', None) is not None:
			return sampler_cls(train_triples, nentity, num_neg_t, num_negatives_h=num_neg_h)
		num_neg = int(getattr(args, 'n_sample', 1))
		return sampler_cls(train_triples, nentity, num_neg)

	raise AttributeError(f'Module {sampler_path} must define build_sampler or FilteredSubsampler')


def config_float(args, name: str, default: float) -> float:
	value = getattr(args, name, None)
	return default if value is None else float(value)


def config_int(args, name: str, default: int | None = None) -> int | None:
	value = getattr(args, name, None)
	if value is None:
		return default
	return int(value)


def config_bool(args, name: str, default: bool = False) -> bool:
	value = getattr(args, name, None)
	if value is None:
		return default
	return bool(value)


def build_optimizer(args, parameters, weight_decay: float):
	from torch import optim
	from torch.optim import Adam

	lr = float(getattr(args, 'lr', getattr(args, 'learning_rate', 1e-3)))
	optim_name = str(getattr(args, 'optim', 'adam')).lower()
	if optim_name == 'adagrad':
		return optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
	if optim_name == 'sgd':
		return optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
	return Adam(parameters, lr=lr, weight_decay=weight_decay)


def build_lr_scheduler(args, optimizer):
	"""Build an optional LR scheduler for index-based KGE training."""

	from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

	name = str(getattr(args, 'lr_scheduler', '') or '').lower()
	if name in ('', 'none', 'constant'):
		return None
	if name == 'reducelronplateau':
		return ReduceLROnPlateau(
			optimizer,
			mode=str(getattr(args, 'lr_scheduler_mode', 'max')),
			factor=float(getattr(args, 'lr_scheduler_factor', 0.95)),
			patience=int(getattr(args, 'lr_scheduler_patience', 7)),
			threshold=float(getattr(args, 'lr_scheduler_threshold', 1e-4)),
		)
	if name in ('step', 'steplr', 'stepdecay'):
		step_size = max(int(getattr(args, 'lr_scheduler_step_size', 50) or 50), 1)
		return StepLR(
			optimizer,
			step_size=step_size,
			gamma=float(getattr(args, 'lr_scheduler_factor', 0.95)),
		)
	return None


def step_lr_scheduler(lr_scheduler, metric_dict: dict | None = None) -> None:
	"""Advance the LR scheduler after an epoch (metric-driven or fixed step decay)."""

	if lr_scheduler is None:
		return
	from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

	if isinstance(lr_scheduler, ReduceLROnPlateau):
		if metric_dict and 'mrr' in metric_dict:
			lr_scheduler.step(metric_dict['mrr'])
		return
	if isinstance(lr_scheduler, StepLR):
		before = lr_scheduler.get_last_lr()
		lr_scheduler.step()
		after = lr_scheduler.get_last_lr()
		if after != before:
			from utils.logger import logger
			logger.info('StepLR decay: %s -> %s', before, after)


def apply_kge_regularization(
	loss: torch.Tensor,
	model: nn.Module,
	args,
	*,
	batch_triples: torch.Tensor | None = None,
) -> torch.Tensor:
	del args  # regularization reads weights from ``model.args``
	model_obj = get_model_obj(model)
	reg_fn = getattr(model_obj, 'regularization_term', None)
	reg_term = reg_fn(batch_triples=batch_triples) if callable(reg_fn) else None
	if reg_term is None:
		return loss
	return loss + reg_term


def _eval_batch_size(args) -> int:
	eval_batch_size = getattr(args, 'eval_batch_size', None)
	if eval_batch_size is not None:
		return max(int(eval_batch_size), 1)
	test_batch_size = getattr(args, 'test_batch_size', None)
	if test_batch_size is not None:
		return max(int(test_batch_size), 1)
	return 128


def load_loss_fn_for_paradigm(args, paradigm: str):
	"""Load a loss factory from ``model_loss_path`` for negsamp/1vsall fallbacks."""

	loss_path = getattr(args, 'model_loss_path', '') or ''
	if not loss_path:
		raise ValueError(f'model_loss_path is required for {paradigm} training')

	module = import_module_from_path(loss_path)
	for builder_name in (f'build_{paradigm}_loss_fn', 'build_loss_fn'):
		builder = getattr(module, builder_name, None)
		if callable(builder):
			return builder(args)

	compute_fn = getattr(module, 'compute_loss', None)
	if callable(compute_fn):
		if len(inspect.signature(compute_fn).parameters) == 1:
			built = compute_fn(args)
			if callable(built):
				return built
		return compute_fn

	raise AttributeError(
		f'Module {loss_path} must define build_{paradigm}_loss_fn, build_loss_fn, or compute_loss'
	)


def _resolve_nentity(args, model: nn.Module | None) -> int:
	from data.dict_hub import get_entity_dict
	from utils.device import get_model_obj

	for attr in ('nentity', 'ent_total'):
		value = getattr(args, attr, None)
		if value is not None:
			return int(value)

	model_obj = get_model_obj(model) if model is not None else None
	if model_obj is not None:
		if hasattr(model_obj, 'entity_embedding'):
			return int(model_obj.entity_embedding.size(0))
		ent_embedder = getattr(model_obj, 'ent_embedder', None)
		if ent_embedder is not None:
			if hasattr(ent_embedder, 'embedding'):
				return int(ent_embedder.embedding.num_embeddings)
			if hasattr(ent_embedder, 'weight'):
				return int(ent_embedder.weight.size(0))
		if hasattr(model_obj, 'ent_embeddings') and hasattr(model_obj.ent_embeddings, 'embedding'):
			return int(model_obj.ent_embeddings.embedding.num_embeddings)

	return len(get_entity_dict())


def init_index_kge_trainer(trainer, model: nn.Module, args) -> None:
	"""Attach shared evaluation/checkpoint state to index-based KGE strategies."""

	from base.evaluator import Evaluator
	from utils.memory import PhaseMemoryTracker

	trainer.model = model
	trainer.encoder = model
	trainer.args = args
	trainer.entity_dict = get_entity_dict()
	trainer.entity_ids = [ex.entity_id for ex in trainer.entity_dict.entity_exs]
	trainer.evaluator = Evaluator(args)
	trainer.best_metric = None
	trainer.best_checkpoint_path = None
	trainer.train_time = 0.0
	trainer.valid_time = 0.0
	trainer.total_time = 0.0
	trainer.memory_tracker = PhaseMemoryTracker()
	trainer._cached_valid_exs = None
	trainer._cached_valid_backward_exs = None

	if torch.cuda.is_available():
		trainer.model.cuda()
	trainer.device = next(trainer.model.parameters()).device


def _kge_validation_interval(args) -> int:
	# Prefer epoch_per_eval; accept legacy eval_every_epoch used by older DaBR configs.
	raw = getattr(args, 'epoch_per_eval', None)
	if raw is None:
		raw = getattr(args, 'eval_every_epoch', None)
	if raw is not None:
		interval = int(raw)
		if interval <= 0:
			return max(int(getattr(args, 'epochs', 1)), 1)
		if interval > int(getattr(args, 'epochs', 1)):
			return max(int(getattr(args, 'epochs', 1)), 1)
		return interval
	# Step-based runs iterate epochs only for shuffling; avoid implicit every-epoch eval.
	from utils.training_cadence import resolve_valid_steps, uses_step_cadence

	if uses_step_cadence(args) and resolve_valid_steps(args) is not None:
		return max(int(getattr(args, 'epochs', 1)), 1)
	return 1


def _kge_should_validate(args, epoch: int) -> bool:
	interval = _kge_validation_interval(args)
	epoch_number = epoch + 1
	max_epochs = max(int(getattr(args, 'epochs', 1)), 1)
	return epoch_number % interval == 0 or epoch_number >= max_epochs


def _kge_should_validate_at_epoch_end(args, epoch: int, *, stopping: bool = False) -> bool:
	"""Epoch-boundary validation during step-based training (honors ``epoch_per_eval`` only)."""

	if stopping:
		return True
	raw = getattr(args, 'epoch_per_eval', None)
	if raw is None:
		raw = getattr(args, 'eval_every_epoch', None)
	if raw is None:
		return False
	interval = int(raw)
	if interval <= 0:
		return False
	return (epoch + 1) % interval == 0


def _kge_get_valid_examples(trainer):
	from data.dataset import Example, load_data, reverse_triplet
	from utils.device import get_model_obj
	from utils.eval_modes import uses_forward_examples_for_backward_eval

	if trainer._cached_valid_exs is not None:
		return trainer._cached_valid_exs, trainer._cached_valid_backward_exs

	valid_path = getattr(trainer.args, 'valid_path', '')
	valid_exs = load_data(valid_path, add_forward_triplet=True, add_backward_triplet=False)
	if uses_forward_examples_for_backward_eval(trainer.args):
		valid_backward_exs = valid_exs
	else:
		valid_backward_exs = [
			Example(**reverse_triplet({
				'head_id': ex.head_id,
				'head': ex.head,
				'relation': ex.relation,
				'tail_id': ex.tail_id,
				'tail': ex.tail,
			}))
			for ex in valid_exs
		]
	trainer._cached_valid_exs = valid_exs
	trainer._cached_valid_backward_exs = valid_backward_exs
	return valid_exs, valid_backward_exs


def _kge_resolve_link_prediction_path(path: str) -> str:
	if not path:
		return path
	if path.endswith('.json'):
		parent_dir = os.path.dirname(os.path.dirname(path))
		candidate = os.path.join(parent_dir, os.path.basename(path)[:-5])
		if os.path.exists(candidate):
			return candidate
		candidate = os.path.join(os.path.dirname(path), os.path.basename(path)[:-5])
		if os.path.exists(candidate):
			return candidate
	return path


def _kge_resolve_monitor_metric(args) -> str:
	return str(getattr(args, 'valid_metric', None) or 'mrr')


def _kge_normalize_metric_name(metric_name: str) -> str:
	return str(metric_name or '').lower().replace('-', '_')


def _kge_is_scorer_valid_metric(metric_name: str) -> bool:
	"""True when valid_metric names an LP scorer (cosine / original / lp_distance_l*)."""

	name = _kge_normalize_metric_name(metric_name)
	if name in {'cosine', 'original', 'native', 'lp_distance', 'distance', 'lp', 'l_distance'}:
		return True
	return name.startswith('lp_distance_l')


def _kge_resolve_valid_lp_score_mode(args) -> str | None:
	"""Map scorer-style valid_metric to an lp_score_mode override, else None."""

	name = _kge_normalize_metric_name(getattr(args, 'valid_metric', None))
	if not name or not _kge_is_scorer_valid_metric(name):
		return None
	if name == 'cosine':
		return 'cosine'
	if name in {'original', 'native'}:
		return 'original'
	return 'lp_distance'


def _kge_resolve_valid_distance_degree(args) -> float | None:
	"""Parse Lp degree from valid_metric like lp_distance_l2; else None (use config default)."""

	name = _kge_normalize_metric_name(getattr(args, 'valid_metric', None))
	if not name.startswith('lp_distance_l'):
		return None
	suffix = name[len('lp_distance_l'):]
	if not suffix:
		return None
	try:
		degree = float(suffix)
	except ValueError:
		return None
	if degree <= 0:
		raise ValueError(f'Invalid lp distance degree in valid_metric: {name}')
	return degree


def _kge_valid_scorer_label(args, score_mode: str | None = None) -> str:
	"""Human-readable scorer label for validation logs."""

	mode = score_mode or _kge_resolve_valid_lp_score_mode(args)
	if mode == 'lp_distance':
		from base.evaluator import _lp_distance_label

		degree = _kge_resolve_valid_distance_degree(args)
		if degree is not None:
			if float(degree).is_integer():
				return f'lp_distance_l{int(degree)}'
			return f'lp_distance_l{degree:g}'
		return _lp_distance_label(args)
	if mode:
		return mode
	monitor = _kge_normalize_metric_name(_kge_resolve_monitor_metric(args))
	if _kge_is_scorer_valid_metric(monitor):
		return monitor
	return ''


def _kge_metric_value(metric_dict: dict | None, metric_name: str):
	if not metric_dict:
		return None
	if metric_name in metric_dict:
		value = metric_dict[metric_name]
		# Nested dual-scorer metrics: monitor MRR under that scorer.
		if isinstance(value, dict):
			return value.get('mrr')
		return value
	aliases = {
		'hit@10': ('hits@10',),
		'hits@10': ('hit@10',),
		'hit@3': ('hits@3',),
		'hits@3': ('hit@3',),
		'hit@1': ('hits@1',),
		'hits@1': ('hit@1',),
	}
	for alt in aliases.get(metric_name, ()):
		if alt in metric_dict:
			return metric_dict[alt]
	# Scorer-style valid_metric on a flat single-scorer validation dict.
	if _kge_is_scorer_valid_metric(metric_name):
		name = _kge_normalize_metric_name(metric_name)
		for key, value in metric_dict.items():
			if isinstance(value, dict) and _kge_normalize_metric_name(key) == name:
				return value.get('mrr')
		if name.startswith('lp_distance'):
			for key, value in metric_dict.items():
				if isinstance(value, dict) and _kge_normalize_metric_name(key).startswith('lp_distance'):
					return value.get('mrr')
		return metric_dict.get('mrr')
	return None


def _kge_extract_monitor_value(metric_dict: dict, train_loss: float | None = None, args=None):
	monitor_name = _kge_resolve_monitor_metric(args) if args is not None else 'mrr'
	value = _kge_metric_value(metric_dict, monitor_name)
	if value is not None:
		return value
	if metric_dict and 'mrr' in metric_dict:
		return metric_dict['mrr']
	if metric_dict and 'loss' in metric_dict:
		return -metric_dict['loss']
	if train_loss is not None:
		return -float(train_loss)
	for key, value in (metric_dict or {}).items():
		if isinstance(value, (int, float)):
			return value
	return None


def eval_index_kge_epoch(trainer, epoch: int, *, step: int | None = None) -> dict:
	"""Run validation link prediction for index-based KGE strategies."""

	from contextlib import nullcontext

	from base.evaluator import log_bidirectional_link_metrics, lp_score_mode_context

	trainer.model.eval()
	if hasattr(trainer, 'optimizer') and trainer.optimizer is not None:
		trainer.optimizer.zero_grad(set_to_none=True)
	if torch.cuda.is_available():
		torch.cuda.empty_cache()

	metric_dict = {}
	valid_path = getattr(trainer.args, 'valid_path', '')
	valid_eval_path = _kge_resolve_link_prediction_path(valid_path)
	if not valid_eval_path or not os.path.exists(valid_eval_path):
		return metric_dict

	valid_exs, valid_backward_exs = _kge_get_valid_examples(trainer)
	valid_output_path = os.path.join(trainer.args.output_dir, 'valid_link_prediction.log')
	eval_batch_size = _eval_batch_size(trainer.args)
	score_mode = _kge_resolve_valid_lp_score_mode(trainer.args)
	distance_degree = _kge_resolve_valid_distance_degree(trainer.args)
	scorer_label = _kge_valid_scorer_label(trainer.args, score_mode)
	score_ctx = (
		lp_score_mode_context(trainer.model, score_mode, distance_degree)
		if score_mode
		else nullcontext()
	)
	with score_ctx:
		forward_metrics = trainer.evaluator.evaluate_link_prediction_inplace(
			trainer.model, valid_eval_path, trainer.entity_dict, valid_output_path,
			batch_size=eval_batch_size, eval_forward=True, examples=valid_exs,
		)
		backward_metrics = trainer.evaluator.evaluate_link_prediction_inplace(
			trainer.model, valid_eval_path, trainer.entity_dict, valid_output_path,
			batch_size=eval_batch_size, eval_forward=False, examples=valid_backward_exs,
		)
	if forward_metrics or backward_metrics:
		prefix = f'[STEP {step}] Valid' if step is not None else f'[EPOCH {epoch + 1}] Valid'
		scope_label = f'{prefix} ({scorer_label} scorer)' if scorer_label else prefix
		metric_dict.update(
			log_bidirectional_link_metrics(scope_label, forward_metrics, backward_metrics)
		)
	return metric_dict


def _save_index_kge_checkpoint(
	trainer,
	epoch: int,
	is_best: bool,
	*,
	step: int | None = None,
) -> str:
	from utils.checkpoint import best_model_path, checkpoint_path, delete_old_ckt, last_model_path, save_checkpoint
	from utils.device import get_model_obj

	filename = (
		checkpoint_path(trainer.args.output_dir, epoch, step)
		if step is not None
		else last_model_path(trainer.args.output_dir)
	)
	saved_checkpoint_path = save_checkpoint(
		{
			'epoch': epoch,
			'step': step,
			'best_epoch': epoch if is_best else None,
			'best_metric': trainer.best_metric,
			'args': trainer.args.__dict__,
			'state_dict': get_model_obj(trainer.model).state_dict(),
		},
		is_best=is_best,
		filename=filename,
	)
	if is_best:
		trainer.best_checkpoint_path = best_model_path(trainer.args.output_dir)
	elif trainer.best_checkpoint_path is None:
		trainer.best_checkpoint_path = saved_checkpoint_path
	delete_old_ckt(
		path_pattern='{}/checkpoint_*.mdl'.format(trainer.args.output_dir),
		keep=getattr(trainer.args, 'max_to_keep', 5),
	)
	return saved_checkpoint_path


def _kge_train_loop_result(trainer) -> dict:
	from utils.memory import format_memory
	from utils.logger import logger, log_run_timing

	num_train_epochs = getattr(trainer, 'num_train_epochs', None)
	epoch_time = log_run_timing(
		train_time=trainer.train_time,
		valid_time=trainer.valid_time,
		total_time=trainer.total_time,
		num_train_epochs=num_train_epochs,
	)
	logger.info('[Memory] Training peak: %s', format_memory(trainer.memory_tracker.train_peak_mb))
	logger.info('[Memory] Eval peak: %s', format_memory(trainer.memory_tracker.eval_peak_mb))
	logger.info('[Memory] Peak memory: %s', format_memory(trainer.memory_tracker.peak_memory_mb))
	return {
		'best_epoch': None if trainer.best_metric is None else trainer.best_metric.get('epoch', 0) + 1,
		'best_step': None if trainer.best_metric is None else trainer.best_metric.get('step'),
		'best_mrr': None if trainer.best_metric is None else trainer.best_metric.get('score'),
		'best_monitor_metric': None if trainer.best_metric is None else trainer.best_metric.get('metric'),
		'best_monitor_score': None if trainer.best_metric is None else trainer.best_metric.get('score'),
		'train_time': trainer.train_time,
		'valid_time': trainer.valid_time,
		'total_time': trainer.total_time,
		'num_train_epochs': num_train_epochs,
		'time_per_train_epoch': epoch_time,
		'best_checkpoint_path': trainer.best_checkpoint_path,
		**trainer.memory_tracker.to_dict(),
	}


def _update_index_kge_best_metric(trainer, metric_dict: dict, epoch: int, step: int | None = None) -> bool:
	monitor_name = _kge_resolve_monitor_metric(trainer.args)
	monitor_value = _kge_metric_value(metric_dict, monitor_name)
	if monitor_value is None:
		return False
	is_best = trainer.best_metric is None or monitor_value > trainer.best_metric.get('score', float('-inf'))
	if is_best:
		payload = {
			'score': monitor_value,
			'metric': monitor_name,
			'metrics': metric_dict,
			'epoch': epoch,
		}
		if step is not None:
			payload['step'] = step
		trainer.best_metric = payload
	return is_best


def on_index_kge_training_step(trainer, epoch: int) -> int:
	"""Advance global step counter and run step-cadence validation, checkpoints, and LR decay."""

	import time

	from utils.training_cadence import (
		increment_trainer_global_step,
		maybe_decay_lr_at_step,
		should_save_checkpoint_at_step,
		should_stop_at_step,
		should_validate_at_step,
	)

	step = increment_trainer_global_step(trainer)
	maybe_decay_lr_at_step(trainer, step)

	is_best = False
	if should_validate_at_step(step, trainer.args):
		eval_start = time.time()
		trainer.memory_tracker.begin_phase()
		metric_dict = eval_index_kge_epoch(trainer, epoch, step=step)
		trainer.memory_tracker.end_phase('eval')
		trainer.valid_time += time.time() - eval_start
		is_best = _update_index_kge_best_metric(trainer, metric_dict, epoch, step=step)

	if should_save_checkpoint_at_step(step, trainer.args) or is_best:
		_save_index_kge_checkpoint(trainer, epoch, is_best, step=step)

	if should_stop_at_step(trainer, step):
		trainer._stop_training = True

	return step


def _kge_resolve_early_stopping(args) -> tuple[int | None, int, Any]:
	patience = config_int(args, 'early_stopping_patience', None)
	min_epochs = config_int(args, 'early_stopping_min_epochs', None) or 0
	min_metric = getattr(args, 'early_stopping_min_metric', None)
	if patience is not None and patience <= 0:
		patience = None
	return patience, min_epochs, min_metric


def _kge_update_early_stopping_bad_count(
	trainer,
	*,
	metric_dict: dict,
	is_best: bool,
	bad_counts: int,
	min_metric,
) -> int:
	monitor_name = _kge_resolve_monitor_metric(trainer.args)
	if _kge_metric_value(metric_dict, monitor_name) is None:
		return bad_counts
	if is_best:
		return 0
	best_score = None if trainer.best_metric is None else trainer.best_metric.get('score')
	if min_metric is None or (best_score is not None and best_score >= float(min_metric)):
		return bad_counts + 1
	return bad_counts


def run_epoch_based_kge_train_loop(trainer, dataloader=None) -> dict:
	"""Epoch-driven training with optional early stopping for index KGE."""

	import time

	from utils.logger import logger

	total_start = time.time()
	max_epochs = max(int(getattr(trainer.args, 'epochs', 1)), 1)
	patience, min_epochs, min_metric = _kge_resolve_early_stopping(trainer.args)
	bad_counts = 0

	for epoch in range(max_epochs):
		train_start = time.time()
		trainer.memory_tracker.begin_phase()
		train_loss = trainer.train_epoch(dataloader, epoch)
		trainer.memory_tracker.end_phase('train')
		trainer.train_time += time.time() - train_start

		validated = _kge_should_validate(trainer.args, epoch)
		metric_dict = {}
		if validated:
			eval_start = time.time()
			trainer.memory_tracker.begin_phase()
			metric_dict = eval_index_kge_epoch(trainer, epoch)
			trainer.memory_tracker.end_phase('eval')
			trainer.valid_time += time.time() - eval_start

		is_best = False
		if validated and metric_dict:
			monitor_name = _kge_resolve_monitor_metric(trainer.args)
			if _kge_metric_value(metric_dict, monitor_name) is not None:
				is_best = _update_index_kge_best_metric(trainer, metric_dict, epoch)
				bad_counts = _kge_update_early_stopping_bad_count(
					trainer,
					metric_dict=metric_dict,
					is_best=is_best,
					bad_counts=bad_counts,
					min_metric=min_metric,
				)

		_save_index_kge_checkpoint(trainer, epoch, is_best)

		lr_scheduler = getattr(trainer, 'lr_scheduler', None)
		step_lr_scheduler(lr_scheduler, metric_dict)

		if patience is not None and bad_counts >= patience and (epoch + 1) >= min_epochs:
			logger.info(
				'[EARLY STOP] No validation %s improvement for %d evaluations.',
				_kge_resolve_monitor_metric(trainer.args),
				patience,
			)
			break

	trainer.num_train_epochs = epoch + 1
	trainer.total_time = time.time() - total_start
	return _kge_train_loop_result(trainer)


def run_step_based_kge_train_loop(trainer, dataloader=None) -> dict:
	"""Step-driven training (RotatE-style max_steps / valid_steps / save_checkpoint_steps)."""

	import time

	from utils.logger import logger
	from utils.training_cadence import (
		get_trainer_global_step,
		init_step_cadence_state,
		resolve_max_steps,
		should_stop_at_step,
		trainer_supports_step_batches,
		uses_step_cadence,
	)

	if not uses_step_cadence(trainer.args):
		raise ValueError('run_step_based_kge_train_loop requires step-based training cadence')
	if not trainer_supports_step_batches(trainer):
		raise ValueError(
			f'{type(trainer).__name__} does not implement iter_training_batches/train_batch '
			'required for step-based training'
		)

	init_step_cadence_state(trainer)
	max_steps = resolve_max_steps(trainer.args)
	if max_steps is None:
		raise ValueError('step-based training requires max_steps')

	total_start = time.time()
	epoch = 0
	max_epochs = max(int(getattr(trainer.args, 'epochs', 1)), 1)
	patience, min_epochs, min_metric = _kge_resolve_early_stopping(trainer.args)
	bad_counts = 0

	while get_trainer_global_step(trainer) < max_steps and epoch < max_epochs:
		if getattr(trainer, '_stop_training', False):
			break

		train_start = time.time()
		trainer.memory_tracker.begin_phase()
		loss_total = 0.0
		batch_count = 0
		for batch in trainer.iter_training_batches(epoch, dataloader):
			if get_trainer_global_step(trainer) >= max_steps or getattr(trainer, '_stop_training', False):
				break
			loss = trainer.train_batch(batch, epoch)
			loss_total += float(loss)
			batch_count += 1
			step = on_index_kge_training_step(trainer, epoch)
			if should_stop_at_step(trainer, step) or getattr(trainer, '_stop_training', False):
				break

		trainer.memory_tracker.end_phase('train')
		trainer.train_time += time.time() - train_start
		current_step = get_trainer_global_step(trainer)
		if batch_count > 0:
			logger.info(
				'[EPOCH %s] Train | Loss: %.4f | Step: %s',
				epoch + 1,
				loss_total / batch_count,
				current_step,
			)

		stopping = should_stop_at_step(trainer, current_step)
		scheduled_epoch_eval = _kge_should_validate_at_epoch_end(trainer.args, epoch, stopping=False)
		validated = _kge_should_validate_at_epoch_end(trainer.args, epoch, stopping=stopping)
		metric_dict = {}
		is_best = False
		if validated:
			eval_start = time.time()
			trainer.memory_tracker.begin_phase()
			metric_dict = eval_index_kge_epoch(trainer, epoch)
			trainer.memory_tracker.end_phase('eval')
			trainer.valid_time += time.time() - eval_start
			is_best = _update_index_kge_best_metric(trainer, metric_dict, epoch, step=current_step)
			if scheduled_epoch_eval:
				bad_counts = _kge_update_early_stopping_bad_count(
					trainer,
					metric_dict=metric_dict,
					is_best=is_best,
					bad_counts=bad_counts,
					min_metric=min_metric,
				)

		if batch_count > 0:
			_save_index_kge_checkpoint(trainer, epoch, is_best, step=current_step)

		if (
			not stopping
			and patience is not None
			and bad_counts >= patience
			and (epoch + 1) >= min_epochs
		):
			logger.info('[EARLY STOP] No validation MRR improvement for %d evaluations.', patience)
			break

		if stopping:
			logger.info('[STOP] Reached max_steps=%d at epoch %d', max_steps, epoch + 1)
			break
		epoch += 1

	if trainer.best_checkpoint_path is None and get_trainer_global_step(trainer) > 0:
		_save_index_kge_checkpoint(
			trainer,
			max(epoch, 0),
			is_best=False,
			step=get_trainer_global_step(trainer),
		)

	trainer.num_train_epochs = epoch + 1
	trainer.total_time = time.time() - total_start
	return _kge_train_loop_result(trainer)


def run_index_kge_train_loop(trainer, dataloader=None) -> dict:
	"""Dispatch epoch-based or step-based training for index KGE strategies."""

	from utils.training_cadence import uses_step_cadence

	if uses_step_cadence(trainer.args):
		return run_step_based_kge_train_loop(trainer, dataloader)
	return run_epoch_based_kge_train_loop(trainer, dataloader)


def run_kge_train_loop(trainer) -> dict:
	"""Shared epoch shell for in-batch / contrastive trainers (SimKGC-style cadence)."""

	import time

	from utils.logger import logger, log_run_timing
	from utils.memory import format_memory
	from utils.training_cadence import get_trainer_global_step, resolve_max_steps, uses_step_cadence

	if getattr(trainer.args, 'use_amp', False) and not hasattr(trainer, 'scaler'):
		trainer.scaler = torch.cuda.amp.GradScaler()

	total_start = time.time()
	max_steps = resolve_max_steps(trainer.args) if uses_step_cadence(trainer.args) else None
	num_train_epochs = 0
	for epoch in range(trainer.args.epochs):
		epoch_train_start = time.time()
		trainer.memory_tracker.begin_phase()
		trainer.train_epoch(epoch)
		trainer.memory_tracker.end_phase('train')
		trainer.train_time += time.time() - epoch_train_start
		num_train_epochs = epoch + 1
		if max_steps is None:
			if _kge_should_validate(trainer.args, epoch):
				trainer._run_eval(epoch=epoch)
		elif getattr(trainer, '_stop_training', False) or get_trainer_global_step(trainer) >= max_steps:
			break

	if max_steps is not None and getattr(trainer, '_stop_training', False):
		logger.info('[STOP] Reached max_steps=%d', max_steps)

	trainer.num_train_epochs = num_train_epochs
	trainer.total_time = time.time() - total_start
	epoch_time = log_run_timing(
		train_time=trainer.train_time,
		valid_time=trainer.valid_time,
		total_time=trainer.total_time,
		num_train_epochs=num_train_epochs,
	)
	logger.info('[Memory] Training peak: %s', format_memory(trainer.memory_tracker.train_peak_mb))
	logger.info('[Memory] Eval peak: %s', format_memory(trainer.memory_tracker.eval_peak_mb))
	logger.info('[Memory] Peak memory: %s', format_memory(trainer.memory_tracker.peak_memory_mb))

	return {
		'best_epoch': None if trainer.best_metric is None else trainer.best_metric.get('epoch', 0) + 1,
		'best_mrr': None if trainer.best_metric is None else trainer.best_metric.get('score'),
		'best_monitor_metric': None if trainer.best_metric is None else trainer.best_metric.get('metric'),
		'best_monitor_score': None if trainer.best_metric is None else trainer.best_metric.get('score'),
		'train_time': trainer.train_time,
		'valid_time': trainer.valid_time,
		'total_time': trainer.total_time,
		'num_train_epochs': num_train_epochs,
		'time_per_train_epoch': epoch_time,
		'best_checkpoint_path': trainer.best_checkpoint_path,
		**trainer.memory_tracker.to_dict(),
	}


def load_strategy_class(args):
	strategy_path = getattr(args, 'model_strategy_path', '') or ''
	for attr in ('Strategy', 'NegSampStrategy', 'OneVsAllStrategy', 'KvsAllStrategy', 'KGAUStrategy', 'InBatchStrategy', 'SimKGCStrategy'):
		try:
			cls = load_attr_from_path(strategy_path, attr)
		except AttributeError:
			continue
		if isinstance(cls, type):
			return cls
	raise AttributeError(f'Could not find Strategy class in {strategy_path}')


def _resolve_relation_index(relation: str, relation_to_idx: dict) -> int:
	from utils.relations import resolve_relation_index

	return resolve_relation_index(relation, relation_to_idx)


def _relation_index_map(model: nn.Module | None) -> dict[str, int]:
	if model is not None:
		rel_map = getattr(model, 'rel_to_idx', None)
		if rel_map:
			return rel_map
	return get_relation_id_map()


def _load_train_examples(args, model: nn.Module | None = None):
	del model
	add_backward = use_reciprocal_relations(args) and not use_kbc_reciprocal_relations(args)
	strategy_path = (getattr(args, 'model_strategy_path', '') or '').replace('\\', '/').lower()
	if 'kvsall' in strategy_path:
		from models.strategies.kvsall_strategy import kvsall_uses_rt_training
		if kvsall_uses_rt_training(args):
			add_backward = False
	return load_data(args.train_path, add_forward_triplet=True, add_backward_triplet=add_backward)


def _examples_to_tensors(examples, entity_dict, relation_to_idx):
	head_indices = torch.tensor([entity_dict.entity_to_idx(example.head_id) for example in examples], dtype=torch.long)
	relation_indices = torch.tensor(
		[_resolve_relation_index(example.relation, relation_to_idx) for example in examples],
		dtype=torch.long,
	)
	tail_indices = torch.tensor([entity_dict.entity_to_idx(example.tail_id) for example in examples], dtype=torch.long)
	return head_indices, relation_indices, tail_indices


def _augment_kbc_1vsall_train_tensors(
	src: torch.Tensor,
	rel: torch.Tensor,
	dst: torch.Tensor,
	relation_to_idx: dict[str, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	"""Stack forward triples with swapped (t, r + n_forward, h) copies (kbc ``get_train``)."""

	n_forward = kbc_forward_relation_count(relation_to_idx)
	if n_forward <= 0:
		return src, rel, dst
	return (
		torch.cat([src, dst]),
		torch.cat([rel, rel + n_forward]),
		torch.cat([dst, src]),
	)


def _prepare_1vsall_train_data(
	args,
	model: nn.Module | None,
	train_examples,
	entity_dict,
	relation_to_idx: dict[str, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	src, rel, dst = _examples_to_tensors(train_examples, entity_dict, relation_to_idx)
	if use_kbc_reciprocal_relations(args):
		src, rel, dst = _augment_kbc_1vsall_train_tensors(src, rel, dst, relation_to_idx)
	return src, rel, dst


def _prepare_train_triples(args, model: nn.Module) -> torch.Tensor:
	entity_dict = get_entity_dict()
	relation_to_idx = _relation_index_map(model)
	train_examples = _load_train_examples(args, model)
	if not train_examples:
		raise ValueError(f'No training examples loaded from {args.train_path}')
	src, rel, dst = _examples_to_tensors(train_examples, entity_dict, relation_to_idx)
	return torch.stack([src, rel, dst], dim=-1)


def _prepare_pointwise_dataloader(args):
	from data.dataloader import collate_pointwise
	from utils.openke_batch_sampling import (
		resolve_openke_batch_size,
		resolve_openke_n_batches,
		uses_openke_batch_sampling,
	)

	train_examples = _load_train_examples(args)
	train_dataset = PointwiseDataset(train_examples)
	num_examples = len(train_examples)
	batch_size = resolve_openke_batch_size(num_examples, args)
	args.batch_size = batch_size
	loader_kwargs = {
		'collate_fn': collate_pointwise,
		'num_workers': getattr(args, 'workers', 0),
		'pin_memory': torch.cuda.is_available(),
	}
	# OpenKE / DaBR: each positive is drawn independently with replacement
	# (``i = rand_max(id, trainTotal)``), for a fixed number of batches per epoch.
	if uses_openke_batch_sampling(args):
		n_batches = resolve_openke_n_batches(num_examples, batch_size, args)
		num_samples = batch_size * n_batches
		sampler = torch.utils.data.RandomSampler(
			train_dataset,
			replacement=True,
			num_samples=num_samples,
		)
		batch_sampler = torch.utils.data.BatchSampler(
			sampler,
			batch_size=batch_size,
			drop_last=True,
		)
		return torch.utils.data.DataLoader(
			train_dataset,
			batch_sampler=batch_sampler,
			**loader_kwargs,
		)
	return torch.utils.data.DataLoader(
		train_dataset,
		batch_size=batch_size,
		shuffle=True,
		drop_last=True,
		**loader_kwargs,
	)


def _strategy_init_kwargs(args, strategy_path: str, model: nn.Module, train_triples: torch.Tensor | None):
	paradigm = _strategy_paradigm(strategy_path)
	kwargs: dict[str, Any] = {}

	if paradigm == 'negsamp':
		sampler_path = (getattr(args, 'model_sampler_path', '') or '').replace('\\', '/')
		if sampler_path.endswith('uniform_pointwise_sampler.py'):
			kwargs['train_dataloader'] = _prepare_pointwise_dataloader(args)
		elif sampler_path.endswith('filtered_1_to_n_sampler.py'):
			from models.samplers.filtered_1_to_n_sampler import build_filtered_negsamp_dataloaders

			triples = train_triples if train_triples is not None else _prepare_train_triples(args, model)
			kwargs['train_triples'] = triples
			kwargs['train_dataloader'] = build_filtered_negsamp_dataloaders(
				args, triples, _resolve_nentity(args, model),
			)
		else:
			kwargs['train_triples'] = train_triples if train_triples is not None else _prepare_train_triples(args, model)
	elif paradigm == 'kvsall':
		kwargs['train_triples'] = train_triples if train_triples is not None else _prepare_train_triples(args, model)
	elif paradigm == '1vsall':
		entity_dict = get_entity_dict()
		train_examples = _load_train_examples(args, model)
		rel_map = _relation_index_map(model)
		kwargs['train_data'] = _prepare_1vsall_train_data(
			args, model, train_examples, entity_dict, rel_map,
		)

	return kwargs


def build_pipeline(args, ngpus_per_node: int = 1):
	"""Wire all five pillars: embedder, scorer, loss, sampler, strategy."""

	strategy_path = getattr(args, 'model_strategy_path', '') or ''
	paradigm = _strategy_paradigm(strategy_path)
	StrategyClass = load_strategy_class(args)

	ent_embedder = build_entity_embedder(args)
	scorer = build_scorer_module(args)
	if embedder_input_mode(ent_embedder) == 'tokens':
		model = bind_model(args, ent_embedder, None, scorer, aux_embedders=_build_aux_embedders(args))
	else:
		rel_embedder = build_relation_embedder(args)
		model = bind_model(args, ent_embedder, rel_embedder, scorer, aux_embedders=_build_aux_embedders(args))
	if torch.cuda.is_available():
		model.cuda()

	if paradigm in ('negsamp', 'kvsall', '1vsall', 'inbatch'):
		loss_fn = load_loss_fn_for_paradigm(args, paradigm)
	else:
		loss_fn = load_loss_fn(args)
	train_triples = (
		_prepare_train_triples(args, model)
		if paradigm in ('negsamp', 'kvsall')
		or (paradigm == 'kgau' and config_bool(args, 'au_hybrid_adversarial_bce', False))
		else None
	)
	if paradigm in ('kvsall', '1vsall'):
		sampler = None
	elif paradigm == 'kgau':
		sampler = (
			load_sampler(args, model, train_triples)
			if config_bool(args, 'au_hybrid_adversarial_bce', False)
			else None
		)
	elif paradigm == 'inbatch':
		sampler = load_sampler(args)
	else:
		sampler = load_sampler(args, model, train_triples)
	strategy_kwargs = _strategy_init_kwargs(args, strategy_path, model, train_triples)

	trainer = StrategyClass(model, sampler, loss_fn, args, ngpus_per_node=ngpus_per_node, **strategy_kwargs)
	return trainer
