"""Abstract evaluation loop shared by KG evaluators."""

from contextlib import contextmanager
from typing import List, Optional, Sequence, Tuple
import inspect
import os
from types import SimpleNamespace

import torch
import tqdm

from base.model import KGEModel, TextKGEModel
from utils.logger import logger
from utils.device import get_model_obj, move_to_cuda

from data.dict_hub import get_all_triplet_dict, get_entity_dict
from data.dataset import Example, load_data
from data.dataloader import collate
from metrics.ranking import ranking_metrics_from_ranks, ranks_from_score_matrix
from metrics.classification import classification_metrics, find_global_threshold
from models.losses.bce_loss import bce_logit_offset, uses_logit_classification_scores

from utils.eval_modes import resolve_head_eval_mode, uses_forward_examples_for_backward_eval
from utils.relations import build_forward_to_inverse_index_tensor
from configs.config import args as global_args
from data.dict_hub import build_tokenizer
from models.builder import import_module_from_path, is_index_kge_model, load_attr_from_path
from utils.checkpoint import load_state_dict_clean, load_checkpoint, best_model_path, checkpoint_path


@contextmanager
def normalize_lp_scores_context(model, normalize: bool):
	"""Temporarily set link-prediction scoring mode (cosine vs native scorer)."""

	model_obj = get_model_obj(model)
	if not hasattr(model_obj, 'normalize_lp_scores'):
		yield
		return
	previous = bool(model_obj.normalize_lp_scores)
	model_obj.normalize_lp_scores = bool(normalize)
	try:
		yield
	finally:
		model_obj.normalize_lp_scores = previous


@contextmanager
def lp_score_mode_context(model, mode: str, distance_degree: float | None = None):
	"""Temporarily set link-prediction scoring mode for evaluation."""

	model_obj = get_model_obj(model)
	if not hasattr(model_obj, 'lp_score_mode'):
		yield
		return
	previous_mode = getattr(model_obj, 'lp_score_mode', 'original')
	previous_normalize = bool(getattr(model_obj, 'normalize_lp_scores', False))
	previous_degree = getattr(model_obj, 'lp_distance_degree', None)
	normalized_mode = str(mode).lower().replace('-', '_')
	if normalized_mode in {'distance', 'l_distance', 'lp'}:
		normalized_mode = 'lp_distance'
	model_obj.lp_score_mode = normalized_mode
	model_obj.normalize_lp_scores = normalized_mode == 'cosine'
	if distance_degree is not None and hasattr(model_obj, 'lp_distance_degree'):
		model_obj.lp_distance_degree = float(distance_degree)
	try:
		yield
	finally:
		model_obj.lp_score_mode = previous_mode
		model_obj.normalize_lp_scores = previous_normalize
		if previous_degree is not None and hasattr(model_obj, 'lp_distance_degree'):
			model_obj.lp_distance_degree = previous_degree


def _lp_distance_degree(args=None) -> float:
	from base.model import KGEModel

	return KGEModel.resolve_lp_distance_degree(args if args is not None else global_args)


def _lp_distance_label(args=None) -> str:
	degree = _lp_distance_degree(args)
	if float(degree).is_integer():
		return f'lp_distance_l{int(degree)}'
	return f'lp_distance_l{degree:g}'


def _score_query_entity_matrix(
	query_vectors: torch.Tensor,
	entity_vectors: torch.Tensor,
	*,
	mode: str,
	distance_degree: float,
) -> torch.Tensor:
	mode = str(mode).lower().replace('-', '_')
	if mode == 'cosine':
		query_vectors = torch.nn.functional.normalize(query_vectors, p=2, dim=-1)
		entity_vectors = torch.nn.functional.normalize(entity_vectors, p=2, dim=-1)
		return torch.mm(query_vectors, entity_vectors.t())
	if mode in {'lp_distance', 'distance'}:
		return -torch.cdist(query_vectors, entity_vectors, p=float(distance_degree))
	return torch.mm(query_vectors, entity_vectors.t())


def average_link_metrics(forward_metrics: dict, backward_metrics: dict) -> dict:
	"""Average numeric link-prediction metrics from forward and backward passes."""

	if not forward_metrics or not backward_metrics:
		return forward_metrics or backward_metrics or {}

	averaged_metrics = {}
	for key in forward_metrics.keys() & backward_metrics.keys():
		forward_value = forward_metrics[key]
		backward_value = backward_metrics[key]
		if isinstance(forward_value, (int, float)) and isinstance(backward_value, (int, float)):
			averaged_metrics[key] = (forward_value + backward_value) / 2
	return averaged_metrics


def format_link_prediction_metrics(scope_label: str, direction: str, metrics: dict) -> str:
	"""Format link prediction metrics for a single direction or their average."""

	return (
		f'{scope_label} ({direction}) | '
		f'MR: {metrics.get("mr", metrics.get("mean_rank", 0.0)):.4f} | '
		f'MRR: {metrics.get("mrr", 0.0):.4f} | '
		f'H@1: {metrics.get("hit@1", metrics.get("hits@1", 0.0)):.4f} | '
		f'H@3: {metrics.get("hit@3", metrics.get("hits@3", 0.0)):.4f} | '
		f'H@10: {metrics.get("hit@10", metrics.get("hits@10", 0.0)):.4f}'
	)


def log_bidirectional_link_metrics(
	scope_label: str,
	forward_metrics: dict | None,
	backward_metrics: dict | None,
	*,
	round_digits: int | None = 4,
) -> dict:
	"""Log forward, backward, and averaged link metrics; return metrics for monitoring."""

	forward_metrics = forward_metrics or {}
	backward_metrics = backward_metrics or {}

	if forward_metrics:
		logger.info(format_link_prediction_metrics(scope_label, 'Fwd', forward_metrics))
	if backward_metrics:
		logger.info(format_link_prediction_metrics(scope_label, 'Bwd', backward_metrics))

	if forward_metrics and backward_metrics:
		result = average_link_metrics(forward_metrics, backward_metrics)
		logger.info(format_link_prediction_metrics(scope_label, 'Avg', result))
		if round_digits is not None:
			return {
				key: round(value, round_digits) if isinstance(value, (int, float)) else value
				for key, value in result.items()
			}
		return result
	return forward_metrics or backward_metrics or {}
from configs.config import apply_train_args
import numpy as np
import json


FILTER_MASK_VALUE = -1e30


class ModelInterfaceError(RuntimeError):
	"""Custom error for when a model does not conform to the expected evaluation interface."""
	pass


def _model_uses_token_inputs(model) -> bool:
	"""Return True when the model expects tokenized training/eval inputs."""

	model_obj = get_model_obj(model)
	return getattr(model_obj, 'training_input_mode', 'indices') == 'tokens'


def _supports_kge_1vsall_eval(model) -> bool:
	"""Return True when the model exposes 1-vs-all tail/head scoring."""

	model_obj = get_model_obj(model)
	if getattr(model_obj, 'training_input_mode', 'indices') == 'tokens':
		return False
	return hasattr(model_obj, 'predict_tail_hr_') or isinstance(model_obj, KGEModel)


def _supports_simkgc_link_eval(model) -> bool:
	"""Return True for token-input models that encode queries/entities like SimKGC."""

	model_obj = get_model_obj(model)
	return (
		getattr(model_obj, 'training_input_mode', 'indices') == 'tokens'
		and hasattr(model_obj, 'predict_by_examples')
		and hasattr(model_obj, 'predict_by_entities')
	)


def _resolve_relation_index(relation: str, relation_to_idx: dict) -> int:
	"""Map a relation string to its embedding index."""

	from utils.relations import resolve_relation_index

	return resolve_relation_index(relation, relation_to_idx)


def _relation_lookup(model):
	"""Return a callable that maps relation strings to embedding indices."""

	if hasattr(model, '_relation_to_idx') and callable(model._relation_to_idx):
		return model._relation_to_idx
	rel_to_idx = getattr(model, 'rel_to_idx', None)
	if rel_to_idx is None:
		raise RuntimeError('Model is missing a relation index lookup for fast evaluation')
	return lambda relation: _resolve_relation_index(relation, rel_to_idx)


def _build_filter_index_maps(all_triplet_dict, entity_dict, relation_lookup) -> tuple[dict, dict]:
	"""Build filtered-evaluation maps over integer (h, r) and (r, t) keys."""

	entity_to_idx = entity_dict.entity2idx
	hr_to_tails: dict[tuple[int, int], list[int]] = {}
	for (head_id, relation), tail_ids in all_triplet_dict.hr2tails.items():
		try:
			h_idx = entity_to_idx[head_id]
			r_idx = relation_lookup(relation)
		except KeyError:
			continue
		tails = [entity_to_idx[tail_id] for tail_id in tail_ids if tail_id in entity_to_idx]
		if tails:
			hr_to_tails[(h_idx, r_idx)] = tails

	rt_to_heads: dict[tuple[int, int], list[int]] = {}
	for (relation, tail_id), head_ids in all_triplet_dict.rt2heads.items():
		try:
			r_idx = relation_lookup(relation)
			t_idx = entity_to_idx[tail_id]
		except KeyError:
			continue
		heads = [entity_to_idx[head_id] for head_id in head_ids if head_id in entity_to_idx]
		if heads:
			rt_to_heads[(r_idx, t_idx)] = heads
	return hr_to_tails, rt_to_heads


def _apply_filter_mask(
	scores: torch.Tensor,
	h_idx: torch.Tensor,
	r_idx: torch.Tensor,
	t_idx: torch.Tensor,
	filter_map: dict[tuple[int, int], list[int]],
	*,
	predict_head: bool,
) -> torch.Tensor:
	"""Mask known alternative true entities to ``FILTER_MASK_VALUE``, keeping the target."""

	rows: list[int] = []
	cols: list[int] = []
	for i in range(h_idx.size(0)):
		if predict_head:
			key = (int(r_idx[i].item()), int(t_idx[i].item()))
			target = int(h_idx[i].item())
		else:
			key = (int(h_idx[i].item()), int(r_idx[i].item()))
			target = int(t_idx[i].item())
		for candidate in filter_map.get(key, ()):
			if candidate != target:
				rows.append(i)
				cols.append(candidate)
	if rows:
		row_tensor = torch.tensor(rows, device=scores.device, dtype=torch.long)
		col_tensor = torch.tensor(cols, device=scores.device, dtype=torch.long)
		scores[row_tensor, col_tensor] = FILTER_MASK_VALUE
	return scores


def _map_forward_relations_to_inverse(
	r_idx: torch.Tensor,
	inverse_map: torch.Tensor | None,
	device: torch.device,
) -> torch.Tensor:
	if inverse_map is None:
		return r_idx
	inverse_map = inverse_map.to(device)
	mapped = inverse_map[r_idx.long()]
	if bool((mapped < 0).any().item()):
		raise RuntimeError('Missing inverse relation index mapping for head evaluation')
	return mapped


def _evaluate_kge_1vsall_batch(
	model,
	h_idx: torch.Tensor,
	r_idx: torch.Tensor,
	t_idx: torch.Tensor,
	hr_filter: dict[tuple[int, int], list[int]],
	rt_filter: dict[tuple[int, int], list[int]],
	*,
	head_eval_mode: str,
	filter_known: bool,
	all_entity_embs: torch.Tensor | None = None,
	inverse_map: torch.Tensor | None = None,
) -> list[int]:
	"""Score and rank one batch with full-matrix ``hr_`` or ``_rt`` broadcasting."""

	device = next(model.parameters()).device
	h_idx = h_idx.to(device)
	r_idx = r_idx.to(device)
	t_idx = t_idx.to(device)

	if head_eval_mode == 'tail':
		scores = model.predict_tail_hr_(h_idx, r_idx, all_t_embs=all_entity_embs)
		if filter_known:
			scores = _apply_filter_mask(scores, h_idx, r_idx, t_idx, hr_filter, predict_head=False)
		target_indices = t_idx
	elif head_eval_mode == 'rt_forward':
		scores = model.predict_head_rt_(r_idx, t_idx, all_h_embs=all_entity_embs)
		if filter_known:
			scores = _apply_filter_mask(scores, h_idx, r_idx, t_idx, rt_filter, predict_head=True)
		target_indices = h_idx
	elif head_eval_mode == 'rt_inverse':
		r_inv = _map_forward_relations_to_inverse(r_idx, inverse_map, device)
		scores = model.predict_head_rt_(r_inv, t_idx, all_h_embs=all_entity_embs)
		if filter_known:
			scores = _apply_filter_mask(scores, h_idx, r_idx, t_idx, rt_filter, predict_head=True)
		target_indices = h_idx
	elif head_eval_mode == 'hr_inverse':
		r_inv = _map_forward_relations_to_inverse(r_idx, inverse_map, device)
		scores = model.predict_tail_hr_(t_idx, r_inv, all_t_embs=all_entity_embs)
		if filter_known:
			scores = _apply_filter_mask(scores, h_idx, r_idx, t_idx, rt_filter, predict_head=True)
		target_indices = h_idx
	else:
		raise ValueError(f'Unsupported head_eval_mode: {head_eval_mode}')

	return _ranks_from_score_matrix(scores, target_indices)


def _encode_simkgc_entity_embeddings(
	model,
	entity_dict,
	batch_size: int,
	*,
	args=None,
) -> torch.Tensor:
	"""Encode all entity vectors once for SimKGC link prediction."""

	from data.dataset import Dataset, Example
	from data.dataloader import collate
	from utils.device import call_model_forward, move_to_cuda

	eval_args = args if args is not None else global_args
	use_cuda = torch.cuda.is_available()
	entity_examples = [
		Example(head_id='', relation='', tail_id=entity_ex.entity_id)
		for entity_ex in entity_dict.entity_exs
	]
	logger.info('[EVAL] Encoding %d entities for link prediction...', len(entity_examples))
	entity_loader = torch.utils.data.DataLoader(
		Dataset(path='', examples=entity_examples, task=eval_args.dataset),
		num_workers=0,
		batch_size=max(batch_size, 512),
		collate_fn=collate,
		shuffle=False,
	)
	entity_vectors = []
	for batch_dict in entity_loader:
		batch_dict['only_ent_embedding'] = True
		if use_cuda:
			batch_dict = move_to_cuda(batch_dict)
		outputs = call_model_forward(model, batch_dict)
		entity_vectors.append(outputs['ent_vectors'])
	return torch.cat(entity_vectors, dim=0)


def _evaluate_simkgc_link_prediction(
	model,
	examples: Sequence[Example],
	entity_dict,
	batch_size: int,
	*,
	filter_known: bool = True,
	args=None,
	all_entity_embs: torch.Tensor | None = None,
	score_mode: str = 'cosine',
) -> dict:
	"""Chunked link prediction for token-input models (reference SimKGC ``compute_metrics`` path)."""

	from data.dataset import Dataset
	from data.dataloader import collate_hr
	from metrics.ranking import rerank_by_graph
	from utils.device import call_model_forward, move_to_cuda

	eval_args = args if args is not None else global_args
	model.eval()
	use_cuda = torch.cuda.is_available()

	scoring_examples = list(examples)
	logger.info(
		'[EVAL] Encoding %d query vectors for link prediction (%s scorer)...',
		len(scoring_examples),
		score_mode,
	)
	hr_vectors = []
	data_loader = torch.utils.data.DataLoader(
		Dataset(path='', examples=scoring_examples, task=eval_args.dataset),
		num_workers=0,
		batch_size=batch_size,
		collate_fn=collate_hr,
		shuffle=False,
	)
	for batch_dict in data_loader:
		batch_dict['encode_hr_only'] = True
		if use_cuda:
			batch_dict = move_to_cuda(batch_dict)
		outputs = call_model_forward(model, batch_dict)
		hr_vectors.append(outputs['hr_vector'])
	hr_tensor = torch.cat(hr_vectors, dim=0)

	if all_entity_embs is None:
		entities_tensor = _encode_simkgc_entity_embeddings(
			model, entity_dict, batch_size, args=eval_args,
		)
	else:
		logger.info('[EVAL] Reusing cached entity embeddings (%d entities)...', all_entity_embs.size(0))
		entities_tensor = all_entity_embs

	device = hr_tensor.device
	entities_tensor = entities_tensor.to(device)
	entity_cnt = entities_tensor.size(0)
	chunk_size = getattr(eval_args, 'chunk_size', None) or 8192
	chunk_size = max(int(chunk_size), 1)
	all_triplet_dict = get_all_triplet_dict()
	ranks: list[int] = []

	for start in range(0, hr_tensor.size(0), batch_size):
		end = min(start + batch_size, hr_tensor.size(0))
		batch_hr = hr_tensor[start:end]
		batch_examples = scoring_examples[start:end]
		batch_score = torch.zeros(
			batch_hr.size(0),
			entity_cnt,
			device=device,
			dtype=batch_hr.dtype,
		)
		for entity_start in range(0, entity_cnt, chunk_size):
			entity_end = min(entity_start + chunk_size, entity_cnt)
			batch_score[:, entity_start:entity_end] = _score_query_entity_matrix(
				batch_hr,
				entities_tensor[entity_start:entity_end],
				mode=score_mode,
				distance_degree=_lp_distance_degree(eval_args),
			)

		rerank_by_graph(batch_score, batch_examples, entity_dict)

		if filter_known:
			_filter_known(batch_score, batch_examples, all_triplet_dict, entity_dict)

		target_indices = _infer_target_indices(batch_examples, entity_dict, predict_head=False).to(device)
		ranks.extend(_ranks_from_score_matrix(batch_score, target_indices))

	return ranking_metrics_from_ranks(ranks)


def _evaluate_kge_link_prediction(
	model,
	examples: Sequence[Example],
	entity_dict,
	batch_size: int,
	*,
	eval_forward: bool,
	filter_known: bool,
	args=None,
	all_entity_embs: torch.Tensor | None = None,
) -> list[int]:
	"""Fast filtered link prediction for ``KGEModel`` instances."""

	eval_args = args if args is not None else global_args
	head_eval_mode = resolve_head_eval_mode(eval_args, eval_forward=eval_forward)
	relation_lookup = _relation_lookup(model)
	hr_filter, rt_filter = _build_filter_index_maps(get_all_triplet_dict(), entity_dict, relation_lookup)
	scoring_examples = (
		_coerce_forward_examples(examples)
		if head_eval_mode in {'rt_forward', 'rt_inverse', 'hr_inverse'}
		else list(examples)
	)
	h_all, r_all, t_all = _examples_to_query_index_tensors(scoring_examples, entity_dict, model)
	if all_entity_embs is None:
		logger.info('[EVAL] Encoding %d entities for link prediction (%s)...', len(entity_dict.entity_exs), head_eval_mode)
		all_entity_embs = model.embed_all_entities()
	inverse_map = None
	if head_eval_mode in {'rt_inverse', 'hr_inverse'}:
		rel_to_idx = getattr(model, 'rel_to_idx', None) or {}
		inverse_map = build_forward_to_inverse_index_tensor(rel_to_idx)

	ranks: list[int] = []
	iterator = range(0, len(scoring_examples), batch_size)
	for start in tqdm.tqdm(iterator, disable=len(scoring_examples) <= batch_size):
		end = min(start + batch_size, len(scoring_examples))
		batch_ranks = _evaluate_kge_1vsall_batch(
			model,
			h_all[start:end],
			r_all[start:end],
			t_all[start:end],
			hr_filter,
			rt_filter,
			head_eval_mode=head_eval_mode,
			filter_known=filter_known,
			all_entity_embs=all_entity_embs,
			inverse_map=inverse_map,
		)
		ranks.extend(batch_ranks)
	return ranks


def _filter_known(batch_score: torch.Tensor, examples: List[Example], all_triplet_dict, entity_dict) -> None:
    """Mask known neighbors for filtered link-prediction evaluation."""
    for idx, ex in enumerate(examples):
        gold_neighbor_ids = all_triplet_dict.get_neighbors(ex.head_id, ex.relation)
        if not gold_neighbor_ids:
            continue

        mask_indices = [
            entity_dict.entity_to_idx(entity_id)
            for entity_id in gold_neighbor_ids
            if entity_id != ex.tail_id
        ]
        if not mask_indices:
            continue

        mask_tensor = torch.LongTensor(mask_indices).to(batch_score.device)
        batch_score[idx].index_fill_(0, mask_tensor, float('-inf'))


def _filter_known_heads(batch_score: torch.Tensor, examples: List[Example], all_triplet_dict, entity_dict) -> None:
    """Mask other known heads for filtered head-prediction evaluation."""

    for idx, ex in enumerate(examples):
        gold_head_ids = all_triplet_dict.get_heads(ex.relation, ex.tail_id)
        if not gold_head_ids:
            continue

        mask_indices = [
            entity_dict.entity_to_idx(entity_id)
            for entity_id in gold_head_ids
            if entity_id != ex.head_id
        ]
        if not mask_indices:
            continue

        mask_tensor = torch.LongTensor(mask_indices).to(batch_score.device)
        batch_score[idx].index_fill_(0, mask_tensor, float('-inf'))


def _coerce_forward_examples(examples: Sequence[Example]) -> List[Example]:
    """Normalize backward/reversed examples to forward (head, relation, tail) form."""

    normalized: List[Example] = []
    for ex in examples:
        relation = ex.relation
        head_id = ex.head_id
        tail_id = ex.tail_id
        head = getattr(ex, 'head', head_id)
        tail = getattr(ex, 'tail', tail_id)

        if str(relation).startswith('inverse '):
            relation = relation[len('inverse '):]
            head_id, tail_id = ex.tail_id, ex.head_id
            head = getattr(ex, 'tail', tail_id)
            tail = getattr(ex, 'head', head_id)

        normalized.append(Example(
            head_id=head_id,
            head=head,
            relation=relation,
            tail_id=tail_id,
            tail=tail,
            label=getattr(ex, 'label', None),
        ))
    return normalized


def _uses_head_batch_scoring(model) -> bool:
    """Return True when the model exposes native head-batch link-prediction scoring."""

    return bool(getattr(model, 'bidirectional_score_batch', False))


def _score_batch_supports_mode(model) -> bool:
    """Return True when score_batch accepts an explicit batch mode argument."""

    if not hasattr(model, 'score_batch'):
        return False
    return 'mode' in inspect.signature(model.score_batch).parameters


def _examples_to_query_index_tensors(examples: Sequence[Example], entity_dict, model) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert examples to head/relation/tail index tensors once per evaluation pass."""

    head_indices = [entity_dict.entity_to_idx(ex.head_id) for ex in examples]
    tail_indices = [entity_dict.entity_to_idx(ex.tail_id) for ex in examples]
    relation_lookup = _relation_lookup(model)
    relation_indices = [relation_lookup(ex.relation) for ex in examples]
    return (
        torch.tensor(head_indices, dtype=torch.long),
        torch.tensor(relation_indices, dtype=torch.long),
        torch.tensor(tail_indices, dtype=torch.long),
    )


def _entity_indices(entity_dict, entity_ids: Sequence[str]) -> torch.Tensor:
    """Convert entity id strings to a single index tensor."""

    return torch.tensor([entity_dict.entity_to_idx(entity_id) for entity_id in entity_ids], dtype=torch.long)


def _infer_target_indices(examples: Sequence[Example], entity_dict, predict_head: bool = False) -> torch.Tensor:
    """Infer target entity indices for a batch of examples."""

    if predict_head:
        target_indices = [entity_dict.entity_to_idx(ex.head_id) for ex in examples]
    else:
        target_indices = [entity_dict.entity_to_idx(ex.tail_id) for ex in examples]
    return torch.LongTensor(target_indices)


def _tie_handling_kwargs(args=None) -> dict:
	"""Resolve tie-handling options for ranking (``tie_handling`` / ``tie_method``)."""

	eval_args = args if args is not None else global_args
	return {
		'tie_handling': str(getattr(eval_args, 'tie_handling', 'rounded_mean_rank') or 'rounded_mean_rank'),
		'tie_rtol': float(getattr(eval_args, 'tie_rtol', 1e-4) or 1e-4),
		'tie_atol': float(getattr(eval_args, 'tie_atol', 1e-5) or 1e-5),
	}


def _ranks_from_score_matrix(
	score: torch.Tensor,
	target_indices: torch.Tensor,
	args=None,
) -> list[int]:
	"""Compute 1-based filtered ranks with configurable tie handling."""

	return ranks_from_score_matrix(score, target_indices, **_tie_handling_kwargs(args))


def _load_labeled_examples(label_path: str) -> List[Example]:
	"""Load triple-classification examples that carry binary labels."""

	return [
		ex
		for ex in load_data(
			label_path,
			add_forward_triplet=label_path.endswith('.json'),
			add_backward_triplet=False,
		)
		if ex.label is not None
	]


def _resolve_label_split_path(args, split: str) -> str:
	"""Resolve a labeled split path (``valid`` or ``test``) from config or data dirs."""

	attr = f'{split}_w_label_path'
	label_path = getattr(args, attr, '') or ''
	if label_path and os.path.exists(label_path):
		return label_path

	candidate_dirs = []
	for source_path in [
		getattr(args, 'valid_w_label_path', ''),
		getattr(args, 'test_w_label_path', ''),
		getattr(args, 'valid_path', ''),
		getattr(args, 'test_path', ''),
		getattr(args, 'train_path', ''),
	]:
		if source_path:
			candidate_dirs.append(os.path.dirname(source_path))
	candidate_dirs.append(os.path.join('data', getattr(args, 'dataset', '')))
	candidate_dirs.append(os.path.join('data', getattr(args, 'dataset', ''), 'preprocessed'))

	for candidate_dir in candidate_dirs:
		for candidate_name in [
			f'{split}_w_label.txt.json',
			f'{split}_w_label.txt',
			f'{split}_label.txt.json',
			f'{split}_label.txt',
		]:
			candidate_path = os.path.join(candidate_dir, candidate_name)
			if os.path.exists(candidate_path):
				return candidate_path
	return ''


def _scores_to_classification_probs(scores: torch.Tensor, args) -> torch.Tensor:
	"""Map KGE scores to values used for TC thresholding and ranking AUCs.

	BCE / logistic losses: apply optional logit offset then ``sigmoid`` (true probs).
	Margin-ranking / AU / other ranking losses: return raw scores so threshold search
	is not crushed by sigmoid saturation on large ``margin - distance`` values.
	"""

	if not isinstance(scores, torch.Tensor):
		scores = torch.as_tensor(scores, dtype=torch.float32)
	if not uses_logit_classification_scores(args):
		return scores
	offset = bce_logit_offset(args)
	if offset != 0.0:
		scores = scores + offset
	return torch.sigmoid(scores)


def _collect_index_triple_classification_probs(
	model,
	examples: Sequence[Example],
	entity_dict,
	batch_size: int,
	tc_args,
) -> List[float]:
	"""Score labeled triples for classification (sigmoid probs or raw ranking scores)."""

	y_prob: List[float] = []
	for i in range(0, len(examples), batch_size):
		batch = examples[i:i + batch_size]
		scores = _score_triple_classification_batch(model, batch, entity_dict)
		prob = _scores_to_classification_probs(scores, tc_args).detach().cpu().numpy().reshape(-1)
		y_prob.extend(prob.tolist())
	return y_prob


def _score_triple_classification_batch(
	model,
	batch: Sequence[Example],
	entity_dict,
) -> torch.Tensor:
	"""Score labeled triples for classification (index KGE or legacy ``score_batch``)."""

	model_obj = get_model_obj(model)
	if hasattr(model_obj, 'score_batch'):
		scores = model_obj.score_batch(
			[ex.head_id for ex in batch],
			[ex.relation for ex in batch],
			[ex.tail_id for ex in batch],
		)
	elif hasattr(model_obj, 'score_hrt'):
		h_idx, r_idx, t_idx = _examples_to_query_index_tensors(batch, entity_dict, model_obj)
		device = next(model_obj.parameters()).device
		scores = model_obj.score_hrt(
			h_idx.to(device),
			r_idx.to(device),
			t_idx.to(device),
		)
	else:
		raise ModelInterfaceError(
			'Model must expose score_batch or score_hrt for index-based triple classification.'
		)

	if not isinstance(scores, torch.Tensor):
		scores = torch.tensor(scores)
	if scores.dim() == 2 and scores.size(0) == scores.size(1):
		scores = scores.diag()
	return scores


def _bert_encoder_configured(args) -> bool:
	"""Return True when a HuggingFace encoder name is available for text triple classification."""

	for key in ('bert_encoder', 'encoder', 'pretrained_model'):
		if str(getattr(args, key, '') or '').strip():
			return True
	return False


def _should_use_index_triple_classification(model, args) -> bool:
	"""Return True when triple classification should use index KGE scoring (not BERT)."""

	if is_index_kge_model(args):
		return True
	model_obj = get_model_obj(model)
	return (
		isinstance(model_obj, KGEModel)
		or hasattr(model_obj, 'score_batch')
		or hasattr(model_obj, 'score_hrt')
	)


def _eval_args_for_triple_classification(evaluator, args=None):
	"""Prefer checkpoint training args when the evaluator loaded a model from disk."""

	if getattr(evaluator, 'train_args', None) is not None:
		return evaluator.train_args
	if args is not None:
		return args
	return global_args


def _score_by_embedding_adapter(model, examples: List[Example], entity_tensor: torch.Tensor) -> torch.Tensor:
    """Score examples using the model's embedding adapters."""

    hr_tensor = model.hr_embeddings(examples, entity_tensor.device)
    if hr_tensor.size(1) != entity_tensor.size(1):
        raise ValueError('hr_embeddings and entity_embeddings must have the same hidden size')
    return hr_tensor


def evaluate_model(
    model,
    eval_path: str,
    entity_dict=None,
    all_triplet_dict=None,
    device: Optional[torch.device] = None,
    batch_size: int = 256,
    chunk_size: Optional[int] = None,
    topk: int = 10,
    filter_known: bool = True,
) -> Tuple[List[List[float]], List[List[int]], dict]:
    """Evaluate a KG model on link prediction.

    Returns:
        topk_scores, topk_indices, metrics
    """

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if entity_dict is None:
        entity_dict = get_entity_dict()
    if all_triplet_dict is None:
        all_triplet_dict = get_all_triplet_dict()

    examples = load_data(eval_path, add_forward_triplet=True, add_backward_triplet=False)
    total = len(examples)

    if total == 0:
        raise ValueError(f'No examples found in {eval_path}')

    model = get_model_obj(model)
    if _supports_simkgc_link_eval(model):
        metrics = _evaluate_simkgc_link_prediction(
            model,
            examples,
            entity_dict,
            batch_size,
            filter_known=filter_known,
            score_mode='cosine',
        )
        return [], [], metrics
    if _supports_kge_1vsall_eval(model):
        ranks_all = _evaluate_kge_link_prediction(
            model,
            examples,
            entity_dict,
            batch_size,
            eval_forward=True,
            filter_known=filter_known,
        )
        metrics = ranking_metrics_from_ranks(ranks_all)
        return [], [], metrics

    if chunk_size is None:
        chunk_size = getattr(model, 'chunk_size', 8192)

    use_embedding_path = hasattr(model, 'entity_embeddings') and hasattr(model, 'hr_embeddings')

    topk_scores_all: List[List[float]] = []
    topk_indices_all: List[List[int]] = []
    ranks_all: List[int] = []

    if use_embedding_path:
        entity_tensor = model.entity_embeddings(device).to(device)
        hr_tensor = _score_by_embedding_adapter(model, examples, entity_tensor).to(device)

        for start in tqdm.tqdm(range(0, total, batch_size)):
            end = min(start + batch_size, total)
            batch_hr = hr_tensor[start:end, :]
            batch_examples = examples[start:end]

            batch_score = torch.zeros(
                batch_hr.size(0),
                entity_tensor.size(0),
                device=device,
                dtype=batch_hr.dtype,
            )
            for entity_start in range(0, entity_tensor.size(0), chunk_size):
                entity_end = min(entity_start + chunk_size, entity_tensor.size(0))
                batch_score[:, entity_start:entity_end] = torch.mm(
                    batch_hr,
                    entity_tensor[entity_start:entity_end, :].t(),
                )

            if filter_known:
                _filter_known(batch_score, batch_examples, all_triplet_dict, entity_dict)

            batch_sorted_score, batch_sorted_indices = torch.sort(batch_score, dim=-1, descending=True)
            target_indices = _infer_target_indices(batch_examples, entity_dict).to(device)
            target_rank = torch.nonzero(batch_sorted_indices.eq(target_indices.unsqueeze(-1)).long(), as_tuple=False)
            if target_rank.size(0) != batch_score.size(0):
                raise RuntimeError('Unable to compute one rank per example')

            for idx in range(target_rank.size(0)):
                row = target_rank[idx].tolist()
                if row[0] != idx:
                    raise RuntimeError('Target rank rows are misaligned')
                ranks_all.append(row[1] + 1)

            topk_scores_all.extend(batch_sorted_score[:, :topk].tolist())
            topk_indices_all.extend(batch_sorted_indices[:, :topk].tolist())

    else:
        if not hasattr(model, 'score_batch'):
            raise ModelInterfaceError('Model must expose either embedding-style adapters or `score_batch`.')

        all_entity_ids = [entity_ex.entity_id for entity_ex in entity_dict.entity_exs]

        for start in tqdm.tqdm(range(0, total, batch_size)):
            end = min(start + batch_size, total)
            batch = examples[start:end]

            batch_score = torch.zeros(len(batch), len(all_entity_ids), device=device)
            for entity_start in range(0, len(all_entity_ids), chunk_size):
                entity_end = min(entity_start + chunk_size, len(all_entity_ids))
                entity_chunk = all_entity_ids[entity_start:entity_end]
                scores_chunk = model.score_batch(
                    [ex.head_id for ex in batch],
                    [ex.relation for ex in batch],
                    entity_chunk,
                )
                if not isinstance(scores_chunk, torch.Tensor):
                    scores_chunk = torch.tensor(scores_chunk, device=device)
                batch_score[:, entity_start:entity_end] = scores_chunk.to(device)

            if filter_known:
                _filter_known(batch_score, batch, all_triplet_dict, entity_dict)

            batch_sorted_score, batch_sorted_indices = torch.sort(batch_score, dim=-1, descending=True)
            target_indices = _infer_target_indices(batch, entity_dict).to(device)
            target_rank = torch.nonzero(batch_sorted_indices.eq(target_indices.unsqueeze(-1)).long(), as_tuple=False)
            if target_rank.size(0) != batch_score.size(0):
                raise RuntimeError('Unable to compute one rank per example')

            for idx in range(target_rank.size(0)):
                row = target_rank[idx].tolist()
                if row[0] != idx:
                    raise RuntimeError('Target rank rows are misaligned')
                ranks_all.append(row[1] + 1)

            topk_scores_all.extend(batch_sorted_score[:, :topk].tolist())
            topk_indices_all.extend(batch_sorted_indices[:, :topk].tolist())

    metrics = ranking_metrics_from_ranks(ranks_all)
    return topk_scores_all, topk_indices_all, metrics

class Evaluator:
    """Helper to load encoder checkpoints and run model-based evaluations."""

    def __init__(self, args=None):
        self.args = args if args is not None else global_args
        self.model = None
        self.train_args: SimpleNamespace | None = None
        self.use_cuda = False

    def _eval_batch_size(self, batch_size: int | None = None) -> int:
        if batch_size is not None:
            return max(int(batch_size), 1)
        eval_batch_size = getattr(self.args, 'eval_batch_size', None)
        if eval_batch_size is not None:
            return max(int(eval_batch_size), 1)
        test_batch_size = getattr(self.args, 'test_batch_size', None)
        if test_batch_size is not None:
            return max(int(test_batch_size), 1)
        return 128

    def load(self, ckt_path: str, use_data_parallel: bool = False) -> None:
        """Load checkpoint, apply training args, build tokenizer and model, and load weights."""

        checkpoint = load_checkpoint(ckt_path, map_location='cpu')
        self.checkpoint = checkpoint
        self.train_args = SimpleNamespace(**checkpoint['args'])

        apply_train_args(self.train_args)

        from models.builder import build_model

        self.model = build_model(self.train_args)
        if _model_uses_token_inputs(self.model):
            build_tokenizer(self.train_args)

        load_state_dict_clean(self.model, ckt_path)
        self.model.eval()

        if use_data_parallel and torch.cuda.device_count() > 1:
            logger.info('Use data parallel evaluator model')
            self.model = torch.nn.DataParallel(self.model).cuda()
            self.use_cuda = True
        elif torch.cuda.is_available():
            self.model.cuda()
            self.use_cuda = True

        logger.info('Load model from %s successfully', ckt_path)

    @torch.no_grad()
    def evaluate_triple_classification_inplace(self, model, label_file, output_log_path, batch_size=None) -> dict:
        """Evaluate triple classification using the model's forward pass."""

        batch_size = self._eval_batch_size(batch_size)
        model = get_model_obj(model)
        model.eval()
        if not os.path.exists(label_file):
            logger.info(f"[EVAL] {label_file} not found, skip evaluation.")
            return
        eval_set = 'TEST' if 'test' in label_file else 'VALID'
        eval_exs = _load_labeled_examples(label_file)
        y_true = [int(ex.label) for ex in eval_exs]
        y_prob = []
        tc_args = _eval_args_for_triple_classification(self)
        if _should_use_index_triple_classification(model, tc_args):
            if not eval_exs:
                logger.info(f"[EVAL] {label_file} has no labeled examples, skip evaluation.")
                return
            entity_dict = get_entity_dict()
            y_prob = _collect_index_triple_classification_probs(
                model, eval_exs, entity_dict, batch_size, tc_args
            )
        elif not _bert_encoder_configured(tc_args):
            raise ModelInterfaceError(
                f'Model {getattr(tc_args, "model", "?")} cannot run text triple classification: '
                'bert_encoder is empty and the loaded model has no score_hrt/score_batch.'
            )
        else:
            with torch.no_grad():
                for i in range(0, len(eval_exs), batch_size):
                    batch = eval_exs[i:i + batch_size]
                    batch_vec = [ex.vectorize() for ex in batch]
                    batch_dict = collate(batch_vec)
                    if torch.cuda.is_available():
                        batch_dict = move_to_cuda(batch_dict)
                        model.cuda()
                    output_dict = model(**batch_dict)
                    logits = model.compute_logits(output_dict=output_dict, batch_dict=batch_dict)['logits']
                    prob = torch.sigmoid(logits.diag()).detach().cpu().numpy().reshape(-1)
                    y_prob.extend(prob.tolist())

        threshold = find_global_threshold(y_true, y_prob)
        y_pred = (np.array(y_prob) > threshold).astype(int).tolist()
        metrics_cls = classification_metrics(y_true, y_pred, y_prob)
        log_thresh = f"[{eval_set}] Best threshold: {threshold:.6f}"
        log_cls = f"[{eval_set}] Triple Classification: {json.dumps(metrics_cls)}"
        logger.info(log_thresh)
        logger.info(log_cls)
        with open(output_log_path, 'a', encoding='utf-8') as f:
            f.write(log_thresh + '\n')
            f.write(log_cls + '\n')
        return metrics_cls

    @torch.inference_mode()
    def evaluate_link_prediction_inplace(
        self,
        model,
        eval_path,
        entity_dict,
        output_log_path,
        batch_size=None,
        eval_forward=True,
        examples=None,
        all_entity_embs: torch.Tensor | None = None,
    ) -> dict:
        """Evaluate link prediction using the model's forward pass."""
        batch_size = self._eval_batch_size(batch_size)
        eval_model = model
        inner_model = get_model_obj(model)
        inner_model.eval()
        if not os.path.exists(eval_path):
            logger.info(f"[EVAL] {eval_path} not found, skip link prediction evaluation.")
            return {}
        if examples is None:
            examples = load_data(eval_path, add_forward_triplet=eval_forward, add_backward_triplet=not eval_forward)

        if _supports_simkgc_link_eval(inner_model):
            direction = 'forward' if eval_forward else 'backward'
            logger.info('[EVAL] Link prediction (%s) on %d queries (SimKGC path)...', direction, len(examples))
            return _evaluate_simkgc_link_prediction(
                eval_model,
                examples,
                entity_dict,
                batch_size,
                filter_known=True,
                args=self.args,
                all_entity_embs=all_entity_embs,
                score_mode=getattr(inner_model, 'lp_score_mode', 'cosine'),
            )

        if _supports_kge_1vsall_eval(inner_model):
            direction = 'forward' if eval_forward else 'backward'
            logger.info('[EVAL] Link prediction (%s) on %d queries...', direction, len(examples))
            ranks = _evaluate_kge_link_prediction(
                inner_model,
                examples,
                entity_dict,
                batch_size,
                eval_forward=eval_forward,
                filter_known=True,
                args=self.args,
                all_entity_embs=all_entity_embs,
            )
            return ranking_metrics_from_ranks(ranks)

        model = inner_model
        predict_head = (not eval_forward) and _uses_head_batch_scoring(model)
        scoring_examples = _coerce_forward_examples(examples) if predict_head else list(examples)

        if hasattr(model, 'score_batch'):
            all_entity_ids = [entity_ex.entity_id for entity_ex in entity_dict.entity_exs]
            score_device = next(model.parameters()).device
            head_ids = [ex.head_id for ex in scoring_examples]
            relations = [ex.relation for ex in scoring_examples]
            score_batch_mode = 'head-batch' if predict_head and _score_batch_supports_mode(model) else 'tail-batch'
            use_fast_indices = hasattr(model, 'score_batch_from_indices')
            query_head_idx = query_rel_idx = query_tail_idx = None
            all_entity_idx = None
            if use_fast_indices:
                query_head_idx, query_rel_idx, query_tail_idx = _examples_to_query_index_tensors(
                    scoring_examples, entity_dict, model
                )
                all_entity_idx = _entity_indices(entity_dict, all_entity_ids)

            if (
                score_batch_mode == 'head-batch'
                and hasattr(model, 'prepare_head_prediction_queries')
                and hasattr(model, 'score_head_prediction_full')
            ):
                tail_ids = [ex.tail_id for ex in scoring_examples]
                query_cache = model.prepare_head_prediction_queries(tail_ids, relations)
                score = model.score_head_prediction_full(query_cache)
                if score.size(0) != len(scoring_examples) or score.size(1) != len(all_entity_ids):
                    raise RuntimeError('DaBR fast head-prediction score matrix has unexpected shape')
            elif (
                score_batch_mode == 'tail-batch'
                and hasattr(model, 'prepare_link_prediction_queries')
                and hasattr(model, 'score_link_prediction_full')
            ):
                query_cache = model.prepare_link_prediction_queries(head_ids, relations)
                score = model.score_link_prediction_full(query_cache)
                if score.size(0) != len(scoring_examples) or score.size(1) != len(all_entity_ids):
                    raise RuntimeError('DaBR fast link-prediction score matrix has unexpected shape')
            else:
                score = torch.zeros(len(scoring_examples), len(all_entity_ids), device=score_device)
                entity_chunk_size = getattr(model, 'eval_entity_chunk_size', None)
                if entity_chunk_size is None:
                    entity_chunk_size = getattr(model.config, 'eval_entity_chunk_size', None) if hasattr(model, 'config') else None
                if entity_chunk_size is None:
                    entity_chunk_size = getattr(global_args, 'eval_entity_chunk_size', None)
                if entity_chunk_size is None:
                    entity_chunk_size = max(batch_size, 4096)
                entity_chunk_size = max(int(entity_chunk_size), 1)

                if score_batch_mode == 'head-batch' and hasattr(model, 'prepare_head_prediction_queries'):
                    tail_ids = [ex.tail_id for ex in scoring_examples]
                    query_cache = model.prepare_head_prediction_queries(tail_ids, relations)
                    score_candidates = getattr(model, 'score_head_prediction_candidates', None)
                elif hasattr(model, 'prepare_link_prediction_queries'):
                    query_cache = model.prepare_link_prediction_queries(head_ids, relations)
                    score_candidates = model.score_link_prediction_candidates
                else:
                    query_cache = None
                    score_candidates = None

                use_eval_amp = (
                    score_device.type == 'cuda'
                    and bool(getattr(global_args, 'eval_use_amp', getattr(global_args, 'use_amp', False)))
                )
                for start in range(0, len(all_entity_ids), entity_chunk_size):
                    end = min(start + entity_chunk_size, len(all_entity_ids))
                    if score_candidates is not None:
                        chunk_score = score_candidates(query_cache, (start, end))
                    elif use_fast_indices:
                        candidate_idx = all_entity_idx[start:end].to(score_device)
                        with torch.autocast(device_type='cuda', enabled=use_eval_amp):
                            if score_batch_mode == 'head-batch':
                                chunk_score = model.score_batch_from_indices(
                                    query_rel_idx,
                                    candidate_idx,
                                    mode='head-batch',
                                    query_tail_indices=query_tail_idx,
                                )
                            else:
                                chunk_score = model.score_batch_from_indices(
                                    query_rel_idx,
                                    candidate_idx,
                                    mode='tail-batch',
                                    query_head_indices=query_head_idx,
                                )
                    else:
                        entity_chunk = all_entity_ids[start:end]
                        score_batch_kwargs = {}
                        if score_batch_mode == 'head-batch':
                            score_batch_kwargs['mode'] = 'head-batch'
                            score_batch_kwargs['query_tail_ids'] = [ex.tail_id for ex in scoring_examples]
                        with torch.autocast(device_type='cuda', enabled=use_eval_amp):
                            if score_batch_mode == 'head-batch':
                                chunk_score = model.score_batch(
                                    head_ids,
                                    relations,
                                    entity_chunk,
                                    **score_batch_kwargs,
                                )
                            else:
                                chunk_score = model.score_batch(head_ids, relations, entity_chunk)
                    if not isinstance(chunk_score, torch.Tensor):
                        chunk_score = torch.tensor(chunk_score, device=score_device)
                    score[:, start:end] = chunk_score
        else:
            hr_tensor, _ = model.predict_by_examples(scoring_examples, batch_size=batch_size)
            entity_examples = [Example(head_id='', relation='', tail_id=entity_ex.entity_id) for entity_ex in entity_dict.entity_exs]
            entities_tensor = model.predict_by_entities(entity_examples, batch_size=max(batch_size, 512))

            if torch.cuda.is_available():
                hr_tensor = hr_tensor.cuda()
                entities_tensor = entities_tensor.cuda()
            score = _score_query_entity_matrix(
                hr_tensor,
                entities_tensor,
                mode=getattr(model, 'lp_score_mode', 'original'),
                distance_degree=_lp_distance_degree(self.args),
            )
        all_triplet_dict = get_all_triplet_dict()
        if predict_head:
            _filter_known_heads(score, scoring_examples, all_triplet_dict, entity_dict)
        else:
            _filter_known(score, scoring_examples, all_triplet_dict, entity_dict)
        target_indices = _infer_target_indices(scoring_examples, entity_dict, predict_head=predict_head).to(score.device)
        ranks = _ranks_from_score_matrix(score, target_indices)
        metrics = ranking_metrics_from_ranks(ranks)
        return metrics

    def evaluate_dual_test_link_prediction(
        self,
        eval_path: str,
        entity_dict,
        output_dir: str,
    ) -> dict:
        """Run test link prediction with cosine, native, and Lp-distance scorers."""

        inner_model = get_model_obj(self.model)
        batch_size = self._eval_batch_size()
        distance_degree = _lp_distance_degree(self.args)
        scorer_modes = [
            ('cosine', 'cosine'),
            ('original', 'original'),
            (_lp_distance_label(self.args), 'lp_distance'),
        ]

        if _supports_simkgc_link_eval(inner_model):
            entity_embs = _encode_simkgc_entity_embeddings(
                self.model, entity_dict, batch_size, args=self.args,
            )
            metrics_by_mode: dict[str, dict] = {}
            for label, mode in scorer_modes:
                logger.info('[TEST] Link prediction (SimKGC %s scorer)...', label)
                log_path = os.path.join(output_dir, f'test_link_prediction_{label}.log')
                with lp_score_mode_context(self.model, mode, distance_degree):
                    forward_metrics = self.evaluate_link_prediction_inplace(
                        self.model,
                        eval_path,
                        entity_dict,
                        log_path,
                        eval_forward=True,
                        all_entity_embs=entity_embs,
                    )
                    backward_metrics = self.evaluate_link_prediction_inplace(
                        self.model,
                        eval_path,
                        entity_dict,
                        log_path,
                        eval_forward=False,
                        all_entity_embs=entity_embs,
                    )
                metrics_by_mode[label] = log_bidirectional_link_metrics(
                    f'[TEST] Link prediction ({label} scorer)',
                    forward_metrics,
                    backward_metrics,
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return metrics_by_mode

        cached_entity_embs = None
        if _supports_kge_1vsall_eval(inner_model):
            cached_entity_embs = inner_model.embed_all_entities()

        dual_metrics: dict[str, dict] = {}
        for label, mode in scorer_modes:
            log_path = os.path.join(output_dir, f'test_link_prediction_{label}.log')
            with lp_score_mode_context(self.model, mode, distance_degree):
                forward_metrics = self.evaluate_link_prediction_inplace(
                    self.model,
                    eval_path,
                    entity_dict,
                    log_path,
                    eval_forward=True,
                    all_entity_embs=cached_entity_embs,
                )
                backward_metrics = self.evaluate_link_prediction_inplace(
                    self.model,
                    eval_path,
                    entity_dict,
                    log_path,
                    eval_forward=False,
                    all_entity_embs=cached_entity_embs,
                )
            dual_metrics[label] = log_bidirectional_link_metrics(
                f'[TEST] Link prediction ({label} scorer)',
                forward_metrics,
                backward_metrics,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return dual_metrics

    def evaluate_test_triple_classification(self, epoch=None) -> dict:
        """Evaluate triple classification on the test split using the loaded checkpoint."""

        args = self.args if self.args is not None else global_args
        test_label_path = _resolve_label_split_path(args, 'test')
        if not test_label_path or not os.path.exists(test_label_path):
            logger.info('[TEST] test_w_label.txt not found, skip test evaluation.')
            return {}

        logger.info('[TEST] Evaluating triple classification on test set...')
        test_exs = _load_labeled_examples(test_label_path)
        if not test_exs:
            logger.info(f"[TEST] {test_label_path} has no labeled examples, skip test evaluation.")
            return {}
        y_true = [int(ex.label) for ex in test_exs]
        y_prob = []
        batch_size = self._eval_batch_size()

        if epoch is None:
            ckt_path = getattr(args, 'eval_model_path', '') or best_model_path(args.output_dir)
        else:
            ckt_path = checkpoint_path(args.output_dir, epoch)
            if not os.path.exists(ckt_path):
                ckt_path = checkpoint_path(args.output_dir, epoch, 0)
            if not os.path.exists(ckt_path):
                ckt_path = getattr(args, 'eval_model_path', '') or best_model_path(args.output_dir)

        if self.model is None:
            self.load(ckt_path)
        self.model.eval()

        tc_args = _eval_args_for_triple_classification(self, args)
        if _should_use_index_triple_classification(self.model, tc_args):
            entity_dict = get_entity_dict()
            y_prob = _collect_index_triple_classification_probs(
                self.model, test_exs, entity_dict, batch_size, tc_args
            )
        elif not _bert_encoder_configured(tc_args):
            raise ModelInterfaceError(
                f'Model {getattr(tc_args, "model", "?")} cannot run text triple classification: '
                'bert_encoder is empty and the loaded model has no score_hrt/score_batch.'
            )
        else:
            for i in range(0, len(test_exs), batch_size):
                batch = test_exs[i:i + batch_size]
                batch_vec = [ex.vectorize() for ex in batch]
                batch_dict = collate(batch_vec)
                if torch.cuda.is_available():
                    batch_dict = move_to_cuda(batch_dict)
                    self.model.cuda()
                output_dict = self.model(**batch_dict)
                logits = self.model.compute_logits(output_dict=output_dict, batch_dict=batch_dict)['logits']
                prob = torch.sigmoid(logits.diag()).detach().cpu().numpy().reshape(-1)
                y_prob.extend(prob.tolist())

        threshold = None
        valid_label_path = _resolve_label_split_path(args, 'valid')
        if valid_label_path and os.path.exists(valid_label_path):
            valid_exs = _load_labeled_examples(valid_label_path)
            if valid_exs:
                valid_y_true = [int(ex.label) for ex in valid_exs]
                if _should_use_index_triple_classification(self.model, tc_args):
                    valid_y_prob = _collect_index_triple_classification_probs(
                        self.model, valid_exs, get_entity_dict(), batch_size, tc_args
                    )
                elif not _bert_encoder_configured(tc_args):
                    raise ModelInterfaceError(
                        f'Model {getattr(tc_args, "model", "?")} cannot run text triple classification: '
                        'bert_encoder is empty and the loaded model has no score_hrt/score_batch.'
                    )
                else:
                    valid_y_prob = []
                    for i in range(0, len(valid_exs), batch_size):
                        batch = valid_exs[i:i + batch_size]
                        batch_vec = [ex.vectorize() for ex in batch]
                        batch_dict = collate(batch_vec)
                        if torch.cuda.is_available():
                            batch_dict = move_to_cuda(batch_dict)
                            self.model.cuda()
                        output_dict = self.model(**batch_dict)
                        logits = self.model.compute_logits(output_dict=output_dict, batch_dict=batch_dict)['logits']
                        prob = torch.sigmoid(logits.diag()).detach().cpu().numpy().reshape(-1)
                        valid_y_prob.extend(prob.tolist())
                threshold = find_global_threshold(valid_y_true, valid_y_prob)
                logger.info('[TEST] Threshold tuned on validation set: %.6f', threshold)
        if threshold is None:
            logger.warning(
                '[TEST] Validation label file not found; falling back to tuning threshold on test set.'
            )
            threshold = find_global_threshold(y_true, y_prob)

        y_pred = (np.array(y_prob) > threshold).astype(int).tolist()
        metrics_cls = classification_metrics(y_true, y_pred, y_prob)
        log_thresh = f'[TEST] Classification threshold: {threshold:.6f}'
        log_cls = f'[TEST] Triple Classification: {json.dumps(metrics_cls)}'
        logger.info(log_thresh)
        logger.info(log_cls)
        return metrics_cls
