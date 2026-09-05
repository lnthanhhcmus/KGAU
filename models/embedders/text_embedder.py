"""Text-backed embedders for token-input KGE models (SimKGC)."""

from typing import Any, Optional

import torch
import torch.nn as nn
import tqdm
from transformers import AutoConfig, AutoModel

from data.dataloader import collate, collate_hr
from data.dataset import Dataset, Example
from data.dict_hub import get_entity_dict, get_relation_id_map


def pool_output(
	pooling: str,
	cls_output: torch.Tensor,
	mask: torch.Tensor,
	last_hidden_state: torch.Tensor,
) -> torch.Tensor:
	"""Pool BERT hidden states and L2-normalize (SimKGC default)."""

	if pooling == 'cls':
		output_vector = cls_output
	elif pooling == 'max':
		input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).long()
		last_hidden_state = last_hidden_state.clone()
		last_hidden_state[input_mask_expanded == 0] = -1e4
		output_vector = torch.max(last_hidden_state, 1)[0]
	elif pooling == 'mean':
		input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
		sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
		sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-4)
		output_vector = sum_embeddings / sum_mask
	else:
		raise ValueError(f'Unknown pooling mode: {pooling}')

	return nn.functional.normalize(output_vector, dim=1)


def _resolve_encode_micro_batch_size(args: Any, batch_size: int) -> int:
	"""Cap BERT encode chunks to limit activation memory on single-GPU training.

	SimKGC runs full-batch encodes by default. Set ``encode_micro_batch_size`` to a
	positive value (e.g. 64) only when GPU memory is tight.
	"""

	explicit = getattr(args, 'encode_micro_batch_size', None)
	if explicit is None:
		return batch_size
	explicit = int(explicit)
	return batch_size if explicit <= 0 else min(explicit, batch_size)


def _use_encode_checkpoint(args: Any, *, training: bool) -> bool:
	if not training:
		return False
	return bool(getattr(args, 'encode_checkpoint', False))


class _BertTextEncoder(nn.Module):
	"""Shared BERT encode + pool path."""

	input_mode = 'tokens'

	def __init__(self, args: Any, *, shared_encoder: nn.Module | None = None):
		super().__init__()
		self.args = args
		self.config = AutoConfig.from_pretrained(args.bert_encoder)
		self.encoder = shared_encoder if shared_encoder is not None else AutoModel.from_pretrained(args.bert_encoder)
		self.pooling = getattr(args, 'pooling', 'mean')

	def _encode_once(
		self,
		token_ids: torch.Tensor,
		mask: torch.Tensor,
		token_type_ids: torch.Tensor,
	) -> torch.Tensor:
		outputs = self.encoder(
			input_ids=token_ids,
			attention_mask=mask,
			token_type_ids=token_type_ids,
			return_dict=True,
		)
		last_hidden_state = outputs.last_hidden_state
		cls_output = last_hidden_state[:, 0, :]
		return pool_output(self.pooling, cls_output, mask, last_hidden_state)

	def encode(
		self,
		token_ids: torch.Tensor,
		mask: torch.Tensor,
		token_type_ids: torch.Tensor,
	) -> torch.Tensor:
		batch_size = token_ids.size(0)
		micro = _resolve_encode_micro_batch_size(self.args, batch_size)
		use_checkpoint = _use_encode_checkpoint(self.args, training=self.training)

		if micro >= batch_size:
			if use_checkpoint:
				return torch.utils.checkpoint.checkpoint(
					self._encode_once,
					token_ids,
					mask,
					token_type_ids,
					use_reentrant=False,
				)
			return self._encode_once(token_ids, mask, token_type_ids)

		chunks = []
		for start in range(0, batch_size, micro):
			end = min(start + micro, batch_size)
			chunk_ids = token_ids[start:end]
			chunk_mask = mask[start:end]
			chunk_type_ids = token_type_ids[start:end]
			if use_checkpoint:
				chunks.append(
					torch.utils.checkpoint.checkpoint(
						self._encode_once,
						chunk_ids,
						chunk_mask,
						chunk_type_ids,
						use_reentrant=False,
					)
				)
			else:
				chunks.append(self._encode_once(chunk_ids, chunk_mask, chunk_type_ids))
		return torch.cat(chunks, dim=0)


class TextEntityEmbedder(_BertTextEncoder):
	"""Entity text encoder (``tail_bert`` in SimKGC)."""

	def __init__(self, args: Any, *, shared_encoder: nn.Module | None = None):
		super().__init__(args, shared_encoder=shared_encoder)
		self._full_entity_predict_loader = None

	def forward(
		self,
		token_ids: torch.Tensor,
		mask: torch.Tensor,
		token_type_ids: torch.Tensor,
	) -> torch.Tensor:
		return self.encode(token_ids, mask, token_type_ids)

	def embed(self, indices: torch.Tensor) -> torch.Tensor:
		"""Encode entities by integer index via their text descriptions."""

		entity_dict = get_entity_dict()
		examples = [
			Example(head_id='', relation='', tail_id=entity_dict.get_entity_by_idx(int(idx)).entity_id)
			for idx in indices.tolist()
		]
		return self._encode_examples(examples)

	def embed_all(self) -> torch.Tensor:
		"""Encode the full entity catalog (cached loader for LP eval)."""

		entity_exs = get_entity_dict().entity_exs
		batch_size = max(getattr(self.args, 'batch_size', 512), 1024)
		loader_workers = self._resolve_entity_loader_workers(None, len(entity_exs))
		return self._encode_entity_exs(entity_exs, batch_size=batch_size, num_workers=loader_workers)

	def _encode_examples(self, examples: list[Example]) -> torch.Tensor:
		from utils.device import move_to_cuda

		if not examples:
			device = next(self.parameters()).device
			return torch.empty(0, self.config.hidden_size, device=device)

		data_loader = torch.utils.data.DataLoader(
			Dataset(path='', examples=examples, task=self.args.dataset),
			batch_size=min(len(examples), max(getattr(self.args, 'batch_size', 512), 512)),
			collate_fn=collate,
			shuffle=False,
			num_workers=0,
		)
		vectors = []
		use_cuda = torch.cuda.is_available()
		for batch_dict in data_loader:
			if use_cuda:
				batch_dict = move_to_cuda(batch_dict)
			vectors.append(
				self.encode(
					batch_dict['tail_token_ids'],
					batch_dict['tail_mask'],
					batch_dict['tail_token_type_ids'],
				)
			)
		return torch.cat(vectors, dim=0)

	def _resolve_entity_loader_workers(self, num_workers: Optional[int], entity_count: int) -> int:
		if num_workers is not None:
			return int(num_workers)
		total_entities = len(get_entity_dict().entity_exs)
		if entity_count < total_entities:
			return 0
		return int(getattr(self.args, 'workers', 0))

	def _ensure_full_entity_predict_loader(self, batch_size: int, num_workers: int) -> torch.utils.data.DataLoader:
		if self._full_entity_predict_loader is not None:
			return self._full_entity_predict_loader

		from data.dict_hub import init_dataloader_worker

		examples = []
		for entity_ex in get_entity_dict().entity_exs:
			entity_id = getattr(entity_ex, 'entity_id', None) or getattr(entity_ex, 'tail_id', None)
			if entity_id is None:
				raise AttributeError('Expected entity examples with an entity_id or tail_id attribute')
			examples.append(Example(head_id='', relation='', tail_id=entity_id))

		loader_kwargs = {
			'dataset': Dataset(path='', examples=examples, task=self.args.dataset),
			'batch_size': batch_size,
			'collate_fn': collate,
			'shuffle': False,
			'num_workers': num_workers,
			'pin_memory': torch.cuda.is_available(),
		}
		if num_workers > 0:
			loader_kwargs['worker_init_fn'] = init_dataloader_worker
			loader_kwargs['persistent_workers'] = True

		self._full_entity_predict_loader = torch.utils.data.DataLoader(**loader_kwargs)
		return self._full_entity_predict_loader

	def _encode_entity_exs(
		self,
		entity_exs,
		batch_size: Optional[int] = None,
		num_workers: Optional[int] = None,
		show_progress: bool = False,
	) -> torch.Tensor:
		from utils.device import move_to_cuda

		if batch_size is None:
			batch_size = max(getattr(self.args, 'batch_size', 512), 1024)

		total_entities = len(get_entity_dict().entity_exs)
		loader_workers = self._resolve_entity_loader_workers(num_workers, len(entity_exs))
		use_full_catalog_loader = len(entity_exs) == total_entities

		if use_full_catalog_loader:
			data_loader = self._ensure_full_entity_predict_loader(batch_size, loader_workers)
		else:
			examples = []
			for entity_ex in entity_exs:
				entity_id = getattr(entity_ex, 'entity_id', None) or getattr(entity_ex, 'tail_id', None)
				if entity_id is None:
					raise AttributeError('Expected entity examples with an entity_id or tail_id attribute')
				examples.append(Example(head_id='', relation='', tail_id=entity_id))
			data_loader = torch.utils.data.DataLoader(
				Dataset(path='', examples=examples, task=self.args.dataset),
				num_workers=0,
				batch_size=batch_size,
				collate_fn=collate,
				shuffle=False,
				pin_memory=torch.cuda.is_available(),
			)

		ent_tensor_list = []
		use_cuda = torch.cuda.is_available()
		iterator = tqdm.tqdm(data_loader) if show_progress else data_loader
		for _, batch_dict in enumerate(iterator):
			if use_cuda:
				batch_dict = move_to_cuda(batch_dict)
			ent_tensor_list.append(
				self.encode(
					batch_dict['tail_token_ids'],
					batch_dict['tail_mask'],
					batch_dict['tail_token_type_ids'],
				)
			)
		return torch.cat(ent_tensor_list, dim=0)


class TextQueryEmbedder(_BertTextEncoder):
	"""Joint (head, relation) query encoder (``hr_bert`` in SimKGC)."""

	def __init__(self, args: Any, *, shared_encoder: nn.Module | None = None):
		super().__init__(args, shared_encoder=shared_encoder)

	def embed_hr(self, head_indices: torch.Tensor, relation_indices: torch.Tensor) -> torch.Tensor:
		"""Encode tail-prediction queries ``(h, r)`` from integer indices."""

		entity_dict = get_entity_dict()
		relation_id_map = get_relation_id_map() or {}
		idx_to_relation = {int(value): key for key, value in relation_id_map.items()}

		examples = []
		for head_idx, relation_idx in zip(head_indices.tolist(), relation_indices.tolist()):
			head_entity = entity_dict.get_entity_by_idx(int(head_idx))
			relation = idx_to_relation.get(int(relation_idx), str(int(relation_idx)))
			examples.append(Example(head_id=head_entity.entity_id, relation=relation, tail_id=''))

		return self._encode_hr_examples(examples)

	def _encode_hr_examples(self, examples: list[Example]) -> torch.Tensor:
		from utils.device import move_to_cuda

		if not examples:
			device = next(self.parameters()).device
			return torch.empty(0, self.config.hidden_size, device=device)

		data_loader = torch.utils.data.DataLoader(
			Dataset(path='', examples=examples, task=self.args.dataset),
			batch_size=min(len(examples), max(getattr(self.args, 'batch_size', 512), 512)),
			collate_fn=collate_hr,
			shuffle=False,
			num_workers=0,
		)
		vectors = []
		use_cuda = torch.cuda.is_available()
		for batch_dict in data_loader:
			if use_cuda:
				batch_dict = move_to_cuda(batch_dict)
			vectors.append(
				self.encode(
					batch_dict['hr_token_ids'],
					batch_dict['hr_mask'],
					batch_dict['hr_token_type_ids'],
				)
			)
		return torch.cat(vectors, dim=0)


def build_entity_embedder(args) -> TextEntityEmbedder:
	return TextEntityEmbedder(args)


def build_relation_embedder(args) -> TextQueryEmbedder:
	"""Build standalone query encoder; ``bind_model`` deep-copies entity weights for SimKGC."""

	return TextQueryEmbedder(args)
