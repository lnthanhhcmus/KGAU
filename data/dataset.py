"""Dataset and graph helpers for generic KG workloads."""

from __future__ import annotations

import json
import os
from collections import deque
from dataclasses import dataclass
from typing import Any, List, Optional

import torch
import torch.utils.data.dataset

from configs.config import args
from data.preprocess import _concat_name_desc, _parse_entity_name
from utils.logger import logger

from data.dict_hub import get_entity_dict, get_link_graph, get_tokenizer


def _get_entity_dict() -> EntityDict:
	"""Get the entity dictionary, which provides mapping from entity IDs to their descriptions."""

	return get_entity_dict()


def _get_link_graph() -> LinkGraph:
	"""Get the link graph, which provides neighbor information for entities."""

	return get_link_graph()


def _get_tokenizer() -> Any:
	"""Get the tokenizer, which is used to tokenize text inputs."""

	return get_tokenizer()


def reverse_triplet(obj) -> dict:
	"""Given a triplet object, return a new triplet object with head and tail reversed, and relation modified to indicate inversion."""

	return {
		'head_id': obj['tail_id'],
		'head': obj['tail'],
		'relation': 'inverse {}'.format(obj['relation']),
		'tail_id': obj['head_id'],
		'tail': obj['head'],
	}


@dataclass
class EntityExample:
	"""Data class representing an entity example, including its ID, name, and description."""

	entity_id: str
	entity: str
	entity_desc: str = ''


class TripletDict:
	"""Data structure for storing triplets and providing neighbor information for entities."""

	def __init__(self, path_list: List[str]):
		self.path_list = path_list
		self.relations = set()
		self.hr2tails = {}
		self.rt2heads = {}
		self.triplet_cnt = 0

		for path in self.path_list:
			self._load(path)

	def _load(self, path: str) -> None:
		"""Load triplets from a given path and populate the internal data structures for neighbor retrieval."""

		examples = []
		if path.endswith('.json'):
			examples = json.load(open(path, 'r', encoding='utf-8'))
		elif path.endswith('.txt'):
			with open(path, 'r', encoding='utf-8') as reader:
				for line in reader:
					fields = line.strip().split('\t')
					if len(fields) not in (3, 4):
						continue
					head_id, relation, tail_id = fields[:3]
					examples.append({'head_id': head_id, 'relation': relation, 'tail_id': tail_id})
		else:
			raise ValueError(f'Unsupported format: {path}')

		for ex in examples:
			rt_key = (ex['relation'], ex['tail_id'])
			if rt_key not in self.rt2heads:
				self.rt2heads[rt_key] = set()
			self.rt2heads[rt_key].add(ex['head_id'])

		reversed_examples = [
			{
				'head_id': ex['tail_id'],
				'relation': 'inverse {}'.format(ex['relation']),
				'tail_id': ex['head_id'],
			}
			for ex in examples
		]
		examples += reversed_examples
		for ex in examples:
			self.relations.add(ex['relation'])
			key = (ex['head_id'], ex['relation'])
			if key not in self.hr2tails:
				self.hr2tails[key] = set()
			self.hr2tails[key].add(ex['tail_id'])
		self.triplet_cnt += len(examples)

	def get_neighbors(self, h: str, r: str) -> set:
		"""Given a head entity ID and a relation, return the set of tail entity IDs that are connected to the head via the relation."""

		return self.hr2tails.get((h, r), set())

	def get_heads(self, r: str, t: str) -> set:
		"""Given a relation and tail entity ID, return known head entities for filtered head prediction."""

		return self.rt2heads.get((r, t), set())


class EntityDict:
	"""Data structure for storing entity information and providing mapping from entity IDs to their descriptions."""

	def __init__(self, entity_dict_dir: str, inductive_test_path: str = None):
		path = os.path.join(entity_dict_dir, 'entities.json')
		assert os.path.exists(path)
		self.entity_exs = [EntityExample(**obj) for obj in json.load(open(path, 'r', encoding='utf-8'))]
		self._ensure_entity_coverage(entity_dict_dir)

		if inductive_test_path:
			examples = json.load(open(inductive_test_path, 'r', encoding='utf-8'))
			valid_entity_ids = set()
			for ex in examples:
				valid_entity_ids.add(ex['head_id'])
				valid_entity_ids.add(ex['tail_id'])
			self.entity_exs = [ex for ex in self.entity_exs if ex.entity_id in valid_entity_ids]

		self.id2entity = {ex.entity_id: ex for ex in self.entity_exs}
		self.entity2idx = {ex.entity_id: i for i, ex in enumerate(self.entity_exs)}
		logger.info('Load {} entities from {}'.format(len(self.id2entity), path))

	def _ensure_entity_coverage(self, entity_dict_dir: str) -> None:
		"""Backfill entities that appear in raw split files but are missing from entities.json."""

		from configs.config import args as current_args

		known_entity_ids = {ex.entity_id for ex in self.entity_exs}
		missing_entity_ids = set()
		for split_path in [getattr(current_args, 'train_path', ''), getattr(current_args, 'valid_path', ''), getattr(current_args, 'test_path', '')]:
			if not split_path or not os.path.exists(split_path):
				continue
			if split_path.endswith('.json'):
				with open(split_path, 'r', encoding='utf-8') as reader:
					for obj in json.load(reader):
						missing_entity_ids.add(obj['head_id'])
						missing_entity_ids.add(obj['tail_id'])
			else:
				with open(split_path, 'r', encoding='utf-8') as reader:
					for line in reader:
						fields = line.strip().split('\t')
						if len(fields) not in (3, 4):
							continue
						missing_entity_ids.add(fields[0])
						missing_entity_ids.add(fields[2])

		missing_entity_ids.difference_update(known_entity_ids)
		if not missing_entity_ids:
			return

		definition_candidates = [
			os.path.join(entity_dict_dir, '..', 'wordnet-mlj12-definitions.txt'),
			os.path.join(entity_dict_dir, 'wordnet-mlj12-definitions.txt'),
		]
		entity_text_map = {}
		for candidate in definition_candidates:
			if not os.path.exists(candidate):
				continue
			with open(candidate, 'r', encoding='utf-8') as reader:
				for line in reader:
					fields = line.strip().split('\t')
					if len(fields) != 3:
						continue
					entity_id, word, _ = fields
					entity_text_map[entity_id] = word.replace('__', ' ')
				break

		for entity_id in sorted(missing_entity_ids):
			self.entity_exs.append(EntityExample(
				entity_id=entity_id,
				entity=entity_text_map.get(entity_id, entity_id),
				entity_desc='',
			))

	def entity_to_idx(self, entity_id: str) -> int:
		"""Given an entity ID, return its corresponding index in the entity list."""

		return self.entity2idx[entity_id]

	def get_entity_by_id(self, entity_id: str) -> EntityExample:
		"""Given an entity ID, return the corresponding EntityExample object containing its description."""

		return self.id2entity[entity_id]

	def get_entity_by_idx(self, idx: int) -> EntityExample:
		"""Given an index, return the corresponding EntityExample object."""

		return self.entity_exs[idx]

	def __len__(self) -> int:
		"""Return the total number of entities in the dictionary."""

		return len(self.entity_exs)


class LinkGraph:
	"""Data structure for storing the link graph of entities, which allows retrieval of neighboring entities based on the training triplets."""

	def __init__(self, train_path: str):
		logger.info('Start to build link graph from {}'.format(train_path))
		self.graph = {}
		if train_path.endswith('.json'):
			examples = json.load(open(train_path, 'r', encoding='utf-8'))
		elif train_path.endswith('.txt'):
			examples = []
			with open(train_path, 'r', encoding='utf-8') as reader:
				for line in reader:
					fields = line.strip().split('\t')
					if len(fields) not in (3, 4):
						continue
					head_id, relation, tail_id = fields[:3]
					examples.append({'head_id': head_id, 'relation': relation, 'tail_id': tail_id})
		else:
			raise ValueError(f'Unsupported format: {train_path}')
		for ex in examples:
			head_id, tail_id = ex['head_id'], ex['tail_id']
			if head_id not in self.graph:
				self.graph[head_id] = set()
			self.graph[head_id].add(tail_id)
			if tail_id not in self.graph:
				self.graph[tail_id] = set()
			self.graph[tail_id].add(head_id)
		logger.info('Done build link graph with {} nodes'.format(len(self.graph)))

	def get_neighbor_ids(self, entity_id: str, max_to_keep=10) -> List[str]:
		"""Given an entity ID, return a list of neighboring entity IDs based on the link graph, limited to a maximum number of neighbors."""

		neighbor_ids = self.graph.get(entity_id, set())
		return sorted(list(neighbor_ids))[:max_to_keep]

	def get_n_hop_entity_indices(self, entity_id: str, entity_dict: EntityDict, n_hop: int = 2, max_nodes: int = 100000) -> set:
		"""Given an entity ID, return the set of neighboring entity indices within n hops in the link graph, limited to a maximum number of nodes to prevent explosion."""

		if n_hop < 0:
			return set()

		seen_eids = {entity_id}
		queue = deque([entity_id])
		for _ in range(n_hop):
			len_q = len(queue)
			for _ in range(len_q):
				tp = queue.popleft()
				for node in self.graph.get(tp, set()):
					if node not in seen_eids:
						queue.append(node)
						seen_eids.add(node)
						if len(seen_eids) > max_nodes:
							return set()
		return {entity_dict.entity_to_idx(e_id) for e_id in seen_eids}


class Example:
	"""Data class representing a single triplet example, including methods for vectorization and retrieval of entity descriptions."""

	def __init__(self, head_id, relation, tail_id, label=None, **kwargs):
		self.head_id = head_id
		self.tail_id = tail_id
		self.relation = relation
		self.label = int(label) if label is not None else None

	@property
	def head_desc(self) -> str:
		"""Return the description of the head entity, or an empty string if the head ID is not provided."""

		if not self.head_id:
			return ''
		return _get_entity_dict().get_entity_by_id(self.head_id).entity_desc

	@property
	def tail_desc(self) -> str:
		"""Return the description of the tail entity, or an empty string if the tail ID is not provided."""

		if not self.tail_id:
			return ''
		return _get_entity_dict().get_entity_by_id(self.tail_id).entity_desc

	@property
	def head(self) -> str:
		"""Return the name of the head entity, or an empty string if the head ID is not provided."""

		if not self.head_id:
			return ''
		return _get_entity_dict().get_entity_by_id(self.head_id).entity

	@property
	def tail(self) -> str:
		"""Return the name of the tail entity, or an empty string if the tail ID is not provided."""

		if not self.tail_id:
			return ''
		return _get_entity_dict().get_entity_by_id(self.tail_id).entity

	def vectorize(self) -> dict:
		"""Convert the example into a dictionary of token IDs and token type IDs for the head-relation pair, tail entity, and head entity, including optional neighbor descriptions if specified in the arguments."""

		head_desc, tail_desc = self.head_desc, self.tail_desc
		if args.use_link_graph:
			if len(head_desc.split()) < 20:
				head_desc += ' ' + get_neighbor_desc(head_id=self.head_id, tail_id=self.tail_id)
			if len(tail_desc.split()) < 20:
				tail_desc += ' ' + get_neighbor_desc(head_id=self.tail_id, tail_id=self.head_id)

		head_word = _parse_entity_name(self.head, task=args.dataset)
		head_text = _concat_name_desc(head_word, head_desc)
		hr_encoded_inputs = _custom_tokenize(text=head_text, text_pair=self.relation)

		head_encoded_inputs = _custom_tokenize(text=head_text)

		tail_word = _parse_entity_name(self.tail, task=args.dataset)
		tail_encoded_inputs = _custom_tokenize(text=_concat_name_desc(tail_word, tail_desc))

		out = {
			'hr_token_ids': hr_encoded_inputs['input_ids'],
			'hr_token_type_ids': hr_encoded_inputs['token_type_ids'],
			'tail_token_ids': tail_encoded_inputs['input_ids'],
			'tail_token_type_ids': tail_encoded_inputs['token_type_ids'],
			'head_token_ids': head_encoded_inputs['input_ids'],
			'head_token_type_ids': head_encoded_inputs['token_type_ids'],
			'obj': self,
		}
		if self.label is not None:
			out['label'] = self.label
		return out


def _custom_tokenize(text: str, text_pair: Optional[str] = None) -> dict:
	"""Custom tokenization function that uses the tokenizer to convert text (and an optional text pair) into token IDs and token type IDs, with truncation and special token handling as specified in the arguments."""

	tokenizer = _get_tokenizer()
	encoded_inputs = tokenizer(
		text=text,
		text_pair=text_pair if text_pair else None,
		add_special_tokens=True,
		max_length=args.max_num_tokens,
		return_token_type_ids=True,
		truncation=True,
	)
	return encoded_inputs



def get_neighbor_desc(head_id: str, tail_id: str = None) -> str:
	"""Given a head entity ID and an optional tail entity ID, return a concatenated string of the names of neighboring entities from the link graph, excluding the tail entity if specified."""

	neighbor_ids = _get_link_graph().get_neighbor_ids(head_id)
	if not args.is_test:
		neighbor_ids = [n_id for n_id in neighbor_ids if n_id != tail_id]
	entities = [_parse_entity_name(_get_entity_dict().get_entity_by_id(n_id).entity, task=args.dataset) for n_id in neighbor_ids]
	return ' '.join(entities)


class Dataset(torch.utils.data.dataset.Dataset):
	"""Custom dataset class for loading examples from specified paths, with support for both JSON and TXT formats, and optional addition of forward and backward triplets."""

	def __init__(self, path, task, examples=None):
		self.path_list = path.split(',')
		self.task = task
		assert examples is not None or all(os.path.exists(path) for path in self.path_list if path)
		if examples is not None:
			self.examples = examples
		else:
			self.examples = []
			for path in self.path_list:
				if not self.examples:
					self.examples = load_data(path)
				else:
					self.examples.extend(load_data(path))

	def __len__(self) -> int:
		"""Return the total number of examples in the dataset."""

		return len(self.examples)

	def __getitem__(self, index) -> dict:
		"""Given an index, return the vectorized representation of the corresponding example."""

		return self.examples[index].vectorize()


class PointwiseDataset(torch.utils.data.dataset.Dataset):
	"""Dataset for embedding models (e.g. DaBR) that only need Example objects, not token ids."""

	def __init__(self, examples):
		self.examples = examples

	def __len__(self) -> int:
		return len(self.examples)

	def __getitem__(self, index) -> dict:
		return {'obj': self.examples[index]}


def load_data(path: str, add_forward_triplet: bool = True, add_backward_triplet: bool = True) -> List[Example]:
	"""Load examples from a given path, which can be in JSON or TXT format, and return a list of Example objects. The function also supports adding forward and backward triplets based on the specified flags."""

	examples = []
	if path.endswith('.json'):
		data = json.load(open(path, 'r', encoding='utf-8'))
		logger.info('Load {} examples from {}'.format(len(data), path))
		for i, obj in enumerate(data):
			if add_forward_triplet:
				examples.append(Example(**obj))
			if add_backward_triplet:
				examples.append(Example(**reverse_triplet(obj)))
			data[i] = None
	elif path.endswith('.txt'):
		with open(path, 'r', encoding='utf-8') as f:
			for line in f:
				fs = line.strip().split('\t')
				if len(fs) == 4:
					head_id, relation, tail_id, label = fs
					if str(label) == '1':
						if add_forward_triplet:
							examples.append(Example(head_id=head_id, relation=relation, tail_id=tail_id, label=label))
						if add_backward_triplet:
							examples.append(Example(**reverse_triplet({'head_id': head_id, 'head': '', 'relation': relation, 'tail_id': tail_id, 'tail': ''})))
					elif not (add_forward_triplet or add_backward_triplet):
						examples.append(Example(head_id=head_id, relation=relation, tail_id=tail_id, label=label))
				elif len(fs) == 3:
					head_id, relation, tail_id = fs
					if add_forward_triplet:
						examples.append(Example(head_id=head_id, relation=relation, tail_id=tail_id, label='1'))
					if add_backward_triplet:
						examples.append(Example(**reverse_triplet({'head_id': head_id, 'head': '', 'relation': relation, 'tail_id': tail_id, 'tail': ''})))
	else:
		raise ValueError(f'Unsupported format: {path}')
	return examples
