"""Batch collation helpers for generic KG workloads."""

from typing import List
import torch

from configs.config import args
from data.dict_hub import get_tokenizer
from models.samplers.masking_sampler import construct_mask, construct_self_negative_mask


def to_indices_and_mask(batch_tensor, pad_token_id=0, need_mask=True) -> torch.Tensor | tuple[torch.Tensor, torch.ByteTensor]:
	"""Convert a list of variable-length tensors into a padded tensor and an optional mask."""

	mx_len = max([t.size(0) for t in batch_tensor])
	batch_size = len(batch_tensor)
	indices = torch.LongTensor(batch_size, mx_len).fill_(pad_token_id)
	if need_mask:
		mask = torch.ByteTensor(batch_size, mx_len).fill_(0)
	for i, t in enumerate(batch_tensor):
		indices[i, : len(t)].copy_(t)
		if need_mask:
			mask[i, : len(t)].fill_(1)
	if need_mask:
		return indices, mask
	return indices


def collate(batch_data: List[dict]) -> dict:
	"""Collate a batch of examples into tensors suitable for model input, including constructing masks for contrastive learning."""
	
	hr_token_ids, hr_mask = to_indices_and_mask(
		[torch.LongTensor(ex['hr_token_ids']) for ex in batch_data],
		pad_token_id=get_tokenizer().pad_token_id,
	)
	hr_token_type_ids = to_indices_and_mask(
		[torch.LongTensor(ex['hr_token_type_ids']) for ex in batch_data],
		need_mask=False,
	)

	tail_token_ids, tail_mask = to_indices_and_mask(
		[torch.LongTensor(ex['tail_token_ids']) for ex in batch_data],
		pad_token_id=get_tokenizer().pad_token_id,
	)
	tail_token_type_ids = to_indices_and_mask(
		[torch.LongTensor(ex['tail_token_type_ids']) for ex in batch_data],
		need_mask=False,
	)

	head_token_ids, head_mask = to_indices_and_mask(
		[torch.LongTensor(ex['head_token_ids']) for ex in batch_data],
		pad_token_id=get_tokenizer().pad_token_id,
	)
	head_token_type_ids = to_indices_and_mask(
		[torch.LongTensor(ex['head_token_type_ids']) for ex in batch_data],
		need_mask=False,
	)

	batch_exs = [ex['obj'] for ex in batch_data]
	needs_contrastive_mask = (
		not args.is_test
		and all(getattr(ex, 'tail_id', None) for ex in batch_exs)
	)
	batch_dict = {
		'hr_token_ids': hr_token_ids,
		'hr_mask': hr_mask,
		'hr_token_type_ids': hr_token_type_ids,
		'tail_token_ids': tail_token_ids,
		'tail_mask': tail_mask,
		'tail_token_type_ids': tail_token_type_ids,
		'head_token_ids': head_token_ids,
		'head_mask': head_mask,
		'head_token_type_ids': head_token_type_ids,
		'batch_data': batch_exs,
		'triplet_mask': construct_mask(row_exs=batch_exs) if needs_contrastive_mask else None,
		'self_negative_mask': construct_self_negative_mask(batch_exs) if needs_contrastive_mask else None,
	}

	return batch_dict


def collate_hr(batch_data: List[dict]) -> dict:
	"""Collate head-relation query batches without contrastive masks."""

	hr_token_ids, hr_mask = to_indices_and_mask(
		[torch.LongTensor(ex['hr_token_ids']) for ex in batch_data],
		pad_token_id=get_tokenizer().pad_token_id,
	)
	hr_token_type_ids = to_indices_and_mask(
		[torch.LongTensor(ex['hr_token_type_ids']) for ex in batch_data],
		need_mask=False,
	)

	return {
		'hr_token_ids': hr_token_ids,
		'hr_mask': hr_mask,
		'hr_token_type_ids': hr_token_type_ids,
		'batch_data': [ex['obj'] for ex in batch_data],
	}


def collate_pointwise(batch_data: List[dict]) -> dict:
	"""Collate embedding-based pointwise batches without text tokenization."""

	return {'batch_data': [ex['obj'] for ex in batch_data]}
