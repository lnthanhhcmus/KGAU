"""1-vs-all broadcasting training paradigm (``TrainingJob1vsAll`` flow)."""

import math

import torch

from models.builder import (
	apply_kge_regularization,
	build_lr_scheduler,
	build_optimizer,
	init_index_kge_trainer,
	load_loss_fn_for_paradigm,
	run_index_kge_train_loop,
)
from utils.device import get_model_obj
from utils.logger import logger


class OneVsAllStrategy:
	"""Train by broadcasting (h, r) and (r, t) against all entities with cross-entropy."""

	def __init__(self, model, sampler, loss_fn, args, train_data=None, ngpus_per_node: int = 1, **_kwargs):
		del sampler, ngpus_per_node, _kwargs
		init_index_kge_trainer(self, model, args)
		if train_data is None:
			raise ValueError('OneVsAllStrategy requires train_data from build_pipeline')
		self.train_src, self.train_rel, self.train_dst = train_data
		self.loss_fn = loss_fn if loss_fn is not None else load_loss_fn_for_paradigm(args, '1vsall')

		if torch.cuda.is_available():
			self.train_src = self.train_src.to(self.device)
			self.train_rel = self.train_rel.to(self.device)
			self.train_dst = self.train_dst.to(self.device)

		batch_size = max(getattr(args, 'batch_size', 1), 1)
		num_batches = max(math.ceil(self.train_src.size(0) / batch_size), 1)
		weight_decay = float(getattr(args, 'weight_decay', None) or 0.0)
		self.weight_decay = weight_decay
		self.optimizer = build_optimizer(args, self.model.parameters(), self.weight_decay)
		self.lr_scheduler = build_lr_scheduler(args, self.optimizer)
		self.bidirectional = bool(getattr(args, 'bidirectional_1vsall', True))

	def _iter_batches(self, batch_size: int, src, rel, dst):
		for start in range(0, len(src), batch_size):
			end = start + batch_size
			yield src[start:end], rel[start:end], dst[start:end]

	def iter_training_batches(self, epoch: int, dataloader=None):
		del epoch, dataloader
		perm = torch.randperm(self.train_src.size(0), device=self.train_src.device)
		src = self.train_src.index_select(0, perm).to(self.device)
		rel = self.train_rel.index_select(0, perm).to(self.device)
		dst = self.train_dst.index_select(0, perm).to(self.device)
		batch_size = max(getattr(self.args, 'batch_size', 1), 1)
		yield from self._iter_batches(batch_size, src, rel, dst)

	def train_batch(self, batch, epoch: int) -> float:
		del epoch
		h_idx, r_idx, t_idx = batch
		model_obj = get_model_obj(self.model)
		h_idx = h_idx.to(self.device)
		r_idx = r_idx.to(self.device)
		t_idx = t_idx.to(self.device)

		self.optimizer.zero_grad()
		scores_hr = self.model.score_hr_(h_idx, r_idx)
		loss_hr = self.loss_fn(scores_hr, t_idx)

		if self.bidirectional and getattr(model_obj.get_scorer(), 'bidirectional_score_batch', False):
			scores_rt = self.model.score_rt_(r_idx, t_idx)
			loss_rt = self.loss_fn(scores_rt, h_idx)
			loss = (loss_hr + loss_rt) / 2.0
		else:
			loss = loss_hr

		batch_triples = torch.stack([h_idx, r_idx, t_idx], dim=1)
		loss = apply_kge_regularization(
			loss,
			self.model,
			self.args,
			batch_triples=batch_triples,
		)
		loss.backward()
		self.optimizer.step()
		return float(loss.item())

	def train_epoch(self, dataloader, epoch: int) -> float:
		del dataloader
		self.model.train()
		batch_size = max(getattr(self.args, 'batch_size', 1), 1)
		num_examples = self.train_src.size(0)

		epoch_loss = 0.0
		for batch in self.iter_training_batches(epoch):
			loss = self.train_batch(batch, epoch)
			h_idx = batch[0]
			epoch_loss += loss * h_idx.size(0)

		avg_loss = epoch_loss / max(num_examples, 1)
		logger.info('[EPOCH %s] Train | Loss: %.4f', epoch + 1, avg_loss)
		return avg_loss

	def train_loop(self, dataloader=None) -> dict:
		return run_index_kge_train_loop(self, dataloader)


Strategy = OneVsAllStrategy
SoftmaxStrategy = OneVsAllStrategy
