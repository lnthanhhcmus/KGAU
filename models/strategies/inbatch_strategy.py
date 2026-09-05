"""In-batch contrastive training paradigm (SimKGC-style)."""

import json
import os
import time

import torch
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from base.evaluator import Evaluator, log_bidirectional_link_metrics
from data.dataset import Dataset, load_data
from data.dataloader import collate
from data.dict_hub import build_tokenizer, get_entity_dict
from metrics.ranking import topk_accuracy as accuracy
from models.builder import (
	_kge_resolve_early_stopping,
	_kge_update_early_stopping_bad_count,
	import_module_from_path,
	init_index_kge_trainer,
	load_attr_from_path,
	load_loss_fn,
)
from utils.checkpoint import best_model_path, last_model_path, save_checkpoint
from utils.device import call_model_forward, get_model_obj, move_to_cuda, report_num_trainable_parameters, setup_data_parallel
from utils.logger import AverageMeter, ProgressMeter, logger, log_run_timing
from utils.memory import PhaseMemoryTracker


class InBatchStrategy(Evaluator):
	"""Train with in-batch negatives: broadcast queries against batch targets only."""

	def __init__(self, model, sampler, loss_fn, args, ngpus_per_node=1, **_kwargs):
		del sampler
		Evaluator.__init__(self)
		self.args = args
		self.ngpus_per_node = ngpus_per_node
		build_tokenizer(args)

		if model is None:
			from models.builder import build_model

			model = build_model(args)
		logger.info(model)

		from data.dict_hub import warmup_data_structures

		warmup_data_structures()
		init_index_kge_trainer(self, model, args)
		self._setup_training()
		self.criterion = loss_fn if loss_fn is not None else load_loss_fn(args)
		self._load_loss_helpers(args)

		self.optimizer = AdamW(
			[p for p in self.model.parameters() if p.requires_grad],
			lr=args.lr,
			weight_decay=float(args.weight_decay) if args.weight_decay is not None else 0.0,
		)
		report_num_trainable_parameters(self.model)

		train_dataset = Dataset(path=args.train_path, task=args.dataset)
		valid_dataset = Dataset(path=args.valid_path, task=args.dataset) if args.valid_path else None
		num_training_steps = args.epochs * len(train_dataset) // max(args.batch_size, 1)
		args.warmup = min(args.warmup, num_training_steps // 10)
		logger.info('Total training steps: {}, warmup steps: {}'.format(num_training_steps, args.warmup))
		self.scheduler = self._create_lr_scheduler(num_training_steps)

		train_loader_kwargs = {
			'dataset': train_dataset,
			'batch_size': args.batch_size,
			'shuffle': True,
			'collate_fn': collate,
			'num_workers': args.workers,
			'pin_memory': True,
			'drop_last': True,
		}
		if int(args.workers) > 0:
			train_loader_kwargs['persistent_workers'] = True
			train_loader_kwargs['prefetch_factor'] = 4
		self.train_loader = torch.utils.data.DataLoader(**train_loader_kwargs)

		self.valid_loader = None
		if valid_dataset:
			valid_loader_kwargs = {
				'dataset': valid_dataset,
				'batch_size': args.batch_size * 2,
				'shuffle': True,
				'collate_fn': collate,
				'num_workers': args.workers,
				'pin_memory': True,
			}
			if int(args.workers) > 0:
				valid_loader_kwargs['persistent_workers'] = True
				valid_loader_kwargs['prefetch_factor'] = 4
			self.valid_loader = torch.utils.data.DataLoader(**valid_loader_kwargs)

		self.memory_tracker = PhaseMemoryTracker()
		self.best_epoch = None

	def _setup_training(self) -> None:
		"""Match reference SimKGC: DataParallel on multi-GPU, else single CUDA device."""

		self.model = setup_data_parallel(self.model)
		self.device = next(self.model.parameters()).device

	def _load_loss_helpers(self, args) -> None:
		loss_path = getattr(args, 'model_loss_path', '') or 'models/losses/infonce_loss.py'
		try:
			self.ModelOutput = load_attr_from_path(loss_path, 'ModelOutput')
		except Exception:
			loss_mod = import_module_from_path(loss_path)
			self.ModelOutput = getattr(loss_mod, 'ModelOutput')

	def _create_lr_scheduler(self, num_training_steps):
		if self.args.lr_scheduler == 'linear':
			return get_linear_schedule_with_warmup(
				optimizer=self.optimizer,
				num_warmup_steps=self.args.warmup,
				num_training_steps=num_training_steps,
			)
		if self.args.lr_scheduler == 'cosine':
			return get_cosine_schedule_with_warmup(
				optimizer=self.optimizer,
				num_warmup_steps=self.args.warmup,
				num_training_steps=num_training_steps,
			)
		raise ValueError(f'Unknown lr scheduler: {self.args.lr_scheduler}')

	def _validation_interval(self) -> int:
		raw = getattr(self.args, 'epoch_per_eval', None)
		if raw is not None:
			interval = int(raw)
			max_epochs = max(int(self.args.epochs), 1)
			if interval <= 0 or interval > max_epochs:
				return max_epochs
			return interval
		raw = getattr(self.args, 'eval_interval_epochs', None)
		if raw is not None:
			return max(int(raw), 1)
		return 1

	def _should_validate(self, epoch: int) -> bool:
		interval = self._validation_interval()
		epoch_number = epoch + 1
		max_epochs = max(int(self.args.epochs), 1)
		return epoch_number % interval == 0 or epoch_number >= max_epochs

	def _eval_every_n_step(self) -> int | None:
		raw = getattr(self.args, 'eval_every_n_step', None)
		if raw is None:
			raw = getattr(self.args, 'valid_steps', None)
		if raw is None:
			return None
		return max(int(raw), 1)

	def _should_run_link_prediction(self, epoch: int) -> bool:
		raw = getattr(self.args, 'valid_link_prediction_epochs', None)
		if raw is None:
			return True
		interval = int(raw)
		if interval <= 0:
			return False
		return (epoch + 1) % interval == 0 or epoch == self.args.epochs - 1

	def _encode_all_entities_for_eval(self) -> torch.Tensor:
		from data.dataset import Dataset, Example

		entity_examples = [
			Example(head_id='', relation='', tail_id=entity_ex.entity_id)
			for entity_ex in get_entity_dict().entity_exs
		]
		batch_size = max(int(getattr(self.args, 'eval_batch_size', 128) or 128), 512)
		entity_loader = torch.utils.data.DataLoader(
			Dataset(path='', examples=entity_examples, task=self.args.dataset),
			num_workers=0,
			batch_size=batch_size,
			collate_fn=collate,
			shuffle=False,
		)
		entity_vectors = []
		use_cuda = torch.cuda.is_available()
		for batch_dict in entity_loader:
			batch_dict['only_ent_embedding'] = True
			if use_cuda:
				batch_dict = move_to_cuda(batch_dict)
			outputs = call_model_forward(self.model, batch_dict)
			entity_vectors.append(outputs['ent_vectors'])
		return torch.cat(entity_vectors, dim=0)

	def _grad_clip_value(self) -> float:
		raw = getattr(self.args, 'grad_clip', None)
		return 10.0 if raw is None else float(raw)

	def train_loop(self) -> dict:
		"""Reference SimKGC ``Trainer.train_loop`` cadence (train time separate from validation)."""

		if self.args.use_amp and not hasattr(self, 'scaler'):
			self.scaler = torch.cuda.amp.GradScaler()

		validation_interval = self._validation_interval()
		logger.info('Validation interval: every %d epoch(s)', validation_interval)

		patience, min_epochs, min_metric = _kge_resolve_early_stopping(self.args)
		bad_counts = 0
		if patience is not None:
			logger.info(
				'Early stopping: stop after %d validation(s) without MRR improvement (min_epochs=%d).',
				patience,
				min_epochs,
			)

		total_start = time.time()
		train_time = 0.0
		valid_time = 0.0
		num_train_epochs = 0

		for epoch in range(self.args.epochs):
			epoch_train_start = time.time()
			self.memory_tracker.begin_phase()
			self.train_epoch(epoch)
			self.memory_tracker.end_phase('train')
			train_time += time.time() - epoch_train_start
			num_train_epochs = epoch + 1

			if self._should_validate(epoch):
				val_start = time.time()
				metric_dict, is_best = self._run_eval(epoch=epoch)
				valid_time += time.time() - val_start

				bad_counts = _kge_update_early_stopping_bad_count(
					self,
					metric_dict=metric_dict,
					is_best=is_best,
					bad_counts=bad_counts,
					min_metric=min_metric,
				)
				if patience is not None and bad_counts >= patience and (epoch + 1) >= min_epochs:
					logger.info(
						'[EARLY STOP] No validation MRR improvement for %d evaluations (epoch %d).',
						patience,
						epoch + 1,
					)
					break
			else:
				logger.info(
					'Skip validation at epoch %d (eval every %d epochs)',
					epoch,
					validation_interval,
				)

		self.train_time = train_time
		self.valid_time = valid_time
		self.total_time = time.time() - total_start
		self.num_train_epochs = num_train_epochs
		log_run_timing(
			train_time=train_time,
			valid_time=valid_time,
			total_time=self.total_time,
			num_train_epochs=self.num_train_epochs,
		)
		logger.info('[Timing] Training time (s): %s', round(train_time, 2))
		logger.info('[Timing] Total run time (s): %s', round(self.total_time, 2))
		return {
			'train_time': train_time,
			'valid_time': valid_time,
			'total_time': self.total_time,
			'num_train_epochs': self.num_train_epochs,
			'best_epoch': None if self.best_metric is None else self.best_metric.get('epoch', 0) + 1,
			'best_mrr': None if self.best_metric is None else self.best_metric.get('score'),
			'best_metric': self.best_metric,
			'best_checkpoint_path': self.best_checkpoint_path,
		}

	def _compute_infonce_loss(
		self,
		logits: torch.Tensor,
		labels: torch.Tensor,
		batch_size: int,
		*,
		symmetric: bool,
	) -> torch.Tensor:
		loss = self.criterion(logits, labels)
		if symmetric:
			loss = loss + self.criterion(logits[:, :batch_size].t(), labels)
		return loss

	def train_epoch(self, epoch) -> None:
		losses = AverageMeter('Loss', ':.4')
		top1 = AverageMeter('Acc@1', ':6.2f')
		top3 = AverageMeter('Acc@3', ':6.2f')
		inv_t = AverageMeter('InvT', ':6.2f')
		progress = ProgressMeter(
			len(self.train_loader),
			[losses, inv_t, top1, top3],
			prefix='Epoch: [{}]'.format(epoch),
		)

		eval_every_n_step = self._eval_every_n_step()
		grad_clip = self._grad_clip_value()

		for i, batch_dict in enumerate(self.train_loader):
			self.model.train()

			if torch.cuda.is_available():
				batch_dict = move_to_cuda(batch_dict)
			batch_size = len(batch_dict['batch_data'])

			if self.args.use_amp:
				with torch.amp.autocast(device_type='cuda'):
					outputs = call_model_forward(self.model, batch_dict)
			else:
				outputs = call_model_forward(self.model, batch_dict)

			outputs = get_model_obj(self.model).compute_logits(output_dict=outputs, batch_dict=batch_dict)
			outputs = self.ModelOutput(**outputs)
			logits, labels = outputs.logits, outputs.labels
			assert logits.size(0) == batch_size

			loss = self._compute_infonce_loss(logits, labels, batch_size, symmetric=True)

			acc1, acc3 = accuracy(logits, labels, topk=(1, 3))
			top1.update(acc1.item(), batch_size)
			top3.update(acc3.item(), batch_size)
			inv_t.update(outputs.inv_t.item() if torch.is_tensor(outputs.inv_t) else outputs.inv_t, 1)
			losses.update(loss.item(), batch_size)

			self.optimizer.zero_grad()
			step_taken = True
			if self.args.use_amp:
				prev_scale = self.scaler.get_scale()
				self.scaler.scale(loss).backward()
				self.scaler.unscale_(self.optimizer)
				torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
				self.scaler.step(self.optimizer)
				self.scaler.update()
				step_taken = self.scaler.get_scale() >= prev_scale
			else:
				loss.backward()
				torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
				self.optimizer.step()

			if step_taken:
				self.scheduler.step()

			if i % self.args.print_freq == 0:
				progress.display(i)
			if eval_every_n_step is not None and (i + 1) % eval_every_n_step == 0:
				self._run_eval(epoch=epoch, step=i + 1)

		logger.info('Learning rate: {}'.format(self.scheduler.get_last_lr()[0]))
		log_str = f"[EPOCH {epoch}] Loss: {losses.avg:.4f} | Acc@1: {top1.avg:.2f} | Acc@3: {top3.avg:.2f}"
		logger.info(log_str)

	@torch.no_grad()
	def eval_epoch(self, epoch) -> dict:
		"""In-batch contrastive validation only (link prediction runs in ``_run_eval``)."""

		metric_dict = {}
		if not self.valid_loader:
			return metric_dict

		losses = AverageMeter('Loss', ':.4')
		top1 = AverageMeter('Acc@1', ':6.2f')
		top3 = AverageMeter('Acc@3', ':6.2f')

		for _, batch_dict in enumerate(self.valid_loader):
			self.model.eval()
			if torch.cuda.is_available():
				batch_dict = move_to_cuda(batch_dict)
			batch_size = len(batch_dict['batch_data'])

			outputs = call_model_forward(self.model, batch_dict)
			outputs = get_model_obj(self.model).compute_logits(output_dict=outputs, batch_dict=batch_dict)
			outputs = self.ModelOutput(**outputs)
			logits, labels = outputs.logits, outputs.labels
			loss = self._compute_infonce_loss(logits, labels, batch_size, symmetric=False)
			losses.update(loss.item(), batch_size)

			acc1, acc3 = accuracy(logits, labels, topk=(1, 3))
			top1.update(acc1.item(), batch_size)
			top3.update(acc3.item(), batch_size)

		metric_dict.update({
			'Acc@1': round(top1.avg, 3),
			'Acc@3': round(top3.avg, 3),
			'loss': round(losses.avg, 3),
		})
		if metric_dict:
			logger.info('Epoch {}, valid metric: {}'.format(epoch, json.dumps(metric_dict)))
		return metric_dict

	@torch.no_grad()
	def _run_eval(self, epoch, step=0) -> tuple[dict, bool]:
		logger.info('[EVAL] Starting validation for epoch %d...', epoch)
		eval_start = time.time()
		self.memory_tracker.begin_phase()
		metric_dict = self.eval_epoch(epoch)

		valid_mrr = None
		valid_eval_path = self._resolve_valid_eval_path()
		if valid_eval_path and os.path.exists(valid_eval_path) and self._should_run_link_prediction(epoch):
			valid_entity_dict = get_entity_dict()
			valid_output_path = os.path.join(self.args.output_dir, 'valid_link_prediction.log')
			entity_embs = self._encode_all_entities_for_eval()
			forward_metrics = self.evaluate_link_prediction_inplace(
				self.model, valid_eval_path, valid_entity_dict, valid_output_path,
				eval_forward=True,
				all_entity_embs=entity_embs,
			)
			backward_metrics = self.evaluate_link_prediction_inplace(
				self.model, valid_eval_path, valid_entity_dict, valid_output_path,
				eval_forward=False,
				all_entity_embs=entity_embs,
			)
			if forward_metrics and backward_metrics:
				metric_dict.update(
					log_bidirectional_link_metrics(f'[EPOCH {epoch}] Valid', forward_metrics, backward_metrics)
				)
				try:
					valid_mrr = round((forward_metrics.get('mrr', 0) + backward_metrics.get('mrr', 0)) / 2, 4)
					metric_dict['mrr'] = valid_mrr
				except Exception:
					valid_mrr = None

		self.memory_tracker.end_phase('eval')
		eval_elapsed = time.time() - eval_start
		logger.info('[EVAL] Finished validation for epoch %d in %.1fs', epoch, eval_elapsed)

		is_best = valid_mrr is not None and (
			self.best_metric is None or valid_mrr > self.best_metric.get('mrr', self.best_metric.get('score', float('-inf')))
		)
		if is_best:
			self.best_metric = {'mrr': valid_mrr, 'score': valid_mrr, 'metrics': metric_dict, 'epoch': epoch}
			self.best_epoch = epoch
			logger.info('[BEST] epoch=%d valid_mrr=%s', epoch + 1, valid_mrr)

		saved_checkpoint_path = save_checkpoint({
			'epoch': epoch,
			'best_epoch': epoch if is_best else getattr(self, 'best_epoch', None),
			'best_metric': self.best_metric,
			'args': self.args.__dict__,
			'state_dict': get_model_obj(self.model).state_dict(),
		}, is_best=is_best, filename=last_model_path(self.args.output_dir))
		if is_best:
			self.best_checkpoint_path = best_model_path(self.args.output_dir)
		elif self.best_checkpoint_path is None:
			self.best_checkpoint_path = saved_checkpoint_path
		return metric_dict, is_best

	def _resolve_valid_eval_path(self) -> str | None:
		if not self.args.valid_path:
			return None
		if os.path.exists(self.args.valid_path):
			return self.args.valid_path
		if self.args.valid_path.endswith('_w_label.txt'):
			cand_json = self.args.valid_path.replace('valid_w_label.txt', 'valid.txt.json')
			cand_txt = self.args.valid_path.replace('valid_w_label.txt', 'valid.txt')
			if os.path.exists(cand_json):
				return cand_json
			if os.path.exists(cand_txt):
				return cand_txt
		if self.args.valid_path.endswith('.txt.json') or self.args.valid_path.endswith('.txt'):
			return self.args.valid_path if os.path.exists(self.args.valid_path) else None
		return None


SimKGCStrategy = InBatchStrategy
Strategy = InBatchStrategy
