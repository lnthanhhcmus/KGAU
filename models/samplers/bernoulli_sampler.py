"""Bernoulli listwise sampler for DistMult and ComplEx training."""

from collections import defaultdict

import numpy as np
import torch
from numpy.random import choice


def get_bern_prob(data, n_ent, n_rel) -> torch.Tensor:
	"""Compute relation-specific Bernoulli corruption probabilities."""

	src, rel, dst = data
	edges = defaultdict(lambda: defaultdict(set))
	rev_edges = defaultdict(lambda: defaultdict(set))
	for s, r, t in zip(src, rel, dst):
		edges[int(r)][int(s)].add(int(t))
		rev_edges[int(r)][int(t)].add(int(s))

	bern_prob = torch.zeros(n_rel)
	for r in edges.keys():
		tph = sum(len(tails) for tails in edges[r].values()) / max(len(edges[r]), 1)
		htp = sum(len(heads) for heads in rev_edges[r].values()) / max(len(rev_edges[r]), 1)
		bern_prob[r] = tph / (tph + htp) if (tph + htp) > 0 else 0.5
	return bern_prob


class BernoulliListwiseSampler(object):
	"""Generate listwise candidate triples with the true triple in column 0."""

	def __init__(self, data, n_ent, n_rel, n_sample):
		self.bern_prob = get_bern_prob(data, n_ent, n_rel)
		self.n_ent = n_ent
		self.n_sample = n_sample

	def corrupt(self, src, rel, dst, keep_truth=True) -> (torch.Tensor, torch.Tensor, torch.Tensor):
		"""Corrupt triples with Bernoulli sampling to generate listwise candidates."""

		n = len(src)
		prob = self.bern_prob[rel]
		selection = torch.bernoulli(prob).cpu().numpy().astype('bool')
		src_np = src.cpu().numpy()
		dst_np = dst.cpu().numpy()

		src_out = np.tile(src_np, (self.n_sample, 1)).transpose()
		dst_out = np.tile(dst_np, (self.n_sample, 1)).transpose()
		rel_out = rel.unsqueeze(1).expand(n, self.n_sample)

		if keep_truth:
			ent_random = choice(self.n_ent, (n, self.n_sample - 1))
			src_out[selection, 1:] = ent_random[selection]
			dst_out[~selection, 1:] = ent_random[~selection]
		else:
			ent_random = choice(self.n_ent, (n, self.n_sample))
			src_out[selection, :] = ent_random[selection]
			dst_out[~selection, :] = ent_random[~selection]

		return torch.from_numpy(src_out).long(), rel_out.long(), torch.from_numpy(dst_out).long()


def _resolve_nentity(args, model) -> int:
	for attr in ('nentity', 'ent_total'):
		value = getattr(args, attr, None)
		if value is not None:
			return int(value)
	if model is not None:
		if hasattr(model, 'ent_embedder') and hasattr(model.ent_embedder, 'embedding'):
			return int(model.ent_embedder.embedding.num_embeddings)
		if hasattr(model, 'entity_embedding'):
			return int(model.entity_embedding.size(0))
	from data.dict_hub import get_entity_dict
	return len(get_entity_dict())


def _resolve_nrelation(args, model) -> int:
	for attr in ('nrelation', 'rel_total'):
		value = getattr(args, attr, None)
		if value is not None:
			return int(value)
	if model is not None and hasattr(model, 'rel_embedder'):
		rel_emb = model.rel_embedder
		if hasattr(rel_emb, 'embedding'):
			return int(rel_emb.embedding.num_embeddings)
		if hasattr(rel_emb, 'rel_re'):
			return int(rel_emb.rel_re.num_items)
	from utils.relations import load_relation_to_idx
	return max(len(load_relation_to_idx(args)), 1)


def build_sampler(args, train_triples, model):
	"""Construct a Bernoulli listwise sampler for DistMult/ComplEx training."""

	if train_triples is None:
		raise ValueError('train_triples is required for BernoulliListwiseSampler')
	src, rel, dst = train_triples[:, 0], train_triples[:, 1], train_triples[:, 2]
	n_ent = _resolve_nentity(args, model)
	n_rel = _resolve_nrelation(args, model)
	n_sample = getattr(args, 'n_sample', None)
	n_sample = 100 if n_sample is None else int(n_sample)
	return BernoulliListwiseSampler((src, rel, dst), n_ent, n_rel, n_sample)