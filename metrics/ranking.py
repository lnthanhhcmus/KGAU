"""Ranking metrics for evaluating KG models."""

from typing import Sequence, List, Tuple
import torch

from configs.config import args
from data.dataset import EntityDict, Example
from data.dict_hub import get_link_graph


def topk_accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1,)) -> List[torch.Tensor]:
    """Compute top-k classification accuracy (percentage) for each k in `topk`.

    Returns a list of tensors containing the percentage accuracy for each requested k.
    """

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        results: List[torch.Tensor] = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            results.append(correct_k.mul_(100.0 / batch_size))
        return results


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)) -> list:
    """Backward-compatible alias for top-k accuracy."""

    return topk_accuracy(output, target, topk=topk)


def ranks_from_score_matrix(
	score: torch.Tensor,
	target_indices: torch.Tensor,
	*,
	tie_handling: str = 'rounded_mean_rank',
	tie_rtol: float = 1e-4,
	tie_atol: float = 1e-5,
) -> list[int]:
	"""Compute 1-based filtered ranks with configurable tie handling."""

	scores = score.clone()
	scores[torch.isnan(scores)] = float('-inf')
	target_scores = scores.gather(1, target_indices.unsqueeze(1))
	target_scores = target_scores.clone()
	target_scores[torch.isnan(target_scores)] = float('-inf')

	is_close = torch.isclose(scores, target_scores, rtol=tie_rtol, atol=tie_atol)
	is_greater = scores > target_scores
	num_ties = torch.sum(is_close, dim=1, dtype=torch.long)
	rank_zero = torch.sum(is_greater & ~is_close, dim=1, dtype=torch.long)

	if tie_handling == 'rounded_mean_rank':
		ranks_zero = rank_zero + num_ties // 2
	elif tie_handling == 'best_rank':
		ranks_zero = rank_zero
	elif tie_handling == 'worst_rank':
		ranks_zero = rank_zero + num_ties - 1
	else:
		raise ValueError(f'Unsupported tie_handling={tie_handling!r}')

	return ranks_zero.add(1).tolist()


def ranking_metrics_from_ranks(ranks: Sequence[int]) -> dict:
    """Compute link-prediction metrics from 1-based ranks.

    Returns a dictionary containing 'mr', 'mrr', and hit@k metrics.
    """

    ranks_list = list(ranks)
    if not ranks_list:
        raise ValueError('ranks must not be empty')

    total = float(len(ranks_list))
    mr = sum(ranks_list) / total
    mrr = sum(1.0 / rank for rank in ranks_list) / total
    hit_at_1 = sum(1 for rank in ranks_list if rank <= 1) / total
    hit_at_3 = sum(1 for rank in ranks_list if rank <= 3) / total
    hit_at_10 = sum(1 for rank in ranks_list if rank <= 10) / total
    return {
        'mr': round(mr, 4),
        'mrr': round(mrr, 4),
        'hit@1': round(hit_at_1, 4),
        'hit@3': round(hit_at_3, 4),
        'hit@10': round(hit_at_10, 4),
    }


def ranking_metrics_from_scores(scores: torch.Tensor, targets: torch.Tensor, topk: Tuple[int, ...] = (1, 3, 10)) -> Tuple[List[List[float]], List[List[int]], dict, List[int]]:
    """Compute link-prediction metrics from a score matrix and target indices.

    Args:
        scores: Tensor of shape (batch_size, num_entities), higher is better.
        targets: Tensor of shape (batch_size,) or (batch_size, 1) with target entity indices.
        topk: Retained for symmetry with accuracy-style helpers.

    Returns:
        A tuple of (topk_scores, topk_indices, metrics, ranks).
    """

    with torch.no_grad():
        if targets.dim() == 2 and targets.size(1) == 1:
            targets = targets.view(-1)
        elif targets.dim() != 1:
            raise ValueError('targets must have shape (batch_size,) or (batch_size, 1)')

        maxk = max(topk)
        sorted_scores, sorted_indices = torch.sort(scores, dim=-1, descending=True)
        target_rank = torch.nonzero(sorted_indices.eq(targets.unsqueeze(-1)).long(), as_tuple=False)
        if target_rank.size(0) != scores.size(0):
            raise RuntimeError('Unable to locate one target rank per example')

        ranks = ranks_from_score_matrix(scores, targets.view(-1))
        metrics = ranking_metrics_from_ranks(ranks)
        topk_scores = sorted_scores[:, :maxk].tolist()
        topk_indices = sorted_indices[:, :maxk].tolist()
        return topk_scores, topk_indices, metrics, ranks


def link_prediction_metrics(ranks: Sequence[int]) -> dict:
    """Alias for ranking_metrics_from_ranks for link prediction tasks."""

    return ranking_metrics_from_ranks(ranks)


def rerank_by_graph(batch_score: torch.Tensor, examples: Sequence[Example], entity_dict: EntityDict) -> None:
    """Re-rank entity scores using the local link graph.

    Modifies `batch_score` in-place by adding a small delta to entities
    that are within `args.rerank_n_hop` hops in the training graph.
    """
    neighbor_weight = 0.0 if args.neighbor_weight is None else args.neighbor_weight
    rerank_n_hop = 2 if args.rerank_n_hop is None else args.rerank_n_hop

    if args.dataset == 'wiki5m_ind':
        assert neighbor_weight < 1e-6, 'Inductive setting can not use re-rank strategy'

    if neighbor_weight < 1e-6:
        return

    for idx in range(batch_score.size(0)):
        cur_ex = examples[idx]
        n_hop_indices = get_link_graph().get_n_hop_entity_indices(
            cur_ex.head_id,
            entity_dict=entity_dict,
            n_hop=rerank_n_hop,
        )
        delta = torch.tensor([neighbor_weight for _ in n_hop_indices]).to(batch_score.device)
        n_hop_indices = torch.LongTensor(list(n_hop_indices)).to(batch_score.device)

        batch_score[idx].index_add_(0, n_hop_indices, delta)

