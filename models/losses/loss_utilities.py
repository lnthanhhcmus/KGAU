"""Shared loss utilities for softmax-family and BCE-family objectives."""

import torch
import torch.nn.functional as F


def _ensure_index_targets(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(targets):
        return torch.as_tensor(targets, device=scores.device, dtype=torch.long)
    return targets.to(device=scores.device, dtype=torch.long)


def _labels_as_matrix(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Convert index labels to a one-hot/multi-hot matrix when needed."""

    if labels.dim() == 2:
        return labels.to(device=scores.device, dtype=scores.dtype)
    index_labels = labels.long().to(scores.device)
    matrix = torch.zeros(scores.shape, device=scores.device, dtype=scores.dtype)
    matrix[torch.arange(scores.size(0), device=scores.device), index_labels] = 1.0
    return matrix


def compute_softmax_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute CE/KL-style softmax loss from hard or soft targets."""

    if targets.dim() == 1:
        index_targets = _ensure_index_targets(scores, targets)
        return F.cross_entropy(scores, index_targets, reduction=reduction)

    target_matrix = targets.to(device=scores.device, dtype=scores.dtype)
    log_probs = F.log_softmax(scores, dim=1)
    target_probs = F.normalize(target_matrix, p=1, dim=1)
    return F.kl_div(log_probs, target_probs, reduction=reduction)


def compute_bce_loss(
    scores: torch.Tensor,
    targets: torch.Tensor | None = None,
    *,
    pos_scores: torch.Tensor | None = None,
    neg_scores: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    offset: float = 0.0,
    reduction: str = "mean",
    adversarial_temp: float | None = None,
) -> torch.Tensor:
    """Compute BCE loss for broadcast or negative-sampling training.

    Broadcast mode:
        - Provide ``scores`` and ``targets``.
    Negative-sampling mode:
        - Provide ``pos_scores`` and ``neg_scores`` (or ``scores`` as ``pos_scores``).
        - Optional ``weights`` for weighted batch averaging.
        - Optional ``adversarial_temp`` for adversarial negative weighting.
    """

    # Negative-sampling mode
    if neg_scores is not None or pos_scores is not None:
        if pos_scores is None:
            pos_scores = scores
        if pos_scores is None or neg_scores is None:
            raise ValueError("Negative-sampling BCE requires both pos_scores and neg_scores")

        pos_scores = pos_scores.reshape(-1)
        if neg_scores.dim() == 3:
            neg_scores = neg_scores.squeeze(-1)
        neg_scores = neg_scores.to(pos_scores.device)
        batch_size = max(pos_scores.size(0), 1)

        if offset != 0.0:
            pos_scores = pos_scores + offset
            neg_scores = neg_scores + offset

        if adversarial_temp is not None:
            if weights is None:
                weights = torch.ones_like(pos_scores)
            weights = weights.to(pos_scores.device).reshape(-1)
            pos_loss = -F.logsigmoid(pos_scores)
            neg_weights = F.softmax(neg_scores * adversarial_temp, dim=-1).detach()
            neg_loss = -(neg_weights * F.logsigmoid(-neg_scores)).sum(dim=-1)
            per_row = (pos_loss + neg_loss) / 2.0
            return (per_row * weights).sum() / weights.sum().clamp_min(1e-12)

        scores_mat = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
        labels = torch.zeros_like(scores_mat)
        labels[:, 0] = 1.0
        per_row = F.binary_cross_entropy_with_logits(scores_mat, labels, reduction="none").sum(dim=1)
        if weights is not None:
            weights = weights.to(scores_mat.device).reshape(-1)
            return (per_row * weights).sum() / weights.sum().clamp_min(1e-12)
        return per_row.sum() / batch_size

    # Broadcast mode
    if scores is None or targets is None:
        raise ValueError("Broadcast BCE requires scores and targets")
    target_matrix = _labels_as_matrix(scores, targets)
    if offset != 0.0:
        scores = scores + offset
    return F.binary_cross_entropy_with_logits(scores, target_matrix, reduction=reduction)


def compute_softplus_loss(
    scores: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    *,
    pos_scores: torch.Tensor | None = None,
    neg_scores: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute pointwise logistic (softplus) loss for higher-is-better scores.

    Explicit mode:
        - Provide ``scores`` and ``labels`` with values in ``{+1, -1}``.
    Negative-sampling mode:
        - Provide ``pos_scores`` and ``neg_scores`` (or ``scores`` as ``pos_scores``).
        - Optional ``weights`` for weighted batch averaging.
    """

    if neg_scores is not None or pos_scores is not None:
        if pos_scores is None:
            pos_scores = scores
        if pos_scores is None or neg_scores is None:
            raise ValueError("Negative-sampling softplus loss requires both pos_scores and neg_scores")

        pos_scores = pos_scores.reshape(-1)
        if neg_scores.dim() == 3:
            neg_scores = neg_scores.squeeze(-1)
        neg_scores = neg_scores.to(pos_scores.device)
        batch_size = max(pos_scores.size(0), 1)

        if weights is not None:
            weights = weights.to(pos_scores.device).reshape(-1)
            pos_loss = F.softplus(-pos_scores)
            neg_flat = neg_scores.reshape(-1)
            if neg_flat.size(0) == batch_size:
                neg_2d = neg_flat.unsqueeze(1)
            elif neg_flat.size(0) % batch_size == 0:
                neg_2d = neg_flat.view(batch_size, -1)
            else:
                labels_cat = torch.cat(
                    [torch.ones_like(pos_scores), -torch.ones_like(neg_flat)],
                    dim=0,
                )
                scores_cat = torch.cat([pos_scores, neg_flat], dim=0)
                per_example = F.softplus(-scores_cat * labels_cat)
                return per_example.mean() if reduction == "mean" else per_example.sum()
            neg_loss = F.softplus(neg_2d).mean(dim=1)
            per_row = (pos_loss + neg_loss) / 2.0
            return (per_row * weights).sum() / weights.sum().clamp_min(1e-12)

        labels_cat = torch.cat(
            [torch.ones_like(pos_scores), -torch.ones_like(neg_scores.reshape(-1))],
            dim=0,
        )
        scores_cat = torch.cat([pos_scores, neg_scores.reshape(-1)], dim=0)
        per_example = F.softplus(-scores_cat * labels_cat)
        if reduction == "sum":
            return per_example.sum()
        if reduction == "none":
            return per_example
        return per_example.mean()

    if scores is None or labels is None:
        raise ValueError("Explicit softplus loss requires scores and labels")

    scores_flat = scores.view(-1)
    labels_flat = labels.view(-1).to(scores.device, dtype=scores.dtype)
    per_example = F.softplus(-scores_flat * labels_flat)
    if reduction == "sum":
        return per_example.sum()
    if reduction == "none":
        return per_example
    return per_example.mean()


def compute_bpr_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute ``softplus(neg - pos)`` averaged over positive–negative pairs.

    Equivalent to ``-log(sigmoid(pos - neg))``. Assumes higher scores are better.
    """

    if pos_scores.dim() == 1 and neg_scores.dim() == 2:
        pos_scores = pos_scores.unsqueeze(1).expand_as(neg_scores)
    per_pair = F.softplus(neg_scores - pos_scores)
    if weights is not None:
        weights = weights.to(pos_scores.device).reshape(-1, 1)
        return (per_pair * weights).sum() / weights.sum().clamp_min(1e-12)
    return per_pair.mean()


def compute_margin_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    *,
    margin: float,
) -> torch.Tensor:
    """Compute ``max(0, margin - pos_score + neg_score)`` averaged over pairs.

    Assumes higher scores are better (similarity). For distance-based scores,
    flip the sign convention before calling this function.
    """

    if pos_scores.dim() == 1 and neg_scores.dim() == 2:
        pos_scores = pos_scores.unsqueeze(1).expand_as(neg_scores)
    loss = F.relu(float(margin) - pos_scores + neg_scores)
    return loss.mean()


def _margin_broadcast_1vsall(
    scores: torch.Tensor,
    targets: torch.Tensor,
    *,
    margin: float,
    reduction: str,
) -> torch.Tensor:
    batch_size, num_entities = scores.shape
    index_targets = _ensure_index_targets(scores, targets)
    batch_idx = torch.arange(batch_size, device=scores.device)
    pos_scores = scores[batch_idx, index_targets]

    pair_loss = F.relu(float(margin) - pos_scores.unsqueeze(1) + scores)
    pair_loss[batch_idx, index_targets] = 0.0

    num_neg = max(num_entities - 1, 1)
    row_loss = pair_loss.sum(dim=1) / num_neg
    if reduction == "sum":
        return row_loss.sum()
    return row_loss.mean()


def _margin_broadcast_kvsall(
    scores: torch.Tensor,
    target_matrix: torch.Tensor,
    *,
    margin: float,
    reduction: str,
) -> torch.Tensor:
    num_entities = target_matrix.size(1)
    pos_threshold = 1.0 / num_entities
    row_losses: list[torch.Tensor] = []

    for row_idx in range(scores.size(0)):
        pos_mask = target_matrix[row_idx] > pos_threshold
        neg_mask = ~pos_mask
        if not bool(pos_mask.any()) or not bool(neg_mask.any()):
            continue
        pos = scores[row_idx, pos_mask]
        neg = scores[row_idx, neg_mask]
        pair_loss = F.relu(float(margin) - pos.unsqueeze(1) + neg.unsqueeze(0))
        row_losses.append(pair_loss.sum())

    if not row_losses:
        return scores.new_zeros(())

    stacked = torch.stack(row_losses)
    if reduction == "sum":
        return stacked.sum()
    return stacked.mean()


def compute_margin_broadcast_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    *,
    margin: float,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute margin ranking over positive-vs-negative entity pairs for broadcast scores.

    :param scores: ``[batch_size, num_entities]`` logits from ``score_hr_()`` / ``score_rt_()``
    :param targets: ``[batch_size]`` entity indices (1vsAll) or ``[batch_size, num_entities]`` multi-hot labels (KvsAll)
    :param margin: Margin hyperparameter (``train.loss_arg`` for margin_ranking)
    :param reduction: ``mean`` (1vsAll default) or ``sum`` (KvsAll default; strategy divides by batch size)
    """

    if targets.dim() == 1:
        return _margin_broadcast_1vsall(scores, targets, margin=margin, reduction=reduction)

    target_matrix = _labels_as_matrix(scores, targets)
    return _margin_broadcast_kvsall(scores, target_matrix, margin=margin, reduction=reduction)


def compute_margin_ranking_loss(
    scores: torch.Tensor | None = None,
    targets: torch.Tensor | None = None,
    *,
    pos_scores: torch.Tensor | None = None,
    neg_scores: torch.Tensor | None = None,
    margin: float,
    reduction: str = "mean",
) -> torch.Tensor:
    """Thin dispatcher for margin ranking (negsamp vs broadcast)."""

    if pos_scores is not None or neg_scores is not None:
        if pos_scores is None or neg_scores is None:
            raise ValueError("Pairwise margin dispatch requires both pos_scores and neg_scores")
        if reduction != "mean":
            raise ValueError("Pairwise margin loss only supports reduction='mean'")
        return compute_margin_loss(pos_scores, neg_scores, margin=margin)

    if scores is None or targets is None:
        raise ValueError("Broadcast margin ranking dispatch requires scores and targets")
    return compute_margin_broadcast_loss(scores, targets, margin=margin, reduction=reduction)
