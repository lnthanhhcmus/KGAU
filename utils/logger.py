"""Logging utilities for KGDirectAU."""

import json
import logging
from pathlib import Path
from typing import Optional, Union

from utils.memory import format_memory


DEFAULT_LOG_FORMAT = '[%(asctime)s %(levelname)s] %(name)s: %(message)s'


def get_logger(
    name: str = 'kg',
    log_file: Optional[Union[str, Path]] = None,
    level: int = logging.INFO,
    console: bool = True,
    propagate: bool = False,
) -> logging.Logger:
    """Create or reconfigure a logger with optional file output.

    Repeated calls replace existing handlers so the logger can be reused across
    different scripts without duplicate messages.
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = propagate

    formatter = logging.Formatter(DEFAULT_LOG_FORMAT)
    handlers = []

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    logger.handlers = handlers
    return logger


def setup_logger(name: str = 'kg', log_file: Optional[Union[str, Path]] = None) -> logging.Logger:
    """Compatibility wrapper for older call sites."""

    return get_logger(name=name, log_file=log_file)


logger = get_logger()


class AverageMeter:
    """Compute and store the average and current value."""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        """Reset all statistics."""

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1) -> None:
        """Update statistics with a new value."""

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        """String representation showing current and average values."""

        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    """Display progress during training or evaluation."""

    def __init__(self, num_batches, meters, prefix=''):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch: int) -> None:
        """Display the current progress and meter values."""

        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches: int) -> str:
        """Generate a format string for batch progress display."""

        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def format_duration(seconds: float) -> str:
    """Format a duration both as raw seconds and as h/m/s."""

    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f'{seconds:.2f}s ({hours}h {minutes}m {secs}s)'


def time_per_train_epoch(train_time: Optional[float], num_train_epochs: Optional[int]) -> Optional[float]:
    """Average wall-clock training time per completed epoch."""

    if train_time is None or not num_train_epochs:
        return None
    return train_time / num_train_epochs


def log_run_timing(
    *,
    train_time: float,
    valid_time: float,
    total_time: float,
    num_train_epochs: Optional[int] = None,
) -> Optional[float]:
    """Log training/validation/total timing."""

    epoch_time = time_per_train_epoch(train_time, num_train_epochs)
    logger.info('[Timing] Training time (s): %.2f', round(train_time, 2))
    logger.info('[Timing] Valid time (s): %.2f', round(valid_time, 2))
    logger.info('[Timing] Total run time (s): %.2f', round(total_time, 2))
    if epoch_time is not None:
        logger.info('[Timing] Time per training epoch (s): %.2f', round(epoch_time, 2))
    return epoch_time


def _format_metric_key(key: str) -> str:
    mapping = {
        'mr': 'MR',
        'mrr': 'MRR',
        'hit@1': 'Hit@1',
        'hit@3': 'Hit@3',
        'hit@10': 'Hit@10',
        'accuracy': 'Accuracy',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1': 'F1-Score',
        'pr_auc': 'PR-AUC',
        'roc_auc': 'ROC-AUC',
    }
    return mapping.get(key, key)


def _format_metric_value(value) -> str:
    if isinstance(value, float):
        return f'{value:.6f}'
    return str(value)


def _append_link_metrics_lines(lines: list[str], link_metrics: dict, section_title: str = 'Link Prediction') -> None:
    """Append formatted link-prediction metrics to a report line list."""

    lines.append(section_title)
    for key in ['mr', 'mrr', 'hit@1', 'hit@3', 'hit@10']:
        if key in link_metrics:
            lines.append(f'  {_format_metric_key(key)}: {_format_metric_value(link_metrics[key])}')
    lines.append('')


def write_results_report(path: Union[str, Path], *, link_metrics: Optional[dict] = None, triple_metrics: Optional[dict] = None, best_epoch: Optional[int] = None, best_mrr: Optional[float] = None, train_time: Optional[float] = None, valid_time: Optional[float] = None, test_time: Optional[float] = None, total_time: Optional[float] = None, time_per_train_epoch: Optional[float] = None, train_peak_mb: Optional[float] = None, eval_peak_mb: Optional[float] = None, peak_memory_mb: Optional[float] = None, configs: Optional[dict] = None, extra_sections: Optional[dict] = None) -> str:
    """Write a structured results summary to disk."""

    lines = []

    if link_metrics:
        multi_scorers = (
            'cosine' in link_metrics
            and 'original' in link_metrics
            and isinstance(link_metrics.get('cosine'), dict)
            and isinstance(link_metrics.get('original'), dict)
        )
        if multi_scorers:
            for scorer_label, scorer_metrics in link_metrics.items():
                if isinstance(scorer_metrics, dict):
                    _append_link_metrics_lines(
                        lines,
                        scorer_metrics,
                        f'Link Prediction ({scorer_label} scorer)',
                    )
        else:
            _append_link_metrics_lines(lines, link_metrics)

    if triple_metrics:
        lines.append('Triple Classification')
        for key in ['accuracy', 'precision', 'recall', 'f1', 'pr_auc', 'roc_auc']:
            if key in triple_metrics:
                lines.append(f'  {_format_metric_key(key)}: {_format_metric_value(triple_metrics[key])}')
        lines.append('')

    if best_epoch is not None or best_mrr is not None:
        lines.append('Best Valid')
        if best_epoch is not None:
            lines.append(f'  Best Epoch: {best_epoch}')
        if best_mrr is not None:
            lines.append(f'  Best MRR: {best_mrr:.6f}')
        lines.append('')

    if train_time is not None or valid_time is not None or test_time is not None or total_time is not None or time_per_train_epoch is not None:
        lines.append('Time')
        if train_time is not None:
            lines.append(f'  Training Time: {format_duration(train_time)}')
        if time_per_train_epoch is not None:
            lines.append(f'  Time Per Train Epoch: {format_duration(time_per_train_epoch)}')
        if valid_time is not None:
            lines.append(f'  Valid Time: {format_duration(valid_time)}')
        if test_time is not None:
            lines.append(f'  Test Time: {format_duration(test_time)}')
        if total_time is not None:
            lines.append(f'  Total Time: {format_duration(total_time)}')
        lines.append('')

    if train_peak_mb is not None or eval_peak_mb is not None or peak_memory_mb is not None:
        lines.append('Memory')
        if train_peak_mb is not None:
            lines.append(f'  Training Peak: {format_memory(train_peak_mb)}')
        if eval_peak_mb is not None:
            lines.append(f'  Eval Peak: {format_memory(eval_peak_mb)}')
        if peak_memory_mb is not None:
            lines.append(f'  Peak Memory: {format_memory(peak_memory_mb)}')
        lines.append('')

    if configs is not None:
        lines.append('Configs')
        lines.append(json.dumps(configs, indent=2, sort_keys=True, ensure_ascii=False))
        lines.append('')

    if extra_sections:
        for section_name, section_value in extra_sections.items():
            lines.append(section_name)
            if isinstance(section_value, dict):
                for key, value in section_value.items():
                    lines.append(f'  {key}: {_format_metric_value(value)}')
            else:
                lines.append(f'  {section_value}')
            lines.append('')

    report = '\n'.join(lines).rstrip() + '\n'
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding='utf-8')
    return str(path)
