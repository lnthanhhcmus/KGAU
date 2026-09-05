import json
import os
import time

import torch

from configs.config import args
from base.evaluator import Evaluator
from data.dict_hub import get_entity_dict
from models.builder import build_pipeline, config_bool
from utils.device import init_hardware
from utils.checkpoint import best_model_path, last_model_path
from utils.logger import setup_logger, write_results_report, _format_metric_key, time_per_train_epoch
from utils.memory import PhaseMemoryTracker


logger = setup_logger(log_file=os.path.join(args.output_dir, 'run.log'))


def _resolve_test_lp_path(current_args) -> str:
    """Resolve the test path for link prediction evaluation, trying multiple candidates in order of preference."""

    candidates = []
    for source_path in [current_args.test_path, current_args.valid_path, current_args.train_path]:
        if not source_path:
            continue
        source_dir = os.path.dirname(source_path)
        candidates.append(os.path.join(source_dir, 'test.txt.json'))
        candidates.append(os.path.join(source_dir, 'test.txt'))

    candidates.append(os.path.join('data', getattr(current_args, 'dataset', ''), 'preprocessed', 'test.txt.json'))
    candidates.append(os.path.join('data', getattr(current_args, 'dataset', ''), 'preprocessed', 'test.txt'))
    candidates.append(os.path.join('data', getattr(current_args, 'dataset', ''), 'test.txt.json'))
    candidates.append(os.path.join('data', getattr(current_args, 'dataset', ''), 'test.txt'))

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return ''


def _release_gpu_memory() -> None:
    """Return cached GPU memory to the allocator between heavy eval passes."""

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _is_dual_link_metrics(link_metrics: dict | None) -> bool:
    """Return True when link metrics contain per-scorer results (cosine, original, lp_distance, ...)."""

    if not link_metrics:
        return False
    return (
        'cosine' in link_metrics
        and 'original' in link_metrics
        and isinstance(link_metrics['cosine'], dict)
        and isinstance(link_metrics['original'], dict)
    )


def _log_link_metrics(link_metrics: dict, *, title: str) -> None:
    """Log link-prediction metrics (single or dual-scorer layout)."""

    if _is_dual_link_metrics(link_metrics):
        for scorer_label, metrics in link_metrics.items():
            logger.info(
                '%s (%s scorer):\n%s',
                title,
                scorer_label,
                json.dumps(metrics, indent=4),
            )
        return
    logger.info('%s:\n%s', title, json.dumps(link_metrics, indent=4))


def _resolve_test_checkpoint_evals(current_args, train_summary: dict | None = None) -> list[tuple[str, str]]:
    """Return ``(label, checkpoint_path)`` pairs to evaluate after training."""

    output_dir = current_args.output_dir
    evals: list[tuple[str, str]] = []
    seen_paths: set[str] = set()

    def _add(label: str, path: str | None) -> None:
        if not path or not os.path.exists(path):
            return
        normalized = os.path.normpath(path)
        if normalized in seen_paths:
            return
        evals.append((label, path))
        seen_paths.add(normalized)

    if config_bool(current_args, 'test_eval_last', False):
        _add('last', last_model_path(output_dir))
    if config_bool(current_args, 'test_eval_best', False):
        best_path = best_model_path(output_dir)
        if train_summary:
            summary_best = train_summary.get('best_checkpoint_path')
            if summary_best and os.path.exists(summary_best):
                best_path = summary_best
        _add('best', best_path)

    if evals:
        return evals

    explicit = getattr(current_args, 'eval_model_path', None)
    if explicit and os.path.exists(explicit):
        return [('checkpoint', explicit)]

    fallback = train_summary.get('best_checkpoint_path') if train_summary else None
    for candidate in (fallback, best_model_path(output_dir), last_model_path(output_dir)):
        if candidate and os.path.exists(candidate):
            return [('checkpoint', candidate)]
    return []


def _run_test_link_prediction(evaluator, test_lp_path: str, entity_dict, output_dir: str) -> dict:
    """Evaluate test link prediction with cosine, native, and Lp-distance scorers."""

    return evaluator.evaluate_dual_test_link_prediction(test_lp_path, entity_dict, output_dir)


def _run_post_train_test_evaluations(
    current_args,
    train_summary: dict | None,
    *,
    run_lp: bool,
    run_tc: bool,
) -> tuple[dict | None, dict | None, float, PhaseMemoryTracker]:
    """Load each requested checkpoint and run test LP / TC."""

    checkpoint_evals = _resolve_test_checkpoint_evals(current_args, train_summary)
    if not checkpoint_evals:
        raise FileNotFoundError(
            f'No checkpoint found under {current_args.output_dir}. '
            'Training must save at least last_model.mdl or best_model.mdl before test evaluation.'
        )

    memory_tracker = PhaseMemoryTracker()
    if train_summary:
        memory_tracker.update_from_summary(train_summary)

    test_start = time.time()
    memory_tracker.begin_phase()
    test_lp_path = _resolve_test_lp_path(current_args)
    entity_dict = get_entity_dict() if (run_lp and test_lp_path) else None

    link_metrics: dict | None = None
    triple_metrics: dict | None = None

    if len(checkpoint_evals) == 1:
        label, eval_model_path = checkpoint_evals[0]
        logger.info('Test evaluation checkpoint (%s): %s', label, eval_model_path)
        evaluator = Evaluator(current_args)
        evaluator.load(eval_model_path)
        if run_lp and test_lp_path and entity_dict is not None:
            link_metrics = _run_test_link_prediction(evaluator, test_lp_path, entity_dict, current_args.output_dir)
            _release_gpu_memory()
        if run_tc:
            triple_metrics = evaluator.evaluate_test_triple_classification()
        del evaluator
    else:
        by_checkpoint: dict[str, dict] = {}
        for label, eval_model_path in checkpoint_evals:
            logger.info('Test evaluation checkpoint (%s): %s', label, eval_model_path)
            evaluator = Evaluator(current_args)
            evaluator.load(eval_model_path)
            if run_lp and test_lp_path and entity_dict is not None:
                by_checkpoint[label] = _run_test_link_prediction(
                    evaluator, test_lp_path, entity_dict, current_args.output_dir,
                )
                _release_gpu_memory()
            if run_tc:
                triple_metrics = evaluator.evaluate_test_triple_classification()
            del evaluator
        link_metrics = {'by_checkpoint': by_checkpoint} if by_checkpoint else None

    memory_tracker.end_phase('eval')
    test_time = time.time() - test_start
    return link_metrics, triple_metrics, test_time, memory_tracker


def _write_results(
    current_args,
    train_summary,
    link_metrics,
    triple_metrics,
    test_time,
    configs_snapshot,
    memory_tracker=None,
) -> None:
    """Write the evaluation results and training summary to a report file."""

    if link_metrics:
        if 'by_checkpoint' in link_metrics:
            for label, metrics in link_metrics['by_checkpoint'].items():
                _log_link_metrics(metrics, title=f'Link prediction metrics on test set ({label} checkpoint)')
        elif _is_dual_link_metrics(link_metrics):
            for scorer_label, metrics in link_metrics.items():
                logger.info(
                    'Link prediction metrics on test set (%s scorer):\n%s',
                    scorer_label,
                    json.dumps(metrics, indent=4),
                )
        else:
            logger.info('Link prediction metrics on test set:\n{}'.format(json.dumps(link_metrics, indent=4)))
    if triple_metrics:
        logger.info('Triple classification metrics on test set:\n{}'.format(json.dumps(triple_metrics, indent=4)))

    best_epoch = train_summary.get('best_epoch') if train_summary else None
    best_mrr = train_summary.get('best_mrr') if train_summary else None
    best_monitor_metric = train_summary.get('best_monitor_metric') if train_summary else None
    best_monitor_score = train_summary.get('best_monitor_score') if train_summary else None

    train_time = train_summary.get('train_time') if train_summary else None
    valid_time = train_summary.get('valid_time') if train_summary else None
    num_train_epochs = train_summary.get('num_train_epochs') if train_summary else None
    epoch_time = train_summary.get('time_per_train_epoch') if train_summary else None
    if epoch_time is None:
        epoch_time = time_per_train_epoch(train_time, num_train_epochs)
    total_time = None
    if train_summary and train_summary.get('total_time') is not None:
        total_time = train_summary['total_time'] + test_time

    memory_summary = memory_tracker.to_dict() if memory_tracker is not None else {}
    train_peak_mb = memory_summary.get('train_peak_mb')
    eval_peak_mb = memory_summary.get('eval_peak_mb')
    peak_memory_mb = memory_summary.get('peak_memory_mb')

    best_valid_extra = {}
    if best_monitor_metric and best_monitor_score is not None:
        best_valid_extra[f'Best {_format_metric_key(best_monitor_metric)}'] = best_monitor_score

    report_link_metrics = link_metrics
    extra_sections = {'Best Valid Monitor': best_valid_extra} if best_valid_extra else None
    if link_metrics and 'by_checkpoint' in link_metrics:
        report_link_metrics = None
        checkpoint_sections = extra_sections.copy() if extra_sections else {}
        for label, metrics in link_metrics['by_checkpoint'].items():
            if _is_dual_link_metrics(metrics):
                for scorer_label, scorer_metrics in metrics.items():
                    section_name = f'Test Link Prediction ({label}, {scorer_label})'
                    checkpoint_sections[section_name] = {
                        _format_metric_key(key): scorer_metrics[key]
                        for key in ['mr', 'mrr', 'hit@1', 'hit@3', 'hit@10']
                        if key in scorer_metrics
                    }
            else:
                checkpoint_sections[f'Test Link Prediction ({label})'] = {
                    _format_metric_key(key): metrics[key]
                    for key in ['mr', 'mrr', 'hit@1', 'hit@3', 'hit@10']
                    if key in metrics
                }
        extra_sections = checkpoint_sections or None

    write_results_report(
        os.path.join(current_args.output_dir, 'results.txt'),
        link_metrics=report_link_metrics,
        triple_metrics=triple_metrics,
        best_epoch=best_epoch,
        best_mrr=best_mrr,
        train_time=train_time,
        valid_time=valid_time,
        test_time=test_time,
        total_time=total_time,
        time_per_train_epoch=epoch_time,
        train_peak_mb=train_peak_mb,
        eval_peak_mb=eval_peak_mb,
        peak_memory_mb=peak_memory_mb,
        configs=configs_snapshot,
        extra_sections=extra_sections,
    )


def main():
    ngpus_per_node = init_hardware(args)

    logger.info('Use {} gpus for this run'.format(ngpus_per_node))
    logger.info('Args={}'.format(json.dumps(args.__dict__, ensure_ascii=False, indent=4)))
    config_snapshot = dict(args.__dict__)

    task_flag = (args.task or 'both').lower()
    run_lp = False
    run_tc = False
    if 'both' in task_flag or task_flag == 'both':
        run_lp = True
        run_tc = True
    else:
        if 'link' in task_flag or 'pred' in task_flag or 'lp' in task_flag:
            run_lp = True
        if 'triple' in task_flag or 'class' in task_flag or 'tc' in task_flag:
            run_tc = True

    if args.is_test:
        link_metrics, triple_metrics, test_time, memory_tracker = _run_post_train_test_evaluations(
            args, None, run_lp=run_lp, run_tc=run_tc,
        )
        _write_results(args, None, link_metrics, triple_metrics, test_time, config_snapshot, memory_tracker)
        return

    trainer = build_pipeline(args, ngpus_per_node=ngpus_per_node)
    train_dataloader = getattr(trainer, 'train_dataloader', None)
    if train_dataloader is not None:
        train_summary = trainer.train_loop(train_dataloader)
    else:
        train_summary = trainer.train_loop()
    del trainer
    _release_gpu_memory()
    link_metrics, triple_metrics, test_time, memory_tracker = _run_post_train_test_evaluations(
        args, train_summary, run_lp=run_lp, run_tc=run_tc,
    )
    _write_results(args, train_summary, link_metrics, triple_metrics, test_time, config_snapshot, memory_tracker)


if __name__ == '__main__':
    main()
