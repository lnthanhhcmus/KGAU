"""Config parsing and global args."""

import argparse
import json
import os
import random
import sys
import warnings
from datetime import datetime
from types import SimpleNamespace

import torch
import torch.backends.cudnn as cudnn
from typing import Dict, Any, Sequence


def _normalize_argv(argv: Sequence[str] | None = None) -> list[str]:
    """Expand ``@path/to/config.json`` tokens into ``--config-path path/to/config.json``."""

    src = list(sys.argv if argv is None else argv)
    if not src:
        return src

    normalized: list[str] = [src[0]]
    index = 1
    while index < len(src):
        token = src[index]
        if token.startswith('@') and len(token) > 1:
            normalized.extend(['--config-path', token[1:]])
        else:
            normalized.append(token)
        index += 1
    return normalized


def _snake_to_kebab(name: str) -> str:
    return name.replace('_', '-')


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the KG training and evaluation script."""

    parser = argparse.ArgumentParser(description='Generic KG arguments')

    parser.add_argument('--config-path', default=None, type=str,
                        help='path to a JSON config file in configs/ or an absolute config path')

    parser.add_argument('--model', default=None, type=str,
                        help='model family name, e.g. simkgc, transe, transd, rotate')
    parser.add_argument('--model-embedder-path', default=None, type=str,
                        help='path to embedder module, e.g. models/embedders/lookup_embedder.py')
    parser.add_argument('--model-scorer-path', default=None, type=str,
                        help='path to scorer module, e.g. models/distmult.py')
    parser.add_argument('--model-encoder-path', default=None, type=str,
                        help='(legacy) alias for model_scorer_path when embedder/scorer are not split')
    parser.add_argument('--model-loss-path', default=None, type=str,
                        help='path to loss module, e.g. models/losses/infonce_loss.py')
    parser.add_argument('--model-sampler-path', default=None, type=str,
                        help='path to sampler module, e.g. models/samplers/masking_sampler.py')
    parser.add_argument('--model-strategy-path', default=None, type=str,
                        help='path to strategy module, e.g. models/strategies/simkgc_strategy.py')
    parser.add_argument('--task', default=None, type=str,
                        help='link prediction/triple classification/both')
    parser.add_argument('--bert-encoder', '--encoder', default=None, type=str, dest='bert_encoder',
                        help='pretrained text encoder name or path')
    parser.add_argument('--dataset', default=None, type=str,
                        help='dataset or benchmark name')

    # Core data/model paths.
    parser.add_argument('--train-path', default=None, type=str,
                        help='path to training data')
    parser.add_argument('--valid-path', default=None, type=str,
                        help='path to validation data')
    parser.add_argument('--test-path', default=None, type=str,
                        help='path to test data')
    parser.add_argument('--valid-w-label-path', default=None, type=str,
                        help='path to validation data for triple classification')
    parser.add_argument('--test-w-label-path', default=None, type=str,
                        help='path to test data for triple classification')
    # in default, paths for .txt.json (preprocess) or .txt (unprocessed) are taken by dataset in 'data/<dataset>/preprocessed' folder e.g. data/WN18RR/preprocessed/train.txt.json, data/WN18RR/preprocessed/valid.txt.json, data/WN18RR/preprocessed/test.txt.json

    parser.add_argument('--eval-model-path', default=None, type=str,
                        help='path to model to evaluate')
    # in default, eval_model_path is taken from best_model.mdl in output-dir if exists; otherwise, it needs to be specified.

    parser.add_argument('--output-dir-prefix', default=None, type=str,
                        help='prefix for the directory used to save checkpoints, predictions, and logs; a timestamp will be appended when used')
    # in default, output is saved in 'logs/<model>_<dataset>' folder e.g. logs/SimKGC_WN18RR.
    # This folder will contain: train.log (Text training output), results.txt (Final result metrics + best valid + time), best_model.mdl  (Best model weights)

    # Hyperparameters and settings.
    parser.add_argument('--additive-margin', default=None, type=float,
                        help='additive margin for contrastive loss and AU loss')
    parser.add_argument('-b', '--batch-size', '--batch_size', default=None, type=int,
                        dest='batch_size',
                        help='mini-batch size')
    parser.add_argument('--dim', default=None, type=int,
                        help='embedding dimension for non-text KG models')
    parser.add_argument('--dropout', default=None, type=float,
                        help='dropout rate')
    parser.add_argument('--epochs', default=None, type=int,
                        help='number of epochs to run')
    parser.add_argument('--eval-every-n-step', '--eval_every_n_step', default=None, type=int,
                        dest='eval_every_n_step',
                        help='evaluate every n steps (default: 10000 for SimKGC)')
    parser.add_argument('--eval-interval-epochs', '--eval_interval_epochs', default=None, type=int,
                        dest='eval_interval_epochs',
                        help='run validation every N epochs (always runs on last epoch)')
    parser.add_argument('--valid-link-prediction-epochs', '--valid_link_prediction_epochs', default=None, type=int,
                        dest='valid_link_prediction_epochs',
                        help='run full link-prediction eval every N epochs (0=skip LP during training)')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--enable-extra-epoch-metrics', '--enable_extra_epoch_metrics',
                       dest='enable_extra_epoch_metrics', action='store_true', default=False,
                       help='enable additional expensive per-epoch validation metrics')
    group.add_argument('--no-enable-extra-epoch-metrics', '--no_enable_extra_epoch_metrics',
                       dest='enable_extra_epoch_metrics', action='store_false',
                       help='disable expensive per-epoch validation metrics (default)')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--shared-encoder', '--shared_encoder', dest='shared_encoder',
                       action='store_true', default=False,
                       help='use one BERT encoder for query and entity text (SimKGC)')
    group.add_argument('--no-shared-encoder', '--no_shared_encoder', dest='shared_encoder',
                       action='store_false', help='use separate hr and tail BERT encoders (default)')
    parser.add_argument('--finetune-t', action='store_true',
                        help='make InfoNCE temperature trainable')
    parser.add_argument('--grad-clip', '--grad_clip', default=None, type=float, dest='grad_clip',
                        help='gradient clipping (default: 10.0 for SimKGC training)')
    parser.add_argument('--is-test', action='store_true',
                        help='run test-mode evaluation')
    parser.add_argument('--lr', '--learning-rate', default=None, type=float, dest='lr',
                        help='initial learning rate')
    parser.add_argument('--lr-scheduler', default=None, type=str,
                        help='learning-rate scheduler')
    parser.add_argument('--max-num-tokens', default=None, type=int,
                        help='maximum number of tokens for text-based models')
    parser.add_argument('--encode-micro-batch-size', '--encode_micro_batch_size', default=None, type=int,
                        dest='encode_micro_batch_size',
                        help='chunk size for BERT encode passes (0 or omit = full batch; set e.g. 64 to save GPU memory)')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--encode-checkpoint', '--encode_checkpoint', dest='encode_checkpoint',
                       action='store_true', default=False,
                       help='checkpoint BERT encode chunks to reduce GPU memory during training')
    group.add_argument('--no-encode-checkpoint', '--no_encode_checkpoint', dest='encode_checkpoint',
                       action='store_false', help='disable BERT encode checkpointing (default)')
    parser.add_argument('--uniformity-pdist-gb', '--uniformity_pdist_gb', default=None, type=float,
                        dest='uniformity_pdist_gb',
                        help='Deprecated legacy knob (kept for config compatibility). Exact i<j '
                             'uniformity now uses chunked pairwise blocks; prefer --uniform-pair-chunk-size.')
    parser.add_argument('--max-uniformity-samples', '--max_uniformity_samples', default=None, type=int,
                        dest='max_uniformity_samples',
                        help='maximum number of embeddings used to estimate the AU uniformity term (0=no cap)')
    parser.add_argument('--max-uniformity-pairs', '--max_uniformity_pairs', default=None, type=int,
                        dest='max_uniformity_pairs',
                        help='maximum random pairwise distances for AU uniformity when full pdist is too large')
    parser.add_argument('--max-to-keep', default=None, type=int,
                        help='max rolling checkpoint_epoch*.mdl files to keep; 0 = only best_model.mdl and last_model.mdl')
    parser.add_argument('--neighbor-weight', default=0.0, type=float,
                        help='reranking weight')
    parser.add_argument('--pooling', default=None, type=str,
                        help='pooling strategy for text encoders')
    parser.add_argument('--pre-batch', default=None, type=int,
                        help='number of pre-batch negatives')
    parser.add_argument('--pre-batch-weight', default=None, type=float,
                        help='weight for pre-batch negatives')
    parser.add_argument('-p', '--print-freq', default=None, type=int,
                        help='logging frequency')
    parser.add_argument('--rerank-n-hop', default=2, type=int,
                        help='neighbor hops for reranking during evaluation')
    parser.add_argument('--seed', default=None, type=int,
                        help='random seed')
    parser.add_argument('--infonce-t', '--t', default=None, type=float, dest='infonce_t',
                        help='InfoNCE temperature parameter')
    parser.add_argument('--use-amp', action='store_true',
                        help='use AMP if available')
    parser.add_argument('--use-link-graph', action='store_true',
                        help='use neighbors from link graph as context')
    parser.add_argument('--use-self-negative', action='store_true',
                        help='use head entity as negative')
    parser.add_argument('--wd', '--weight-decay', default=None, type=float,
                        dest='weight_decay', help='weight decay')
    parser.add_argument('-j', '--workers', default=None, type=int,
                        help='DataLoader workers (default 4; 0 = in-process). '
                             'Filtered NegSamp uses workers so CPU negative sampling overlaps GPU.')
    parser.add_argument('--warmup', default=None, type=int,
                        help='warmup steps')

    # Softmax / Bernoulli negative-sampling (DistMult, ComplEx, etc.).
    parser.add_argument('--sample-freq', '--sample_freq', default=None, type=int,
                        help='negative sampling frequency')
    parser.add_argument('-ns', '--n-sample', '--n_sample', default=None, type=int,
                        help='number of negative samples per positive')
    parser.add_argument('--neg-score-chunk-size', '--neg_score_chunk_size', default=None, type=int,
                        dest='neg_score_chunk_size',
                        help='NegSamp: max negatives scored per chunk (default 256; <=0 disables chunking; '
                             'when chunking, uses gradient checkpointing). Alias: --negative-chunk-size')
    parser.add_argument('--negative-chunk-size', '--negative_chunk_size', default=None, type=int,
                        dest='negative_chunk_size',
                        help='Alias of --neg-score-chunk-size (default 256; <=0 disables chunking)')
    parser.add_argument('--neg-weight-chunk-size', '--neg_weight_chunk_size', default=None, type=int,
                        dest='neg_weight_chunk_size',
                        help='Deprecated/unused (kept for config compatibility; adversarial weights use full scores)')
    parser.add_argument('--uniform-pair-chunk-size', '--uniform_pair_chunk_size', default=None, type=int,
                        dest='uniform_pair_chunk_size',
                        help='KGAU: pairwise block size for exact i<j uniformity (0/None=auto, soft-cap 256)')
    parser.add_argument('--lam', default=None, type=float,
                        help='L2 regularization strength (kgau/softmax; overrides weight_decay when set)')

    # KGAU alignment-uniformity hyperparameters (DistMult-AU, ComplEx-AU, etc.).
    parser.add_argument('--alpha', default=None, type=float,
                        help='alignment loss scale (default 1.0)')
    parser.add_argument('--align-alpha', '--align_alpha', default=None, type=float,
                        dest='align_alpha',
                        help='alignment loss exponent on (q-t); 2.0 = squared L2 (default)')
    parser.add_argument('--gamma-q', '--gamma_q', default=None, type=float,
                        help='uniformity weight for query embeddings')
    parser.add_argument('--gamma-t', '--gamma_t', default=None, type=float,
                        help='uniformity weight for target embeddings')
    parser.add_argument('--gamma-h', '--gamma_h', default=None, type=float,
                        help='uniformity weight for head embeddings (default 0 when omitted)')
    parser.add_argument('--gamma-ent', '--gamma_ent', default=None, type=float,
                        help='uniformity weight for all entity embeddings')
    parser.add_argument('--gamma-cross', '--gamma_cross', default=None, type=float,
                        help='uniformity weight on pooled query+tail vectors (joint LP space)')
    parser.add_argument('--learnable-au-scales', '--learnable_au_scales',
                        dest='learnable_au_scales', action='store_true', default=False,
                        help='learnable alpha + independent learnable gammas with opposing epoch schedules')
    parser.add_argument('--learnable-au-alpha', '--learnable_au_alpha',
                        dest='learnable_au_alpha', action='store_true', default=False,
                        help='make AU alignment scale alpha learnable (log re-parameterization)')
    parser.add_argument('--log-au-alpha-lr', '--log_au_alpha_lr', default=None, type=float,
                        dest='log_au_alpha_lr',
                        help='learning rate for learnable log-alpha parameter')
    parser.add_argument('--alpha-linear-schedule', '--alpha_linear_schedule',
                        dest='alpha_linear_schedule', action='store_true', default=False,
                        help='linearly increase AU alpha multiplier from 1.0 to alpha_schedule_end')
    parser.add_argument('--no-alpha-linear-schedule', dest='alpha_linear_schedule',
                        action='store_false',
                        help='disable alpha linear schedule (even with learnable_au_alpha)')
    parser.add_argument('--alpha-schedule-end', '--alpha_schedule_end', default=None, type=float,
                        dest='alpha_schedule_end',
                        help='alpha schedule multiplier at the last scheduled epoch')
    parser.add_argument('--alpha-schedule-start-epoch', '--alpha_schedule_start_epoch', default=None, type=int,
                        dest='alpha_schedule_start_epoch',
                        help='epoch index (0-based) to begin alpha linear schedule')
    parser.add_argument('--alpha-schedule-epochs', '--alpha_schedule_epochs', default=None, type=int,
                        dest='alpha_schedule_epochs',
                        help='epochs over which to schedule alpha multiplier (0: full training)')
    parser.add_argument('--learnable-au-gammas', '--learnable_au_gammas',
                        dest='learnable_au_gammas', action='store_true', default=False,
                        help='make AU uniformity gammas learnable (log re-parameterization)')
    parser.add_argument('--log-au-gamma-lr', '--log_au_gamma_lr', default=None, type=float,
                        dest='log_au_gamma_lr',
                        help='learning rate for learnable log-gamma parameters')
    log_au_gamma_lr_schedule_group = parser.add_mutually_exclusive_group()
    log_au_gamma_lr_schedule_group.add_argument(
        '--log-au-gamma-lr-linear-schedule', '--log_au_gamma_lr_linear_schedule',
        dest='log_au_gamma_lr_linear_schedule',
        action='store_true', default=None,
        help='linearly ramp log_au_gamma_lr from its initial value to gamma_schedule_end')
    log_au_gamma_lr_schedule_group.add_argument(
        '--no-log-au-gamma-lr-linear-schedule', '--no_log_au_gamma_lr_linear_schedule',
        dest='log_au_gamma_lr_linear_schedule',
        action='store_false',
        help='disable log_au_gamma_lr linear schedule (even with learnable_au_gammas)')
    parser.add_argument('--gamma-linear-schedule', '--gamma_linear_schedule',
                        dest='gamma_linear_schedule', action='store_true', default=False,
                        help='linearly decay AU gamma multiplier from 1.0 to gamma_schedule_end')
    parser.add_argument('--no-gamma-linear-schedule', dest='gamma_linear_schedule',
                        action='store_false',
                        help='disable gamma linear schedule (even with learnable_au_gammas)')
    parser.add_argument('--gamma-schedule-end', '--gamma_schedule_end', default=None, type=float,
                        dest='gamma_schedule_end',
                        help='gamma schedule multiplier at the last scheduled epoch')
    parser.add_argument('--gamma-schedule-start-epoch', '--gamma_schedule_start_epoch', default=None, type=int,
                        dest='gamma_schedule_start_epoch',
                        help='epoch index (0-based) to begin gamma linear schedule')
    parser.add_argument('--gamma-schedule-epochs', '--gamma_schedule_epochs', default=None, type=int,
                        dest='gamma_schedule_epochs',
                        help='epochs over which to schedule gamma multiplier (0: full training)')
    parser.add_argument('--tuni', default=None, type=float,
                        help='AU temperature: uniformity potential scale; also alignment scale with tuni_as_alpha')
    tuni_as_alpha_group = parser.add_mutually_exclusive_group()
    tuni_as_alpha_group.add_argument(
        '--tuni-as-alpha', '--tuni_as_alpha',
        dest='tuni_as_alpha',
        action='store_true',
        default=None,
        help='use the same tuni for alignment (replaces alpha) and uniformity temperature',
    )
    tuni_as_alpha_group.add_argument(
        '--no-tuni-as-alpha', '--no_tuni_as_alpha',
        dest='tuni_as_alpha',
        action='store_false',
        help='use alpha for alignment and tuni for uniformity only (default)',
    )
    parser.add_argument('--learnable-uniformity-scale', '--learnable_uniformity_scale',
                        dest='learnable_uniformity_scale', action='store_true', default=False,
                        help='make tuni learnable (uniformity temp; alignment too when tuni_as_alpha)')
    parser.add_argument('--log-uniformity-lr', '--log_uniformity_lr', default=None, type=float,
                        dest='log_uniformity_lr',
                        help='learning rate for learnable log-tuni')
    parser.add_argument('--tuni-linear-schedule', '--tuni_linear_schedule',
                        dest='tuni_linear_schedule', action='store_true', default=False,
                        help='linearly increase tuni from start to end across training epochs')
    parser.add_argument('--tuni-schedule-start', '--tuni_schedule_start', default=None, type=float,
                        dest='tuni_schedule_start',
                        help='starting tuni for linear schedule (default: --tuni)')
    parser.add_argument('--tuni-schedule-end', '--tuni_schedule_end', default=None, type=float,
                        dest='tuni_schedule_end',
                        help='ending tuni for linear schedule (default: --tuni)')
    parser.add_argument('--tuni-schedule-start-epoch', '--tuni_schedule_start_epoch', default=None, type=int,
                        dest='tuni_schedule_start_epoch',
                        help='epoch index (0-based) to begin tuni linear schedule')
    parser.add_argument('--tuni-schedule-epochs', '--tuni_schedule_epochs', default=None, type=int,
                        dest='tuni_schedule_epochs',
                        help='epochs over which to schedule tuni (0: use --epochs through last epoch)')
    au_deduplicate_group = parser.add_mutually_exclusive_group()
    au_deduplicate_group.add_argument(
        '--au-deduplicate', '--au_deduplicate',
        dest='au_deduplicate',
        action='store_true',
        default=None,
        help='KGAU: deduplicate query/tail/head/entity ids before uniformity loss (default: true)',
    )
    au_deduplicate_group.add_argument(
        '--no-au-deduplicate', '--no_au_deduplicate',
        dest='au_deduplicate',
        action='store_false',
        help='KGAU: use all batch rows for uniformity (DirectAU-style, no id dedup)',
    )
    average_uniformity_group = parser.add_mutually_exclusive_group()
    average_uniformity_group.add_argument(
        '--average-uniformity-terms', '--average_uniformity_terms',
        dest='average_uniformity_terms',
        action='store_true',
        default=None,
        help='KGAU: divide summed uniformity terms by active term count (GB-Magic au style)',
    )
    average_uniformity_group.add_argument(
        '--no-average-uniformity-terms', '--no_average_uniformity_terms',
        dest='average_uniformity_terms',
        action='store_false',
        help='KGAU: sum uniformity terms without averaging (default)',
    )
    entity_uniformity_batch_group = parser.add_mutually_exclusive_group()
    entity_uniformity_batch_group.add_argument(
        '--entity-uniformity-batch', '--entity_uniformity_batch',
        dest='entity_uniformity_batch',
        action='store_true',
        default=None,
        help='KGAU: gamma_ent on batch cat(embed_h, embed_t) of triple endpoints '
             '(GB-Magic; independent of head/tail-batch mode) instead of full entity table',
    )
    entity_uniformity_batch_group.add_argument(
        '--no-entity-uniformity-batch', '--no_entity_uniformity_batch',
        dest='entity_uniformity_batch',
        action='store_false',
        help='KGAU: entity uniformity on full entity table (default for index KGE)',
    )
    kgau_bidirectional_group = parser.add_mutually_exclusive_group()
    kgau_bidirectional_group.add_argument(
        '--kgau-bidirectional', '--kgau_bidirectional',
        dest='kgau_bidirectional',
        action='store_true',
        default=None,
        help='KGAU: alternate tail/head batches each step (GB-Magic BidirectionalOneShotIterator); '
             'eval head via rt_forward',
    )
    kgau_bidirectional_group.add_argument(
        '--no-kgau-bidirectional', '--no_kgau_bidirectional',
        dest='kgau_bidirectional',
        action='store_false',
        help='KGAU: legacy reciprocal-relation tail training (default)',
    )
    uniformity_full_pdist_group = parser.add_mutually_exclusive_group()
    uniformity_full_pdist_group.add_argument(
        '--uniformity-full-pdist', '--uniformity_full_pdist',
        dest='uniformity_full_pdist',
        action='store_true',
        default=None,
        help='KGAU: exact full-batch i<j uniformity via chunked pairwise blocks '
             '(replaces torch.pdist; no Monte Carlo pairs)',
    )
    uniformity_full_pdist_group.add_argument(
        '--no-uniformity-full-pdist', '--no_uniformity_full_pdist',
        dest='uniformity_full_pdist',
        action='store_false',
        help='KGAU: subsampled/random-pair uniformity (default)',
    )
    parser.add_argument('--lr-decay-preserve-optimizer', '--lr_decay_preserve_optimizer',
                        dest='lr_decay_preserve_optimizer', action='store_true', default=None,
                        help='On warm-up LR decay, scale param-group LRs in-place (GB-Magic) instead of rebuilding Adam')
    parser.add_argument('--no-lr-decay-preserve-optimizer', '--no_lr_decay_preserve_optimizer',
                        dest='lr_decay_preserve_optimizer', action='store_false',
                        help='Rebuild optimizer on LR decay (legacy KGDirectAU default)')
    parser.add_argument('--test-eval-last', '--test_eval_last', dest='test_eval_last',
                        action='store_true', default=None,
                        help='Evaluate test set from last_model.mdl (GB-Magic final checkpoint)')
    parser.add_argument('--no-test-eval-last', '--no_test_eval_last', dest='test_eval_last',
                        action='store_false', help='Do not evaluate test from last_model.mdl')
    parser.add_argument('--test-eval-best', '--test_eval_best', dest='test_eval_best',
                        action='store_true', default=None,
                        help='Evaluate test set from best_model.mdl (best valid MRR checkpoint)')
    parser.add_argument('--no-test-eval-best', '--no_test_eval_best', dest='test_eval_best',
                        action='store_false', help='Do not evaluate test from best_model.mdl')
    normalize_group = parser.add_mutually_exclusive_group()
    normalize_group.add_argument(
        '--normalize-lp-scores', '--normalize_lp_scores',
        dest='normalize_lp_scores',
        action='store_true',
        default=False,
        help='L2-normalize query/tail vectors for link-prediction scoring',
    )
    normalize_group.add_argument(
        '--no-normalize-lp-scores', '--no_normalize_lp_scores',
        dest='normalize_lp_scores',
        action='store_false',
        help='disable normalized link-prediction scoring',
    )
    parser.add_argument('--lp-score-mode', '--lp_score_mode', default=None, type=str,
                        dest='lp_score_mode',
                        help='link-prediction scoring mode: original, cosine, or lp_distance')
    parser.add_argument('--lp-distance-degree', '--lp_distance_degree', '--distance-degree-l', '--distance_degree_l',
                        default=2.0, type=float, dest='lp_distance_degree',
                        help='Lp distance degree for lp_distance evaluation (2=L2, 3=L3)')
    au_normalize_group = parser.add_mutually_exclusive_group()
    au_normalize_group.add_argument(
        '--normalize-au-vectors', '--normalize_au_vectors',
        dest='normalize_au_vectors',
        action='store_true',
        default=None,
        help='L2-normalize query/tail vectors for KGAU training (default: on for *-AU except pRotatE-AU)',
    )
    au_normalize_group.add_argument(
        '--no-normalize-au-vectors', '--no_normalize_au_vectors',
        dest='normalize_au_vectors',
        action='store_false',
        help='use raw embeddings for KGAU training vectors',
    )
    parser.add_argument('--alignment-mode', '--alignment_mode', default=None, type=str,
                        dest='alignment_mode',
                        help='KGAU alignment: cosine, phase_residual, or sin_phase')
    au_hybrid_group = parser.add_mutually_exclusive_group()
    au_hybrid_group.add_argument(
        '--au-hybrid-adversarial-bce', '--au_hybrid_adversarial_bce',
        dest='au_hybrid_adversarial_bce',
        action='store_true',
        default=None,
        help='add filtered adversarial BCE on native KGE scores alongside KGAU loss',
    )
    au_hybrid_group.add_argument(
        '--no-au-hybrid-adversarial-bce', '--no_au_hybrid_adversarial_bce',
        dest='au_hybrid_adversarial_bce',
        action='store_false',
        help='disable KGAU + adversarial BCE hybrid training',
    )
    parser.add_argument('--au-hybrid-au-weight', '--au_hybrid_au_weight', default=None, type=float,
                        dest='au_hybrid_au_weight',
                        help='weight on KGAU loss term in hybrid training (default 1.0)')
    parser.add_argument('--au-hybrid-kge-weight', '--au_hybrid_kge_weight', default=None, type=float,
                        dest='au_hybrid_kge_weight',
                        help='weight on adversarial BCE term in hybrid training (default 1.0)')
    parser.add_argument('--transe-norm', '--transe_norm', default=None, type=int,
                        dest='transe_norm',
                        help='TransE distance norm: 1 (classic L1) or 2 (L2, TransE-AU)')
    parser.add_argument('--dabr-distance-norm', '--dabr_distance_norm', default=None, type=int,
                        dest='dabr_distance_norm',
                        help='DaBR additive-branch distance norm: 1 (paper L1) or 2 (L2, Option A)')
    triple_relation_group = parser.add_mutually_exclusive_group()
    triple_relation_group.add_argument(
        '--triple-relation-embedding', '--triple_relation_embedding',
        dest='triple_relation_embedding',
        action='store_true',
        default=None,
        help='Use 3x relation embedding width (TransERR)',
    )
    triple_relation_group.add_argument(
        '--no-triple-relation-embedding', '--no_triple_relation_embedding',
        dest='triple_relation_embedding',
        action='store_false',
        help='Disable 3x relation embedding width',
    )

    # Index KGE training (DistMult, ComplEx, KvsAll, reciprocal relations).
    parser.add_argument('--add-reciprocal-relations', '--add_reciprocal_relations',
                        dest='add_reciprocal_relations', action='store_true',
                        help='train with inverse relations (reciprocal_relations_model)')
    parser.add_argument('--kbc-reciprocal-relations', '--kbc_reciprocal_relations',
                        dest='kbc_reciprocal_relations', action='store_true', default=None,
                        help='kbc-style reciprocal relations: inverse id = forward id + n_forward, doubled train triples')
    parser.add_argument('--bidirectional-1vsall', '--bidirectional_1vsall',
                        dest='bidirectional_1vsall', action='store_true', default=None,
                        help='train both tail (hr_) and head (rt_) 1-vs-all CE losses')
    parser.add_argument('--sparse-embeddings', '--sparse_embeddings',
                        dest='sparse_embeddings', action='store_true', default=None,
                        help='use sparse Embedding tables (kbc-style Adagrad)')
    parser.add_argument('--label-smoothing', '--label_smoothing', default=None, type=float,
                        dest='label_smoothing', help='KvsAll label smoothing')
    parser.add_argument('--loss-arg', '--loss_arg', default=None, type=float,
                        dest='loss_arg', help='BCE loss offset (train.loss_arg)')
    parser.add_argument('--entity-dropout', '--entity_dropout', default=None, type=float,
                        dest='entity_dropout', help='entity embedding dropout')
    parser.add_argument('--relation-dropout', '--relation_dropout', default=None, type=float,
                        dest='relation_dropout', help='relation embedding dropout')
    parser.add_argument('--entity-regularize-weight', '--entity_regularize_weight',
                        default=None, type=float, dest='entity_regularize_weight',
                        help='L3 entity embedding regularization weight')
    parser.add_argument('--relation-regularize-weight', '--relation_regularize_weight',
                        default=None, type=float, dest='relation_regularize_weight',
                        help='L3 relation embedding regularization weight')
    parser.add_argument('--regularizer', default=None, type=str,
                        help='regularization backend: n3_kbc (kbc ComplEx N3) or L3 (default)')
    parser.add_argument('--regularize-weight', '--regularize_weight', default=None, type=float,
                        dest='regularize_weight', help='kbc N3 regularization weight')
    parser.add_argument('--init-method', '--init_method', default=None, type=str,
                        dest='init_method', help='lookup init: uniform_, xavier_uniform_, scaled, kbc')
    parser.add_argument('--init-scale', '--init_scale', default=None, type=float,
                        dest='init_scale', help='kbc init scale applied after default Embedding init')
    parser.add_argument('--init-uniform-a', '--init_uniform_a', default=None, type=float,
                        dest='init_uniform_a', help='uniform_ lower bound (upper defaults to -a)')
    parser.add_argument('--init-uniform-b', '--init_uniform_b', default=None, type=float,
                        dest='init_uniform_b', help='uniform_ upper bound')
    parser.add_argument('--init-xavier-gain', '--init_xavier_gain', default=None, type=float,
                        dest='init_xavier_gain', help='xavier init gain')
    parser.add_argument('--eval-batch-size', '--eval_batch_size', default=None, type=int,
                        dest='eval_batch_size', help='link-prediction evaluation batch size')
    parser.add_argument('--chunk-size', '--chunk_size', default=None, type=int,
                        dest='chunk_size',
                        help='entity chunk size for SimKGC-style link-prediction scoring')
    parser.add_argument('--early-stopping-patience', '--early_stopping_patience',
                        default=None, type=int, dest='early_stopping_patience',
                        help='epochs without valid MRR improvement before early stop')
    parser.add_argument('--early-stopping-min-epochs', '--early_stopping_min_epochs',
                        default=None, type=int, dest='early_stopping_min_epochs',
                        help='minimum epochs before early stopping can trigger')
    parser.add_argument('--early-stopping-min-metric', '--early_stopping_min_metric',
                        default=None, type=float, dest='early_stopping_min_metric',
                        help='only count plateau epochs once best valid MRR reaches this value')
    parser.add_argument('--lr-scheduler-mode', '--lr_scheduler_mode', default=None, type=str,
                        dest='lr_scheduler_mode', help='ReduceLROnPlateau mode')
    parser.add_argument('--lr-scheduler-factor', '--lr_scheduler_factor', default=None, type=float,
                        dest='lr_scheduler_factor',
                        help='LR decay factor (ReduceLROnPlateau factor / StepLR gamma)')
    parser.add_argument('--lr-scheduler-patience', '--lr_scheduler_patience', default=None, type=int,
                        dest='lr_scheduler_patience', help='ReduceLROnPlateau patience')
    parser.add_argument('--lr-scheduler-threshold', '--lr_scheduler_threshold', default=None, type=float,
                        dest='lr_scheduler_threshold', help='ReduceLROnPlateau threshold')
    parser.add_argument('--lr-scheduler-step-size', '--lr_scheduler_step_size', default=None, type=int,
                        dest='lr_scheduler_step_size',
                        help='StepLR: multiply LR by factor every this many epochs')

    # RotatE / pRotatE and adversarial negative sampling.
    adversarial_training_group = parser.add_mutually_exclusive_group()
    adversarial_training_group.add_argument(
        '--adversarial-training', '--adversarial_training',
        dest='adversarial_training',
        action='store_true',
        default=None,
        help='Use KnowledgeGraphEmbedding-style adversarial negative sampling (RotatE/DistMult/ComplEx)',
    )
    adversarial_training_group.add_argument(
        '--no-adversarial-training', '--no_adversarial_training',
        dest='adversarial_training',
        action='store_false',
        help='Disable adversarial negative sampling',
    )
    parser.add_argument('--margin', default=None, type=float,
                        help='RotatE/pRotatE embedding margin (gamma in KnowledgeGraphEmbedding)')
    parser.add_argument('--epsilon', default=None, type=float,
                        help='RotatE/pRotatE embedding_range epsilon (default 2.0; range = (margin+epsilon)/dim)')
    parser.add_argument('--l-norm', '--l_norm', default=None, type=float, dest='l_norm',
                        help='RotatE distance Lp norm')
    parser.add_argument('--adversarial-temperature', '--adversarial_temperature',
                        default=None, type=float, dest='adversarial_temperature',
                        help='Adversarial BCE temperature (pRotatE)')
    parser.add_argument('--test-batch-size', '--test_batch_size', default=None, type=int,
                        dest='test_batch_size', help='Test evaluation batch size')
    normalize_phases_group = parser.add_mutually_exclusive_group()
    normalize_phases_group.add_argument(
        '--normalize-phases', '--normalize_phases',
        dest='normalize_phases',
        action='store_true',
        default=None,
        help='Keep relation phases in [-pi, pi] after each step (RotatE/pRotatE)',
    )
    normalize_phases_group.add_argument(
        '--no-normalize-phases', '--no_normalize_phases',
        dest='normalize_phases',
        action='store_false',
        help='Disable relation phase normalization',
    )

    # Training cadence, optimizer, and KvsAll query layout.
    parser.add_argument('--training-cadence', '--training_cadence', default=None, type=str,
                        dest='training_cadence', help='Training cadence: step or epoch (default: step if max_steps set)')
    parser.add_argument('--max-steps', '--max_steps', default=None, type=int,
                        dest='max_steps', help='Stop training after this many optimizer steps')
    parser.add_argument('--valid-steps', '--valid_steps', default=None, type=int,
                        dest='valid_steps', help='Validate every N optimizer steps (alias: eval_every_n_step)')
    parser.add_argument('--save-checkpoint-steps', '--save_checkpoint_steps', default=None, type=int,
                        dest='save_checkpoint_steps', help='Save checkpoint every N optimizer steps')
    parser.add_argument('--warm-up-steps', '--warm_up_steps', default=None, type=int,
                        dest='warm_up_steps', help='Decay LR by lr_decay_factor at this step (default: max_steps // 2)')
    parser.add_argument('--warm-up-ratio', '--warm_up_ratio', default=None,
                        nargs='+', type=float, dest='warm_up_ratio',
                        help='LR decay ratios over training budget: max_steps (step cadence) '
                             'or epochs*steps_per_epoch (epoch cadence), e.g. 0.2 0.5 0.8')
    parser.add_argument('--warm-up-epochs', '--warm_up_epochs', default=None, type=int,
                        dest='warm_up_epochs',
                        help='Decay LR after this many epochs (overrides warm_up_steps when warm_up_steps unset)')
    parser.add_argument('--lr-decay-factor', '--lr_decay_factor', default=None, type=float,
                        dest='lr_decay_factor', help='LR multiplier applied at warm_up_steps (default: 0.1)')
    parser.add_argument('--epoch-per-eval', '--epoch_per_eval', default=None, type=int,
                        dest='epoch_per_eval', help='Run validation every N epochs')
    parser.add_argument('--optim', default=None, type=str,
                        help='Optimizer: adam, adamw, adagrad, or sgd')
    parser.add_argument('--kvsall-query-types', '--kvsall_query_types', default=None,
                        nargs='+', type=str, dest='kvsall_query_types',
                        help='KvsAll query types (e.g. hr_ _rt)')

    # DaBR-specific hyperparameters.
    parser.add_argument('--lmbda', default=None, type=float,
                        help='DaBR primary lambda')
    parser.add_argument('--lmbda-two', '--lmbda_two', default=None, type=float,
                        dest='lmbda_two', help='DaBR secondary lambda')
    parser.add_argument('--entity-reg-weight', '--entity_reg_weight', default=None, type=float,
                        dest='entity_reg_weight', help='DaBR entity quaternion regularization weight')
    parser.add_argument('--relation-reg-weight', '--relation_reg_weight', default=None, type=float,
                        dest='relation_reg_weight', help='DaBR relation/drift quaternion regularization weight')
    dabr_reg_neg_group = parser.add_mutually_exclusive_group()
    dabr_reg_neg_group.add_argument(
        '--dabr-reg-include-negatives', '--dabr_reg_include_negatives',
        dest='dabr_reg_include_negatives', action='store_true', default=None,
        help='DaBR: regularize over the full positive+negative batch (matches the reference implementation)')
    dabr_reg_neg_group.add_argument(
        '--no-dabr-reg-include-negatives', '--no_dabr_reg_include_negatives',
        dest='dabr_reg_include_negatives', action='store_false',
        help='DaBR: regularize over positive triples only (default)')
    dabr_semantic_only_group = parser.add_mutually_exclusive_group()
    dabr_semantic_only_group.add_argument(
        '--dabr-au-semantic-only', '--dabr_au_semantic_only',
        dest='dabr_au_semantic_only', action='store_true', default=None,
        help='DaBR-AU: single-sphere AU on the semantic (quaternion) branch only; '
             'drop the distance/translation component in training and link prediction')
    dabr_semantic_only_group.add_argument(
        '--no-dabr-au-semantic-only', '--no_dabr_au_semantic_only',
        dest='dabr_au_semantic_only', action='store_false',
        help='DaBR-AU: disable semantic-only mode')
    dabr_distance_only_group = parser.add_mutually_exclusive_group()
    dabr_distance_only_group.add_argument(
        '--dabr-au-distance-only', '--dabr_au_distance_only',
        dest='dabr_au_distance_only', action='store_true', default=None,
        help='DaBR-AU: TransE-style single-sphere AU on h+dr ↔ t only; '
             'drop the semantic/quaternion component in training and link prediction')
    dabr_distance_only_group.add_argument(
        '--no-dabr-au-distance-only', '--no_dabr_au_distance_only',
        dest='dabr_au_distance_only', action='store_false',
        help='DaBR-AU: disable distance-only mode')
    dabr_independent_group = parser.add_mutually_exclusive_group()
    dabr_independent_group.add_argument(
        '--dabr-au-independent-spheres', '--dabr_au_independent_spheres',
        dest='dabr_au_independent_spheres', action='store_true', default=None,
        help='DaBR-AU: separate entity tables for semantic and distance hyperspheres '
             '(no shared entity params); fuse LP as ⟨h⊗r,t⊗r⁻¹⟩ + λ·cos_dist')
    dabr_independent_group.add_argument(
        '--no-dabr-au-independent-spheres', '--no_dabr_au_independent_spheres',
        dest='dabr_au_independent_spheres', action='store_false',
        help='DaBR-AU: share entity embeddings across semantic and distance components')
    parser.add_argument('--n-batches', '--n_batches', default=None, type=int,
                        dest='n_batches', help='Training batches per epoch (OpenKE nbatches; sets batch_size)')
    openke_batch_group = parser.add_mutually_exclusive_group()
    openke_batch_group.add_argument(
        '--openke-batch-sampling', '--openke_batch_sampling',
        dest='openke_batch_sampling', action='store_true', default=None,
        help='Sample training positives with replacement each batch (OpenKE/DaBR getBatch); '
             'defaults on for DaBR / DaBR-AU',
    )
    openke_batch_group.add_argument(
        '--no-openke-batch-sampling', '--no_openke_batch_sampling',
        dest='openke_batch_sampling', action='store_false',
        help='Disable OpenKE with-replacement positive sampling (epoch shuffle / without replacement)',
    )
    parser.add_argument('--valid-metric', '--valid_metric', default=None, type=str,
                        dest='valid_metric',
                        help='Validation metric for checkpointing/early stop '
                             '(e.g. mrr, hit@10, or scorer label lp_distance_l2)')

    # L3 regularization and weighted regularization flags.
    parser.add_argument('--regularize-p', '--regularize_p', default=None, type=int,
                        dest='regularize_p', help='L3 regularization norm order (LibKGE-style)')
    parser.add_argument('--regularization-p', '--regularization_p', default=None, type=int,
                        dest='regularization_p',
                        help='Lp norm order for legacy scalar regularization (GB-Magic; default 3)')
    parser.add_argument('--regularization', default=None, type=float,
                        dest='regularization',
                        help='Legacy scalar Lp embedding regularization coefficient')
    entity_regularize_weighted_group = parser.add_mutually_exclusive_group()
    entity_regularize_weighted_group.add_argument(
        '--entity-regularize-weighted', '--entity_regularize_weighted',
        dest='entity_regularize_weighted',
        action='store_true',
        default=None,
        help='Scale entity L3 regularization by embedding dimension',
    )
    entity_regularize_weighted_group.add_argument(
        '--no-entity-regularize-weighted', '--no_entity_regularize_weighted',
        dest='entity_regularize_weighted',
        action='store_false',
        help='Use unweighted entity L3 regularization',
    )
    relation_regularize_weighted_group = parser.add_mutually_exclusive_group()
    relation_regularize_weighted_group.add_argument(
        '--relation-regularize-weighted', '--relation_regularize_weighted',
        dest='relation_regularize_weighted',
        action='store_true',
        default=None,
        help='Scale relation L3 regularization by embedding dimension',
    )
    relation_regularize_weighted_group.add_argument(
        '--no-relation-regularize-weighted', '--no_relation_regularize_weighted',
        dest='relation_regularize_weighted',
        action='store_false',
        help='Use unweighted relation L3 regularization',
    )

    # Separate head/tail negative sample counts (RotatE, etc.).
    parser.add_argument('--n-sample-t', '--n_sample_t', default=None, type=int,
                        dest='n_sample_t', help='Number of tail negative samples per positive')
    parser.add_argument('--n-sample-h', '--n_sample_h', default=None, type=int,
                        dest='n_sample_h', help='Number of head negative samples per positive')

    # Link-prediction evaluation.
    parser.add_argument('--head-eval-mode', '--head_eval_mode', default=None, type=str,
                        dest='head_eval_mode',
                        help='Backward LP head scoring: rt_forward (direct head, forward r), '
                             'rt_inverse (KvsAll _rt), hr_inverse (kbc CE inverse triple), '
                             'or auto (infer from strategy; default when omitted)')
    parser.add_argument('--eval-entity-chunk-size', '--eval_entity_chunk_size',
                        default=None, type=int, dest='eval_entity_chunk_size',
                        help='Entity chunk size for chunked RotatE evaluation')
    parser.add_argument('--tie-handling', '--tie_handling', default=None, type=str,
                        dest='tie_handling', help='Rank tie handling strategy')
    parser.add_argument('--tie-rtol', '--tie_rtol', default=None, type=float,
                        dest='tie_rtol', help='Relative tolerance for rank ties')
    parser.add_argument('--tie-atol', '--tie_atol', default=None, type=float,
                        dest='tie_atol', help='Absolute tolerance for rank ties')

    return parser


def _resolve_output_dir() -> str:
    """Determine the output directory for checkpoints and logs, creating it if necessary."""

    def _default_run_dir() -> str:
        """Construct a default run directory based on the model and dataset names, with a timestamp for uniqueness."""

        base_dir = os.path.join(os.getcwd(), 'logs')
        run_name = f'{_format_model_name(args.model)}_{_format_dataset_name(args.dataset)}'
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        return os.path.join(base_dir, f'{run_name}_{timestamp}')

    def _is_default_placeholder(path: str) -> bool:
        """Check if the given path is empty or matches the default placeholder pattern for this model and dataset."""
        
        if not path:
            return True
        placeholder = os.path.join('logs', f'{_format_model_name(args.model)}_{_format_dataset_name(args.dataset)}')
        normalized_path = os.path.normpath(path)
        normalized_placeholder = os.path.normpath(placeholder)
        absolute_placeholder = os.path.normpath(os.path.join(os.getcwd(), placeholder))
        return normalized_path in {normalized_placeholder, absolute_placeholder}
    # starting candidate list: explicit output_dir, prefix, or fallback defaults
    candidates = [getattr(args, 'output_dir', None) or '', getattr(args, 'output_dir_prefix', None) or '']

    if args.eval_model_path:
        candidates.append(os.path.dirname(args.eval_model_path))
    candidates.append(_default_run_dir())
    candidates.append(os.getcwd())

    for candidate in candidates:
        if _is_default_placeholder(candidate):
            continue
        try:
            os.makedirs(candidate, exist_ok=True)
        except OSError:
            continue
        if os.access(candidate, os.W_OK):
            return candidate

    return os.getcwd()


def _format_model_name(model: str) -> str:
    """Format the model name for consistent config lookup and output naming."""

    if not model:
        return ''
    mapping = {
        'dabr': 'DaBR',
        'dabr-au': 'DaBR-AU',
        'simkgc': 'SimKGC',
        'transe': 'TransE',
        'transe-au': 'TransE-AU',
        'transerr': 'TransERR',
        'transerr-au': 'TransERR-AU',
        'transd': 'TransD',
        'rotate': 'RotatE',
    }
    return mapping.get(model.lower(), model)


def _format_dataset_name(dataset: str) -> str:
    """Format the dataset name for consistent config lookup and output naming."""

    if not dataset:
        return ''
    mapping = {
        'wn18rr': 'WN18RR',
        'fb15k237': 'FB15k237',
        'wiki5m_ind': 'Wiki5M_Ind',
    }
    return mapping.get(dataset.lower(), dataset)


def _resolve_config_path() -> str:
    """Resolve the config JSON path, preferring an explicit path and then configs/ fallbacks."""

    explicit_path = getattr(args, 'config_path', None) or ''
    if explicit_path:
        if os.path.exists(explicit_path):
            return explicit_path
        candidate = os.path.join('configs', explicit_path)
        if os.path.exists(candidate):
            return candidate

    candidates = [
        os.path.join('configs', f'{_format_model_name(args.model)}_{_format_dataset_name(args.dataset)}.json'),
        os.path.join('configs', f'{(args.model or "").lower()}_{(args.dataset or "").lower()}.json'),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def _load_json_defaults(path: str) -> Dict[str, Any]:
    """Load configuration defaults from a JSON object file if it exists."""

    if not path or not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f'Config file must contain a JSON object: {path}')
    return cfg


def _resolve_case_insensitive_path(path: str) -> str:
    """Resolve an existing path on case-sensitive filesystems when only letter case differs."""

    if not path:
        return path
    norm_path = os.path.normpath(path)
    if os.path.exists(norm_path):
        return norm_path

    drive, tail = os.path.splitdrive(norm_path)
    current = drive + os.sep if drive else os.sep if norm_path.startswith(os.sep) else ''
    parts = [p for p in tail.split(os.sep) if p]

    if not parts:
        return norm_path

    for part in parts:
        if not current or not os.path.isdir(current):
            return norm_path
        try:
            entries = os.listdir(current)
        except OSError:
            return norm_path

        matched = None
        lower_part = part.lower()
        for entry in entries:
            if entry.lower() == lower_part:
                matched = entry
                break

        if matched is None:
            return norm_path
        current = os.path.join(current, matched)

    return current if os.path.exists(current) else norm_path


def _resolve_data_path(path: str) -> str:
    """Resolve data paths, falling back from preprocessed *.txt.json to raw *.txt when needed."""

    if not path:
        return path

    candidate = _resolve_case_insensitive_path(path)
    if os.path.exists(candidate):
        return candidate

    if candidate.endswith('.txt.json'):
        raw_candidate = candidate[:-5]
        raw_candidate = _resolve_case_insensitive_path(raw_candidate)
        if os.path.exists(raw_candidate):
            return raw_candidate

    return candidate


def _replace_split_suffix(path: str, source_suffix: str, target_suffix: str) -> str:
    """Replace a dataset split suffix inside a file name while preserving the directory."""

    if not path:
        return path

    directory, basename = os.path.split(path)
    if source_suffix not in basename:
        return path
    return os.path.join(directory, basename.replace(source_suffix, target_suffix))


def _derive_split_variant(path: str, *, split_name: str, labeled: bool) -> str:
    """Map between the raw and labeled split variants for a given split name."""

    if not path:
        return path

    if labeled:
        source_suffix = f'{split_name}.txt'
        target_suffix = f'{split_name}_w_label.txt'
    else:
        source_suffix = f'{split_name}_w_label.txt'
        target_suffix = f'{split_name}.txt'

    return _replace_split_suffix(path, source_suffix, target_suffix)


def _cuda_unavailable_reason() -> str:
    """Return a human-readable reason when CUDA is unavailable in the current Python env."""

    torch_cuda = getattr(torch.version, 'cuda', None)
    torch_version = getattr(torch, '_version_', 'unknown')
    executable = sys.executable

    if not torch_cuda:
        return (
            'CPU-only PyTorch build detected '
            f'(python={executable}, torch={torch_version}). '
            'Install a CUDA wheel in this same environment.'
        )

    return (
        'CUDA runtime is bundled with PyTorch but no GPU is usable in this environment '
        f'(python={executable}, torch={torch_version}, torch_cuda={torch_cuda}). '
        'This is commonly caused by a CUDA-runtime/driver mismatch or running with a different Python env than expected.'
    )


def _resolve_data_path(path: str) -> str:
    """Resolve a dataset path against the repo root and common layout variants."""

    if not path:
        return path
    if os.path.isabs(path) and os.path.exists(path):
        return path

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        path,
        os.path.join(os.getcwd(), path),
        os.path.join(repo_root, path),
    ]

    if '/preprocessed/' in path:
        candidates.append(path.replace('/preprocessed/', '/'))
        candidates.append(os.path.join(repo_root, path.replace('/preprocessed/', '/')))

    if path.endswith('.json'):
        candidates.append(path[:-5])
        candidates.append(os.path.join(repo_root, path[:-5]))
    elif path.endswith('.txt'):
        candidates.append(path + '.json')
        candidates.append(os.path.join(repo_root, path + '.json'))

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return path


def _filter_json_defaults(config_defaults: Dict[str, Any]) -> tuple[Dict[str, Any], list]:
    """Drop non-hyperparameter keys from JSON configs (saved arg dumps, etc.)."""

    if not config_defaults:
        return {}, []
    filtered = dict(config_defaults)
    json_cli_tokens = filtered.pop('unparsed_args', None) or []
    if not isinstance(json_cli_tokens, list):
        json_cli_tokens = []
    return filtered, json_cli_tokens


def _apply_extra_cli_tokens(parser, args, tokens) -> list:
    """Re-parse leftover CLI tokens (e.g. copied into JSON ``unparsed_args``) against the full parser."""

    if not tokens:
        return []
    return parser.parse_known_args(list(tokens), namespace=args)[1]


def _register_dynamic_json_args(parser: argparse.ArgumentParser, config_defaults: Dict[str, Any]) -> None:
    """Register argparse options for JSON keys not yet on the parser (future-proof CLI overrides)."""

    registered = {action.dest for action in parser._actions if action.dest != argparse.SUPPRESS}
    for key, value in config_defaults.items():
        if key in registered or key == 'unparsed_args':
            continue
        flag = f'--{_snake_to_kebab(key)}'
        snake_flag = f'--{key}'
        if isinstance(value, bool):
            group = parser.add_mutually_exclusive_group()
            group.add_argument(flag, snake_flag, dest=key, action='store_true', default=None)
            group.add_argument(f'--no-{_snake_to_kebab(key)}', f'--no_{key}',
                               dest=key, action='store_false')
        elif isinstance(value, int):
            parser.add_argument(flag, snake_flag, dest=key, default=None, type=int)
        elif isinstance(value, float):
            parser.add_argument(flag, snake_flag, dest=key, default=None, type=float)
        elif isinstance(value, list):
            parser.add_argument(flag, snake_flag, dest=key, default=None, nargs='+')
        elif value is None:
            parser.add_argument(flag, snake_flag, dest=key, default=None)
        else:
            parser.add_argument(flag, snake_flag, dest=key, default=None, type=str)


_cli_argv = _normalize_argv()[1:]

parser = build_parser()
args, unknown_args = parser.parse_known_args(_cli_argv)

config_path = _resolve_config_path()
config_defaults = _load_json_defaults(config_path)
json_cli_tokens: list = []
if config_defaults:
    config_defaults, json_cli_tokens = _filter_json_defaults(config_defaults)
    _register_dynamic_json_args(parser, config_defaults)
    parser.set_defaults(**config_defaults)
    args, unknown_args = parser.parse_known_args(_cli_argv)

extra_cli_tokens = list(unknown_args) + list(json_cli_tokens)
args.unparsed_args = _apply_extra_cli_tokens(parser, args, extra_cli_tokens)

# JSON null or omitted optional AU weights must not propagate as None.
for _name, _default in (('gamma_h', 0.0), ('gamma_ent', 0.0), ('gamma_cross', 0.0)):
    if getattr(args, _name, None) is None:
        setattr(args, _name, _default)

_model_name = str(getattr(args, 'model', '') or '').lower()
if any(tag in _model_name for tag in ('rotate', 'protate', 'transe', 'transerr')):
    if getattr(args, 'margin', None) is None:
        args.margin = 6.0
    if getattr(args, 'epsilon', None) is None:
        args.epsilon = 2.0

if _model_name in {'transerr', 'transerr-au'} and getattr(args, 'triple_relation_embedding', None) is None:
    args.triple_relation_embedding = True

if getattr(args, 'workers', None) is None:

    args.workers = 2


def _infer_data_paths_from_dataset() -> None:
    """Fill standard preprocessed split paths from ``dataset`` when omitted in JSON."""

    dataset = getattr(args, 'dataset', None) or ''
    if not dataset:
        return
    base = os.path.join('data', dataset, 'preprocessed')
    defaults = {
        'train_path': os.path.join(base, 'train.txt.json'),
        'valid_path': os.path.join(base, 'valid.txt.json'),
        'test_path': os.path.join(base, 'test.txt.json'),
        'valid_w_label_path': os.path.join(base, 'valid_w_label.txt.json'),
        'test_w_label_path': os.path.join(base, 'test_w_label.txt.json'),
    }
    for key, path in defaults.items():
        if not getattr(args, key, None):
            setattr(args, key, path)


_infer_data_paths_from_dataset()

args.train_path = _resolve_data_path(getattr(args, 'train_path', None) or '')
args.valid_path = _resolve_data_path(_derive_split_variant(getattr(args, 'valid_path', None) or '', split_name='valid', labeled=False))
args.test_path = _resolve_data_path(_derive_split_variant(getattr(args, 'test_path', None) or '', split_name='test', labeled=False))
args.valid_w_label_path = _resolve_data_path(
    getattr(args, 'valid_w_label_path', None) or ''
    or _derive_split_variant(args.valid_path, split_name='valid', labeled=True)
)
args.test_w_label_path = _resolve_data_path(
    getattr(args, 'test_w_label_path', None) or ''
    or _derive_split_variant(args.test_path, split_name='test', labeled=True)
)
assert not args.train_path or os.path.exists(args.train_path)
if args.pooling is not None:
    assert args.pooling in ['cls', 'mean', 'max']
_model_name_for_scheduler = (args.model or '').lower()
_is_index_kge_model = _model_name_for_scheduler in {
    'distmult', 'distmult-au', 'distmult-adversarial', 'distmult-adversarial-au',
    'complex', 'complex-au', 'dabr', 'dabr-au', 'rotate', 'rotate-au', 'protate', 'protate-au',
    'transe', 'transe-au', 'transerr', 'transerr-au',
}
if args.lr_scheduler is not None:
    if _is_index_kge_model:
        assert args.lr_scheduler.lower() in {
            'linear', 'cosine', 'none', 'constant', 'reducelronplateau',
            'step', 'steplr', 'stepdecay',
        }
    else:
        assert args.lr_scheduler in ['linear', 'cosine']

args.config_path = config_path

_model_name = (args.model or '').lower()
_is_text_model = _model_name not in {
    'distmult', 'distmult-au', 'complex', 'complex-au', 'dabr', 'dabr-au',
    'transe', 'transe-au', 'transerr', 'transerr-au',
}

if getattr(args, 'normalize_phases', None) is None and _model_name in {'protate', 'protate-au', 'rotate-au'}:
    args.normalize_phases = True

if _is_text_model:
    args.encoder = args.bert_encoder
    args.pretrained_model = args.bert_encoder
else:
    args.bert_encoder = ''
    args.encoder = ''
    args.pretrained_model = ''

if not args.model_strategy_path:
    if _model_name in {'distmult', 'distmult-au', 'complex', 'complex-au', 'dabr', 'dabr-au'}:
        args.model_strategy_path = 'models/strategies/1vsall_strategy.py'
    else:
        args.model_strategy_path = 'models/strategies/inbatch_strategy.py'

if not args.model_encoder_path:
    if _model_name == 'distmult':
        args.model_encoder_path = 'models/distmult.py'
    elif _model_name == 'distmult-au':
        args.model_encoder_path = 'models/distmult.py'
    elif _model_name == 'complex':
        args.model_encoder_path = 'models/complex.py'
    elif _model_name == 'complex-au':
        args.model_encoder_path = 'models/complex.py'
    elif _model_name == 'dabr':
        args.model_encoder_path = 'models/dabr.py'
    elif _model_name == 'dabr-au':
        args.model_encoder_path = 'models/dabr.py'
    elif _model_name == 'rotate':
        args.model_encoder_path = 'models/rotate.py'
    elif _model_name == 'rotate-au':
        args.model_encoder_path = 'models/rotate.py'
    elif _model_name in {'protate', 'protate-au'}:
        args.model_encoder_path = 'models/protate.py'
    elif _model_name in {'transe', 'transe-au'}:
        args.model_encoder_path = 'models/transe.py'
    elif _model_name in {'transerr', 'transerr-au'}:
        args.model_encoder_path = 'models/transerr.py'
    else:
        args.model_encoder_path = 'models/simkgc.py'

if not args.model_scorer_path:
    args.model_scorer_path = args.model_encoder_path

if not args.model_embedder_path:
    if 'simkgc' in _model_name:
        args.model_embedder_path = 'models/embedders/text_embedder.py'
    else:
        args.model_embedder_path = 'models/embedders/lookup_embedder.py'

if not args.model_sampler_path:
    if _model_name in {'distmult', 'distmult-au', 'complex', 'complex-au', 'dabr', 'dabr-au'}:
        args.model_sampler_path = 'models/samplers/bernoulli_sampler.py'
    else:
        args.model_sampler_path = 'models/samplers/masking_sampler.py'

if not args.model_loss_path and _model_name in {'distmult', 'distmult-au', 'complex', 'complex-au', 'dabr', 'dabr-au'}:
    args.model_loss_path = 'models/losses/infonce_loss.py'

# --task is a separate flag controlling which evaluations to run
# (link prediction / triple classification / both). Do NOT overwrite it
# with args.dataset here so users can specify evaluation task independently.

if args.seed is not None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    try:
        torch.cuda.manual_seed_all(args.seed)
    except Exception:
        # cuda may not be available in all environments
        pass
    cudnn.deterministic = True

try:
    if args.use_amp:
        import torch.cuda.amp
except Exception:
    args.use_amp = False
    warnings.warn('AMP training is not available, set use_amp=False')

if not torch.cuda.is_available():
    args.use_amp = False
    args.print_freq = 1
    warnings.warn(
        'GPU is not available, set use_amp=False and print_freq=1. '
        + _cuda_unavailable_reason()
    )

# Ensure args exposes output_dir (parser flags were removed).
if not hasattr(args, 'output_dir'):
    args.output_dir = ''

# If a user provided an output_dir_prefix (e.g., "logs/Model_Dataset"),
# convert it into a timestamped run directory and prefer it when writable.
if getattr(args, 'output_dir_prefix', None):
    prefix = args.output_dir_prefix.rstrip('/\\')
    import re, datetime
    ts_pattern = re.compile(r'.*\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$')
    if ts_pattern.match(prefix):
        chosen = prefix
    else:
        chosen = prefix + '_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    try:
        os.makedirs(chosen, exist_ok=True)
        if os.access(chosen, os.W_OK):
            args.output_dir = chosen
    except Exception:
        # ignore and fall back to resolver
        pass

# If no explicit output_dir was chosen above, resolve a sensible default.
if not args.output_dir:
    args.output_dir = _resolve_output_dir()
    
def apply_train_args(train_args: SimpleNamespace) -> SimpleNamespace:
    """Merge training-time args from a checkpoint with current global args.

    Ensures any missing flags are filled from current parser defaults and
    updates global args for evaluation flags like use_link_graph and is_test.
    """

    train_args_dict = vars(train_args)
    for k, v in vars(args).items():
        if k not in train_args_dict:
            train_args_dict[k] = v

    # Export training flags to global args used at runtime
    args.use_link_graph = getattr(train_args, 'use_link_graph', args.use_link_graph)
    # When applying training args for evaluation, prefer explicit test flag if present,
    # otherwise set evaluation mode to True to indicate we're loading a checkpoint for eval.
    args.is_test = getattr(train_args, 'is_test', True)
    return train_args


def _merge_with_defaults(cfg: Dict[str, Any]) -> SimpleNamespace:
    """Return a SimpleNamespace merged with current parser defaults.

    This fills in any missing keys from the current args defaults so
    downstream code can rely on a complete args namespace (useful when
    loading hyperparameters from JSON files).
    """

    merged = dict(vars(args))
    merged.update(cfg)
    return SimpleNamespace(**merged)


def load_args_from_json(path: str) -> SimpleNamespace:
    """Load args from a JSON file and merge with parser defaults.

    Returns a SimpleNamespace suitable to pass to apply_train_args.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    return _merge_with_defaults(cfg)


def save_args_to_json(namespace: SimpleNamespace, path: str) -> None:
    """Save an args namespace to a JSON file (converting to plain dict)."""
    
    d = dict(vars(namespace))
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(d, f, indent=2, sort_keys=True)
