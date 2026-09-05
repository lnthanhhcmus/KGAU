"""Checkpoint saving and loading utilities for KG models."""

import glob
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from collections import OrderedDict

from utils.logger import logger


BEST_MODEL_FILENAME = 'best_model.mdl'
LAST_MODEL_FILENAME = 'last_model.mdl'


def _candidate_paths(path: str):
    candidate = Path(path)
    candidates = [candidate]
    if candidate.suffix:
        alternate_suffix = '.mdl' if candidate.suffix != '.mdl' else '.pth'
        candidates.append(candidate.with_suffix(alternate_suffix))
    else:
        candidates.extend([candidate.with_suffix('.mdl'), candidate.with_suffix('.pth')])
    if candidate.name == BEST_MODEL_FILENAME:
        candidates.append(candidate.with_name('model_best.pth'))
    if candidate.name == LAST_MODEL_FILENAME:
        candidates.append(candidate.with_name('model_last.pth'))
    for item in candidates:
        yield item


def best_model_path(model_dir: str) -> str:
    return str(Path(model_dir) / BEST_MODEL_FILENAME)


def last_model_path(model_dir: str) -> str:
    return str(Path(model_dir) / LAST_MODEL_FILENAME)


def checkpoint_path(model_dir: str, epoch: int, step: Optional[int] = None) -> str:
    if step is None:
        return str(Path(model_dir) / f'checkpoint_epoch{epoch}.mdl')
    return str(Path(model_dir) / f'checkpoint_{epoch}_{step}.mdl')


def save_checkpoint(state: Dict[str, Any], is_best: bool, filename: str) -> str:
    """Save a training checkpoint to disk."""

    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)

    if is_best:
        best_path = path.with_name(BEST_MODEL_FILENAME)
        if best_path != path:
            shutil.copyfile(path, best_path)
    last_path = path.with_name(LAST_MODEL_FILENAME)
    if last_path != path:
        shutil.copyfile(path, last_path)
    return str(path)


def load_checkpoint(path: str, map_location: Optional[str] = 'cpu') -> Dict[str, Any]:
    """Load a checkpoint dictionary from disk."""

    resolved_path = None
    for candidate in _candidate_paths(path):
        if os.path.exists(candidate):
            resolved_path = str(candidate)
            break
    if resolved_path is None:
        raise FileNotFoundError(path)
    return torch.load(resolved_path, map_location=map_location)


def save_model_weights(model: torch.nn.Module, path: str, **metadata: Any) -> str:
    """Save only model weights to a `.mdl` file."""

    payload: Dict[str, Any] = {'state_dict': model.state_dict()}
    payload.update(metadata)
    return save_checkpoint(payload, is_best=False, filename=path)


def save_last_model_weights(model: torch.nn.Module, model_dir: str, **metadata: Any) -> str:
    """Convenience helper to save model weights directly as `last_model.mdl` in `model_dir`."""

    payload: Dict[str, Any] = {'state_dict': model.state_dict()}
    payload.update(metadata)
    target = Path(model_dir) / LAST_MODEL_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)
    return str(target)


def load_model_weights(model: torch.nn.Module, path: str, map_location: Optional[str] = 'cpu', strict: bool = True) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Load weights from a `.mdl` checkpoint into a model."""

    checkpoint = load_checkpoint(path, map_location=map_location)
    state_dict = checkpoint.get('state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=strict)
    return model, checkpoint


def delete_old_checkpoints(path_pattern: str, keep: int = 5) -> None:
    """Delete older checkpoints, keeping only the newest `keep` files."""

    files = sorted(glob.glob(path_pattern), key=os.path.getmtime, reverse=True)
    for file_path in files[keep:]:
        logger.info('Delete old checkpoint %s', file_path)
        try:
            os.remove(file_path)
        except FileNotFoundError:
            continue


delete_old_ckt = delete_old_checkpoints


def load_state_dict_clean(model: torch.nn.Module, ckt_path: str, map_location: Optional[str] = 'cpu', strict: bool = True) -> Dict[str, Any]:
    """Load a checkpoint and strip the `module.` prefix introduced by DataParallel.

    Returns the loaded checkpoint dict.
    """

    checkpoint = load_checkpoint(ckt_path, map_location=map_location)
    state_dict = checkpoint.get('state_dict', checkpoint)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_key = k[len('module.'):]
        else:
            new_key = k
        new_state_dict[new_key] = v

    from utils.checkpoint_migration import migrate_simkgc_state_dict

    migrated_state_dict, migrated = migrate_simkgc_state_dict(new_state_dict)
    if migrated:
        logger.warning('Applied legacy SimKGC checkpoint key migration')

    model.load_state_dict(migrated_state_dict, strict=strict)
    return checkpoint