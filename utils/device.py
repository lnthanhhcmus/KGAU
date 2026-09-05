"""Device management utilities for PyTorch models."""

from typing import Any, Optional

import torch
import torch.nn as nn

from utils.logger import logger

import torch.backends.cudnn as cudnn


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Get the appropriate torch.device based on CUDA availability and user preference."""

    return torch.device('cuda' if prefer_cuda and torch.cuda.is_available() else 'cpu')


def init_hardware(args) -> int:
    """Initialize CUDA/cuDNN once at program start and return GPU count."""

    ngpus_per_node = torch.cuda.device_count()
    cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass
    return ngpus_per_node


def call_model_forward(model: nn.Module, batch_dict: dict) -> dict:
    """Call ``model.forward`` with explicit kwargs (DataParallel-safe, SimKGC-style)."""

    return model(
        hr_token_ids=batch_dict['hr_token_ids'],
        hr_mask=batch_dict['hr_mask'],
        hr_token_type_ids=batch_dict['hr_token_type_ids'],
        tail_token_ids=batch_dict.get('tail_token_ids'),
        tail_mask=batch_dict.get('tail_mask'),
        tail_token_type_ids=batch_dict.get('tail_token_type_ids'),
        head_token_ids=batch_dict.get('head_token_ids'),
        head_mask=batch_dict.get('head_mask'),
        head_token_type_ids=batch_dict.get('head_token_type_ids'),
        only_ent_embedding=batch_dict.get('only_ent_embedding', False),
        encode_hr_only=batch_dict.get('encode_hr_only', False),
    )


def setup_data_parallel(model: nn.Module) -> nn.Module:
    """Wrap *model* with DataParallel when multiple GPUs are available."""

    if torch.cuda.device_count() > 1:
        return torch.nn.DataParallel(model).cuda()
    if torch.cuda.is_available():
        return model.cuda()
    logger.info('No gpu will be used')
    return model


def setup_cuda(enable: bool = True, seed: Optional[int] = None, benchmark: bool = True, deterministic: bool = False) -> torch.device:
    """Initialize CUDA/cuDNN settings and return the selected device."""

    use_cuda = enable and torch.cuda.is_available()

    if seed is not None:
        torch.manual_seed(seed)
        if use_cuda:
            torch.cuda.manual_seed_all(seed)

    if use_cuda:
        torch.backends.cudnn.benchmark = benchmark
        torch.backends.cudnn.deterministic = deterministic

    return get_device(use_cuda)


def move_to_device(sample: Any, device: Optional[torch.device] = None) -> Any:
    """Recursively move nested tensors to a target device."""

    if device is None:
        device = get_device()

    if sample is None:
        return None

    def _move(value) -> Any:
        if torch.is_tensor(value):
            return value.to(device, non_blocking=True)
        if isinstance(value, dict):
            return {key: _move(val) for key, val in value.items()}
        if isinstance(value, list):
            return [_move(item) for item in value]
        if isinstance(value, tuple):
            return tuple(_move(item) for item in value)
        return value

    return _move(sample)


def move_to_cuda(sample: Any) -> Any:
    """Backward-compatible alias that moves tensors to CUDA when available."""

    return move_to_device(sample, get_device())


def get_model_obj(model: nn.Module) -> nn.Module:
    """Unwrap DataParallel / DistributedDataParallel modules."""

    return model.module if hasattr(model, 'module') else model


def report_num_trainable_parameters(model: torch.nn.Module) -> int:
    """Log and return the number of trainable parameters."""

    assert isinstance(model, torch.nn.Module), 'Argument must be nn.Module'

    num_parameters = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count = param.numel()
            num_parameters += param_count
            logger.info('%s: %s', name, param_count)

    logger.info('Number of parameters: %sM', num_parameters // 10**6)
    return num_parameters