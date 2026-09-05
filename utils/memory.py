"""GPU/CPU memory tracking utilities."""

import sys
from typing import Optional

import torch

BYTES_PER_MB = 1024 ** 2


def format_memory(megabytes: float) -> str:
    """Format a memory size in megabytes, using GB when large enough."""

    if megabytes >= 1024:
        return f'{megabytes / 1024:.2f} GB ({megabytes:.2f} MB)'
    return f'{megabytes:.2f} MB'


def reset_peak_memory() -> None:
    """Reset peak memory counters before a timed phase."""

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def current_peak_memory_mb() -> float:
    """Return peak GPU memory since the last reset, or process RSS on CPU."""

    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / BYTES_PER_MB
    return _current_process_memory_mb()


def _current_process_memory_mb() -> float:
    """Best-effort current process RSS in megabytes."""

    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == 'darwin':
            return usage / BYTES_PER_MB
        return usage / 1024
    except (ImportError, OSError, AttributeError):
        return 0.0


class PhaseMemoryTracker:
    """Track peak memory usage for training and evaluation."""

    def __init__(self) -> None:
        self.train_peak_mb = 0.0
        self.eval_peak_mb = 0.0
        self.peak_memory_mb = 0.0

    def begin_phase(self) -> None:
        """Reset peak counters at the start of a phase."""

        reset_peak_memory()

    def end_phase(self, phase: str) -> None:
        """Record the peak memory used during the phase that just finished."""

        peak_mb = current_peak_memory_mb()
        if phase == 'train':
            self.train_peak_mb = max(self.train_peak_mb, peak_mb)
        elif phase in ('eval', 'valid', 'test'):
            self.eval_peak_mb = max(self.eval_peak_mb, peak_mb)
        else:
            raise ValueError(f'Unknown memory phase: {phase}')
        self.peak_memory_mb = max(self.peak_memory_mb, peak_mb)

    def update_from_summary(self, summary: Optional[dict]) -> None:
        """Load phase peaks from a training summary dictionary."""

        if not summary:
            return
        self.train_peak_mb = float(summary.get('train_peak_mb') or 0.0)
        self.eval_peak_mb = float(summary.get('eval_peak_mb') or 0.0)
        self.peak_memory_mb = float(summary.get('peak_memory_mb') or 0.0)

    def to_dict(self) -> dict:
        """Serialize tracked peaks for result reporting."""

        return {
            'train_peak_mb': self.train_peak_mb,
            'eval_peak_mb': self.eval_peak_mb,
            'peak_memory_mb': self.peak_memory_mb,
        }
