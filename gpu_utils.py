"""
GPU/CPU device configuration utilities for TensorFlow.

Usage examples:
  from gpu_utils import configure_gpu, force_cpu
  device = configure_gpu(memory_growth=True, quiet=False)
  # or
  device = force_cpu()

Notes:
- Environment variables (e.g., TF_CPP_MIN_LOG_LEVEL) are set here to keep callers clean.
- TensorFlow is imported inside functions to avoid side effects during module import.
"""

import os
from typing import Optional

# Default environment settings (can be overridden by caller before import if needed)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')  # Suppress INFO/WARN
os.environ.setdefault('CUDA_HOME', '/usr/local/cuda')
os.environ.setdefault(
    'LD_LIBRARY_PATH',
    '/usr/local/cuda/targets/x86_64-linux/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
)


def configure_gpu(memory_growth: bool = True, quiet: bool = False) -> str:
    """Configure TensorFlow to use GPU if available.

    - Enables memory growth on all visible GPUs (if memory_growth=True).
    - Returns the preferred device string ('/GPU:0' if GPU present, else '/CPU:0').
    - Prints a short message unless quiet=True.
    """
    import tensorflow as tf  # Local import to avoid imposing TF on unrelated scripts

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        if memory_growth:
            try:
                for gpu in gpus:
                    # Use experimental API for broader TF version compatibility
                    try:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    except Exception:
                        # Fallback to non-experimental if available
                        tf.config.set_memory_growth(gpu, True)
            except Exception as e:
                if not quiet:
                    print(f"GPU configuration warning: {e}")
        if not quiet:
            try:
                names = [gpu.name for gpu in gpus]
            except Exception:
                names = [str(g) for g in gpus]
            print(f"Found {len(gpus)} GPU(s): {names}")
        return '/GPU:0'
    else:
        if not quiet:
            print("No GPUs found - using CPU")
        return '/CPU:0'


essential_visible_devices_cached = False

def force_cpu(quiet: bool = False) -> str:
    """Force TensorFlow to run on CPU by hiding all GPUs.

    Returns '/CPU:0'.
    """
    import tensorflow as tf
    try:
        tf.config.set_visible_devices([], 'GPU')
    except Exception:
        # Older TF API
        try:
            tf.config.experimental.set_visible_devices([], 'GPU')
        except Exception:
            pass
    if not quiet:
        print("Forced CPU: all GPUs hidden from TensorFlow")
    return '/CPU:0'
