"""Block 2 Autoencoder (AE) DataLoader Utilities.

Provides PyTorch DataLoader construction utilities for:
- Real Autoencoder training loader (real images only, label_a == 0).
- AI Autoencoder training loader (AI images only, label_a == 1).
- Shared evaluation loaders (both classes, class_filter=None) for validation and test.
- Class-filtered evaluation loaders for cross-class reconstruction error assessment.

Key Guarantees:
- Correct shuffle defaults: True for 'train', False for 'validation' and 'test'.
- Automatic pin_memory detection (active when CUDA is available unless overridden).
- Multiprocessing safe (num_workers >= 0) with clean worker state serialization.
- Preserves sample metadata (sample_idx, image_id) across batches via PyTorch default collate.
- No redundant persistent datasets or feature caches created.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Union

import torch
from torch.utils.data import DataLoader

from .dataset import AEDataset, AIAEDataset, RealAEDataset


def create_ae_loader(
    split: str = "train",
    class_filter: Optional[Union[str, int]] = None,
    batch_size: int = 32,
    shuffle: Optional[bool] = None,
    num_workers: int = 4,
    pin_memory: Optional[bool] = None,
    data_dir: Optional[str] = None,
    transform: Optional[Callable] = None,
    return_metadata: bool = True,
    **kwargs: Any,
) -> DataLoader:
    """Create a PyTorch DataLoader for Autoencoder workflows.

    Args:
        split: "train", "validation", or "test". Defaults to "train".
        class_filter: "real" (0), "ai" (1), or None for all images. Defaults to None.
        batch_size: Batch size. Defaults to 32.
        shuffle: Whether to shuffle. If None, defaults to True for 'train' and False for
                 evaluation splits ('validation', 'test').
        num_workers: Multiprocessing worker count. Defaults to 4.
        pin_memory: Whether to pin memory for fast GPU transfer. If None, defaults to
                    torch.cuda.is_available().
        data_dir: Optional custom data directory. Defaults to None (canonical Block 1 paths).
        transform: Optional symmetric transformation applied before tensor conversion.
        return_metadata: If True (default), batches include metadata dictionary.
        **kwargs: Additional keyword arguments forwarded directly to torch.utils.data.DataLoader.

    Returns:
        Configured torch.utils.data.DataLoader instance.
    """
    if shuffle is None:
        shuffle = split == "train"

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    dataset = AEDataset(
        split=split,
        class_filter=class_filter,
        data_dir=data_dir,
        transform=transform,
        return_metadata=return_metadata,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **kwargs,
    )


def create_real_ae_loader(
    split: str = "train",
    batch_size: int = 32,
    shuffle: Optional[bool] = None,
    num_workers: int = 4,
    pin_memory: Optional[bool] = None,
    data_dir: Optional[str] = None,
    transform: Optional[Callable] = None,
    return_metadata: bool = True,
    **kwargs: Any,
) -> DataLoader:
    """Create a DataLoader selecting only real images (label_a == 0).

    Used primarily for training the Real Autoencoder on real images (7,000 samples for 'train'),
    or evaluating its reconstruction performance on real validation/test images.
    """
    return create_ae_loader(
        split=split,
        class_filter="real",
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        data_dir=data_dir,
        transform=transform,
        return_metadata=return_metadata,
        **kwargs,
    )


def create_ai_ae_loader(
    split: str = "train",
    batch_size: int = 32,
    shuffle: Optional[bool] = None,
    num_workers: int = 4,
    pin_memory: Optional[bool] = None,
    data_dir: Optional[str] = None,
    transform: Optional[Callable] = None,
    return_metadata: bool = True,
    **kwargs: Any,
) -> DataLoader:
    """Create a DataLoader selecting only AI-generated images (label_a == 1).

    Used primarily for training the AI Autoencoder on AI images (35,000 samples for 'train'),
    or evaluating its reconstruction performance on AI validation/test images.
    """
    return create_ae_loader(
        split=split,
        class_filter="ai",
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        data_dir=data_dir,
        transform=transform,
        return_metadata=return_metadata,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Evaluation and convenience helpers
# ---------------------------------------------------------------------------


def create_ae_eval_loader(
    split: str = "validation",
    class_filter: Optional[Union[str, int]] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: Optional[bool] = None,
    data_dir: Optional[str] = None,
    transform: Optional[Callable] = None,
    return_metadata: bool = True,
    **kwargs: Any,
) -> DataLoader:
    """Create an evaluation DataLoader (shuffle=False).

    When class_filter=None (default), returns the shared evaluation dataset with both
    real and AI images so both autoencoders can be evaluated on the exact same images.
    """
    return create_ae_loader(
        split=split,
        class_filter=class_filter,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        data_dir=data_dir,
        transform=transform,
        return_metadata=return_metadata,
        **kwargs,
    )


def create_real_ae_train_loader(
    batch_size: int = 32,
    num_workers: int = 4,
    **kwargs: Any,
) -> DataLoader:
    """Convenience helper for Real AE training (split='train', class_filter='real', shuffle=True)."""
    return create_real_ae_loader(
        split="train",
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        **kwargs,
    )


def create_real_ae_val_loader(
    batch_size: int = 32,
    num_workers: int = 4,
    **kwargs: Any,
) -> DataLoader:
    """Convenience helper for Real AE evaluation on real validation images (shuffle=False)."""
    return create_real_ae_loader(
        split="validation",
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        **kwargs,
    )


def create_real_ae_test_loader(
    batch_size: int = 32,
    num_workers: int = 4,
    **kwargs: Any,
) -> DataLoader:
    """Convenience helper for Real AE evaluation on real test images (shuffle=False)."""
    return create_real_ae_loader(
        split="test",
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        **kwargs,
    )


def create_ai_ae_train_loader(
    batch_size: int = 32,
    num_workers: int = 4,
    **kwargs: Any,
) -> DataLoader:
    """Convenience helper for AI AE training (split='train', class_filter='ai', shuffle=True)."""
    return create_ai_ae_loader(
        split="train",
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        **kwargs,
    )


def create_ai_ae_val_loader(
    batch_size: int = 32,
    num_workers: int = 4,
    **kwargs: Any,
) -> DataLoader:
    """Convenience helper for AI AE evaluation on AI validation images (shuffle=False)."""
    return create_ai_ae_loader(
        split="validation",
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        **kwargs,
    )


def create_ai_ae_test_loader(
    batch_size: int = 32,
    num_workers: int = 4,
    **kwargs: Any,
) -> DataLoader:
    """Convenience helper for AI AE evaluation on AI test images (shuffle=False)."""
    return create_ai_ae_loader(
        split="test",
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        **kwargs,
    )
