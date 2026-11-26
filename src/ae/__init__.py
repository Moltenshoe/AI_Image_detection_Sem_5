"""Block 2 — Autoencoder (AE) Data Interface.

Independent Block 2 module providing data loading infrastructure for two class-specific
Convolutional Autoencoders (CAEs):
- Real Autoencoder: Trained exclusively on real images (label_a == 0).
- AI Autoencoder: Trained exclusively on AI-generated images (label_a == 1).

Public API:
    Datasets:
        AEDataset: General-purpose AE dataset supporting class filtering and shared evaluation.
        RealAEDataset: Convenience dataset pre-configured for real images.
        AIAEDataset: Convenience dataset pre-configured for AI images.

    Loaders:
        create_ae_loader: General-purpose DataLoader factory.
        create_real_ae_loader: DataLoader factory for real images.
        create_ai_ae_loader: DataLoader factory for AI images.
        create_ae_eval_loader: Evaluation loader (shared or class-specific).
        create_real_ae_train_loader: Training loader for Real AE.
        create_real_ae_val_loader: Validation loader for Real AE.
        create_real_ae_test_loader: Test loader for Real AE.
        create_ai_ae_train_loader: Training loader for AI AE.
        create_ai_ae_val_loader: Validation loader for AI AE.
        create_ai_ae_test_loader: Test loader for AI AE.
"""

from __future__ import annotations

from .dataset import (
    AEDataset,
    AIAEDataset,
    RealAEDataset,
)
from .loader import (
    create_ae_eval_loader,
    create_ae_loader,
    create_ai_ae_loader,
    create_ai_ae_test_loader,
    create_ai_ae_train_loader,
    create_ai_ae_val_loader,
    create_real_ae_loader,
    create_real_ae_test_loader,
    create_real_ae_train_loader,
    create_real_ae_val_loader,
)

__all__ = [
    "AEDataset",
    "RealAEDataset",
    "AIAEDataset",
    "create_ae_loader",
    "create_real_ae_loader",
    "create_ai_ae_loader",
    "create_ae_eval_loader",
    "create_real_ae_train_loader",
    "create_real_ae_val_loader",
    "create_real_ae_test_loader",
    "create_ai_ae_train_loader",
    "create_ai_ae_val_loader",
    "create_ai_ae_test_loader",
]
