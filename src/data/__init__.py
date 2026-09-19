# src/data/__init__.py
# Block 1 — Data Loading + Preprocessing
#
# Public API:
#
#   preprocessing.py       — canonical image preprocessing (crop + resize)
#   loader.py              — DefactifyDataset: on-demand raw Parquet reader
#   materializer.py        — materialize_training_dataset(): write processed Parquet
#   processed_loader.py    — ProcessedTrainDataset: read materialized processed Parquet

from .preprocessing import (
    preprocess,
    centered_square_crop,
    resize_to_canonical,
    CANONICAL_SIZE,
    INTERPOLATION,
)
from .loader import DefactifyDataset
from .materializer import (
    materialize_training_dataset,
    get_processed_train_dir,
    PREPROCESSING_VERSION,
)
from .processed_loader import ProcessedTrainDataset

__all__ = [
    # preprocessing
    "preprocess",
    "centered_square_crop",
    "resize_to_canonical",
    "CANONICAL_SIZE",
    "INTERPOLATION",
    # on-demand eval loader
    "DefactifyDataset",
    # materialization
    "materialize_training_dataset",
    "get_processed_train_dir",
    "PREPROCESSING_VERSION",
    # materialized training reader
    "ProcessedTrainDataset",
]
