"""Feature Registry for the Canonical 111 Candidate Forensic Feature Bank.

Provides machine-readable schema, descriptions, domain categories, and validation
for all 111 features across Branches A, B, C_LBP, D_MFR, and E.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import torch
from src.forensics.pipeline import ForensicPipeline

CANONICAL_BRANCHES: Tuple[str, ...] = ("A", "B", "C_LBP", "D_MFR", "E")
EXPECTED_FEATURE_COUNT: int = 111
EXPECTED_BRANCH_COUNTS: Dict[str, int] = {
    "A": 34,
    "B": 30,
    "C_LBP": 16,
    "D_MFR": 5,
    "E": 26,
}


@dataclass(frozen=True)
class FeatureMetadata:
    """Metadata describing a single candidate forensic feature."""

    index: int
    name: str
    branch: str
    sub_branch: str
    domain: str
    data_type: str
    description: str
    sensitivity: str


def _generate_feature_metadata(index: int, name: str) -> FeatureMetadata:
    """Derive deterministic metadata for a given feature name and index."""
    # Branch A
    if name.startswith("fft_"):
        return FeatureMetadata(
            index=index,
            name=name,
            branch="A",
            sub_branch="A1_FFT",
            domain="frequency",
            data_type="float32",
            description=f"Standard 2D Fourier radial spectral feature: {name}",
            sensitivity="frequency_energy",
        )
    if name.startswith("synth_"):
        is_hf = "fft_highfreq_ratio" in name
        return FeatureMetadata(
            index=index,
            name=name,
            branch="A",
            sub_branch="A2_SYNTH",
            domain="frequency_periodicity",
            data_type="float32",
            description=f"Synthbuster-inspired cross-difference periodicity feature: {name}",
            sensitivity="periodicity_highfreq" if is_hf else "periodicity_peaks",
        )

    # Branch B
    if name.startswith("LL3_") or any(
        name.startswith(f"{b}_")
        for b in ["LH3", "HL3", "HH3", "LH2", "HL2", "HH2", "LH1", "HL1", "HH1"]
    ) or name == "detail_energy_ratio":
        subband = name.split("_")[0]
        return FeatureMetadata(
            index=index,
            name=name,
            branch="B",
            sub_branch="B_HAAR",
            domain="wavelet",
            data_type="float32",
            description=f"3-level 2D Haar DWT multiscale wavelet feature ({subband}): {name}",
            sensitivity="wavelet_multiscale",
        )

    # Branch C (C_LBP)
    if name.startswith("lbp_"):
        return FeatureMetadata(
            index=index,
            name=name,
            branch="C_LBP",
            sub_branch="C_LBP",
            domain="texture",
            data_type="float32",
            description=f"Standard rotation-invariant uniform LBP texture feature: {name}",
            sensitivity="local_texture",
        )

    # Branch D (D_MFR)
    if name.startswith("mfr_"):
        return FeatureMetadata(
            index=index,
            name=name,
            branch="D_MFR",
            sub_branch="D_MFR",
            domain="residual",
            data_type="float32",
            description=f"Nonlinear 3x3 median filter residual noise feature: {name}",
            sensitivity="nonlinear_residual",
        )

    # Branch E
    if name.startswith("dct_"):
        return FeatureMetadata(
            index=index,
            name=name,
            branch="E",
            sub_branch="E1_DCT",
            domain="compression_transform",
            data_type="float32",
            description=f"8x8 block DCT basis transform & zonal distribution feature: {name}",
            sensitivity="transform_dct",
        )
    if name.startswith("ela_"):
        return FeatureMetadata(
            index=index,
            name=name,
            branch="E",
            sub_branch="E2_RESP",
            domain="compression_response",
            data_type="float32",
            description=f"Controlled multi-quality JPEG recompression response feature: {name}",
            sensitivity="compression_response",
        )
    if name.startswith("phase_"):
        return FeatureMetadata(
            index=index,
            name=name,
            branch="E",
            sub_branch="E3_PHASE",
            domain="compression_phase",
            data_type="float32",
            description=f"Compression-stable Fourier phase spectrum response feature: {name}",
            sensitivity="phase_stability",
        )
    if name.startswith("grid_"):
        return FeatureMetadata(
            index=index,
            name=name,
            branch="E",
            sub_branch="E4_GRID",
            domain="compression_grid",
            data_type="float32",
            description=f"Canonical 8x8 JPEG grid boundary discontinuity feature: {name}",
            sensitivity="grid_discontinuity",
        )

    raise ValueError(f"Unrecognized feature name: '{name}'. Cannot assign metadata.")


class FeatureRegistry:
    """Registry managing the canonical 111 candidate forensic feature bank."""

    def __init__(self, pipeline: Optional[ForensicPipeline] = None) -> None:
        """Initialize registry from canonical ForensicPipeline."""
        if pipeline is None:
            self._pipeline = ForensicPipeline(branches=CANONICAL_BRANCHES)
        else:
            self._pipeline = pipeline

        self._feature_names: List[str] = self._pipeline.get_feature_names()
        self._validate_contract()

        self._entries: List[FeatureMetadata] = [
            _generate_feature_metadata(i, name)
            for i, name in enumerate(self._feature_names)
        ]
        self._by_name: Dict[str, FeatureMetadata] = {
            e.name: e for e in self._entries
        }

    def _validate_contract(self) -> None:
        """Validate exact 111-feature contract, uniqueness, and branch allocations."""
        total = len(self._feature_names)
        if total != EXPECTED_FEATURE_COUNT:
            raise ValueError(
                f"Feature count mismatch: expected exactly {EXPECTED_FEATURE_COUNT}, got {total}."
            )

        unique = set(self._feature_names)
        if len(unique) != total:
            raise ValueError(
                f"Duplicate feature names detected! Total: {total}, Unique: {len(unique)}."
            )

        # Probe individual branches to ensure branch counts match expectations
        for branch_name, expected_count in EXPECTED_BRANCH_COUNTS.items():
            branch_pipe = ForensicPipeline(branches=[branch_name])
            b_names = branch_pipe.get_feature_names()
            if len(b_names) != expected_count:
                raise ValueError(
                    f"Branch '{branch_name}' count mismatch: expected {expected_count}, got {len(b_names)}."
                )

    @property
    def feature_names(self) -> List[str]:
        """List of all 111 feature names in deterministic order."""
        return list(self._feature_names)

    @property
    def entries(self) -> List[FeatureMetadata]:
        """List of metadata entries for all 111 features."""
        return list(self._entries)

    def get_by_name(self, name: str) -> FeatureMetadata:
        """Retrieve metadata entry by feature name."""
        if name not in self._by_name:
            raise KeyError(f"Feature '{name}' not found in registry.")
        return self._by_name[name]

    def get_by_index(self, index: int) -> FeatureMetadata:
        """Retrieve metadata entry by feature index."""
        if index < 0 or index >= len(self._entries):
            raise IndexError(f"Index {index} out of range [0, {len(self._entries)-1}].")
        return self._entries[index]

    def get_features_by_branch(self, branch: str) -> List[FeatureMetadata]:
        """Retrieve all features belonging to a specific branch."""
        return [e for e in self._entries if e.branch == branch]

    def get_features_by_sub_branch(self, sub_branch: str) -> List[FeatureMetadata]:
        """Retrieve all features belonging to a specific sub-branch."""
        return [e for e in self._entries if e.sub_branch == sub_branch]

    def export_csv(self, output_path: str) -> str:
        """Export machine-readable feature registry to CSV."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fieldnames = [
            "index",
            "name",
            "branch",
            "sub_branch",
            "domain",
            "data_type",
            "description",
            "sensitivity",
        ]
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for entry in self._entries:
                writer.writerow(asdict(entry))
        return output_path

    def export_json(self, output_path: str) -> str:
        """Export machine-readable feature registry to JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        data = [asdict(e) for e in self._entries]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return output_path
