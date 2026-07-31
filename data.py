#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Data loading utilities for the reservoir scheduling workflow."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from feature_engineering import DataTransformer, FeatureEngineer

try:
    from config_loader import get_config
except Exception:  # pragma: no cover - optional dependency
    get_config = None  # type: ignore


RESERVOIR_NAMES: Tuple[str, ...] = (
    "乌东德",
    "白鹤滩",
    "溪洛渡",
    "向家坝",
    "三峡",
    "葛洲坝",
)


def _sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names for easier pattern matching."""

    renamed: Dict[str, str] = {}
    for col in df.columns:
        new_col = str(col)
        for token in ("(", ")", "（", "）", " ", "m³/s", "m3/s", "m^3/s", "M3/S"):
            new_col = new_col.replace(token, "")
        new_col = new_col.replace("区间来水", "_区间来水")
        renamed[col] = new_col.strip()
    return df.rename(columns=renamed)


def _load_constraints_table(script_dir: str) -> pd.DataFrame:
    path = os.path.join(script_dir, "shuxing", "约束条件.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("约束文件缺失: %s" % path)
    return _sanitize_columns(pd.read_csv(path, encoding="utf-8-sig"))



class EnhancedReservoirDataset(Dataset):
    """Legacy sliding-window dataset (kept for compatibility)."""

    def __init__(
        self,
        data_dir: str,
        years: List[int],
        sequence_length: int = 3,
        use_log_transform: bool = True,
        normalize: bool = True,
        fit_transforms: bool = True,
        precomputed_transforms: Optional[Dict[str, object]] = None,
    ) -> None:
        self.data_dir = data_dir
        self.years = years
        self.sequence_length = sequence_length
        self.use_log_transform = use_log_transform
        self.normalize = normalize
        self.fit_transforms = fit_transforms
        self.precomputed_transforms = precomputed_transforms

        self.feature_engineer = FeatureEngineer()
        self.data_transformer = DataTransformer()
        self.data_transformer.use_log_transform = self.use_log_transform
        self.data_transformer.normalize = self.normalize

        self.sequences: List[np.ndarray] = []
        self.targets: List[np.ndarray] = []
        self.original_outflows: List[np.ndarray] = []

        self._load_and_preprocess()

    def _load_and_preprocess(self) -> None:
        script_dir = os.path.dirname(os.path.abspath(__file__))

        inflow_all: List[np.ndarray] = []
        outflow_all: List[np.ndarray] = []
        feature_all: List[np.ndarray] = []

        from exceptions import DataFileError
        
        for year in self.years:
            path = os.path.join(script_dir, self.data_dir, f"{year}.csv")
            if not os.path.exists(path):
                raise DataFileError(
                    f"数据文件缺失: {path}\n"
                    f"请检查 {self.data_dir} 目录下是否存在 {year}.csv 文件。"
                )

            df = _sanitize_columns(pd.read_csv(path, encoding="utf-8-sig"))

            inflow_cols = [col for col in df.columns if "入库流量" in col]
            outflow_cols = [col for col in df.columns if "出库流量" in col]
            interval_cols = [col for col in df.columns if "_区间来水" in col]

            inflow = df[inflow_cols].to_numpy(dtype=np.float32)
            outflow = df[outflow_cols].to_numpy(dtype=np.float32)
            interval = df[interval_cols].to_numpy(dtype=np.float32)

            combined = np.hstack([inflow, interval]).astype(np.float32)
            features = self.feature_engineer.create_features(combined, df=df).astype(np.float32)

            inflow_all.append(combined)
            outflow_all.append(outflow)
            feature_all.append(features)

        inflow_all_arr = np.vstack(inflow_all)
        outflow_all_arr = np.vstack(outflow_all)
        feature_all_arr = np.vstack(feature_all)

        if self.precomputed_transforms is not None:
            self.data_transformer.load_state(self.precomputed_transforms)
            _, outflow_norm, feature_norm = self.data_transformer.transform(
                inflow_all_arr, outflow_all_arr, feature_all_arr
            )
        else:
            if self.fit_transforms:
                _, outflow_norm, feature_norm = self.data_transformer.fit_transform(
                    inflow_all_arr, outflow_all_arr, feature_all_arr
                )
            else:
                outflow_norm, feature_norm = outflow_all_arr, feature_all_arr

        # build sliding windows
        for idx in range(len(feature_norm)):
            start = max(0, idx - self.sequence_length + 1)
            window = feature_norm[start : idx + 1]
            if len(window) < self.sequence_length:
                pad = np.repeat(window[0:1], self.sequence_length - len(window), axis=0)
                window = np.vstack([pad, window])
            self.sequences.append(window.astype(np.float32))
            self.targets.append(outflow_norm[idx].astype(np.float32))
            self.original_outflows.append(outflow_all_arr[idx].astype(np.float32))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.sequences[index], dtype=torch.float32),
            torch.tensor(self.targets[index], dtype=torch.float32),
        )

    # Convenience helpers -------------------------------------------------
    def inverse_transform_outflow(self, outflow_normalized):
        return self.data_transformer.inverse_transform_outflow(outflow_normalized)

    def export_transforms(self) -> Dict[str, object]:
        return self.data_transformer.export_transforms()

    def save_transforms(self, filepath: str) -> None:
        self.data_transformer.save_transforms(filepath)

    @classmethod
    def load_transforms(cls, filepath: str) -> Dict[str, object]:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Transform file not found: {filepath}")
        transforms_dict = joblib.load(filepath)
        if not isinstance(transforms_dict, dict):
            raise ValueError("Transform file does not contain a valid transforms dictionary")
        return transforms_dict


class AnnualReservoirDataset(Dataset):
    """Dataset providing full 36-period annual sequences with constraint info."""

    def __init__(
        self,
        data_dir: str,
        years: List[int],
        sequence_length: int = 36,
        use_log_transform: bool = True,
        normalize: bool = True,
        fit_transforms: bool = True,
        precomputed_transforms: Optional[Dict[str, object]] = None,
    ) -> None:
        self.data_dir = data_dir
        self.years = years
        self.sequence_length = sequence_length
        self.use_log_transform = use_log_transform
        self.normalize = normalize
        self.fit_transforms = fit_transforms
        self.precomputed_transforms = precomputed_transforms

        self.feature_engineer = FeatureEngineer()
        self.data_transformer = DataTransformer()
        self.data_transformer.use_log_transform = self.use_log_transform
        self.data_transformer.normalize = self.normalize

        self.feature_sequences: List[np.ndarray] = []
        self.feature_sequences_original: List[np.ndarray] = []
        self.target_sequences: List[np.ndarray] = []
        self.head_inflows: List[np.ndarray] = []
        self.interval_inflows: List[np.ndarray] = []
        self.original_outflows: List[np.ndarray] = []
        self.q_in_sequences: List[np.ndarray] = []

        self.q_min: np.ndarray = np.zeros((self.sequence_length, 0), dtype=np.float32)
        self.q_max: np.ndarray = np.zeros((self.sequence_length, 0), dtype=np.float32)
        self.V0_vector: np.ndarray = np.zeros(0, dtype=np.float32)
        self.V_target_vector: np.ndarray = np.zeros(0, dtype=np.float32)

        self._load_and_preprocess()
        self._load_constraints()

    # ------------------------------------------------------------------
    def _load_and_preprocess(self) -> None:
        script_dir = os.path.dirname(os.path.abspath(__file__))

        inflow_all: List[np.ndarray] = []
        outflow_all: List[np.ndarray] = []
        feature_all: List[np.ndarray] = []
        feature_all_original: List[np.ndarray] = []
        year_lengths: List[int] = []

        head_sequences: List[np.ndarray] = []
        interval_sequences: List[np.ndarray] = []
        outflow_sequences: List[np.ndarray] = []

        from exceptions import DataFileError
        
        for year in self.years:
            path = os.path.join(script_dir, self.data_dir, f"{year}.csv")
            if not os.path.exists(path):
                raise DataFileError(
                    f"数据文件缺失: {path}\n"
                    f"请检查 {self.data_dir} 目录下是否存在 {year}.csv 文件。"
                )

            df = _sanitize_columns(pd.read_csv(path, encoding="utf-8-sig"))

            inflow_cols = [col for col in df.columns if "入库流量" in col]
            outflow_cols = [col for col in df.columns if "出库流量" in col]
            interval_cols = [col for col in df.columns if "_区间来水" in col]

            inflow = df[inflow_cols].to_numpy(dtype=np.float32)
            outflow = df[outflow_cols].to_numpy(dtype=np.float32)
            interval = df[interval_cols].to_numpy(dtype=np.float32)

            if len(inflow) != self.sequence_length:
                raise ValueError(f"Year {year} expected {self.sequence_length} periods, found {len(inflow)}")

            combined = np.hstack([inflow, interval]).astype(np.float32)
            features = self.feature_engineer.create_features(combined, df=df).astype(np.float32)

            inflow_all.append(combined)
            outflow_all.append(outflow)
            feature_all.append(features)
            feature_all_original.append(features.copy())
            year_lengths.append(len(inflow))
            head_sequences.append(inflow[:, 0].copy())
            interval_sequences.append(interval.copy())
            outflow_sequences.append(outflow.copy())

        inflow_all_arr = np.vstack(inflow_all)
        outflow_all_arr = np.vstack(outflow_all)
        feature_all_arr = np.vstack(feature_all)
        feature_all_original_arr = np.vstack(feature_all_original)

        if self.precomputed_transforms is not None:
            self.data_transformer.load_state(self.precomputed_transforms)
            _, outflow_norm, feature_norm = self.data_transformer.transform(
                inflow_all_arr, outflow_all_arr, feature_all_arr
            )
        else:
            if self.fit_transforms:
                _, outflow_norm, feature_norm = self.data_transformer.fit_transform(
                    inflow_all_arr, outflow_all_arr, feature_all_arr
                )
            else:
                outflow_norm, feature_norm = outflow_all_arr, feature_all_arr

        cursor = 0
        for idx, length in enumerate(year_lengths):
            next_cursor = cursor + length
            self.feature_sequences.append(feature_norm[cursor:next_cursor].astype(np.float32))
            self.feature_sequences_original.append(feature_all_original_arr[cursor:next_cursor].astype(np.float32))
            self.target_sequences.append(outflow_norm[cursor:next_cursor].astype(np.float32))
            self.head_inflows.append(head_sequences[idx].astype(np.float32))
            self.interval_inflows.append(interval_sequences[idx].astype(np.float32))
            self.original_outflows.append(outflow_sequences[idx].astype(np.float32))

            interval_seq = interval_sequences[idx]
            q_in_seq = np.zeros_like(outflow_sequences[idx], dtype=np.float32)
            q_in_seq[:, 0] = head_sequences[idx]
            num_interval_cols = interval_seq.shape[1] if interval_seq.ndim == 2 else 0
            # Conservative inflow prior: only exogenous inflows are exposed to the model.
            for reservoir_idx in range(1, outflow_sequences[idx].shape[1]):
                interval_vals = (
                    interval_seq[:, reservoir_idx - 1]
                    if num_interval_cols >= reservoir_idx
                    else 0.0
                )
                q_in_seq[:, reservoir_idx] = interval_vals
            self.q_in_sequences.append(q_in_seq.astype(np.float32))
            cursor = next_cursor

    # ------------------------------------------------------------------
    def _load_constraints(self) -> None:
        if not self.target_sequences:
            return

        script_dir = os.path.dirname(os.path.abspath(__file__))
        df = _load_constraints_table(script_dir)

        q_min_cols = [f"{name}最小出库流量" for name in RESERVOIR_NAMES]
        q_max_cols = [f"{name}最大出库流量" for name in RESERVOIR_NAMES]

        missing = [col for col in q_min_cols + q_max_cols if col not in df.columns]
        if missing:
            raise KeyError(f"约束文件缺少必需列：{missing}")
        
        self.q_min = df[q_min_cols].iloc[: self.sequence_length].to_numpy(dtype=np.float32)
        self.q_max = df[q_max_cols].iloc[: self.sequence_length].to_numpy(dtype=np.float32)
        self.q_min = np.minimum(self.q_min, self.q_max)
        
        # Populate storage vectors from config (required)
        from exceptions import ConfigurationError
        
        if get_config is None:
            raise ConfigurationError(
                "config_loader 模块未能加载。请确保 config.yaml 存在且格式正确。"
            )
        
        cfg_loader = get_config()
        V0_storage = cfg_loader.get("constraints.initial_storage", None)
        VT_storage = cfg_loader.get("constraints.target_storage", None)
        
        if V0_storage is None or VT_storage is None:
            raise ConfigurationError(
                "config.yaml 缺少必需字段: constraints.initial_storage 和/或 constraints.target_storage"
            )
        
        try:
            V0_arr = np.asarray(V0_storage, dtype=np.float32)
            VT_arr = np.asarray(VT_storage, dtype=np.float32)
        except (ValueError, TypeError) as e:
            raise ConfigurationError(
                f"配置中的 initial_storage/target_storage 格式错误，应为数值列表: {e}"
            )
        
        if V0_arr.size != len(RESERVOIR_NAMES) or VT_arr.size != len(RESERVOIR_NAMES):
            raise ConfigurationError(
                f"配置中的库容数组长度不匹配。期望 {len(RESERVOIR_NAMES)} 个水库，"
                f"但 initial_storage 有 {V0_arr.size} 个，target_storage 有 {VT_arr.size} 个。"
            )
        
        self.V0_vector = V0_arr.copy()
        self.V_target_vector = VT_arr.copy()

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.feature_sequences)


    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            "features": torch.tensor(self.feature_sequences[index], dtype=torch.float32),
            "features_original": torch.tensor(self.feature_sequences_original[index], dtype=torch.float32),
            "targets": torch.tensor(self.target_sequences[index], dtype=torch.float32),
            "head_inflow": torch.tensor(self.head_inflows[index], dtype=torch.float32),
            "interval_inflow": torch.tensor(self.interval_inflows[index], dtype=torch.float32),
            "original_outflow": torch.tensor(self.original_outflows[index], dtype=torch.float32),
            "q_min": torch.tensor(self.q_min, dtype=torch.float32),
            "q_max": torch.tensor(self.q_max, dtype=torch.float32),
            "q_in": torch.tensor(self.q_in_sequences[index], dtype=torch.float32),
            "V0": torch.tensor(self.V0_vector, dtype=torch.float32),
            "V_target": torch.tensor(self.V_target_vector, dtype=torch.float32),
            "year": torch.tensor(self.years[index], dtype=torch.long),
        }

    # Convenience -------------------------------------------------------
    def inverse_transform_outflow(self, outflow_normalized):
        np_input = (
            outflow_normalized.detach().cpu().numpy()
            if isinstance(outflow_normalized, torch.Tensor)
            else np.asarray(outflow_normalized)
        )
        if np_input.ndim == 3:
            batch, steps, reservoirs = np_input.shape
            flat = np_input.reshape(-1, reservoirs)
            restored = self.data_transformer.inverse_transform_outflow(flat)
            restored = np.nan_to_num(restored, nan=0.0, posinf=1e9, neginf=0.0)
            return restored.reshape(batch, steps, reservoirs)
        restored2d = self.data_transformer.inverse_transform_outflow(np_input)
        return np.nan_to_num(restored2d, nan=0.0, posinf=1e9, neginf=0.0)

    def export_transforms(self) -> Dict[str, object]:
        return self.data_transformer.export_transforms()

    def save_transforms(self, filepath: str) -> None:
        self.data_transformer.save_transforms(filepath)

    @classmethod
    def load_transforms(cls, filepath: str) -> Dict[str, object]:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Transform file not found: {filepath}")
        transforms_dict = joblib.load(filepath)
        if not isinstance(transforms_dict, dict):
            raise ValueError("Transform file does not contain a valid transforms dictionary")
        return transforms_dict




