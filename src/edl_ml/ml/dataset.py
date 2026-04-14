"""Torch dataset helpers for the capacitance surrogate."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset

from edl_ml.data.features import FEATURE_COLUMNS, TARGET_COLUMN

INPUT_COLUMNS: tuple[str, ...] = FEATURE_COLUMNS + ("electrode_potential_v",)
"""Columns fed to the MLP (features + electrode potential)."""


@dataclass(slots=True)
class StandardScalerTensor:
    """Simple mean/std standardiser stored as tensors.

    Fitting is performed on the training set so no test information leaks
    into the normalisation statistics.
    """

    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def fit(cls, array: NDArray[np.float64]) -> StandardScalerTensor:
        """Fit mean/std on the given 2D array and return a scaler."""
        mean = torch.as_tensor(array.mean(axis=0), dtype=torch.float32)
        std = torch.as_tensor(array.std(axis=0), dtype=torch.float32)
        std = torch.where(std > 1e-8, std, torch.ones_like(std))
        return cls(mean=mean, std=std)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Scale ``x`` using the fitted mean and std."""
        return (x - self.mean) / self.std

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Invert the transformation."""
        return x * self.std + self.mean


class CapacitanceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """PyTorch dataset wrapping a long-format capacitance DataFrame.

    Parameters
    ----------
    df
        DataFrame produced by
        :func:`edl_ml.data.generate.build_capacitance_dataset`.
    x_scaler
        Fitted scaler for the input features. Required.
    y_scaler
        Fitted scaler for the target. Required.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        x_scaler: StandardScalerTensor,
        y_scaler: StandardScalerTensor,
    ) -> None:
        self._x = torch.as_tensor(
            df[list(INPUT_COLUMNS)].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )
        self._y = torch.as_tensor(
            df[[TARGET_COLUMN]].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )
        self._x = x_scaler.transform(self._x)
        self._y = y_scaler.transform(self._y)

    def __len__(self) -> int:
        return int(self._x.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._x[idx], self._y[idx]


@dataclass(frozen=True, slots=True)
class LoaderBundle:
    """Container for the three dataloaders and the fitted scalers.

    Attributes
    ----------
    train, val, test
        DataLoaders for each split.
    x_scaler, y_scaler
        Scalers fitted on the training split only.
    """

    train: DataLoader[tuple[torch.Tensor, torch.Tensor]]
    val: DataLoader[tuple[torch.Tensor, torch.Tensor]]
    test: DataLoader[tuple[torch.Tensor, torch.Tensor]]
    x_scaler: StandardScalerTensor
    y_scaler: StandardScalerTensor


def build_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int = 256,
    num_workers: int = 0,
    seed: int | None = 0,
) -> LoaderBundle:
    """Fit scalers on ``train_df`` and wrap each split in a DataLoader.

    Parameters
    ----------
    train_df, val_df, test_df
        Splits returned by
        :func:`edl_ml.data.generate.split_by_sample`.
    batch_size
        Mini-batch size.
    num_workers
        DataLoader workers. Keep at 0 on macOS MPS.
    seed
        Seed for the training shuffle generator.
    """
    x_scaler = StandardScalerTensor.fit(train_df[list(INPUT_COLUMNS)].to_numpy(dtype=np.float32))
    y_scaler = StandardScalerTensor.fit(train_df[[TARGET_COLUMN]].to_numpy(dtype=np.float32))
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)

    def _loader(
        df: pd.DataFrame, *, shuffle: bool
    ) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        ds = CapacitanceDataset(df, x_scaler=x_scaler, y_scaler=y_scaler)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            generator=gen if shuffle else None,
            drop_last=False,
        )

    return LoaderBundle(
        train=_loader(train_df, shuffle=True),
        val=_loader(val_df, shuffle=False),
        test=_loader(test_df, shuffle=False),
        x_scaler=x_scaler,
        y_scaler=y_scaler,
    )
