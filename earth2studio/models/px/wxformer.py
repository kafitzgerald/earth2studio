# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CREDIT WxFormer prognostic model wrappers for Earth2Studio.

WxFormer is a CrossFormer-based global weather model developed by NSF NCAR MILES.
It operates on ERA5 hybrid model levels and predicts upper-air and surface variables.

References
----------
- Schreck et al. (2024) https://arxiv.org/abs/2411.07814
- https://github.com/NCAR/miles-credit
"""

from __future__ import annotations

import warnings
from collections import OrderedDict
from collections.abc import Generator, Iterator
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from loguru import logger

from earth2studio.data import DataSource, ForecastSource, fetch_data
from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.px.base import PrognosticModel
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.utils import handshake_coords, handshake_dim, handshake_size
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.type import CoordSystem

try:
    from credit.models.crossformer import CrossFormer
except ImportError:
    OptionalDependencyFailure("wxformer")
    CrossFormer = None

# ── Variable definitions ─────────────────────────────────────────────────────

# ERA5 hybrid model levels used by WxFormer (16 of the 137 ERA5 levels)
HYBRID_LEVELS = [10, 30, 40, 50, 60, 70, 80, 90, 95, 100, 105, 110, 120, 130, 136, 137]

# State variables: upper-air (U, V, T, Q × 16 levels) + surface (7 vars) = 71 total.
# Naming convention: {var}{level}hl where 'hl' denotes ERA5 hybrid level.
VARIABLES: list[str] = []
for _v in ["u", "v", "t", "q"]:
    for _lev in HYBRID_LEVELS:
        VARIABLES.append(f"{_v}{_lev}hl")
VARIABLES += ["sp", "t2m", "v500", "u500", "t500", "z500", "q500"]

# CREDIT internal names used in the normalization NetCDF files
_CREDIT_UPPER = ["U", "V", "T", "Q"]
_CREDIT_SURFACE = ["SP", "t2m", "V500", "U500", "T500", "Z500", "Q500"]

# Indices of Q hybrid-level and Q500 channels in VARIABLES (clamped to ≥ 1e-8 by 1H)
_Q_HL_INDICES = list(range(3 * len(HYBRID_LEVELS), 4 * len(HYBRID_LEVELS)))  # q channels
_Q500_INDEX = VARIABLES.index("q500")
_Q_INDICES = _Q_HL_INDICES + [_Q500_INDEX]

# ── Shared CrossFormer architecture ──────────────────────────────────────────

_ARCH_COMMON = dict(
    image_height=640,
    image_width=1280,
    patch_height=1,
    patch_width=1,
    frames=1,
    channels=4,
    surface_channels=7,
    input_only_channels=3,
    output_only_channels=0,
    levels=16,
    dim=(128, 256, 512, 1024),
    depth=(2, 2, 8, 2),
    global_window_size=(10, 5, 2, 1),
    local_window_size=10,
    cross_embed_kernel_sizes=((4, 8, 16, 32), (2, 4), (2, 4), (2, 4)),
    cross_embed_strides=(2, 2, 2, 2),
    attn_dropout=0.0,
    ff_dropout=0.0,
    post_conf={"activate": False},  # tracer-fixing is applied in __call__ instead
)

_ARCH_6H = dict(
    **_ARCH_COMMON,
    interp=True,
    padding_conf={"activate": True, "mode": "mirror", "pad_lon": 80, "pad_lat": 80},
)

_ARCH_1H = dict(
    **_ARCH_COMMON,
    interp=False,
    padding_conf={"activate": True, "mode": "earth", "pad_lon": 80, "pad_lat": 80},
)

# ── Coordinate helpers ────────────────────────────────────────────────────────

_LAT = np.linspace(90, -90, 640, endpoint=False)
_LON = np.linspace(0, 360, 1280, endpoint=False)


# ── Normalization helpers ─────────────────────────────────────────────────────


def _load_norm_tensors(
    mean_path: str | Path, std_path: str | Path
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    """Build [71]-length mean/std tensors matching VARIABLES ordering.

    Returns state mean, state std (shape [71]), plus scalar tsi_mean and tsi_std.
    """
    mean_ds = xr.open_dataset(mean_path)
    std_ds = xr.open_dataset(std_path)

    means, stds = [], []
    for var in _CREDIT_UPPER:
        vals_m = mean_ds[var].values  # shape (16,)
        vals_s = std_ds[var].values
        means.extend(vals_m.tolist())
        stds.extend(vals_s.tolist())
    for var in _CREDIT_SURFACE:
        means.append(float(mean_ds[var].values))
        stds.append(float(std_ds[var].values))

    tsi_mean = float(mean_ds["tsi"].values)
    tsi_std = float(std_ds["tsi"].values)

    return (
        torch.tensor(means, dtype=torch.float32),
        torch.tensor(stds, dtype=torch.float32),
        tsi_mean,
        tsi_std,
    )


def _load_static(static_path: str | Path) -> torch.Tensor:
    """Load pre-normalized static fields (Z_GDS4_SFC, LSM) from NetCDF.

    Returns tensor of shape [2, 640, 1280].
    """
    ds = xr.open_dataset(static_path)
    z = torch.tensor(ds["Z_GDS4_SFC"].values, dtype=torch.float32)
    lsm = torch.tensor(ds["LSM"].values, dtype=torch.float32)
    return torch.stack([z, lsm], dim=0)  # [2, 640, 1280]


def _load_checkpoint(ckpt_path: str | Path, model: torch.nn.Module) -> None:
    """Load a CREDIT checkpoint into *model*, handling both full and model-only formats."""
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    msg = model.load_state_dict(state_dict, strict=False)
    if msg.missing_keys:
        logger.debug(f"WxFormer checkpoint: {len(msg.missing_keys)} missing keys (expected)")
    if msg.unexpected_keys:
        logger.warning(f"WxFormer checkpoint: unexpected keys: {msg.unexpected_keys}")


# ── Base class ────────────────────────────────────────────────────────────────


class _WxFormerBase(torch.nn.Module, AutoModelMixin, PrognosticMixin):
    """Shared implementation for WxFormer1H and WxFormer6H.

    Parameters
    ----------
    core_model : CrossFormer
        Instantiated CrossFormer backbone.
    mean : torch.Tensor
        Per-variable normalization means, shape [71].
    std : torch.Tensor
        Per-variable normalization standard deviations, shape [71].
    tsi_mean : float
        Mean for top solar irradiance normalization.
    tsi_std : float
        Standard deviation for top solar irradiance normalization.
    static : torch.Tensor
        Pre-normalized static fields (Z_GDS4_SFC, LSM), shape [2, 640, 1280].
    tsi_data_source : DataSource | ForecastSource | None, optional
        Data source providing top solar irradiance (``tsi`` variable) on the
        0.25-degree lat/lon grid. Required for inference. Can be set after
        construction via ``model.tsi_data_source = source``.
    tsi_variable : str, optional
        Variable name to request from ``tsi_data_source``, by default ``"tsi"``.
    fix_tracers : bool, optional
        If True, clamp Q hybrid-level and Q500 outputs to ≥ 1e-8 after
        denormalization. Enabled by default for WxFormer1H, disabled for 6H.
    """

    # Subclasses must define DT
    DT: np.timedelta64

    def __init__(
        self,
        core_model: torch.nn.Module,
        mean: torch.Tensor,
        std: torch.Tensor,
        tsi_mean: float,
        tsi_std: float,
        static: torch.Tensor,
        tsi_data_source: DataSource | ForecastSource | None = None,
        tsi_variable: str = "tsi",
        fix_tracers: bool = False,
    ) -> None:
        super().__init__()
        self.model = core_model
        self.register_buffer("mean", mean.reshape(1, 1, 1, -1, 1, 1))
        self.register_buffer("std", std.reshape(1, 1, 1, -1, 1, 1))
        self.register_buffer("tsi_mean", torch.tensor(tsi_mean))
        self.register_buffer("tsi_std", torch.tensor(tsi_std))
        self.register_buffer("static", static)  # [2, H, W]
        self.tsi_data_source = tsi_data_source
        self.tsi_variable = tsi_variable
        self.fix_tracers = fix_tracers

        if tsi_data_source is None:
            warnings.warn(
                "No TSI data source was provided to WxFormer. "
                "Set model.tsi_data_source before running inference.",
                stacklevel=2,
            )

    # ── Coordinate system ─────────────────────────────────────────────────────

    def input_coords(self) -> CoordSystem:
        """Input coordinate system.

        Returns
        -------
        CoordSystem
            Ordered dict with keys: batch, time, lead_time, variable, lat, lon.
        """
        return OrderedDict(
            {
                "batch": np.empty(0),
                "time": np.empty(0),
                "lead_time": np.array([np.timedelta64(0, "h")]),
                "variable": np.array(VARIABLES),
                "lat": _LAT,
                "lon": _LON,
            }
        )

    @batch_coords()
    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        """Output coordinate system.

        Parameters
        ----------
        input_coords : CoordSystem
            Input coordinate system.

        Returns
        -------
        CoordSystem
            Coordinate system with lead_time advanced by one model time step.
        """
        target = self.input_coords()
        handshake_dim(input_coords, "lead_time", 2)
        handshake_dim(input_coords, "variable", 3)
        handshake_dim(input_coords, "lat", 4)
        handshake_dim(input_coords, "lon", 5)
        handshake_coords(input_coords, target, "variable")
        handshake_coords(input_coords, target, "lat")
        handshake_coords(input_coords, target, "lon")

        out = input_coords.copy()
        out["lead_time"] = input_coords["lead_time"] + self.DT
        return out

    # ── Forward ───────────────────────────────────────────────────────────────

    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Single time-step forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input state tensor of shape [batch, time, lead_time, variable, lat, lon].
        coords : CoordSystem
            Coordinate system corresponding to *x*.

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Predicted state and updated coordinate system.
        """
        if self.tsi_data_source is None:
            raise RuntimeError(
                "WxFormer requires a TSI data source. "
                "Set model.tsi_data_source before calling."
            )

        B, T, lt, V, H, W = x.shape
        handshake_dim(coords, "lead_time", 2)
        handshake_size(coords, "lead_time", 1)
        handshake_dim(coords, "variable", 3)
        handshake_coords(coords, self.input_coords(), "variable")
        handshake_dim(coords, "lat", 4)
        handshake_dim(coords, "lon", 5)

        output_coords = self.output_coords(coords)

        # Normalize state: [B, T, 1, 71, H, W]
        x_norm = (x - self.mean) / self.std

        # ── Fetch and normalize TSI ────────────────────────────────────────────
        tsi_data, _ = fetch_data(
            self.tsi_data_source,
            time=coords["time"],
            variable=np.array([self.tsi_variable]),
            lead_time=coords["lead_time"],
            device=x.device,
        )
        # tsi_data shape after fetch_data: [time, lead_time, variable, lat, lon]
        # Normalize and reshape to [B, T, 1, 1, H, W]
        tsi_norm = (tsi_data - self.tsi_mean) / self.tsi_std
        tsi_norm = tsi_norm.unsqueeze(0).expand(B, -1, -1, -1, -1, -1)

        # ── Build CrossFormer input ────────────────────────────────────────────
        # Static: [2, H, W] -> [B, T, 1, 2, H, W]
        static = (
            self.static.unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(B, T, 1, -1, -1, -1)
        )

        # Channel order (static_first=True): [upper_air(64), surface(7), static(2), tsi(1)]
        x_in = torch.cat([x_norm, static, tsi_norm], dim=3)  # [B, T, 1, 74, H, W]

        # Reshape to [B*T, 74, 1, H, W] for CrossFormer (frames=1 mode)
        x_in = x_in.reshape(B * T, 74, 1, H, W)

        x_out = self.model(x_in)  # [B*T, 71, 1, H, W]

        # Reshape back: [B, T, 1, 71, H, W]
        x_out = x_out.reshape(B, T, 1, V, H, W)

        # Denormalize
        x_out = x_out * self.std + self.mean

        # Tracer fixing: clamp Q to physical minimum (WxFormer1H only)
        if self.fix_tracers:
            x_out[..., _Q_INDICES, :, :] = x_out[..., _Q_INDICES, :, :].clamp(
                min=1e-8
            )

        return x_out, output_coords

    # ── Iterator ──────────────────────────────────────────────────────────────

    @batch_func()
    def _default_generator(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> Generator[tuple[torch.Tensor, CoordSystem], None, None]:
        coords = coords.copy()
        self.output_coords(coords)
        yield x, coords

        if self.tsi_data_source is None:
            raise ValueError(
                "A TSI data source must be set before using WxFormer as an iterator."
            )

        while True:
            x, coords = self.front_hook(x, coords)
            x, coords = self.__call__(x, coords)
            x, coords = self.rear_hook(x, coords)
            yield x, coords

    def create_iterator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        """Create a time-stepping iterator.

        Parameters
        ----------
        x : torch.Tensor
            Initial condition tensor.
        coords : CoordSystem
            Coordinate system of *x*.

        Yields
        ------
        tuple[torch.Tensor, CoordSystem]
            Predicted state and coordinate system at each lead time.
        """
        yield from self._default_generator(x, coords)

    # ── Package loading (shared) ──────────────────────────────────────────────

    @classmethod
    @check_optional_dependencies()
    def _load_from_package(
        cls,
        package: Package,
        arch: dict,
        mean_filename: str,
        std_filename: str,
        fix_tracers: bool,
        tsi_data_source: DataSource | ForecastSource | None,
        tsi_variable: str,
    ) -> PrognosticModel:
        """Shared load logic for both model variants."""
        ckpt_path = package.resolve("finetune_final/model_checkpoint.pt")
        mean_path = package.resolve(f"finetune_final/{mean_filename}")
        std_path = package.resolve(f"finetune_final/{std_filename}")

        mean, std, tsi_mean, tsi_std = _load_norm_tensors(mean_path, std_path)

        # Static fields (Z_GDS4_SFC and LSM).
        # Requires static_norm.nc to be present in the HF package.
        # To prepare this file, see: earth2studio.models.px.wxformer._prepare_static
        static_path = Path(package.resolve("static_norm.nc"))
        if not static_path.exists():
            raise FileNotFoundError(
                f"static_norm.nc not found in package at {static_path}. "
                "This file containing Z_GDS4_SFC and LSM on the 0.25° grid is "
                "required. Contact the package maintainer or use "
                "earth2studio.models.px.wxformer.prepare_static_file() to create it."
            )
        static = _load_static(static_path)

        core = CrossFormer(**arch)
        core.eval()
        _load_checkpoint(ckpt_path, core)

        return cls(
            core_model=core,
            mean=mean,
            std=std,
            tsi_mean=tsi_mean,
            tsi_std=tsi_std,
            static=static,
            tsi_data_source=tsi_data_source,
            tsi_variable=tsi_variable,
            fix_tracers=fix_tracers,
        )


# ── Concrete model classes ────────────────────────────────────────────────────


class WxFormer6H(_WxFormerBase):
    """CREDIT WxFormer 6-hour global weather forecasting model.

    CrossFormer-based model trained on ERA5 hybrid model-level data at 0.25°
    horizontal resolution. Predicts upper-air (U, V, T, Q on 16 hybrid levels)
    and surface (SP, T2m, V/U/T/Z/Q at 500 hPa) variables with a 6-hour time step.

    Note
    ----
    Requires a top-solar-irradiance data source. Set ``model.tsi_data_source``
    before running inference. The variable name requested from the source is
    controlled by ``model.tsi_variable`` (default: ``"tsi"``).

    Parameters
    ----------
    core_model : CrossFormer
        Instantiated CrossFormer backbone.
    mean : torch.Tensor
        Per-variable normalization means, shape [71].
    std : torch.Tensor
        Per-variable normalization standard deviations, shape [71].
    tsi_mean : float
        Mean for top solar irradiance normalization.
    tsi_std : float
        Standard deviation for top solar irradiance normalization.
    static : torch.Tensor
        Pre-normalized static fields (Z_GDS4_SFC, LSM), shape [2, 640, 1280].
    tsi_data_source : DataSource | ForecastSource | None, optional
        Data source for top solar irradiance, by default None.
    tsi_variable : str, optional
        Variable name to request from *tsi_data_source*, by default ``"tsi"``.

    Badges
    ------
    domain:global res:0.25deg step:6h lev:model_level product:atmos year:2024
    """

    DT = np.timedelta64(6, "h")

    @classmethod
    def load_default_package(cls) -> Package:
        """Default pre-trained WxFormer 6-hour model package.

        Returns
        -------
        Package
            Package pointing to ``djgagne2/wxformer_6h`` on Hugging Face Hub.
        """
        return Package(
            "hf://djgagne2/wxformer_6h",
            cache_options={
                "cache_storage": Package.default_cache("wxformer6h"),
                "same_names": True,
            },
        )

    @classmethod
    @check_optional_dependencies()
    def load_model(
        cls,
        package: Package,
        tsi_data_source: DataSource | ForecastSource | None = None,
        tsi_variable: str = "tsi",
    ) -> PrognosticModel:
        """Load WxFormer 6-hour model from a package.

        Parameters
        ----------
        package : Package
            Model package, typically from :meth:`load_default_package`.
        tsi_data_source : DataSource | ForecastSource | None, optional
            Data source providing top solar irradiance on the 0.25° grid,
            by default None (must be set before inference).
        tsi_variable : str, optional
            Variable name to request from *tsi_data_source*, by default ``"tsi"``.

        Returns
        -------
        PrognosticModel
            Loaded WxFormer6H model.
        """
        return cls._load_from_package(
            package=package,
            arch=_ARCH_6H,
            mean_filename="mean_6h_1979_2018_16lev_0.25deg.nc",
            std_filename="std_6h_1979_2018_16lev_0.25deg.nc",
            fix_tracers=False,
            tsi_data_source=tsi_data_source,
            tsi_variable=tsi_variable,
        )


class WxFormer1H(_WxFormerBase):
    """CREDIT WxFormer 1-hour global weather forecasting model.

    CrossFormer-based model trained on ERA5 hybrid model-level data at 0.25°
    horizontal resolution. Predicts upper-air (U, V, T, Q on 16 hybrid levels)
    and surface (SP, T2m, V/U/T/Z/Q at 500 hPa) variables with a 1-hour time step.
    Specific humidity (Q) outputs are clamped to ≥ 1e-8 for physical consistency.

    Note
    ----
    Requires a top-solar-irradiance data source. Set ``model.tsi_data_source``
    before running inference. The variable name requested from the source is
    controlled by ``model.tsi_variable`` (default: ``"tsi"``).

    Parameters
    ----------
    core_model : CrossFormer
        Instantiated CrossFormer backbone.
    mean : torch.Tensor
        Per-variable normalization means, shape [71].
    std : torch.Tensor
        Per-variable normalization standard deviations, shape [71].
    tsi_mean : float
        Mean for top solar irradiance normalization.
    tsi_std : float
        Standard deviation for top solar irradiance normalization.
    static : torch.Tensor
        Pre-normalized static fields (Z_GDS4_SFC, LSM), shape [2, 640, 1280].
    tsi_data_source : DataSource | ForecastSource | None, optional
        Data source for top solar irradiance, by default None.
    tsi_variable : str, optional
        Variable name to request from *tsi_data_source*, by default ``"tsi"``.

    Badges
    ------
    domain:global res:0.25deg step:1h lev:model_level product:atmos year:2024
    """

    DT = np.timedelta64(1, "h")

    @classmethod
    def load_default_package(cls) -> Package:
        """Default pre-trained WxFormer 1-hour model package.

        Returns
        -------
        Package
            Package pointing to ``djgagne2/wxformer_1h`` on Hugging Face Hub.
        """
        return Package(
            "hf://djgagne2/wxformer_1h",
            cache_options={
                "cache_storage": Package.default_cache("wxformer1h"),
                "same_names": True,
            },
        )

    @classmethod
    @check_optional_dependencies()
    def load_model(
        cls,
        package: Package,
        tsi_data_source: DataSource | ForecastSource | None = None,
        tsi_variable: str = "tsi",
    ) -> PrognosticModel:
        """Load WxFormer 1-hour model from a package.

        Parameters
        ----------
        package : Package
            Model package, typically from :meth:`load_default_package`.
        tsi_data_source : DataSource | ForecastSource | None, optional
            Data source providing top solar irradiance on the 0.25° grid,
            by default None (must be set before inference).
        tsi_variable : str, optional
            Variable name to request from *tsi_data_source*, by default ``"tsi"``.

        Returns
        -------
        PrognosticModel
            Loaded WxFormer1H model.
        """
        return cls._load_from_package(
            package=package,
            arch=_ARCH_1H,
            mean_filename="mean_1h_1979_2018_16lev_0.25deg.nc",
            std_filename="std_1h_1979_2018_16lev_0.25deg.nc",
            fix_tracers=True,
            tsi_data_source=tsi_data_source,
            tsi_variable=tsi_variable,
        )
