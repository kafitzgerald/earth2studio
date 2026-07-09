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

from collections import OrderedDict
from collections.abc import Iterable

import numpy as np
import pytest
import torch

from earth2studio.data import Random, fetch_data
from earth2studio.models.px import WxFormer1H, WxFormer6H
from earth2studio.models.px.wxformer import HYBRID_LEVELS, VARIABLES, _Q_INDICES
from earth2studio.utils import handshake_dim

# ── Constants ─────────────────────────────────────────────────────────────────

N_VARS = len(VARIABLES)  # 71
H, W = 640, 1280
TSI_VARIABLE = "tsi"


# ── Test doubles ──────────────────────────────────────────────────────────────


class _FakeCrossFormer(torch.nn.Module):
    """Returns zeros with the correct state-variable shape."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 74, 1, H, W]  →  out: [B, 71, 1, H, W]
        B = x.shape[0]
        return torch.zeros(B, N_VARS, 1, x.shape[-2], x.shape[-1], device=x.device)


class _FakeTsiSource:
    """Returns a ones tensor as TSI on the WxFormer grid."""

    def __call__(
        self,
        time: np.ndarray,
        variable: np.ndarray,
        lead_time: np.ndarray,
    ) -> tuple[np.ndarray, OrderedDict]:
        T, lt = len(time), len(lead_time)
        data = np.ones((T, lt, 1, H, W), dtype=np.float32)
        coords = OrderedDict(
            {
                "time": time,
                "lead_time": lead_time,
                "variable": variable,
                "lat": np.linspace(90, -90, H, endpoint=False),
                "lon": np.linspace(0, 360, W, endpoint=False),
            }
        )
        return data, coords


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_model(cls, fix_tracers: bool = False):
    """Build a WxFormer wrapper around a fake CrossFormer."""
    mean = torch.zeros(N_VARS)
    std = torch.ones(N_VARS)
    static = torch.zeros(2, H, W)
    return cls(
        core_model=_FakeCrossFormer(),
        mean=mean,
        std=std,
        tsi_mean=0.0,
        tsi_std=1.0,
        static=static,
        tsi_data_source=_FakeTsiSource(),
        tsi_variable=TSI_VARIABLE,
        fix_tracers=fix_tracers,
    )


@pytest.fixture
def model6h() -> WxFormer6H:
    return _make_model(WxFormer6H)


@pytest.fixture
def model1h() -> WxFormer1H:
    return _make_model(WxFormer1H, fix_tracers=True)


def _input(
    model,
    time: np.ndarray,
    device: str = "cpu",
) -> tuple[torch.Tensor, OrderedDict]:
    ic = model.input_coords()
    del ic["batch"], ic["time"], ic["lead_time"], ic["variable"]
    ds = Random(ic)
    lead_time = model.input_coords()["lead_time"]
    variable = model.input_coords()["variable"]
    return fetch_data(ds, time, variable, lead_time, device=device)


# ── Variable list tests ───────────────────────────────────────────────────────


def test_variables_count():
    assert N_VARS == 4 * len(HYBRID_LEVELS) + 7


def test_variables_hybrid_level_naming():
    assert "u10hl" in VARIABLES
    assert "q137hl" in VARIABLES
    assert "sp" in VARIABLES
    assert "q500" in VARIABLES


def test_q_indices_correct():
    assert all(VARIABLES[i].startswith("q") and VARIABLES[i].endswith("hl") for i in _Q_INDICES[:-1])
    assert VARIABLES[_Q_INDICES[-1]] == "q500"


# ── Coordinate tests ──────────────────────────────────────────────────────────


def test_wxformer6h_input_coords(model6h: WxFormer6H):
    ic = model6h.input_coords()
    assert list(ic.keys()) == ["batch", "time", "lead_time", "variable", "lat", "lon"]
    assert ic["lead_time"][0] == np.timedelta64(0, "h")
    assert len(ic["variable"]) == N_VARS
    assert ic["lat"][0] == pytest.approx(90.0, abs=1.0)
    assert ic["lat"][-1] == pytest.approx(-90.0, abs=1.0)


def test_wxformer6h_output_coords(model6h: WxFormer6H):
    time = np.array([np.datetime64("2020-01-01T00:00")])
    x, coords = _input(model6h, time)
    oc = model6h.output_coords(coords)
    assert oc["lead_time"][0] == np.timedelta64(6, "h")
    np.testing.assert_array_equal(oc["variable"], np.array(VARIABLES))


def test_wxformer1h_output_coords(model1h: WxFormer1H):
    time = np.array([np.datetime64("2020-01-01T00:00")])
    x, coords = _input(model1h, time)
    oc = model1h.output_coords(coords)
    assert oc["lead_time"][0] == np.timedelta64(1, "h")


# ── Forward pass tests ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "time",
    [
        np.array([np.datetime64("2020-01-01T00:00")]),
        np.array([np.datetime64("2020-01-01T00:00"), np.datetime64("2020-01-02T00:00")]),
    ],
)
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU"),
        ),
    ],
)
def test_wxformer6h_call(model6h: WxFormer6H, time: np.ndarray, device: str):
    model6h = model6h.to(device)
    x, coords = _input(model6h, time, device=device)

    out, out_coords = model6h(x, coords)

    assert out.shape == torch.Size([len(time), 1, N_VARS, H, W])
    assert torch.isfinite(out).all()
    assert out_coords["lead_time"][0] == np.timedelta64(6, "h")
    handshake_dim(out_coords, "time", 0)
    handshake_dim(out_coords, "lead_time", 1)
    handshake_dim(out_coords, "variable", 2)
    handshake_dim(out_coords, "lat", 3)
    handshake_dim(out_coords, "lon", 4)


def test_wxformer1h_call(model1h: WxFormer1H):
    time = np.array([np.datetime64("2020-01-01T00:00")])
    x, coords = _input(model1h, time)

    out, out_coords = model1h(x, coords)

    assert out.shape == torch.Size([1, 1, N_VARS, H, W])
    assert out_coords["lead_time"][0] == np.timedelta64(1, "h")


# ── Tracer fixing ─────────────────────────────────────────────────────────────


def test_fix_tracers_clamps_q(model1h: WxFormer1H):
    """WxFormer1H should clamp Q outputs to >= 1e-8."""
    # Patch the core to return large negative values for Q channels
    class _NegQModel(torch.nn.Module):
        def forward(self, x):
            out = torch.zeros(x.shape[0], N_VARS, 1, x.shape[-2], x.shape[-1])
            out[:, _Q_INDICES] = -1.0
            return out

    model1h.model = _NegQModel()
    time = np.array([np.datetime64("2020-01-01T00:00")])
    x, coords = _input(model1h, time)
    out, _ = model1h(x, coords)
    assert (out[..., _Q_INDICES, :, :] >= 1e-8).all()


def test_fix_tracers_disabled_6h(model6h: WxFormer6H):
    """WxFormer6H should NOT clamp Q outputs."""

    class _NegQModel(torch.nn.Module):
        def forward(self, x):
            out = torch.zeros(x.shape[0], N_VARS, 1, x.shape[-2], x.shape[-1])
            out[:, _Q_INDICES] = -1.0
            return out

    model6h.model = _NegQModel()
    time = np.array([np.datetime64("2020-01-01T00:00")])
    x, coords = _input(model6h, time)
    out, _ = model6h(x, coords)
    # No clamping applied — negative Q values remain
    assert (out[..., _Q_INDICES, :, :] < 0).any()


# ── Iterator tests ────────────────────────────────────────────────────────────


def test_wxformer6h_iterator(model6h: WxFormer6H):
    time = np.array([np.datetime64("2020-01-01T00:00")])
    x, coords = _input(model6h, time)

    it = model6h.create_iterator(x, coords)
    assert isinstance(it, Iterable)

    initial, initial_coords = next(it)
    assert initial_coords["lead_time"][0] == np.timedelta64(0, "h")

    for i, (out, out_coords) in enumerate(it):
        assert out.shape == torch.Size([1, 1, N_VARS, H, W])
        assert out_coords["lead_time"][0] == np.timedelta64(6 * (i + 1), "h")
        if i >= 3:
            break


# ── Input validation (exception) tests ───────────────────────────────────────


@pytest.mark.parametrize(
    "coords_update",
    [
        {"variable": np.array(["bad_var", *VARIABLES[1:]])},
        {"lat": np.linspace(-90, 90, H)},
        {"lon": np.linspace(0, 360, W + 1, endpoint=False)},
    ],
)
def test_wxformer6h_bad_coords_raises(model6h: WxFormer6H, coords_update: dict):
    time = np.array([np.datetime64("2020-01-01T00:00")])
    x, coords = _input(model6h, time)
    coords.update(coords_update)

    with pytest.raises((KeyError, ValueError)):
        model6h(x, coords)


def test_no_tsi_source_raises():
    """Calling without a TSI data source should raise RuntimeError."""
    model = _make_model(WxFormer6H)
    model.tsi_data_source = None
    time = np.array([np.datetime64("2020-01-01T00:00")])
    x, coords = _input(model, time)

    with pytest.raises(RuntimeError, match="TSI data source"):
        model(x, coords)


# ── Package loading (slow, requires network) ──────────────────────────────────


@pytest.mark.slow
@pytest.mark.parametrize("cls", [WxFormer6H, WxFormer1H])
def test_wxformer_package_loads(cls):
    pkg = cls.load_default_package()
    model = cls.load_model(pkg)
    assert isinstance(model, cls)
