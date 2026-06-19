"""Regression tests for the rolling-map leading-edge traversability artifact and its fix.

Mechanism (verified against recorded rosbags): on a rolling robot-centered map the
visibility-cleanup ray loop stamps ``is_upper_bound=1`` and a stepped ``upper_bound`` height
on freshly revealed frontier cells. The edge-detecting traversability filter renders that
height step as a near-zero (obstacle) traversability and writes it to layer 3. The legacy
publish mask kept any cell where ``is_valid OR is_upper_bound``, so the frontier obstacle was
published as a false non-traversable line that scrolled inward and never cleared.

Fix: ``ElevationMap.get_traversability`` masks by ``is_valid`` only when
``traversability_mask_use_upper_bound`` is False (the new default), turning ray-stamped
upper-bound-only cells into unknown (NaN) instead of a finite obstacle.

The first test runs anywhere (numpy + the deployed weights) and guards the mechanism: the real
trained filter must convert a height step into a low-traversability value. The second test needs
cupy and guards the actual publish-mask selection in ``get_traversability``.
"""

import pickle
from pathlib import Path

import numpy as np
import pytest


def _find_weights():
    for parent in Path(__file__).resolve().parents:
        cand = parent / "config" / "core" / "weights.dat"
        if cand.exists():
            return cand
    raise FileNotFoundError("config/core/weights.dat not found above test file")


def _load_weights():
    with open(_find_weights(), "rb") as f:
        w = pickle.load(f)
    return w["conv1.weight"], w["conv2.weight"], w["conv3.weight"], w["conv_final.weight"]


def _dilated_conv2d(image, weights, dilation):
    """Mirror of TraversabilityFilterCupy._dilated_conv2d in numpy."""
    height, width = image.shape
    out_channels = weights.shape[0]
    out_h, out_w = height - 2 * dilation, width - 2 * dilation
    output = np.zeros((out_channels, out_h, out_w), dtype=np.float32)
    for row in range(3):
        rs = row * dilation
        for col in range(3):
            cs = col * dilation
            window = image[rs : rs + out_h, cs : cs + out_w]
            output += window[None] * weights[:, 0, row, col].reshape(out_channels, 1, 1)
    return output


def _traversability(elevation, w):
    """Mirror of TraversabilityFilterCupy.__call__ in numpy."""
    w1, w2, w3, w_out = w
    out1 = _dilated_conv2d(elevation, w1, 1)[:, 2:-2, 2:-2]
    out2 = _dilated_conv2d(elevation, w2, 2)[:, 1:-1, 1:-1]
    out3 = _dilated_conv2d(elevation, w3, 3)
    feats = np.concatenate((out1, out2, out3), axis=0)
    cost = np.sum(np.abs(feats) * w_out[0], axis=0)  # w_out[0]: (12,1,1) broadcasts over (12,H,W)
    return np.exp(-cost)


def _min_trav_for_step(w, step_m, n=40):
    field = np.zeros((n, n), dtype=np.float32)
    field[:, n // 2 :] = step_m
    return float(_traversability(field, w).min())


def test_real_filter_renders_height_step_as_obstacle():
    """The deployed trained weights turn a frontier height step into a low (obstacle) value."""
    w = _load_weights()

    flat = np.zeros((40, 40), dtype=np.float32)
    assert float(_traversability(flat, w).min()) > 0.95, "flat ground must stay traversable (~1)"

    # A frontier step (as ray-stamping creates between measured ground and ray-sampled height)
    # must read as an obstacle, and larger steps must read as stronger obstacles.
    m_flat = _min_trav_for_step(w, 0.0)
    m_10cm = _min_trav_for_step(w, 0.10)
    m_30cm = _min_trav_for_step(w, 0.30)
    assert m_10cm < 0.5, f"0.1 m step should read as obstacle, got {m_10cm:.3f}"
    assert m_30cm < 0.1, f"0.3 m step should read as a hard obstacle, got {m_30cm:.3f}"
    assert m_30cm < m_10cm < m_flat, "larger steps must produce lower traversability"


def _tiny_param(cp_required=True):
    from elevation_mapping_cupy.parameter import Parameter

    param = Parameter()
    param.use_chainer = False
    param.resolution = 0.1
    param.map_length = 2.0  # -> small map, fast kernel compile
    param.update()
    return param


def test_get_traversability_excludes_upper_bound_only_by_default():
    """get_traversability publishes ray-stamped (is_upper_bound-only) cells as NaN by default,
    and as a finite value only when traversability_mask_use_upper_bound is True (legacy).
    """
    pytest.importorskip("cupy")
    import cupy as cp

    from elevation_mapping_cupy.elevation_mapping import ElevationMap

    param = _tiny_param()
    em = ElevationMap(param)

    # Reset relevant layers, then plant two obstacle cells well inside the published interior.
    em.elevation_map[2] *= 0.0  # is_valid
    em.elevation_map[6] *= 0.0  # is_upper_bound
    em.elevation_map[3] *= 0.0  # traversability (0 == obstacle)

    # Plant exactly two obstacle (traversability==0) cells well inside the published interior:
    # one directly measured (is_valid), one ray-stamped only (is_upper_bound). Everything else
    # stays masked (NaN), so we can assert on the count of finite published cells.
    interior = em.cell_n // 2
    measured = (interior, interior)  # is_valid=1  -> a real obstacle
    ray_only = (interior, interior + 3)  # is_upper_bound=1 only -> ray-stamped artifact
    em.elevation_map[2][measured] = 1.0
    em.elevation_map[6][ray_only] = 1.0

    param.traversability_mask_use_upper_bound = False
    t_new = cp.asnumpy(em.get_traversability())
    finite_new = np.isfinite(t_new)
    assert finite_new.sum() == 1, "only the measured cell should be published (ray-stamp-only -> NaN)"
    assert t_new[finite_new][0] == 0.0, "measured obstacle must be preserved as obstacle"

    param.traversability_mask_use_upper_bound = True
    t_legacy = cp.asnumpy(em.get_traversability())
    assert np.isfinite(t_legacy).sum() == 2, "legacy mask must additionally publish the upper-bound-only cell"
