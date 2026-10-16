"""Regression checks for the native triangular contour stage."""
from __future__ import annotations

import numpy as np
import pytest
import rasterio
import tifffile

from bora.tri_mesh import (LIBRARY, TriMeshConfig, refine_tri_mesh,
                           refine_tri_mesh_streaming, scale_for_tile_size)


def test_mesh_scale_tracks_tile_size():
    base = TriMeshConfig(mesh_spacing=64, support_radius=80, band_radius=216)
    assert scale_for_tile_size(base, 56) == TriMeshConfig(16, 24, 56)
    assert scale_for_tile_size(base, 90) == TriMeshConfig(24, 32, 90)
    assert scale_for_tile_size(base, 216) == base


@pytest.mark.skipif(not LIBRARY.is_file(), reason="C++ tri-mesh library not built")
def test_mesh_reduces_staircase_without_changing_background_or_far_cores():
    yy, xx = np.mgrid[:256, :256]
    border = 64 + 16 * (yy // 32)
    labels = np.where(xx < border, 1, 2).astype(np.uint16)
    labels[:20] = 0
    original = labels.copy()
    refined, changed = refine_tri_mesh(
        labels, config=TriMeshConfig(mesh_spacing=32, support_radius=40,
                                     band_radius=50, kernel_threads=2))
    assert changed > 0
    assert np.array_equal(refined == 0, original == 0)
    assert refined[100, 10] == 1
    assert refined[100, 240] == 2
    original_edges = np.count_nonzero(original[:, 1:] != original[:, :-1])
    refined_edges = np.count_nonzero(refined[:, 1:] != refined[:, :-1])
    assert refined_edges <= original_edges


@pytest.mark.skipif(not LIBRARY.is_file(), reason="C++ tri-mesh library not built")
def test_mesh_color_guidance_and_class_retention():
    labels = np.ones((128, 128), np.uint16)
    labels[:, 64:] = 2
    rgb = np.zeros((128, 128, 3), np.uint8)
    rgb[:, :70] = [70, 40, 150]
    rgb[:, 70:] = [210, 160, 140]
    refined, _ = refine_tri_mesh(
        labels, rgb, TriMeshConfig(mesh_spacing=16, support_radius=24,
                                   band_radius=40, color_weight=0.3,
                                   kernel_threads=2))
    assert set(np.unique(refined)) == {1, 2}
    assert refined[64, 8] == 1
    assert refined[64, 120] == 2


@pytest.mark.skipif(not LIBRARY.is_file(), reason="C++ tri-mesh library not built")
def test_post_smoothing_rounds_mesh_result_and_keeps_background():
    yy, xx = np.mgrid[:256, :256]
    labels = np.where(xx < 80 + 24*(yy//48), 1, 2).astype(np.uint16)
    labels[:12] = 0
    mesh, _ = refine_tri_mesh(labels, config=TriMeshConfig(
        48, 64, 80, 0, 2, 0))
    rounded, _ = refine_tri_mesh(labels, config=TriMeshConfig(
        48, 64, 80, 0, 2, 12))
    assert np.any(mesh != rounded)
    assert np.array_equal(rounded == 0, labels == 0)


@pytest.mark.skipif(not LIBRARY.is_file(), reason="C++ tri-mesh library not built")
def test_streamed_blocks_match_whole_array(tmp_path):
    yy, xx = np.mgrid[:320, :320]
    labels = np.where(xx < 72 + 16*(yy//40), 1, 2).astype(np.uint16)
    source = tmp_path / "source.tif"
    output = tmp_path / "output.tif"
    with rasterio.open(source, "w", driver="GTiff", width=320, height=320,
                       count=1, dtype="uint16") as dst:
        dst.write(labels, 1)
    config = TriMeshConfig(mesh_spacing=32, support_radius=40,
                           band_radius=50, kernel_threads=2)
    direct, _ = refine_tri_mesh(labels, config=config)
    report = refine_tri_mesh_streaming(source, output, config=config,
                                       block_size=128, workers=2)
    actual = tifffile.imread(output)
    assert report["blocks"] == 9
    assert np.array_equal(actual, direct)
