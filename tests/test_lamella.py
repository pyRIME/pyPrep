"""The lamella locator (redeposition removal finds the interior with it) on synthetic thick volumes."""

import numpy as np
import pytest
from scipy import ndimage

from pyprep import lamella

APIX = 10.48


def _background(rng, shape):
    """White noise plus smooth, low-frequency noise everywhere, as in a reconstruction (its dark tail is the
    reference the locator compares runs against)."""
    vol = 0.3 * rng.standard_normal(shape).astype(np.float32)
    bg = ndimage.gaussian_filter(rng.standard_normal(shape).astype(np.float32), 8.0)
    return vol + bg * (0.6 / bg.std())


def _volume(rng, slab=(60, 110), ny=96, nz=160, nx=96, offset_deg=0.0):
    """Background noise, plus band-pass texture (the 'cells') in a slab tilted about Y by ``offset_deg``."""
    vol = _background(rng, (ny, nz, nx))
    tex = ndimage.gaussian_filter(rng.standard_normal((ny, nz, nx)).astype(np.float32), 2.0)
    tex *= 0.6 / tex.std()
    zz = np.arange(nz)[None, :, None]
    xx = np.arange(nx)[None, None, :]
    z0 = slab[0] + np.tan(np.radians(offset_deg)) * (xx - nx / 2)
    vol += np.where((zz >= z0) & (zz < z0 + slab[1] - slab[0]), tex, 0)
    return vol


def test_find_interior_locates_a_textured_slab():
    vol = _volume(np.random.default_rng(0))
    r = lamella.find_interior(vol, APIX, down=2, tile_nm=40.0)
    assert 40 <= r["thickness_vox_median"] <= 60                       # 50 voxels
    centre = r["z_shift_vox"] + (vol.shape[1] - 1) / 2
    assert abs(centre - 85) < 6
    assert r["global_round"] == 1


def test_find_interior_measures_the_tilt_about_y():
    vol = _volume(np.random.default_rng(1), offset_deg=8.0)
    r = lamella.find_interior(vol, APIX, down=2, tile_nm=40.0)
    assert abs(r["angle_offset_deg"] - 8.0) < 2.5


def test_find_interior_refuses_a_field_without_texture():
    vol = 0.3 * np.random.default_rng(2).standard_normal((96, 160, 96)).astype(np.float32)
    with pytest.raises(RuntimeError):
        lamella.find_interior(vol, APIX, down=2, tile_nm=40.0)


@pytest.mark.xfail(strict=True, reason="known limitation: in smooth noise without cells, round 4 (the blob-texture "
                                       "last resort) can accept a faint band, as it accepts faint field-filling cells")
def test_find_interior_refuses_smooth_noise_without_cells():
    vol = _background(np.random.default_rng(2), (96, 160, 96))
    with pytest.raises(RuntimeError):
        lamella.find_interior(vol, APIX, down=2, tile_nm=40.0)
