"""Tilt QC, stale-output detection after exclusions, presets, TIFF export, disk estimate."""

import json
from pathlib import Path

import numpy as np
import pytest

from pyprep import qc
from pyprep.io import mrc
from pyprep.settings import ProcessingSettings

ANGLES = np.arange(-60, 61, 3.0)


def _intensity(offset=5.0, thickness=0.35, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    return 100 * np.exp(-thickness / np.cos(np.radians(ANGLES - offset))) * (1 + noise * rng.standard_normal(len(ANGLES)))


def test_clean_series_has_no_flags_and_finds_specimen_tilt():
    flags, fit = qc.flag_tilts(ANGLES, _intensity())
    assert not any(f.flags for f in flags)
    assert fit.offset == pytest.approx(5.0, abs=2)


@pytest.mark.parametrize("factor,flag", [(0.6, "dark"), (1.6, "bright")])
def test_dark_and_bright_tilts_are_flagged(factor, flag):
    y = _intensity()
    y[17] *= factor
    flags, _ = qc.flag_tilts(ANGLES, y)
    assert flags[17].flags == [flag]
    assert sum(bool(f.flags) for f in flags) == 1
    assert flags[17].excludable == (flag == "dark")


def test_drift_score_and_convergence_flags():
    n = len(ANGLES)
    drift = np.full(n, 3.0)
    drift[4] = 60.0
    scores = np.full(n, 0.2)
    scores[9] = 0.05
    conv = [True] * n
    conv[11] = False
    flags, _ = qc.flag_tilts(ANGLES, drift=drift, scores=scores, converged=conv)
    assert flags[4].flags == ["drift"] and flags[9].flags == ["low score"] and flags[11].flags == ["not converged"]


def test_saturation_value():
    assert qc.saturation_value(np.uint8) == 255
    assert qc.saturation_value(np.uint8, mode=101) == 15
    assert qc.saturation_value(np.uint16) is None


def test_exclusion_makes_stacks_and_tomogram_stale(tmp_path):
    from pyprep.pipeline import is_complete, recon_complete, result_json_path
    from pyprep.tiltseries import load_tilt_series
    from test_core import TOMO5_MDOC

    (tmp_path / "TS_1.mdoc").write_text(TOMO5_MDOC)
    for name in ["TS_1_001_-0.10_20250113_171527_fractions.mrc", "TS_1_002_1.90_20250113_171617_fractions.mrc",
                 "TS_1_003_-2.10_20250113_171715_fractions.mrc"]:
        mrc.write_mrc(tmp_path / name, np.zeros((4, 48, 64), np.uint8))
    ts = load_tilt_series(tmp_path / "TS_1.mdoc")
    s = ProcessingSettings()
    s.output.bin_levels = [1]
    out = tmp_path / "out"
    series_dir = out / "TS_1"
    series_dir.mkdir(parents=True)
    (series_dir / "TS_1.mrc").write_bytes(b"x")
    frame = {"invert_y": True, "tilt_axis": 93.8, "recorded_tilt_axis": 86.2, "reason": "Tomo5"}
    result_json_path(ts, series_dir).write_text(json.dumps({"status": "complete", "used_tilts": [0, 1, 2],
                                                            "image_frame": frame}))
    recon_dir = series_dir / "imod_bin4"
    recon_dir.mkdir()
    (recon_dir / "TS_1_rec.mrc").write_bytes(b"x")
    (recon_dir / "pyprep_recon.json").write_text(json.dumps(
        {"status": "complete", "preset": "patch", "tomogram": str(recon_dir / "TS_1_rec.mrc"), "used_tilts": [0, 1, 2],
         "invert_y": True}))
    s.recon.enabled = False
    assert is_complete(ts, s, out)
    s.recon.enabled = True
    assert recon_complete(ts, s, out)
    ts.tilts[1].excluded = True
    s.recon.enabled = False
    assert not is_complete(ts, s, out)
    s.recon.enabled = True
    assert not recon_complete(ts, s, out)


def test_presets_roundtrip(tmp_path, monkeypatch):
    from pyprep import settings as st
    monkeypatch.setenv("APPDATA", str(tmp_path))
    names = st.list_presets()                         # built-ins created on first use
    assert any("Tomo5" in n for n in names) and any("EER" in n for n in names)
    s = ProcessingSettings()
    s.output.dtype = "int16"
    st.save_preset("Krios K3/Tomo5", s)               # slash is sanitised
    loaded = [n for n in st.list_presets() if n.startswith("Krios")]
    assert len(loaded) == 1 and st.load_preset(loaded[0]).output.dtype == "int16"
    st.delete_preset(loaded[0])
    assert not [n for n in st.list_presets() if n.startswith("Krios")]


def test_export_tiff_scale_and_range(tmp_path):
    import tifffile
    from pyprep.export import export_tiff
    vol = np.random.default_rng(0).normal(size=(5, 40, 60)).astype(np.float32)
    mrc.write_mrc(tmp_path / "v.mrc", vol, pixel_size=13.2)
    out = export_tiff(tmp_path / "v.mrc", tmp_path / "v.tif", bin_factor=2)
    with tifffile.TiffFile(out) as t:
        data = t.asarray()
        assert t.is_imagej and data.shape == (5, 20, 30) and data.dtype == np.uint8
        assert t.imagej_metadata["unit"] == "nm" and t.imagej_metadata["spacing"] == pytest.approx(2.64, rel=1e-3)
        assert data.min() == 0 and data.max() == 255


def test_estimate_output_bytes(tmp_path):
    from pyprep.pipeline import estimate_output_bytes
    from pyprep.tiltseries import load_tilt_series
    from test_core import TOMO5_MDOC
    (tmp_path / "TS_1.mdoc").write_text(TOMO5_MDOC)
    mrc.write_mrc(tmp_path / "TS_1_001_-0.10_20250113_171527_fractions.mrc", np.zeros((4, 48, 64), np.uint8))
    ts = load_tilt_series(tmp_path / "TS_1.mdoc")
    s = ProcessingSettings()
    s.recon.enabled = False
    s.output.bin_levels = [1, 4]
    assert estimate_output_bytes(ts, s) == (64 * 48 + 16 * 12) * 1 * 4       # one usable tilt, float32


def test_thumbnails_gallery_and_selection(tmp_path):
    import zlib
    from pyprep.thumbs import gallery_entries, load_selection, save_selection, tomogram_thumbnail, write_png_gray
    # PNG writer produces a valid, decodable image
    img = (np.arange(12 * 20) % 256).astype(np.uint8).reshape(12, 20)
    write_png_gray(tmp_path / "a.png", img)
    raw = (tmp_path / "a.png").read_bytes()
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    idat = raw[raw.index(b"IDAT") + 4: raw.index(b"IEND") - 8]
    rows = zlib.decompress(idat)
    assert rows == b"".join(b"\x00" + img[y].tobytes() for y in range(12))
    # tomogram thumbnail = mean of the central slices
    vol = np.zeros((30, 40, 50), np.float32)
    vol[10:20, 5:15, 5:15] = 1.0                      # feature only in the central slices
    mrc.write_mrc(tmp_path / "rec.mrc", vol)
    tomogram_thumbnail(tmp_path / "rec.mrc", tmp_path / "t.png", n_slices=10)
    assert (tmp_path / "t.png").stat().st_size > 50
    # gallery over an output folder with one series that has a tomogram
    d = tmp_path / "out" / "TS_1"
    (d / "imod_bin4").mkdir(parents=True)
    mrc.write_mrc(d / "imod_bin4" / "TS_1_rec.mrc", vol)
    (d / "TS_1_pyprep.json").write_text(json.dumps({"series": "TS_1", "status": "complete", "tilts": [],
                                                     "reconstruction": {"status": "complete",
                                                                        "tomogram": str(d / "imod_bin4" / "TS_1_rec.mrc")}}))
    entries = gallery_entries(tmp_path / "out")
    assert len(entries) == 1 and entries[0]["source"] == "tomogram" and Path(entries[0]["thumbnail"]).exists()
    save_selection(tmp_path / "out", {"TS_1"})
    assert load_selection(tmp_path / "out") == {"TS_1"}
