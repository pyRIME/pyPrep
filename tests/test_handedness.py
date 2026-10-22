"""Handedness: image orientation (Tomo5 Y inversion), tilt axis direction from the CTF, stack mdoc title."""

import numpy as np
import pytest

from pyprep.pipeline import handedness_decision, normalize_angle, set_stack_mdoc_axis


def test_normalize_angle():
    assert normalize_angle(86.2 + 180) == pytest.approx(-93.8)
    assert normalize_angle(-93.8 + 180) == pytest.approx(86.2)
    assert normalize_angle(180) == pytest.approx(-180)
    assert normalize_angle(-190) == pytest.approx(170)


@pytest.mark.parametrize("mode,ctf,flip", [
    ("keep", {"handedness": -1, "handedness_confidence": 1.0, "handedness_votes": 26}, False),
    ("flip", None, True),
    ("flip", {"handedness": 1, "handedness_confidence": 1.0, "handedness_votes": 26}, True),
    ("auto", None, False),                                                            # nothing to decide on
    ("auto", {"handedness": 1, "handedness_confidence": 1.0, "handedness_votes": 26}, False),
    ("auto", {"handedness": -1, "handedness_confidence": 1.0, "handedness_votes": 26}, True),
    ("auto", {"handedness": -1, "handedness_confidence": 0.6, "handedness_votes": 26}, False),   # unclear
    ("auto", {"handedness": -1, "handedness_confidence": 1.0, "handedness_votes": 2}, False),    # too few tilts
    ("auto", {"handedness": -1, "handedness_confidence": 0.96}, True),                # older record, no votes
])
def test_handedness_decision(mode, ctf, flip):
    got, reason = handedness_decision(mode, ctf)
    assert got is flip
    assert reason


def test_set_stack_mdoc_axis(tmp_path):
    f = tmp_path / "s.mrc.mdoc"
    f.write_text("PixelSpacing = 13.2\n\n[T = pyPrep 0.1: motion-corrected tilt series]\n"
                 "[T =     Tilt axis angle = 86.20, binning = 4  spot = 6  camera = 0]\n\n"
                 "[ZValue = 0]\nTiltAngle = -44.11\n")
    assert set_stack_mdoc_axis(f, -93.8)
    text = f.read_text()
    assert "Tilt axis angle = -93.80, binning = 4" in text and "86.20" not in text
    assert "TiltAngle = -44.11" in text
    assert not set_stack_mdoc_axis(tmp_path / "missing.mdoc", 0.0)


# ------------------------------------------------------------------ image orientation (Y inversion)
@pytest.mark.parametrize("title,mode,inv", [
    ("Tomography_5.27.0.12951: TFPC01803B7EFFA", "auto", True),     # Tomo5 stores images upside down
    ("SerialEM: Digitized on EMBL Krios", "auto", False),
    ("Tomography_5.27.0.12951: TFPC01803B7EFFA", "no", False),
    ("SerialEM: Digitized on EMBL Krios", "yes", True),
])
def test_image_frame(title, mode, inv):
    from pathlib import Path

    from pyprep.io.mdoc import parse_mdoc
    from pyprep.pipeline import image_frame
    from pyprep.settings import ProcessingSettings
    from pyprep.tiltseries import TiltSeries

    doc = parse_mdoc(f"PixelSpacing = 2.62\n[T = {title}]\n[ZValue = 0]\nTiltAngle = -8.02\nRotationAngle = -266.73\n")
    ts = TiltSeries(name="P", mdoc_path=Path("P.mdoc"), mdoc=doc, tilts=[], pixel_size=2.62, voltage=300,
                    tilt_axis=-266.73)
    s = ProcessingSettings()
    s.input.invert_y = mode
    f = image_frame(ts, s)
    assert f["invert_y"] is inv and f["reason"]
    assert f["recorded_tilt_axis"] == pytest.approx(-266.73)
    # IMOD measures the axis from Y, so a Y flip maps a to 180 - a (C. difficile series: -266.73 -> 86.73,
    # the same axis as Tomo5's TiltAxisAngle 176.73 measured from X)
    assert f["tilt_axis"] == pytest.approx(86.73 if inv else -266.73)


def _tomo5_series(tmp_path, ny=96, nx=128):
    """Three-tilt Tomo5-style series (Tomography title, RotationAngle 86.2) with synthetic MRC fractions."""
    from synthetic import make_movie

    from pyprep.io import mrc

    sections = []
    for z, angle in enumerate((-2.0, 0.0, 2.0)):
        frames, _ = make_movie(ny=ny, nx=nx, n_frames=4, seed=z)
        name = f"TS5_{z + 1:03d}_{angle:.2f}_20260101_000000_fractions.mrc"
        mrc.write_mrc(tmp_path / name, frames)
        sections.append(f"[ZValue = {z}]\nTiltAngle = {angle}\nExposureDose = 3.0\nPriorRecordDose = {3.0 * z}\n"
                        f"RotationAngle = 86.2\nSubFramePath = X:\\frames\\{name}\nNumSubFrames = 4\n")
    mdoc = tmp_path / "TS5.mdoc"
    mdoc.write_text("PixelSpacing = 2.0\nVoltage = 300\n\n[T = Tomography_5.27.0.12951: test]\n\n"
                    "[T =   TiltAxisAngle = 176.20  Binning = 1  SpotSize = 4]\n\n" + "\n".join(sections))
    return mdoc


def test_tomo5_images_inverted_in_y(tmp_path):
    """Tomo5 series: the aligned images are written with their rows reversed and the stack mdoc carries the
    mirrored axis (180 - a); 'no' writes them as stored. Nothing else changes."""
    from pyprep.io import mrc
    from pyprep.io.mdoc import read_mdoc
    from pyprep.pipeline import process_series
    from pyprep.settings import ProcessingSettings
    from pyprep.tiltseries import load_tilt_series

    ts = load_tilt_series(_tomo5_series(tmp_path))
    got = {}
    for mode in ("auto", "no"):
        s = ProcessingSettings()
        s.use_gpu = False
        s.recon.enabled = False
        s.ctf.enabled = False
        s.output.bin_levels = [1]
        s.motion.align_bin = 2
        s.input.invert_y = mode
        rec = process_series(ts, s, tmp_path / mode)
        assert rec["status"] == "complete"
        stack = tmp_path / mode / "TS5" / "TS5.mrc"
        got[mode] = (rec, mrc.read_mrc(stack), read_mdoc(stack.with_name(stack.name + ".mdoc")).tilt_axis_angle)
    (rec_a, a, axis_a), (rec_n, n, axis_n) = got["auto"], got["no"]
    assert rec_a["image_frame"]["invert_y"] and not rec_n["image_frame"]["invert_y"]
    np.testing.assert_array_equal(a, n[:, ::-1, :])
    assert axis_a == pytest.approx(93.8) and axis_n == pytest.approx(86.2)
    assert rec_a["tilts"][0]["shifts_px"] == rec_n["tilts"][0]["shifts_px"]      # motion in raw-frame coordinates


def test_tomo5_records_from_before_y_inversion_are_stale(tmp_path):
    """Tomo5 stacks written as stored (records without 'image_frame') are rebuilt, and so is their tomogram;
    SerialEM records without the field stay complete."""
    import json

    from pyprep.pipeline import is_complete, recon_complete, result_json_path
    from pyprep.settings import ProcessingSettings
    from pyprep.tiltseries import load_tilt_series

    s = ProcessingSettings()
    s.output.bin_levels = [1]
    s.recon.enabled = False
    for kind, title in (("tomo5", "Tomography_5.27.0.12951: test"), ("serialem", "SerialEM: test")):
        d = tmp_path / kind
        d.mkdir()
        text = _tomo5_series(d).read_text().replace("Tomography_5.27.0.12951: test", title)
        (d / "TS5.mdoc").write_text(text)
        ts = load_tilt_series(d / "TS5.mdoc")
        out = d / "out"
        series_dir = out / "TS5"
        series_dir.mkdir(parents=True)
        (series_dir / "TS5.mrc").write_bytes(b"x")
        result_json_path(ts, series_dir).write_text(json.dumps({"status": "complete", "used_tilts": [0, 1, 2]}))
        assert is_complete(ts, s, out) is (kind == "serialem")
        rec_dir = series_dir / "imod_bin4"
        rec_dir.mkdir()
        (rec_dir / "TS5_rec.mrc").write_bytes(b"x")
        (rec_dir / "pyprep_recon.json").write_text(json.dumps(
            {"status": "complete", "preset": "patch", "tomogram": str(rec_dir / "TS5_rec.mrc"),
             "used_tilts": [0, 1, 2]}))
        # stacks rebuilt in the new orientation, tomogram from before: reconstruct again (Tomo5 only)
        result_json_path(ts, series_dir).write_text(json.dumps(
            {"status": "complete", "used_tilts": [0, 1, 2],
             "image_frame": {"invert_y": kind == "tomo5", "tilt_axis": 93.8 if kind == "tomo5" else 86.2,
                             "recorded_tilt_axis": 86.2}}))
        s.recon.enabled = True
        assert recon_complete(ts, s, out) is (kind == "serialem")
        s.recon.enabled = False
