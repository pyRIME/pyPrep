"""EER support: bitstream agreement with imagecodecs, fractionation, super-resolution,
gain references and a full synthetic EER tilt series through the pipeline."""

import numpy as np
import pytest
import tifffile

from pyprep.io.eer import EerMovie, split_groups
from pyprep.io.frames import open_movie
from pyprep.io.gain import prepare_gain
from pyprep.settings import InputSettings, ProcessingSettings
from pyprep.tiltseries import load_tilt_series

from eer_synth import encode_frame, random_frame, write_eer
from synthetic import specimen


def _movie(tmp_path, n=23, shape=(120, 200), rate=0.03, seed=0, **kw):
    rng = np.random.default_rng(seed)
    events = [random_frame(rng, shape, rate) for _ in range(n)]
    path = tmp_path / "m.eer"
    write_eer(path, [encode_frame(*e, shape, skipbits=kw.get("skipbits", 7)) for e in events], shape, **kw)
    return path, events


@pytest.mark.parametrize("compression,skipbits", [(65001, 7), (65002, 7), (65000, 8)])
def test_eer_fractions_are_exact_event_sums(tmp_path, compression, skipbits):
    path, events = _movie(tmp_path, compression=compression, skipbits=skipbits)
    m = EerMovie(path, fractions=5)
    assert m.n_raw_frames == 23 and m.n_frames == 5 and sum(m.frame_counts()) == 23
    frac = m.read()
    assert frac.dtype == np.uint16
    for k, (a, b) in enumerate(m.groups):
        expected = np.zeros(m.shape, np.uint16)
        for ys, xs, _, _ in events[a:b]:
            expected[ys, xs] += 1
        np.testing.assert_array_equal(frac[k], expected)


def test_eer_super_resolution_uses_subpixel_bits(tmp_path):
    path, events = _movie(tmp_path, n=4)
    m = EerMovie(path, fractions=1, upsampling=2)
    img = m.read()[0]
    expected = np.zeros((240, 400), np.uint16)
    for ys, xs, sy, sx in events:
        np.add.at(expected, (ys * 2 + sy // 2, xs * 2 + sx // 2), 1)
    np.testing.assert_array_equal(img, expected)


def test_split_groups():
    assert split_groups(23, fractions=5) == [(0, 5), (5, 9), (9, 14), (14, 18), (18, 23)]
    assert split_groups(103, group=25) == [(0, 25), (25, 50), (50, 75), (75, 103)]   # small tail folded in
    assert split_groups(120, group=50) == [(0, 50), (50, 120)]                        # 20 < 25: folded
    assert split_groups(130, group=50) == [(0, 50), (50, 100), (100, 130)]            # 30 kept separate
    assert split_groups(3, fractions=10) == [(0, 1), (1, 2), (2, 3)]


def test_open_movie_uses_input_settings(tmp_path):
    path, _ = _movie(tmp_path, n=40)
    m = open_movie(path, InputSettings(eer_group=8))
    assert m.is_eer and m.n_frames == 5 and m.frame_counts() == [8] * 5


def test_gain_divide_rotate_expand(tmp_path):
    g = np.arange(1, 13, dtype=np.float32).reshape(3, 4)
    g[0, 0] = 0                                       # defect
    p = tmp_path / "ref.gain"
    tifffile.imwrite(p, np.rot90(g, -1))              # stored rotated: needs +90 to match frames
    info = prepare_gain(p, (6, 8), rotate=90, upsampling=2, movie_is_eer=True)
    assert info.mode == "divide" and info.multiplier.shape == (6, 8)
    assert info.multiplier[0, 0] == 0 and info.multiplier[5, 7] == pytest.approx(1 / 12)
    with pytest.raises(ValueError, match="transposed"):
        prepare_gain(p, (6, 8), upsampling=2)


def test_doses_for_eer_fractions(tmp_path):
    from pyprep.tiltseries import Tilt
    t = Tilt(zvalue=0, angle=0, section=None, exposure_dose=3.0, frame_doses=[0.01] * 300)
    np.testing.assert_allclose(t.doses_for(3, [100, 100, 100]), [1.0, 1.0, 1.0])      # from mdoc frames
    t2 = Tilt(zvalue=0, angle=0, section=None, exposure_dose=3.0)
    np.testing.assert_allclose(t2.doses_for(2, [100, 200]), [1.0, 2.0])             # proportional split


def _eer_tilt_series(tmp_path, drift, n_eer=48, shape=(192, 256), rate=0.25, gain=None, seed=1):
    """Three-tilt EER series of a drifting specimen; returns the mdoc path."""
    rng = np.random.default_rng(seed)
    H, W = shape
    spec = specimen(H * 4, W * 4, rng, contrast=0.5)
    sections = []
    for z, angle in enumerate((0.0, 3.0, -3.0)):
        frames = []
        for i in range(n_eer):
            shift = (drift[0] * i / n_eer, drift[1] * i / n_eer)
            e = random_frame(rng, shape, rate, specimen=spec, shift=shift, up=4, detector_gain=gain)
            frames.append(encode_frame(*e, shape))
        name = f"TSE_{z + 1:03d}_{angle:.2f}_20260101_000000_EER.eer"
        write_eer(tmp_path / name, frames, shape)
        sections.append(f"[ZValue = {z}]\nTiltAngle = {angle}\nExposureDose = 3.0\nPriorRecordDose = {3.0 * z}\n"
                        f"RotationAngle = 86.2\nSubFramePath = X:\\frames\\{name}\n")
    mdoc = tmp_path / "TSE.mdoc"
    mdoc.write_text("PixelSpacing = 2.0\nVoltage = 300\n\n[T = Tomography: test]\n\n" + "\n".join(sections))
    return mdoc


def test_eer_tilt_series_through_pipeline(tmp_path):
    """Drift inside each EER exposure is recovered from the fractions; gain is divided out."""
    from pyprep.pipeline import process_series

    yy, xx = np.mgrid[0:192, 0:256]
    gain = (1 + 0.3 * np.sin(2 * np.pi * xx / 90) * np.cos(2 * np.pi * yy / 70)).astype(np.float32)
    tifffile.imwrite(tmp_path / "ref.gain", gain)
    drift = (6.0, -4.0)                                    # physical pixels over the exposure
    mdoc = _eer_tilt_series(tmp_path, drift, gain=gain)
    ts = load_tilt_series(mdoc)
    assert len(ts.usable) == 3
    s = ProcessingSettings()
    s.recon.enabled = False
    s.output.bin_levels = [1]
    s.motion.align_bin = 2
    # As stored, so the gain pattern is compared in its own orientation (Y inversion: test_handedness).
    s.input = InputSettings(eer_fractions=6, gain_path=str(tmp_path / "ref.gain"), invert_y="no")
    rec = process_series(ts, s, tmp_path / "out")
    assert rec["status"] == "complete"
    for t in rec["tilts"]:
        sh = np.asarray(t["shifts_px"])
        assert sh.shape == (6, 2)
        # The sampler shows the specimen displaced by -shift (same convention as the K3
        # test movies), so aligning needs +shift.  Fraction centres span 5/6 of the exposure.
        total = sh[-1] - sh[0]
        expected = np.array(drift) * (5 / 6)
        np.testing.assert_allclose(total, expected, atol=1.0)
    from pyprep.io import mrc
    h = mrc.read_header(tmp_path / "out" / "TSE" / "TSE.mrc")
    assert (h.nx, h.ny, h.nz) == (256, 192, 3) and h.pixel_size == pytest.approx(2.0)

    # Gain is really divided out: the sum no longer carries the +/-30 % detector pattern.
    def gain_corr(out_root):
        img = mrc.read_sections(out_root / "TSE" / "TSE.mrc", 0, 1)[0][8:-8, 8:-8]
        return np.corrcoef(img.ravel(), gain[8:-8, 8:-8].ravel())[0, 1]
    assert abs(gain_corr(tmp_path / "out")) < 0.1
    s_nogain = ProcessingSettings.from_dict(s.to_dict())
    s_nogain.input.gain_path = ""
    process_series(ts, s_nogain, tmp_path / "out_nogain")
    assert gain_corr(tmp_path / "out_nogain") > 0.2          # measured ~0.29 vs ~0.0 corrected

    # 8K rendering: same physical output size and pixel, shifts still in physical pixels
    s.input.eer_upsampling = 2
    rec2 = process_series(ts, s, tmp_path / "out8k")
    h2 = mrc.read_header(tmp_path / "out8k" / "TSE" / "TSE.mrc")
    assert (h2.nx, h2.ny) == (256, 192) and h2.pixel_size == pytest.approx(2.0)
    t0 = np.asarray(rec2["tilts"][0]["shifts_px"])
    np.testing.assert_allclose(t0[-1] - t0[0], np.array(drift) * (5 / 6), atol=1.0)
