"""Tomogram positioning: tilt.com editing and the geometry of the correction."""

import math
import sys
from pathlib import PureWindowsPath

import numpy as np
import pytest

from pyprep import imod, positioning as P
from pyprep.io import mrc

needs_imod = pytest.mark.skipif(sys.platform != "win32" or imod.imod_dir() is None,
                                reason="needs IMOD for Windows")

TILT_COM = """# Command file to run Tilt
#
####CreatedVersion####4.11.25
$setenv IMOD_OUTPUT_FORMAT MRC
$tilt -StandardInput
FakeSIRTiterations\t6
InputProjections TS_ali.mrc
OutputFile\tTS_full_rec.mrc
IMAGEBINNED\t1
TILTFILE TS.tlt
THICKNESS\t152
RADIAL .35 .035
XAXISTILT\t-0.06
PERPENDICULAR
MODE 2
FULLIMAGE\t1023 1440
SUBSETSTART\t0 0
AdjustOrigin 1
$if (-e ./savework) ./savework
"""


def test_tiltcom_roundtrip(tmp_path):
    f = tmp_path / "tilt.com"
    f.write_text(TILT_COM)
    tc = P.TiltCom.read(f)
    assert tc.before[-1] == "$tilt -StandardInput"
    assert tc.after == ["$if (-e ./savework) ./savework"]
    assert tc.get("thickness") == "152"
    assert tc.get("PERPENDICULAR") == ""
    assert tc.floats("FULLIMAGE", 2) == [1023.0, 1440.0]
    assert tc.floats("OFFSET", 2) == [0.0, 0.0]            # missing -> zeros
    tc.set("THICKNESS", 200)
    tc.set("SHIFT", "0.0 -5.0")
    tc.remove("FakeSIRTiterations")
    tc.write()
    again = P.TiltCom.read(f)
    assert again.get("THICKNESS") == "200"
    assert again.floats("SHIFT", 2) == [0.0, -5.0]
    assert again.get("FakeSIRTiterations") is None
    assert [k for k, _ in again.params][-1] == "SHIFT"       # new keys go at the end, before $if
    assert f.read_text().rstrip().endswith("$if (-e ./savework) ./savework")


def test_tiltcom_needs_tilt_command(tmp_path):
    f = tmp_path / "tilt.com"
    f.write_text("# nothing here\n")
    with pytest.raises(ValueError):
        P.TiltCom.read(f)


def test_correction_flat_centred():
    # Trial (Z, Y, X) = (100, 80, 200); specimen already flat and centred, 20 trial px thick.
    c = P.correction(((0, 60), (200, 60)), ((0, 40), (200, 40)),
                     ((0, 60), (80, 60)), ((0, 40), (80, 40)), (100, 80, 200), scale=2)
    assert c.offset_add == pytest.approx(0)
    assert c.xtilt_add == pytest.approx(0)
    assert c.shift_add == pytest.approx(0)
    assert c.thickness_px == 40                               # reconstruction px = trial px x scale


def test_correction_tilted_and_raised():
    nz, ny, nx, scale = 100, 80, 200, 2
    a, b = 10.0, -6.0
    ta, tb = math.tan(math.radians(a)), math.tan(math.radians(b))
    zc = nz / 2 + 12                                          # mid-plane 12 trial px above the centre

    def xz(z):
        return ((0, z - ta * nx / 2), (nx, z + ta * nx / 2))

    def yz(z):
        return ((0, z - tb * ny / 2), (ny, z + tb * ny / 2))

    c = P.correction(xz(zc + 15), xz(zc - 15), yz(zc + 15), yz(zc - 15), (nz, ny, nx), scale)
    ca, cb = math.cos(math.radians(a)), math.cos(math.radians(b))
    assert c.offset_add == pytest.approx(a)
    assert c.xtilt_add == pytest.approx(b)
    assert c.shift_add == pytest.approx(-12 * ca * cb * scale)
    assert c.thickness_px == round(30 * ca * cb * scale)


def test_auto_boundaries_finds_band():
    import numpy as np
    v = np.full((120, 50), 1.0, np.float32)
    v[40:70] = 5.0
    lo, hi = P.auto_boundaries(v)
    assert 36 <= lo <= 42 and 67 <= hi <= 73


def test_apply_positioning_writes_tiltcom(tmp_path, monkeypatch):
    (tmp_path / "tilt.com").write_text(TILT_COM)
    calls = []
    monkeypatch.setattr(P, "run_tilt", lambda d, tc, log=None, cancel=None: calls.append("tilt"))
    monkeypatch.setattr(P, "run_trimvol", lambda d, log=None: calls.append("trimvol"))
    corr = P.Correction(offset_add=1.5, xtilt_add=-2.0, shift_add=-7.3, thickness_px=100)
    geom = P.apply_positioning(tmp_path, "TS", corr, margin_px=10, log=lambda s: None)
    assert calls == ["tilt", "trimvol"]
    assert geom == {"OFFSET": [1.5, 0.0], "XAXISTILT": -2.06, "SHIFT": [0.0, -7.3], "THICKNESS": 120}
    assert (tmp_path / "tilt.com.pyprep_orig").read_text() == TILT_COM
    tc = P.TiltCom.read(tmp_path / "tilt.com")
    assert tc.get("THICKNESS") == "120" and tc.floats("XAXISTILT", 1) == [-2.06]
    # A second pass adds to the current values but keeps the first backup.
    P.apply_positioning(tmp_path, "TS", P.Correction(0.5, 0.0, 0.0, 100), margin_px=0, log=lambda s: None)
    assert P.TiltCom.read(tmp_path / "tilt.com").floats("OFFSET", 1) == [2.0]
    assert (tmp_path / "tilt.com.pyprep_orig").read_text() == TILT_COM
    P.restore_original(tmp_path, log=lambda s: None)
    assert (tmp_path / "tilt.com").read_text() == TILT_COM


def test_apply_positioning_keeps_tiltcom_when_imod_fails(tmp_path, monkeypatch):
    # the correction is relative to the current tilt.com: a retry after a failure must not add it twice
    (tmp_path / "tilt.com").write_text(TILT_COM)
    monkeypatch.setattr(P, "run_tilt", lambda d, tc, log=None, cancel=None: None)

    def trimvol_fails(d, log=None):
        raise RuntimeError("trimvol failed")
    monkeypatch.setattr(P, "run_trimvol", trimvol_fails)
    with pytest.raises(RuntimeError, match="trimvol failed"):
        P.apply_positioning(tmp_path, "TS", P.Correction(1.5, -2.0, -7.3, 100), margin_px=10, log=lambda s: None)
    assert (tmp_path / "tilt.com").read_text() == TILT_COM


def test_com_args_strip_quotes():
    # batchruntomo writes trimvol.com with quoted file names
    assert P._com_args('$trimvol -f -rx "TS_full_rec.mrc" "TS_rec.mrc"') == ["-f", "-rx", "TS_full_rec.mrc", "TS_rec.mrc"]
    assert P._com_args(r'$trimvol -rx "D:\My data\TS_full_rec.mrc" out.mrc') == ["-rx", r"D:\My data\TS_full_rec.mrc",
                                                                                  "out.mrc"]


def test_imod_command_keeps_program_files_path(monkeypatch):
    # plain "cmd /c" cut "C:\Program Files" at the space whenever an argument was quoted
    monkeypatch.setattr(imod, "imod_dir", lambda: PureWindowsPath(r"C:\Program Files\IMOD"))
    assert imod.imod_command("trimvol", ["-rx", r"D:\My data\a.mrc", "b.mrc"]) == \
        r'cmd.exe /d /s /c ""C:\Program Files\IMOD\bin\trimvol.cmd" -rx "D:\My data\a.mrc" b.mrc"'


@needs_imod
def test_trimvol_runs_in_folder_with_space(tmp_path):
    d = tmp_path / "with space"
    d.mkdir()
    mrc.write_mrc(d / "TS_full_rec.mrc", np.random.rand(16, 48, 64).astype(np.float32))
    (d / "trimvol.com").write_text('$trimvol -f -rx "TS_full_rec.mrc" "TS_rec.mrc"\n')
    P.run_trimvol(d, log=lambda s: None)
    assert mrc.read_mrc(d / "TS_rec.mrc").shape == (48, 16, 64)      # -rx swaps Y and Z


@needs_imod
def test_batchruntomo_starts_from_folder_with_space(tmp_path):
    d = tmp_path / "with space"
    d.mkdir()
    lines = []
    imod.run_batchruntomo(d, "TS", d / "missing.adoc", imod.ReconSettings(cpus=1), log=lines.append, end=0)
    text = "\n".join(lines)
    assert "is not recognized" not in text and "does not exist" in text    # batchruntomo itself ran
