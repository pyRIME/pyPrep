"""Interactive tomogram positioning (replaces etomo's manual tomopitch step).

Workflow
    1. ``make_trial``: a quick bin-2 reconstruction of the aligned stack that is
       much thicker than the specimen, using the current ``tilt.com`` geometry.
    2. The user marks the top and bottom of the specimen in XZ and YZ views of
       the trial (``trial_views``) - two lines in each.
    3. ``correction``: from those lines, the tilt-angle offset, X-axis tilt,
       Z shift and thickness that make the specimen flat and centred.
    4. ``apply_positioning``: writes them into ``tilt.com`` (original kept as
       ``tilt.com.pyprep_orig``) and re-runs IMOD ``tilt`` and ``trimvol``.

Conventions (measured with IMOD tilt on synthetic specimens, see
docs/how-it-works.md): in the reconstruction (rows = Z, columns = X, sections =
Y), a specimen whose mid-plane has slope angle ``a`` in XZ and ``b`` in YZ and
centre ``c`` rows above the middle is made flat and centred by adding ``a`` to
OFFSET (tilt-angle offset), ``b`` to XAXISTILT and ``-c cos(a) cos(b)`` to the
Z SHIFT.
"""

from __future__ import annotations

import math
import os
import shlex
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from . import imod
from .io import mrc

TRIAL_BIN = 2
TRIAL_NAME = "pyprep_pos_trial.mrc"
TRIAL_STACK = "pyprep_pos_ali_bin2.mrc"


# ---------------------------------------------------------------------- tilt.com
@dataclass
class TiltCom:
    path: Path
    before: list = field(default_factory=list)   # lines up to and including "$tilt -StandardInput"
    params: list = field(default_factory=list)   # [key, value-string] in order
    after: list = field(default_factory=list)    # remaining lines ($if savework ...)

    @classmethod
    def read(cls, path) -> "TiltCom":
        path = Path(path)
        lines = path.read_text().splitlines()
        tc = cls(path)
        i = 0
        while i < len(lines) and not lines[i].strip().lower().startswith("$tilt"):
            tc.before.append(lines[i])
            i += 1
        if i == len(lines):
            raise ValueError(f"{path.name}: no $tilt command")
        tc.before.append(lines[i])
        i += 1
        while i < len(lines) and not lines[i].strip().startswith("$"):
            raw = lines[i].strip()
            if raw and not raw.startswith("#"):
                key, _, val = raw.replace("\t", " ").partition(" ")
                tc.params.append([key, val.strip()])
            i += 1
        tc.after = lines[i:]
        return tc

    def get(self, key: str) -> str | None:
        for k, v in self.params:
            if k.lower() == key.lower():
                return v
        return None

    def set(self, key: str, value) -> None:
        for p in self.params:
            if p[0].lower() == key.lower():
                p[1] = str(value)
                return
        self.params.append([key, str(value)])

    def remove(self, key: str) -> None:
        self.params = [p for p in self.params if p[0].lower() != key.lower()]

    def floats(self, key: str, n: int) -> list[float]:
        vals = [float(v) for v in (self.get(key) or "").replace(",", " ").split()[:n]]
        return vals + [0.0] * (n - len(vals))

    def text(self) -> str:
        body = [f"{k}\t{v}" if v else k for k, v in self.params]
        return "\n".join(self.before + body + self.after) + "\n"

    def write(self, path=None) -> None:
        Path(path or self.path).write_text(self.text())


# ---------------------------------------------------------------------- running IMOD
def _exe(name: str) -> Path:
    d = imod.imod_dir()
    if d is None:
        raise RuntimeError("IMOD not found")
    return d / "bin" / name


def run_tilt(recon_dir: Path, tc: TiltCom, log: Callable[[str], None] = print,
             cancel: threading.Event | None = None) -> None:
    """Run IMOD tilt with the parameters of ``tc`` (fed on standard input)."""
    env = imod._imod_env()
    env["IMOD_OUTPUT_FORMAT"] = "MRC"
    stdin = "\n".join(f"{k} {v}".strip() for k, v in tc.params) + "\n"
    proc = subprocess.Popen([str(_exe("tilt.exe")), "-StandardInput"], cwd=str(recon_dir), env=env,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    out, _ = proc.communicate(stdin) if cancel is None else _communicate_cancellable(proc, stdin, cancel)
    tail = [l for l in (out or "").splitlines() if l.strip()][-6:]
    if proc.returncode != 0:
        raise RuntimeError("IMOD tilt failed:\n" + "\n".join(tail))
    log("tilt: " + (tail[-1] if tail else "done"))


def _communicate_cancellable(proc, stdin, cancel):
    result = {}

    def talk():
        result["out"] = proc.communicate(stdin)[0]
    t = threading.Thread(target=talk, daemon=True)
    t.start()
    while t.is_alive():
        if cancel.wait(0.3):
            imod._kill_tree(proc)
            t.join(5)
            raise RuntimeError("cancelled")
    return result.get("out", ""), None


def _com_args(line: str) -> list[str]:
    """Arguments of a ``$program ...`` line of an IMOD .com file: quotes removed
    (batchruntomo quotes file names), backslashes kept as in Windows paths."""
    lex = shlex.shlex(line.lstrip("$"), posix=True)
    lex.whitespace_split, lex.escape = True, ""
    return list(lex)[1:]


def run_trimvol(recon_dir: Path, log: Callable[[str], None] = print) -> None:
    """Re-run the trimvol command batchruntomo wrote into trimvol.com (rotation to Z slices)."""
    com = recon_dir / "trimvol.com"
    line = next((l.strip() for l in com.read_text().splitlines() if l.strip().startswith("$trimvol")), None)
    if line is None:
        raise RuntimeError("trimvol.com has no $trimvol command")
    args = _com_args(line)
    r = subprocess.run(imod.imod_command("trimvol", args), cwd=str(recon_dir),
                       env=imod._imod_env(), capture_output=True, text=True,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if r.returncode != 0:
        raise RuntimeError("trimvol failed:\n" + "\n".join((r.stdout + r.stderr).splitlines()[-6:]))
    log("trimvol: " + " ".join(args))


# ---------------------------------------------------------------------- trial tomogram
def _bin_stack(src: Path, dst: Path, b: int) -> None:
    import torch
    from . import gpu
    h = mrc.read_header(src)
    with mrc.MrcStackWriter(dst, h.nx // b, h.ny // b, h.nz, np.float32, (h.pixel_size or 1.0) * b) as w:
        for z in range(h.nz):
            sec = torch.from_numpy(mrc.read_sections(src, z, 1, header=h)[0].astype(np.float32))
            w.write_section(z, gpu.bin_image(sec, b).numpy())


def make_trial(recon_dir, root: str, thickness_px: int, log: Callable[[str], None] = print,
               cancel: threading.Event | None = None) -> Path:
    """Bin-2 trial reconstruction ``thickness_px`` thick (in pixels of the reconstruction)."""
    recon_dir = Path(recon_dir)
    ali = recon_dir / f"{root}_ali.mrc"
    if not ali.exists():
        raise RuntimeError(f"{ali.name} not found - run the reconstruction first")
    t0 = time.perf_counter()
    stack = recon_dir / TRIAL_STACK
    if not stack.exists() or stack.stat().st_mtime < ali.stat().st_mtime:
        log(f"Binning {ali.name} by {TRIAL_BIN} for the trial tomogram")
        _bin_stack(ali, stack, TRIAL_BIN)
    tc = TiltCom.read(recon_dir / "tilt.com")
    tc.set("InputProjections", TRIAL_STACK)
    tc.set("OutputFile", TRIAL_NAME)
    tc.set("IMAGEBINNED", TRIAL_BIN)
    tc.set("THICKNESS", int(thickness_px))
    log(f"Trial tomogram: {thickness_px} px thick at bin {TRIAL_BIN}")
    run_tilt(recon_dir, tc, log, cancel)
    log(f"Trial tomogram ready ({time.perf_counter() - t0:.0f} s)")
    return recon_dir / TRIAL_NAME


def trial_views(trial: Path, highpass: float = 6.0) -> dict:
    """XZ and YZ "detail energy" views of a trial tomogram.

    Plain averages only show smooth reconstruction background; what reveals the
    specimen is where fine detail is.  Each slice is high-pass filtered and its
    squared values are summed along Y (XZ view) and along X (YZ view); the square
    root is returned.  Rows = Z with row 0 at the bottom; ``scale`` converts trial
    pixels to pixels of the full reconstruction.
    """
    from scipy import ndimage
    h = mrc.read_header(trial)                    # sections = Y, rows = Z, cols = X
    exz = np.zeros((h.ny, h.nx), np.float64)
    eyz = np.zeros((h.ny, h.nz), np.float64)
    for s in range(h.nz):
        sec = mrc.read_sections(trial, s, 1, header=h)[0].astype(np.float64)
        e = (sec - ndimage.gaussian_filter(sec, highpass)) ** 2
        exz += e
        eyz[:, s] = e.mean(axis=1)
    return {"xz": np.sqrt(exz / max(h.nz, 1)).astype(np.float32), "yz": np.sqrt(eyz).astype(np.float32),
            "shape": (h.ny, h.nz, h.nx), "scale": TRIAL_BIN, "pixel_size": (h.pixel_size or 1.0)}


def auto_boundaries(view: np.ndarray, frac: float = 0.3) -> tuple[float, float]:
    """First guess of the specimen's bottom and top rows: where the row-averaged detail
    energy rises ``frac`` of the way from the background level to the peak."""
    prof = view.mean(axis=1)
    k = max(3, len(prof) // 40)
    prof = np.convolve(prof, np.ones(k) / k, mode="same")
    base = np.percentile(prof, 10)
    thr = base + frac * (prof.max() - base)
    rows = np.nonzero(prof > thr)[0]
    if len(rows) == 0:
        return len(prof) * 0.35, len(prof) * 0.65
    return float(rows[0]), float(rows[-1])


# ---------------------------------------------------------------------- geometry
@dataclass
class Correction:
    offset_add: float        # deg, added to OFFSET (tilt-angle offset)
    xtilt_add: float         # deg, added to XAXISTILT
    shift_add: float         # reconstruction px, added to the Z SHIFT
    thickness_px: int        # specimen thickness, reconstruction px (without margin)

    def describe(self, pixel_nm: float) -> str:
        return (f"specimen {self.thickness_px * pixel_nm:.0f} nm thick, tilt {self.offset_add:+.2f} deg, "
                f"X-axis tilt {self.xtilt_add:+.2f} deg, Z shift {self.shift_add * pixel_nm:+.0f} nm")


def _line_at(p1, p2, x):
    (x1, z1), (x2, z2) = p1, p2
    if abs(x2 - x1) < 1e-6:
        return (z1 + z2) / 2
    return z1 + (z2 - z1) * (x - x1) / (x2 - x1)


def _slope(p1, p2):
    (x1, z1), (x2, z2) = p1, p2
    return 0.0 if abs(x2 - x1) < 1e-6 else (z2 - z1) / (x2 - x1)


def correction(top_xz, bot_xz, top_yz, bot_yz, shape, scale: float) -> Correction:
    """Lines are ((h1, z1), (h2, z2)) in trial pixels (h = x or y, z = row, row 0 at the bottom);
    ``shape`` = (Z, Y, X) of the trial; ``scale`` = reconstruction px per trial px."""
    nz, ny, nx = shape
    a = math.degrees(math.atan((_slope(*top_xz) + _slope(*bot_xz)) / 2))
    b = math.degrees(math.atan((_slope(*top_yz) + _slope(*bot_yz)) / 2))
    mid_xz = (_line_at(*top_xz, nx / 2) + _line_at(*bot_xz, nx / 2)) / 2
    mid_yz = (_line_at(*top_yz, ny / 2) + _line_at(*bot_yz, ny / 2)) / 2
    centre = (mid_xz + mid_yz) / 2 - nz / 2
    sep_xz = abs(_line_at(*top_xz, nx / 2) - _line_at(*bot_xz, nx / 2))
    sep_yz = abs(_line_at(*top_yz, ny / 2) - _line_at(*bot_yz, ny / 2))
    ca, cb = math.cos(math.radians(a)), math.cos(math.radians(b))
    thick = max(sep_xz, sep_yz) * ca * cb
    return Correction(offset_add=a, xtilt_add=b, shift_add=-centre * ca * cb * scale,
                      thickness_px=int(round(thick * scale)))


# ---------------------------------------------------------------------- apply
def apply_positioning(recon_dir, root: str, corr: Correction, margin_px: int,
                      log: Callable[[str], None] = print, cancel: threading.Event | None = None) -> dict:
    """Write the correction into tilt.com, re-run tilt and trimvol; returns the new geometry."""
    recon_dir = Path(recon_dir)
    com = recon_dir / "tilt.com"
    backup = recon_dir / "tilt.com.pyprep_orig"
    if not backup.exists():
        shutil.copyfile(com, backup)
    tc = TiltCom.read(com)
    off = tc.floats("OFFSET", 2)
    shift = tc.floats("SHIFT", 2)
    xtilt = tc.floats("XAXISTILT", 1)[0]
    geom = {"OFFSET": [round(off[0] + corr.offset_add, 2), off[1]],
            "XAXISTILT": round(xtilt + corr.xtilt_add, 2),
            "SHIFT": [shift[0], round(shift[1] + corr.shift_add, 1)],
            "THICKNESS": int(corr.thickness_px + 2 * margin_px)}
    tc.set("OFFSET", f"{geom['OFFSET'][0]} {geom['OFFSET'][1]}")
    tc.set("XAXISTILT", geom["XAXISTILT"])
    tc.set("SHIFT", f"{geom['SHIFT'][0]} {geom['SHIFT'][1]}")
    tc.set("THICKNESS", geom["THICKNESS"])
    t0 = time.perf_counter()
    run_tilt(recon_dir, tc, log, cancel)
    run_trimvol(recon_dir, log)
    # Only now: the correction is relative to the tilt.com the trial was built from, so writing
    # it before a failed tilt/trimvol would make a retried Apply add the correction twice.
    tc.write()
    log("tilt.com: " + ", ".join(f"{k} {v}" for k, v in geom.items()))
    log(f"Tomogram rebuilt with the new positioning ({time.perf_counter() - t0:.0f} s)")
    return geom


def restore_original(recon_dir, log: Callable[[str], None] = print,
                     cancel: threading.Event | None = None) -> None:
    recon_dir = Path(recon_dir)
    backup = recon_dir / "tilt.com.pyprep_orig"
    if not backup.exists():
        raise RuntimeError("no saved original tilt.com")
    shutil.copyfile(backup, recon_dir / "tilt.com")
    tc = TiltCom.read(recon_dir / "tilt.com")
    run_tilt(recon_dir, tc, log, cancel)
    run_trimvol(recon_dir, log)
    log("Original positioning restored")
