"""Reconstruction with IMOD's batchruntomo (Windows IMOD install).

pyPrep writes a directive file from a preset, copies the binned aligned stack
(with its .rawtlt and .mdoc) into ``<series>/imod_bin<N>/`` and runs
batchruntomo there.  The result is an ordinary etomo project: it can be opened
in etomo (``<series>.edf``) to inspect or redo any step.

Presets
    patch  - patch tracking, no fiducials (default; matches the reference
             Position_9_2 etomo project, scaled to the chosen binning)
    gold   - gold fiducials: auto-seeding, bead tracking, gold erasing
             (the settings of the lab's Linux batchruntomo script)
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

from .io.mdoc import clip_titles

PRESETS = {
    "patch": "Patch tracking (no gold)",
    "gold": "Gold fiducials",
}


@dataclass
class ReconSettings:
    enabled: bool = True
    preset: str = "patch"
    bin: int = 4                           # which pyPrep stack to reconstruct (bin level)
    use_dose_weighted: bool = False        # reconstruct the _DW stack instead of the plain sum
    patch_size_nm: float = 400.0           # patch tracking: patch edge length
    patch_overlap: float = 0.6
    gold_size_nm: float = 10.0             # gold preset
    gold_beads: int = 20
    erase_gold: bool = True
    positioning: str = "auto"              # "auto" = IMOD cryoposition; "fixed" = skip, use thickness_nm
    positioning_thickness_nm: float = 330.0
    thickness_nm: float = 200.0            # fixed thickness, or fallback if auto positioning fails
    sirt_like_iterations: int = 6          # 0 = plain weighted back-projection
    remove_xrays: bool = True
    cpus: int = max(1, (os.cpu_count() or 2) - 1)
    use_gpu: bool = False
    handedness: str = "auto"               # auto (flip if the CTF gradient says so) | keep | flip
    ctf_correct: bool = True               # phase-flip with pyPrep's per-tilt defocus (IMOD ctfphaseflip)
    deconvolve: bool = True                # also write <series>_rec_deconv.mrc (Wiener-like filter)
    deconv_strength: float = 1.0
    deconv_falloff: float = 1.0
    extra_directives: str = ""             # advanced: raw "key = value" lines, override the preset

    def to_dict(self):
        return asdict(self)


# ------------------------------------------------------------------ IMOD discovery
def imod_dir() -> Path | None:
    for cand in (os.environ.get("IMOD_DIR"), r"C:\Program Files\IMOD"):
        if cand and (Path(cand) / "bin").is_dir():
            return Path(cand)
    return None


def imod_version() -> str | None:
    d = imod_dir()
    if d and (d / "VERSION").exists():
        return (d / "VERSION").read_text().strip()
    return None


def imod_status() -> tuple[bool, str]:
    """(usable, human-readable description) for the settings page."""
    d = imod_dir()
    if d is None:
        return False, "IMOD not found (set IMOD_DIR or install IMOD for Windows)"
    if not (d / "bin" / "batchruntomo.cmd").exists():
        return False, f"IMOD at {d} has no batchruntomo.cmd"
    py = shutil.which("python", path=_imod_env()["PATH"])
    return True, f"IMOD {imod_version() or '?'} at {d}" + (f"  (python: {py})" if py else "  (no python on PATH!)")


def _imod_env() -> dict:
    """Environment for IMOD: the system PATH without pyPrep's own Python, so IMOD's
    scripts run on the same interpreter etomo uses."""
    env = dict(os.environ)
    own = {os.path.normcase(os.path.abspath(p)) for p in
           (sys.prefix, os.path.join(sys.prefix, "Scripts"), os.path.join(sys.prefix, "Library", "bin"))}
    parts = [p for p in env.get("PATH", "").split(os.pathsep)
             if p and os.path.normcase(os.path.abspath(p)) not in own]
    d = imod_dir()
    if d is not None:
        env.setdefault("IMOD_DIR", str(d))
        if str(d / "bin") not in parts:
            parts.insert(0, str(d / "bin"))
    env["PATH"] = os.pathsep.join(parts)
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    return env


# ------------------------------------------------------------------ directives
def build_directives(pixel_size_A: float, tilt_axis: float, voltage_kv: float,
                     rs: ReconSettings, ctf: dict | None = None) -> dict:
    """Directive dict for a stack with the given pixel size (A, of the stack being reconstructed).

    ``ctf`` is pyPrep's CTF record (see ``ctf.save_result``); with ``rs.ctf_correct``
    batchruntomo phase-flips the aligned stack using ``<series>.defocus``."""
    nm = pixel_size_A / 10.0
    to_px = lambda length_nm: max(1, int(round(length_nm / nm)))   # noqa: E731
    d = imod_dir()
    template = d / "SystemTemplate" / "cryoSample.adoc" if d else None
    dirs = {
        "setupset.currentStackExt": "mrc",
        "setupset.copyarg.stackext": "mrc",
        "setupset.copyarg.dual": "0",
        "setupset.copyarg.pixel": f"{nm:.4f}",
        "setupset.copyarg.rotation": f"{tilt_axis:.2f}",
        "setupset.copyarg.userawtlt": "1",
        "setupset.copyarg.voltage": f"{int(round(voltage_kv))}",
        "setupset.scanHeader": "1",
        "runtime.Preprocessing.any.removeXrays": "1" if rs.remove_xrays else "0",
        "runtime.Preprocessing.any.archiveOriginal": "0",
        "runtime.AlignedStack.any.binByFactor": "1",
        "comparam.align.tiltalign.RotOption": "1",
        "comparam.align.tiltalign.TiltOption": "5",
        "comparam.align.tiltalign.MagOption": "1",
        "comparam.align.tiltalign.RobustFitting": "1",
        "comparam.align.tiltalign.LocalAlignments": "0",
        "runtime.Postprocess.any.doTrimvol": "1",
        "runtime.Trimvol.any.reorient": "2",
    }
    if rs.positioning == "fixed":
        dirs["runtime.Positioning.any.sampleType"] = "0"
        dirs["comparam.tilt.tilt.THICKNESS"] = str(to_px(rs.thickness_nm))
    else:
        # Cryo positioning finds the specimen slab; sparse samples (e.g. isolated
        # microvilli) can defeat findsection, and batchruntomo then uses the fallback.
        dirs["runtime.Positioning.any.sampleType"] = "2"
        dirs["runtime.Positioning.any.thickness"] = str(to_px(rs.positioning_thickness_nm))
        dirs["runtime.Reconstruction.any.fallbackThickness"] = str(to_px(rs.thickness_nm))
    if template is not None and template.exists():
        dirs["setupset.systemTemplate"] = str(template).replace("\\", "/")
    if rs.sirt_like_iterations > 0:
        dirs["comparam.tilt.tilt.FakeSIRTiterations"] = str(int(rs.sirt_like_iterations))
    if rs.preset == "gold":
        dirs.update({
            "setupset.copyarg.gold": f"{rs.gold_size_nm:g}",
            "runtime.Fiducials.any.trackingMethod": "0",
            "runtime.Fiducials.any.seedingMethod": "1",
            "comparam.autofidseed.autofidseed.TargetNumberOfBeads": str(int(rs.gold_beads)),
            "runtime.Positioning.any.hasGoldBeads": "1",
            "runtime.AlignedStack.any.eraseGold": "1" if rs.erase_gold else "0",
        })
    else:
        size = to_px(rs.patch_size_nm)
        dirs.update({
            "setupset.copyarg.gold": "0",
            "runtime.Fiducials.any.trackingMethod": "1",
            "comparam.xcorr_pt.tiltxcorr.SizeOfPatchesXandY": f"{size},{size}",
            "comparam.xcorr_pt.tiltxcorr.OverlapOfPatchesXandY": f"{rs.patch_overlap:g},{rs.patch_overlap:g}",
            "comparam.xcorr_pt.tiltxcorr.IterateCorrelations": "1",
            "runtime.Positioning.any.hasGoldBeads": "0",
            "runtime.AlignedStack.any.eraseGold": "0",
        })
    if rs.ctf_correct and ctf:
        dirs["runtime.AlignedStack.any.correctCTF"] = "1"
        dirs["setupset.copyarg.defocus"] = f"{ctf['defocus_um'] * 1000:.0f}"
        dirs["setupset.copyarg.Cs"] = f"{ctf.get('settings', {}).get('cs_mm', 2.7):g}"
        if ctf.get("handedness", 1) < 0:
            dirs["comparam.ctfcorrection.ctfphaseflip.InvertTiltAngles"] = "1"
    for line in rs.extra_directives.splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            dirs[k.strip()] = v.strip()
    return dirs


def write_directive_file(path: Path, directives: dict, header: str = "") -> None:
    lines = [f"# {h}" for h in header.splitlines()] if header else []
    lines += [f"{k} = {v}" for k, v in directives.items()]
    Path(path).write_text("\n".join(lines) + "\n")


# ------------------------------------------------------------------ running
def prepare_recon_dir(out_dir: Path, series_name: str, stack: Path, defocus_file: Path | None = None) -> Path:
    """Copy the stack (+ .rawtlt, .mdoc and the defocus file) into ``imod_bin<N>/<series>.mrc``."""
    m = re.search(r"_bin(\d+)", stack.stem)
    recon_dir = out_dir / f"imod_bin{m.group(1) if m else 1}"
    recon_dir.mkdir(parents=True, exist_ok=True)
    # Manual-positioning leftovers belong to the previous run's alignment.
    for old in ("tilt.com.pyprep_orig", "pyprep_pos_trial.mrc", "pyprep_pos_ali_bin2.mrc",
                f"{series_name}.defocus", f"{series_name}_rec_deconv.mrc"):
        (recon_dir / old).unlink(missing_ok=True)
    if defocus_file is not None and Path(defocus_file).exists():
        shutil.copyfile(defocus_file, recon_dir / f"{series_name}.defocus")
    dst = recon_dir / f"{series_name}.mrc"
    shutil.copyfile(stack, dst)
    shutil.copyfile(stack.with_suffix(".rawtlt"), recon_dir / f"{series_name}.rawtlt")
    mdoc = stack.with_name(stack.name + ".mdoc")
    if mdoc.exists():
        # Stacks written before titles were clipped still carry over-long titles that IMOD cannot read.
        text = clip_titles(mdoc.read_text().replace(f"ImageFile = {stack.name}", f"ImageFile = {dst.name}"))
        (recon_dir / f"{dst.name}.mdoc").write_text(text)
    return recon_dir


def _kill_tree(proc: subprocess.Popen) -> None:
    subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)


def imod_command(name: str, args) -> str:
    """Command line that runs IMOD's ``bin/<name>.cmd`` wrapper through cmd.exe.

    ``cmd /c`` strips the outer quotes of a command line holding more than one quoted
    string, which cuts "C:\\Program Files\\IMOD\\..." at the space as soon as any argument
    is quoted (a path with a space). With ``/s`` and one pair of quotes around the whole
    command, cmd removes only that pair."""
    d = imod_dir()
    if d is None:
        raise RuntimeError("IMOD not found")
    return f'cmd.exe /d /s /c ""{d / "bin" / f"{name}.cmd"}" {subprocess.list2cmdline([str(a) for a in args])}"'


def run_batchruntomo(recon_dir: Path, root: str, directive_file: Path, rs: ReconSettings,
                     log: Callable[[str], None] = print, cancel: threading.Event | None = None,
                     start: float | None = None, end: float | None = None) -> int:
    args = ["-DirectiveFile", str(directive_file), "-RootName", root,
            "-CurrentLocation", str(recon_dir), "-CPUMachineList", str(max(1, rs.cpus)),
            "-NiceValue", "0", "-EtomoDebug", "0"]
    if rs.use_gpu:
        args += ["-GPUMachineList", "1"]
    if start is not None:
        args += ["-StartingStep", f"{start:g}"]
    if end is not None:
        args += ["-EndingStep", f"{end:g}"]
    command = imod_command("batchruntomo", args)
    log("Running: batchruntomo " + " ".join(args))
    proc = subprocess.Popen(command, cwd=str(recon_dir), env=_imod_env(), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    watcher = None
    if cancel is not None:
        def watch():
            while proc.poll() is None:
                if cancel.wait(0.5):
                    log("Cancelling batchruntomo...")
                    _kill_tree(proc)
                    return
        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
    failed = False
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            log(line)
            failed = failed or batchruntomo_failed(line)
    rc = proc.wait()
    if rc == 0 and failed:
        rc = 1          # batchruntomo exits 0 even when the data set aborted
    return rc


_BRT_FAILURE_RE = re.compile(r"^ABORT SET:|failures occurred for \d+ data ?sets?")


def batchruntomo_failed(line: str) -> bool:
    """True for the batchruntomo output lines that mean the data set was aborted."""
    return bool(_BRT_FAILURE_RE.search(line.strip()))


def find_tomogram(recon_dir: Path, root: str) -> Path | None:
    """Final (trimmed) reconstruction, falling back to the untrimmed one."""
    for name in (f"{root}_rec.mrc", f"{root}_SIRT_rec.mrc", f"{root}.rec", f"{root}_full_rec.mrc"):
        p = recon_dir / name
        if p.exists():
            return p
    recs = sorted(recon_dir.glob(f"{root}*rec*.mrc"), key=lambda p: p.stat().st_mtime)
    return recs[-1] if recs else None


def reconstruct(out_dir: Path, series_name: str, stack: Path, pixel_size_A: float, tilt_axis: float,
                voltage_kv: float, rs: ReconSettings, log: Callable[[str], None] = print,
                cancel: threading.Event | None = None, ctf: dict | None = None) -> dict:
    """Set up and run batchruntomo on ``stack``; returns a result record."""
    t0 = time.perf_counter()
    # batchruntomo runs inside the reconstruction folder, so every path must be absolute.
    out_dir, stack = Path(out_dir).resolve(), Path(stack).resolve()
    ok, msg = imod_status()
    if not ok:
        raise RuntimeError(msg)
    log(msg)
    use_ctf = bool(rs.ctf_correct and ctf and ctf.get("defocus_file") and Path(ctf["defocus_file"]).exists())
    if rs.ctf_correct and not use_ctf:
        log("CTF correction skipped: no pyPrep CTF estimate for this series")
    recon_dir = prepare_recon_dir(out_dir, series_name, stack, Path(ctf["defocus_file"]) if use_ctf else None)
    if use_ctf:
        # The invert flag follows the handedness relative to the tilt axis used here
        # (a handedness correction rotates the axis by 180 deg, which reverses the gradient).
        from .ctf import read_defocus_file, write_defocus_file
        dfile = recon_dir / f"{series_name}.defocus"
        _, rows = read_defocus_file(dfile)
        write_defocus_file(dfile, [r[1] for r in rows], [r[2] / 1000 for r in rows],
                           invert=ctf.get("handedness", 1) < 0)
    dirs = build_directives(pixel_size_A, tilt_axis, voltage_kv, rs, ctf if use_ctf else None)
    if use_ctf:
        log(f"CTF correction with {series_name}.defocus (defocus {ctf['defocus_um']:.2f} um"
            + (", tilt angles inverted for ctfphaseflip)" if ctf.get("handedness", 1) < 0 else ")"))
    adoc = recon_dir / "pyprep_batchruntomo.adoc"
    write_directive_file(adoc, dirs, f"pyPrep batchruntomo directives - preset: {PRESETS.get(rs.preset, rs.preset)}\n"
                                     f"stack: {stack.name}  ({pixel_size_A:.2f} A/px)")
    log(f"Reconstruction ({PRESETS.get(rs.preset, rs.preset)}) in {recon_dir}")
    if use_ctf:
        # tiltalign refines the tilt angles, and ctfphaseflip matches defocus entries to views by
        # angle: run up to CTF plotting (step 9), re-label the defocus file with the final angles,
        # then continue from 3D gold detection (step 10).
        rc = run_batchruntomo(recon_dir, series_name, adoc, rs, log, cancel, end=9)
        if rc == 0 and not (cancel is not None and cancel.is_set()):
            from .ctf import retarget_defocus_file
            tlt = recon_dir / f"{series_name}.tlt"
            try:
                n = retarget_defocus_file(recon_dir / f"{series_name}.defocus", tlt)
                log(f"{series_name}.defocus: {n} views re-labelled with the aligned tilt angles ({tlt.name})")
            except (OSError, ValueError) as e:
                log(f"WARNING: could not match the defocus file to {tlt.name}: {e}")
            rc = run_batchruntomo(recon_dir, series_name, adoc, rs, log, cancel, start=10)
    else:
        rc = run_batchruntomo(recon_dir, series_name, adoc, rs, log, cancel)
    tomo = find_tomogram(recon_dir, series_name)
    cancelled = cancel is not None and cancel.is_set()
    status = "cancelled" if cancelled else ("complete" if rc == 0 and tomo is not None else "failed")
    rec = {"status": status, "returncode": rc, "preset": rs.preset, "recon_dir": str(recon_dir),
           "tomogram": str(tomo) if tomo else None, "stack": str(stack), "directives": dirs,
           "ctf_corrected": use_ctf, "tilt_axis": round(float(tilt_axis), 2),
           "settings": rs.to_dict(), "seconds": round(time.perf_counter() - t0, 1)}
    (recon_dir / "pyprep_recon.json").write_text(json.dumps(rec, indent=1))
    log(f"Reconstruction {status}" + (f": {tomo.name}" if tomo else "") + f" ({rec['seconds']:.0f} s)")
    return rec
