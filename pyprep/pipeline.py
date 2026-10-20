"""Process a tilt series: align and sum every tilt, write etomo-ready stacks."""

from __future__ import annotations

import csv
import json
import logging
import platform
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from . import __version__, ctf, gpu, qc
from .io import mrc
from .io.frames import open_movie
from .io.mdoc import Mdoc, MdocSection, PYPREP_TAG, write_mdoc
from .motion import FrameLayout, MotionCorrector, MotionSettings
from .settings import ProcessingSettings
from .tiltseries import TiltSeries

ProgressFn = Callable[[int, int, str], None]

KIND_SUFFIX = {"sum": "", "even": "_EVN", "odd": "_ODD", "dw": "_DW"}
EXCLUDE_TOLERANCE = 0.5   # deg, for matching user-excluded tilt angles
_RE_IMOD_STEP = re.compile(r"Reached step\s+([0-9.]+)")


class Cancelled(Exception):
    pass


@dataclass
class StackOutput:
    kind: str
    bin: int
    path: Path

    @property
    def root(self) -> str:
        return self.path.stem


def stack_name(series: str, kind: str, b: int) -> str:
    return f"{series}{KIND_SUFFIX[kind]}{'' if b == 1 else f'_bin{b}'}.mrc"


def recon_stack_key(settings: ProcessingSettings) -> tuple[str, int]:
    r = settings.recon
    return ("dw" if r.use_dose_weighted else "sum", int(r.bin))


def planned_outputs(series: TiltSeries, settings: ProcessingSettings, out_dir: Path) -> list[StackOutput]:
    """Stacks to write: the selected kinds x bin levels, plus the stack reconstruction needs."""
    o = settings.output
    pairs = [(k, int(b)) for k in o.stack_kinds() for b in sorted(set(o.bin_levels))]
    if settings.recon.enabled and recon_stack_key(settings) not in pairs:
        pairs.append(recon_stack_key(settings))
    return [StackOutput(k, b, out_dir / stack_name(series.name, k, b)) for k, b in pairs]


def estimate_output_bytes(series: TiltSeries, settings: ProcessingSettings) -> int:
    """Rough disk space a series will need (stacks, plus the IMOD project if reconstructing)."""
    tilts = [t for t in series.tilts if not t.missing and not t.excluded]
    if not tilts:
        return 0
    try:
        movie = open_movie(tilts[0].frame_path, settings.input)
    except Exception:
        return 0
    up = int(getattr(movie, "upsampling", 1) or 1)
    ny, nx = movie.shape[0] // up, movie.shape[1] // up
    px = 2 if settings.output.dtype == "int16" else 4
    n = len(tilts)
    total = sum((nx // o.bin) * (ny // o.bin) * n * px for o in planned_outputs(series, settings, Path(".")))
    r = settings.recon
    if r.enabled:
        b = int(r.bin)
        stack = (nx // b) * (ny // b) * n * 4
        thick = max(r.thickness_nm, r.positioning_thickness_nm) * 10 / (series.pixel_size * b)
        volume = (nx // b) * (ny // b) * thick * 4
        total += 4 * stack + (4 if r.deconvolve else 3) * volume   # stacks; full/trimmed/trial(/deconvolved)
    return int(total)


def result_json_path(series: TiltSeries, out_dir: Path) -> Path:
    return out_dir / f"{series.name}_pyprep.json"


def is_complete(series: TiltSeries, settings: ProcessingSettings, out_root: Path) -> bool:
    out_dir = Path(out_root) / series.name
    j = result_json_path(series, out_dir)
    if not j.exists():
        return False
    try:
        info = json.loads(j.read_text())
    except (OSError, ValueError):
        return False
    if info.get("status") != "complete":
        return False
    if "used_tilts" in info and sorted(info["used_tilts"]) != used_tilt_ids(series, settings):
        return False                      # tilts were excluded/included since: rebuild
    old, new = frame_of_record(info, series), image_frame(series, settings)
    if bool(old["invert_y"]) != new["invert_y"] or abs(normalize_angle(old["tilt_axis"] - new["tilt_axis"])) > 0.01:
        return False                      # other orientation or axis (Tomo5 stacks from before): rebuild, CTF too
    return all(p.path.exists() for p in planned_outputs(series, settings, out_dir))


def _apply_exclusions(series: TiltSeries, angles) -> None:
    """Exclude tilts near the given angles (adds to any per-tilt exclusions already set)."""
    for t in series.tilts:
        if any(abs(t.angle - a) <= EXCLUDE_TOLERANCE for a in angles):
            t.excluded = True


def _series_logger(log_path: Path, callback: Callable[[str], None] | None, mode: str = "w") -> logging.Logger:
    log = logging.getLogger(f"pyprep.series.{log_path.stem}.{id(log_path)}.{time.time()}")
    log.setLevel(logging.INFO)
    log.propagate = False
    fh = logging.FileHandler(log_path, mode=mode, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))
    log.addHandler(fh)
    if callback is not None:
        class _CB(logging.Handler):
            def emit(self, record):
                callback(self.format(record))
        h = _CB()
        h.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(h)
    return log


def _close_logger(log: logging.Logger) -> None:
    for h in list(log.handlers):
        h.close()
        log.removeHandler(h)


@dataclass
class MovieContext:
    """How a series' movies map onto physical pixels.

    EER frames may be rendered at ``up`` x super-resolution; everything the user
    sets (output binning, alignment binning, shift limits) is in physical pixels,
    so it is scaled here, and shifts are reported back in physical pixels.
    """
    up: int
    frame_pixel: float                   # A per rendered pixel
    motion: MotionSettings               # alignment settings in rendered pixels
    gain: torch.Tensor | None
    description: list

    def frame_bin(self, out_bin: int) -> int:
        return int(out_bin) * self.up


def prepare_movie_context(series: TiltSeries, settings: ProcessingSettings, movie, device) -> MovieContext:
    up = int(getattr(movie, "upsampling", 1) or 1)
    ms = settings.motion
    motion = replace(ms, align_bin=int(ms.align_bin) * up, max_shift=ms.max_shift * up,
                     tolerance=ms.tolerance * up)
    desc = [movie.describe()]
    gain = None
    inp = settings.input
    if inp.gain_path:
        from .io.gain import prepare_gain
        g = prepare_gain(inp.gain_path, movie.shape, inp.gain_mode, inp.gain_rotate, inp.gain_flip,
                         up, bool(getattr(movie, "is_eer", False)))
        gain = torch.from_numpy(g.multiplier).to(device)
        desc.append(g.description)
    elif getattr(movie, "is_eer", False):
        desc.append("WARNING: EER movie without a gain reference - frames are not gain-corrected")
    return MovieContext(up, series.pixel_size / up, motion, gain, desc)


def read_frames(path, settings: ProcessingSettings) -> tuple[np.ndarray, list | None, int | None]:
    """All frames/fractions of one tilt, the raw frame count behind each (EER), and the
    pixel value that means 'clipped' for integer fraction files (None if not applicable)."""
    movie = open_movie(path, settings.input)
    frames = movie.read()
    sat = None
    if not getattr(movie, "is_eer", False):
        header = getattr(movie, "header", None)
        sat = qc.saturation_value(frames.dtype, header.mode if header is not None else None)
    return frames, movie.frame_counts(), sat


def used_tilt_ids(series: TiltSeries, settings: ProcessingSettings) -> list[int]:
    """Acquisition indices of the tilts that go into the stacks with these settings."""
    _apply_exclusions(series, settings.output.exclude_angles)
    return sorted(t.zvalue for t in series.usable)


def _to_host(img: torch.Tensor, dtype: np.dtype) -> tuple[np.ndarray, tuple]:
    """Convert on the device to the output dtype, compute header stats there, then copy to host."""
    if dtype.kind == "i":
        info = np.iinfo(dtype)
        img = img.round().clamp_(info.min, info.max)
    d = img.to(torch.float64) if img.device.type == "cpu" else img
    stats = (float(img.min()), float(img.max()),
             float(d.sum(dtype=torch.float64)), float((d * d).sum(dtype=torch.float64)))
    host = img.to(torch.int16 if dtype == np.int16 else torch.float32).cpu().numpy()
    return host, stats


def _record_qc(record: dict, log) -> None:
    """Flag suspicious tilts in the series record and log them."""
    tl = record["tilts"]
    if not tl:
        return
    flags, fit = qc.flag_tilts([t["angle"] for t in tl],
                               intensity=[t["mean_counts"] / t["exposure"] for t in tl],
                               drift=[t["drift_A"] for t in tl], scores=[float(np.mean(t["scores"])) for t in tl],
                               converged=[t["converged"] for t in tl])
    for t, f in zip(tl, flags):
        t["flags"], t["flag_detail"] = f.flags, f.detail
        if f.flags:
            log.warning(f"QC: tilt {t['acquisition'] + 1:03d} ({t['angle']:+.2f} deg): " + "; ".join(
                [*f.detail, *[x for x in f.flags if x == "not converged"]]))
    sats = [t["saturated"] for t in tl if t.get("saturated") is not None]
    record["qc"] = {"flagged": sum(bool(f.flags) for f in flags),
                    "excludable": [t["acquisition"] for t, f in zip(tl, flags) if f.excludable],
                    "intensity_fit": None if fit is None else {"a": fit.a, "b": fit.b, "offset": fit.offset,
                                                                "spread": fit.spread},
                    "saturation": float(np.median(sats)) if sats else None}
    if sats and np.median(sats) > qc.SATURATION_WARN:
        log.warning(f"QC: fraction files are saturated - {100 * np.median(sats):.1f}% of pixels at the "
                    f"maximum 8-bit value (median over tilts). Counts above it were clipped at acquisition; "
                    f"saving fractions as 16-bit (or more, shorter fractions) avoids this.")
    if not record["qc"]["flagged"]:
        log.info("QC: no tilts flagged")


def _fit_ctf(collector, series: TiltSeries, out_dir: Path, device, log, axis: float) -> dict | None:
    """Fit the collected spectra, write <name>.defocus / _ctf.npz, log a summary.
    ``axis``: tilt axis in the frame of the collected images (see ``image_frame``)."""
    try:
        t0 = time.perf_counter()
        res = ctf.fit_series(collector, axis, series.voltage, device, log.info)
        rec = ctf.save_result(res, out_dir, series.name)
        d = res.defocus_um
        res_a = [t.resolution_A for t in res.tilts]
        log.info(f"CTF: defocus {rec['defocus_um']:.2f} um (low tilts; range {np.nanmin(d):.2f}-{np.nanmax(d):.2f}), "
                 f"median fit resolution {np.nanmedian(res_a):.1f} A ({time.perf_counter() - t0:.1f} s) "
                 f"-> {series.name}.defocus")
        if 0 < res.handedness_confidence < 0.7:
            log.warning(f"CTF: the direction of the defocus gradient is uncertain "
                        f"({100 * res.handedness_confidence:.0f}% agreement)")
        return rec
    except Exception as e:
        log.warning(f"CTF estimation failed: {e}")
        return None


def mdoc_defocus_um(series: TiltSeries) -> float | None:
    """Target defocus from the mdoc (underfocus positive), if recorded."""
    vals = []
    for t in series.tilts:
        try:
            vals.append(abs(float(t.section.get("TargetDefocus", "nan"))))
        except ValueError:
            pass
    vals = [v for v in vals if np.isfinite(v) and v > 0]
    return float(np.median(vals)) if vals else None


def ensure_ctf(series: TiltSeries, settings: ProcessingSettings, out_root, log_callback=None,
               cancel: threading.Event | None = None) -> dict | None:
    """CTF for a series whose stacks already exist (from the bin-1 aligned stack)."""
    out_dir = Path(out_root) / series.name
    j = result_json_path(series, out_dir)
    try:
        info = json.loads(j.read_text())
    except (OSError, ValueError):
        return None
    if info.get("ctf") or not settings.ctf.enabled:
        return info.get("ctf")
    stack = out_dir / stack_name(series.name, "sum", 1)
    say = log_callback or (lambda s: None)
    if not stack.exists():
        say(f"{series.name}: CTF not estimated - it needs the bin 1 aligned stack (tick bin 1 and re-run)")
        return None
    log = _series_logger(out_dir / f"{series.name}_pyprep.log", log_callback, mode="a")
    try:
        log.info(f"CTF estimation from {stack.name}")
        device = gpu.select_device(settings.use_gpu, settings.gpu_id)
        h = mrc.read_header(stack)
        angles = [t["angle"] for t in info.get("tilts", [])]
        if len(angles) != h.nz:
            angles = [float(a) for a in stack.with_suffix(".rawtlt").read_text().split()]
        col = ctf.SpectrumCollector(series.pixel_size, settings.ctf)
        for z in range(h.nz):
            if cancel is not None and cancel.is_set():
                return None
            sec = mrc.read_sections(stack, z, 1, header=h)[0].astype(np.float32)
            col.add(torch.from_numpy(sec).to(device), angles[z])
        rec = _fit_ctf(col, series, out_dir, device, log, frame_of_record(info, series)["tilt_axis"])
    finally:
        _close_logger(log)
    if rec is not None:
        info = json.loads(j.read_text())
        info["ctf"] = rec
        j.write_text(json.dumps(info, indent=1))
    return rec


def deconvolve_series(series: TiltSeries, settings: ProcessingSettings, out_root, rec: dict,
                      log_callback=None) -> dict | None:
    """Deconvolved copy of the series' tomogram (``<tomogram>_deconv.mrc``); updates ``rec``."""
    from . import deconv
    tomo = Path(rec.get("tomogram") or "")
    if not tomo.is_file():
        return None
    out_dir = Path(out_root) / series.name
    log = _series_logger(out_dir / f"{series.name}_pyprep.log", log_callback, mode="a")
    try:
        info = json.loads(result_json_path(series, out_dir).read_text())
    except (OSError, ValueError):
        info = {}
    try:
        c = info.get("ctf")
        if c:
            defocus, source = c["defocus_um"], "pyPrep CTF estimate"
        else:
            defocus, source = mdoc_defocus_um(series) or 3.0, "the mdoc target (no CTF estimate)"
        cs = (c or {}).get("settings", {}).get("cs_mm", settings.ctf.cs_mm)
        p = deconv.DeconvParams(pixel_A=series.pixel_size * int(settings.recon.bin), defocus_um=float(defocus),
                                kv=series.voltage, cs_mm=cs, amplitude=settings.ctf.amplitude_contrast,
                                strength=settings.recon.deconv_strength, falloff=settings.recon.deconv_falloff,
                                phase_flipped=bool(rec.get("ctf_corrected")))
        log.info(f"Deconvolution: defocus from {source}")
        device = gpu.select_device(settings.use_gpu, settings.gpu_id)
        out = deconv.deconvolve_file(tomo, deconv.deconv_name(tomo), p, device, log.info)
        rec["deconvolved"] = str(out)
        rec["deconv"] = dict(p.__dict__)
    except Exception as e:
        log.warning(f"Deconvolution failed: {e}")
        return None
    finally:
        _close_logger(log)
    return rec


def normalize_angle(a: float) -> float:
    """Angle in degrees wrapped to [-180, 180)."""
    return (float(a) + 180.0) % 360.0 - 180.0


TOMO5_INVERT_Y = ("Tomography 5 stores images inverted in Y relative to IMOD's convention (IMOD reports its "
                  "stacks as 'inverted in Y'); checked by subtomogram averaging of 70S ribosomes")


def image_frame(series: TiltSeries, settings: ProcessingSettings) -> dict:
    """Orientation of the images pyPrep writes, and the tilt axis in that frame.

    Tomography 5 stores images upside down (inverted in Y) and records the tilt axis
    for them as stored (``RotationAngle``; its header ``TiltAxisAngle`` is the same axis
    for the upright image, measured from X). Used as stored, the CTF defocus gradient
    looks inverted, and the tomogram comes out as a mirror image whether or not the
    axis is then rotated by 180 deg: the rotation makes the gradient agree but keeps
    the mirror. Inverting the images in Y restores the physical orientation, in which
    IMOD's conventions hold: the recorded axis, mirrored with them, agrees with the
    gradient and the hand is right (verified on C. difficile lamellae, Krios/K3/
    Tomography 5.27, with 70S ribosomes: subtomogram averaging and template matching
    against EMD-50855; the CTF check passes without rotating the axis)."""
    mode = settings.input.invert_y
    recorded = series.tilt_axis if series.tilt_axis is not None else 0.0
    if mode == "yes":
        inv, reason = True, "images inverted in Y (setting: yes)"
    elif mode == "no":
        inv, reason = False, "images as stored (setting: no)"
    elif series.mdoc.is_tomo5:
        inv, reason = True, "images inverted in Y: " + TOMO5_INVERT_Y
    else:
        inv, reason = False, "images as stored (SerialEM convention)"
    # IMOD measures the tilt axis angle from the Y axis, so a Y flip maps it to 180 - a
    axis = normalize_angle(180.0 - recorded) if inv else recorded
    return {"invert_y": inv, "tilt_axis": round(float(axis), 2), "recorded_tilt_axis": round(float(recorded), 2),
            "reason": reason}


def frame_of_record(info: dict, series: TiltSeries) -> dict:
    """Image frame a series record's stacks were written in (older records: as stored)."""
    if info.get("image_frame"):
        return info["image_frame"]
    recorded = series.tilt_axis if series.tilt_axis is not None else 0.0
    return {"invert_y": False, "tilt_axis": recorded, "recorded_tilt_axis": recorded, "reason": "images as stored"}


def handedness_decision(mode: str, ctf_rec: dict | None) -> tuple[bool, str]:
    """(rotate the tilt axis by 180 deg?, reason) for the handedness setting and CTF record.

    The CTF check makes the tilt axis direction agree with the defocus gradient of the
    images as written (IMOD's convention), which ctfphaseflip needs. It fixes a 180 deg
    error in the recorded axis. It cannot see images that were stored mirrored: those
    reconstruct as a mirror image with a consistent gradient. That is handled before,
    by the image orientation (see ``image_frame``)."""
    if mode == "flip":
        return True, "tilt axis rotated by 180 deg (setting: always flip)"
    if mode == "keep":
        return False, "tilt axis as recorded (setting: keep)"
    if not ctf_rec:
        return False, "tilt axis as recorded (no CTF estimate to check the handedness)"
    h = ctf_rec.get("handedness", 1)
    conf = ctf_rec.get("handedness_confidence", 0.0)
    votes = ctf_rec.get("handedness_votes")            # absent in records from before it was stored
    of = f"{100 * conf:.0f}% of " + (f"{votes} tilts beyond 20 deg" if votes is not None else "the tilts beyond 20 deg")
    if h > 0:
        return False, "tilt axis as recorded (CTF defocus gradient matches IMOD's convention)"
    if conf >= 0.8 and (votes is None or votes >= 4):
        return True, f"tilt axis rotated by 180 deg to correct the handedness (CTF defocus gradient inverted in {of})"
    return False, f"tilt axis as recorded: the CTF suggests inverted handedness, but not clearly enough ({of})"


_RE_AXIS_TITLE = re.compile(r"(Tilt axis angle = )(-?\d+(?:\.\d*)?)")


def set_stack_mdoc_axis(mdoc_path: Path, axis: float) -> bool:
    """Rewrite the 'Tilt axis angle = ...' title of a stack's .mrc.mdoc (what etomo reads)."""
    try:
        text = Path(mdoc_path).read_text()
    except OSError:
        return False
    new = _RE_AXIS_TITLE.sub(lambda m: f"{m.group(1)}{axis:.2f}", text, count=1)
    if new != text:
        Path(mdoc_path).write_text(new)
    return True


def sync_handedness(series: TiltSeries, settings: ProcessingSettings, out_root, log_callback=None) -> dict | None:
    """Decide the tilt axis for this series (handedness setting + CTF), record it in
    ``_pyprep.json`` and write it into the stacks' mdoc files."""
    out_dir = Path(out_root) / series.name
    j = result_json_path(series, out_dir)
    try:
        info = json.loads(j.read_text())
    except (OSError, ValueError):
        return None
    flip, reason = handedness_decision(settings.recon.handedness, info.get("ctf"))
    frame = frame_of_record(info, series)
    base = frame["tilt_axis"]                                   # axis of the stacks as written
    axis = normalize_angle(base + 180.0) if flip else base
    h = {"flipped": flip, "tilt_axis": round(axis, 2), "stack_tilt_axis": round(base, 2),
         "recorded_tilt_axis": round(frame["recorded_tilt_axis"], 2), "invert_y": bool(frame["invert_y"]),
         "reason": reason}
    if info.get("handedness") != h:
        for o in info.get("outputs", []):
            p = Path(o["path"])
            p = p if p.exists() else out_dir / p.name
            set_stack_mdoc_axis(p.with_name(p.name + ".mdoc"), axis)
        info["handedness"] = h
        j.write_text(json.dumps(info, indent=1))
        if log_callback:
            log_callback(f"{series.name}: {reason}" + (f" -> {axis:.2f} deg" if flip else ""))
    return h


def write_rawtlt(path: Path, angles) -> None:
    path.write_text("".join(f"{a:8.2f}\n" for a in angles))


def build_stack_mdoc(series: TiltSeries, tilts, out: StackOutput, nx: int, ny: int, mode: int,
                     stats: list, axis: float) -> Mdoc:
    """SerialEM-style mdoc whose ZValues follow the output stack order; ``axis`` is the
    tilt axis of the stack as written (see ``image_frame``)."""
    ps = series.pixel_size * out.bin
    doc = Mdoc(path=None)
    doc.header = {"PixelSpacing": f"{ps:.4f}", "Voltage": f"{series.voltage:.2f}",
                  "ImageFile": out.path.name, "ImageSize": f"{nx} {ny}", "DataMode": str(mode)}
    kind_text = {"sum": "motion-corrected", "even": "even-frame half-sum", "odd": "odd-frame half-sum",
                 "dw": "motion-corrected, DOSE-WEIGHTED"}[out.kind]
    # Date first: titles are clipped to 80 characters (MAX_TITLE_LEN), so a long series name loses its tail.
    doc.titles.append(f"{PYPREP_TAG} {__version__} {time.strftime('%d-%b-%y %H:%M')}: {kind_text} "
                      f"from {series.mdoc_path.name}")
    spot = series.tilts[0].section.get("SpotSize", "0") if series.tilts else "0"
    # This title format is what IMOD looks for to pick up the tilt axis angle.
    doc.titles.append(f"    Tilt axis angle = {axis:.2f}, binning = {out.bin}  spot = {spot}  camera = 0")
    for z, (t, st) in enumerate(zip(tilts, stats)):
        items = dict(t.section.items)
        items["PixelSpacing"] = f"{ps:.4f}"
        try:
            items["Binning"] = f"{float(items.get('Binning', '1').split()[0]) * out.bin:g}"
        except ValueError:
            items["Binning"] = str(out.bin)
        items["MinMaxMean"] = f"{st[0]:.2f} {st[1]:.2f} {st[2]:.2f}"
        items["AcquisitionOrder"] = str(t.zvalue)
        doc.sections.append(MdocSection("ZValue", str(z), items))
    return doc


def process_series(series: TiltSeries, settings: ProcessingSettings, out_root: str | Path,
                   progress: ProgressFn | None = None, cancel: threading.Event | None = None,
                   log_callback: Callable[[str], None] | None = None) -> dict:
    """Motion-correct every tilt of ``series`` and write the requested stacks.

    Returns the result record that is also saved as ``<name>_pyprep.json``.
    """
    t_start = time.perf_counter()
    out_dir = Path(out_root) / series.name
    out_dir.mkdir(parents=True, exist_ok=True)
    log = _series_logger(out_dir / f"{series.name}_pyprep.log", log_callback)
    progress = progress or (lambda i, n, msg: None)
    o = settings.output
    ms = settings.motion
    device = gpu.select_device(settings.use_gpu, settings.gpu_id)
    frame = image_frame(series, settings)
    record = {"pyprep_version": __version__, "series": series.name, "mdoc": str(series.mdoc_path),
              "frames_dir": str(series.frames_dir) if series.frames_dir else None,
              "settings": settings.to_dict(), "device": gpu.device_summary(device),
              "host": platform.node(), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
              "status": "running", "image_frame": frame, "tilts": [], "outputs": []}
    writers: dict = {}
    try:
        _apply_exclusions(series, o.exclude_angles)
        tilts = series.usable
        log.info(series.summary())
        log.info(f"Device: {gpu.device_summary(device)}")
        log.info("Image orientation: " + frame["reason"]
                 + (f"; tilt axis {frame['recorded_tilt_axis']:.2f} -> {frame['tilt_axis']:.2f} deg"
                    if frame["invert_y"] else ""))
        for t in series.missing:
            log.warning(f"MISSING fraction file for tilt {t.zvalue + 1:03d} ({t.angle:+.2f} deg): "
                        f"{t.section.get('SubFramePath', '?')} - left out of the stack")
        for t in series.tilts:
            if t.excluded:
                log.info(f"Excluded by user: tilt {t.zvalue + 1:03d} ({t.angle:+.2f} deg)")
        if not tilts:
            raise RuntimeError("no usable tilts (all missing or excluded)")
        outputs = planned_outputs(series, settings, out_dir)
        if not outputs:
            raise RuntimeError("no outputs selected")
        kinds = {out.kind for out in outputs}
        bins = sorted({out.bin for out in outputs})

        first = open_movie(tilts[0].frame_path, settings.input)
        ny, nx = first.shape
        ctx = prepare_movie_context(series, settings, first, device)
        for line in ctx.description:
            (log.warning if line.startswith("WARNING") else log.info)(line)
        frame_bins = {b: ctx.frame_bin(b) for b in bins}
        collector, ctf_fb = None, None
        if settings.ctf.enabled:
            collector = ctf.SpectrumCollector(series.pixel_size, settings.ctf)
            ctf_fb = ctx.frame_bin(collector.bin)
        sum_bins = sorted(set(frame_bins.values()) | ({ctf_fb} if ctf_fb else set()))
        layout = FrameLayout.for_shape(ny, nx, sum_bins + [int(ctx.motion.align_bin)])
        dtype = np.dtype(o.dtype)
        sample = first.read(0, 1)
        if dtype.kind == "i" and sample.dtype.kind == "f":
            log.warning("Input frames are floating point; writing float32 instead of int16")
            dtype = np.dtype(np.float32)
        label = mrc.make_label(f"pyPrep {__version__}: aligned {series.name}")
        for out in outputs:
            writers[(out.kind, out.bin)] = mrc.MrcStackWriter(
                out.path, nx // frame_bins[out.bin], ny // frame_bins[out.bin], len(tilts), dtype,
                series.pixel_size * out.bin, [label])
        log.info(f"{len(tilts)} tilts -> " + ", ".join(p.path.name for p in outputs))
        log.info(f"Frames {nx}x{ny}, FFT size {layout.W}x{layout.H}; alignment bin {ms.align_bin}, "
                 f"B-factor {ms.bfactor}, axis masking {'on' if ms.mask_axes else 'off'}")

        mc = MotionCorrector(ctx.motion, device)
        stats = {k: [] for k in writers}
        motion_rows = []
        # One thread prefetches the next tilt's frames, another writes finished sections,
        # so disk I/O overlaps GPU work.  At most one tilt of writes is kept in flight.
        read_pool = ThreadPoolExecutor(max_workers=1)
        write_pool = ThreadPoolExecutor(max_workers=1)
        pending = read_pool.submit(read_frames, tilts[0].frame_path, settings)
        writes: list = []
        try:
            for z, t in enumerate(tilts):
                if cancel is not None and cancel.is_set():
                    raise Cancelled()
                progress(z, len(tilts), f"Tilt {z + 1}/{len(tilts)} ({t.angle:+.1f} deg)")
                t0 = time.perf_counter()
                frames, frame_counts, sat_value = pending.result()
                if z + 1 < len(tilts):
                    pending = read_pool.submit(read_frames, tilts[z + 1].frame_path, settings)
                t1 = time.perf_counter()
                if frames.shape[1:] != (ny, nx):
                    raise RuntimeError(f"{t.frame_path.name}: frame size {frames.shape[1:]} != {(ny, nx)}")
                res = mc.align(frames, ctx.frame_pixel, gain=ctx.gain, layout=layout)
                t2 = time.perf_counter()
                imgs = mc.sum_frames(frames, res.shifts, ctx.frame_pixel, bins=sum_bins,
                                     even_odd=bool(kinds & {"even", "odd"}), dose_weight="dw" in kinds,
                                     frame_doses=t.doses_for(len(frames), frame_counts),
                                     prior_dose=t.prior_dose, voltage_kv=series.voltage, gain=ctx.gain,
                                     layout=layout, to_numpy=False)
                if frame["invert_y"]:     # rows reversed: stacks, CTF and everything after see the physical view
                    imgs = {k: torch.flip(v, dims=(-2,)) for k, v in imgs.items()}
                host = {}
                for key in writers:
                    kind, b = key
                    arr, st = _to_host(imgs[(kind, frame_bins[b])], dtype)
                    host[key] = (arr, st)
                    stats[key].append((st[0], st[1], st[2] / arr.size))
                if collector is not None:
                    try:
                        collector.add(imgs[("sum", ctf_fb)], t.angle, already_binned=True)
                    except Exception as e:           # CTF is an extra: never stop the stacks for it
                        log.warning(f"CTF: spectra not collected ({e}); CTF estimation skipped")
                        collector = None
                del imgs
                t3 = time.perf_counter()
                for f in writes:          # previous tilt's writes must finish (bounds memory, surfaces errors)
                    f.result()
                t4 = time.perf_counter()
                writes = [write_pool.submit(w.write_section, z, *host[key]) for key, w in writers.items()]
                drift = res.drift_angstrom(ctx.frame_pixel)
                shifts_phys = res.shifts / ctx.up          # report in physical pixels
                for j, (dy, dx) in enumerate(shifts_phys):
                    motion_rows.append([z, t.zvalue, f"{t.angle:.2f}", j, f"{dx:.3f}", f"{dy:.3f}"])
                record["tilts"].append({
                    "z": z, "acquisition": t.zvalue, "angle": t.angle, "file": t.frame_path.name,
                    "frames": int(len(frames)), "prior_dose": t.prior_dose, "dose": t.exposure_dose,
                    "shifts_px": np.round(shifts_phys, 3).tolist(), "scores": np.round(res.scores, 4).tolist(),
                    "iterations": res.iterations, "converged": res.converged, "drift_A": round(drift, 2),
                    "mean_counts": round(float(frames.mean(dtype=np.float64)) * len(frames), 4),
                    "exposure": qc.exposure_norm(t),
                    "saturated": (round(float(np.count_nonzero(frames == sat_value)) / frames.size, 6)
                                  if sat_value is not None else None)})
                flag = "" if res.converged else "  (not converged)"
                log.info(f"z {z:2d}  {t.angle:+7.2f} deg  {t.frame_path.name}  {len(frames)} frames  "
                         f"drift {drift:6.2f} A  score {res.scores.mean():.3f}  "
                         f"[read-wait {t1 - t0:.2f}  align {t2 - t1:.2f}  sum {t3 - t2:.2f}  "
                         f"write-wait {t4 - t3:.2f} s]{flag}")
            for f in writes:
                f.result()
        finally:
            read_pool.shutdown(wait=True, cancel_futures=True)
            write_pool.shutdown(wait=True)

        for w in writers.values():
            w.close()
        angles = [t.angle for t in tilts]
        for out in outputs:
            write_rawtlt(out.path.with_suffix(".rawtlt"), angles)
            w = writers[(out.kind, out.bin)]
            doc = build_stack_mdoc(series, tilts, out, w.nx, w.ny, w.mode, stats[(out.kind, out.bin)],
                                   frame["tilt_axis"])
            write_mdoc(doc, out.path.with_name(out.path.name + ".mdoc"))
            record["outputs"].append({"kind": out.kind, "bin": out.bin, "path": str(out.path),
                                      "pixel_size": series.pixel_size * out.bin,
                                      "size": [w.nx, w.ny, w.nz]})
        with open(out_dir / f"{series.name}_motion.csv", "w", newline="") as f:
            cw = csv.writer(f)
            cw.writerow(["z", "acquisition", "angle", "frame", "dx_px", "dy_px"])
            cw.writerows(motion_rows)
        _record_qc(record, log)
        if collector is not None and len(collector.spectra) == len(tilts):
            record["ctf"] = _fit_ctf(collector, series, out_dir, device, log, frame["tilt_axis"])
        record["used_tilts"] = sorted(t.zvalue for t in tilts)
        record["missing"] = [{"acquisition": t.zvalue, "angle": t.angle,
                              "file": t.section.get("SubFramePath")} for t in series.missing]
        record["status"] = "complete"
        progress(len(tilts), len(tilts), "Done")
        log.info(f"Finished {series.name} in {time.perf_counter() - t_start:.1f} s")
    except Cancelled:
        record["status"] = "cancelled"
        log.warning("Cancelled by user")
    except Exception as e:
        record["status"] = "failed"
        record["error"] = f"{type(e).__name__}: {e}"
        log.exception(f"FAILED: {e}")
        raise
    finally:
        for w in writers.values():
            w.close()
        record["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        record["seconds"] = round(time.perf_counter() - t_start, 1)
        result_json_path(series, out_dir).write_text(json.dumps(record, indent=1))
        _close_logger(log)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return record


# ---------------------------------------------------------------------- reconstruction
def recon_dir_for(series: TiltSeries, settings: ProcessingSettings, out_root) -> Path:
    return Path(out_root) / series.name / f"imod_bin{int(settings.recon.bin)}"


def recon_complete(series: TiltSeries, settings: ProcessingSettings, out_root) -> bool:
    j = recon_dir_for(series, settings, out_root) / "pyprep_recon.json"
    try:
        rec = json.loads(j.read_text())
    except (OSError, ValueError):
        return False
    if "used_tilts" in rec and sorted(rec["used_tilts"]) != used_tilt_ids(series, settings):
        return False
    try:
        info = json.loads(result_json_path(series, Path(out_root) / series.name).read_text())
    except (OSError, ValueError):
        info = {}
    want = (info.get("handedness") or {}).get("flipped", False)
    if bool(rec.get("handedness_flipped", False)) != bool(want):
        return False                      # handedness decision changed: reconstruct again
    if bool(rec.get("invert_y", False)) != bool(frame_of_record(info, series)["invert_y"]):
        return False                      # stacks rewritten in the other image orientation: reconstruct again
    return (rec.get("status") == "complete" and rec.get("preset") == settings.recon.preset
            and rec.get("tomogram") and Path(rec["tomogram"]).exists())


def reconstruct_series(series: TiltSeries, settings: ProcessingSettings, out_root,
                       progress: ProgressFn | None = None, cancel: threading.Event | None = None,
                       log_callback: Callable[[str], None] | None = None) -> dict:
    """Run IMOD batchruntomo on the series' binned stack (which must already exist)."""
    from . import imod

    out_dir = Path(out_root) / series.name
    kind, b = recon_stack_key(settings)
    stack = out_dir / stack_name(series.name, kind, b)
    log = _series_logger(out_dir / f"{series.name}_pyprep.log", log_callback, mode="a")
    progress = progress or (lambda i, n, msg: None)
    total_steps = 15

    def on_line(line: str):
        log.info(line)
        m = _RE_IMOD_STEP.match(line)
        if m:
            progress(min(int(float(m.group(1))), total_steps), total_steps, f"IMOD step {m.group(1)}")
        elif "(running " in line:
            progress(-1, total_steps, "IMOD: " + line.split("(running")[0].strip()[:60])

    try:
        if not stack.exists():
            raise RuntimeError(f"{stack.name} not found - run frame alignment first")
        log.info(f"===== Reconstruction: {imod.PRESETS.get(settings.recon.preset, settings.recon.preset)}, "
                 f"bin {b} =====")
        progress(0, total_steps, "IMOD setup")
        hand = sync_handedness(series, settings, out_root) or {}
        flipped = bool(hand.get("flipped"))
        try:
            info = json.loads(result_json_path(series, out_dir).read_text())
        except (OSError, ValueError):
            info = {}
        frame = frame_of_record(info, series)
        axis = hand.get("tilt_axis", frame["tilt_axis"])
        ctf_rec = info.get("ctf")
        if ctf_rec and flipped:
            ctf_rec = dict(ctf_rec, handedness=-ctf_rec.get("handedness", 1))   # relative to the rotated axis
        if frame["invert_y"]:
            log.info("Image orientation: " + frame["reason"])
        if hand.get("reason"):
            log.info("Handedness: " + hand["reason"] + f" (tilt axis {axis:.2f} deg)")
        rec = imod.reconstruct(out_dir, series.name, stack, series.pixel_size * b, axis,
                               series.voltage, settings.recon, log=on_line, cancel=cancel, ctf=ctf_rec)
        rec["handedness_flipped"] = flipped
        rec["invert_y"] = bool(frame["invert_y"])
    except Exception as e:
        log.exception(f"Reconstruction FAILED: {e}")
        rec = {"status": "failed", "error": f"{type(e).__name__}: {e}", "preset": settings.recon.preset}
    finally:
        _close_logger(log)
    if rec["status"] == "complete" and settings.recon.deconvolve:
        progress(-1, total_steps, "Deconvolving the tomogram")
        deconvolve_series(series, settings, out_root, rec, log_callback)
        try:
            (Path(rec["recon_dir"]) / "pyprep_recon.json").write_text(json.dumps(rec, indent=1))
        except OSError:
            pass
    # Keep the series record in sync so the Results page can find the tomogram.
    j = result_json_path(series, out_dir)
    try:
        info = json.loads(j.read_text())
        info["reconstruction"] = {k: v for k, v in rec.items() if k != "directives"}
        if "used_tilts" in info and rec.get("recon_dir"):
            rec["used_tilts"] = info["used_tilts"]
            info["reconstruction"]["used_tilts"] = info["used_tilts"]
            (Path(rec["recon_dir"]) / "pyprep_recon.json").write_text(json.dumps(rec, indent=1))
        j.write_text(json.dumps(info, indent=1))
    except (OSError, ValueError):
        pass
    if rec["status"] == "complete":
        progress(total_steps, total_steps, "Tomogram done")
    return rec


def _ensure_deconvolved(series: TiltSeries, settings: ProcessingSettings, out_root, log_callback=None) -> None:
    j = recon_dir_for(series, settings, out_root) / "pyprep_recon.json"
    try:
        rec = json.loads(j.read_text())
    except (OSError, ValueError):
        return
    if rec.get("deconvolved") and Path(rec["deconvolved"]).exists():
        return
    if deconvolve_series(series, settings, out_root, rec, log_callback) is not None:
        j.write_text(json.dumps(rec, indent=1))
        sj = result_json_path(series, Path(out_root) / series.name)
        try:
            info = json.loads(sj.read_text())
            info["reconstruction"] = {k: v for k, v in rec.items() if k != "directives"}
            sj.write_text(json.dumps(info, indent=1))
        except (OSError, ValueError):
            pass


def run_series(series: TiltSeries, settings: ProcessingSettings, out_root,
               progress: ProgressFn | None = None, cancel: threading.Event | None = None,
               log_callback: Callable[[str], None] | None = None, force_recon: bool = False) -> dict:
    """Frame alignment + stacks, then (if enabled) IMOD reconstruction.

    Steps already complete are skipped when ``settings.skip_existing`` is set.
    Returns {"stacks": status, "recon": status or None, "seconds": total}.
    """
    t0 = time.perf_counter()
    say = log_callback or (lambda s: None)
    result = {"stacks": None, "recon": None}
    if settings.skip_existing and is_complete(series, settings, out_root):
        say(f"{series.name}: stacks already complete - skipped")
        result["stacks"] = "skipped"
        if settings.ctf.enabled:
            ensure_ctf(series, settings, out_root, log_callback, cancel)
    if result["stacks"] in ("complete", "skipped"):
        sync_handedness(series, settings, out_root, log_callback)
    else:
        rec = process_series(series, settings, out_root, progress, cancel, log_callback)
        result["stacks"] = rec["status"]
    if settings.recon.enabled and result["stacks"] in ("complete", "skipped"):
        if cancel is not None and cancel.is_set():
            result["recon"] = "cancelled"
        elif settings.skip_existing and not force_recon and recon_complete(series, settings, out_root):
            say(f"{series.name}: tomogram already complete - skipped")
            result["recon"] = "skipped"
            if settings.recon.deconvolve:
                _ensure_deconvolved(series, settings, out_root, log_callback)
        else:
            result["recon"] = reconstruct_series(series, settings, out_root, progress, cancel,
                                                 log_callback)["status"]
    if result["stacks"] in ("complete", "skipped"):
        try:                                   # thumbnail for the session gallery (never fatal)
            from .thumbs import series_thumbnail
            png, source = series_thumbnail(Path(out_root) / series.name)
            if png is not None:
                say(f"{series.name}: gallery thumbnail ({source}) -> {png.name}")
        except Exception as e:
            say(f"{series.name}: could not make the gallery thumbnail: {e}")
    result["seconds"] = round(time.perf_counter() - t0, 1)
    return result
