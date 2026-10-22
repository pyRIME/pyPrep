"""Redeposition removal: dense surface material (FIB redeposition, Pt/GIS grains) is taken out of the aligned tilt
series by support-constrained SIRT, then the tomogram is rebuilt with IMOD tilt.

Needs the ASTRA Toolbox with a CUDA GPU (optional dependency: ``pip install astra-toolbox`` or ``pyprep[redep]``).
Validated on C. difficile lamellae (session of 2026-10-13): it removes the shading and blotches the grains cast into
the interior at 50-400 nm and changes nothing below about 12 nm (even/odd FSC unchanged); see docs/how-it-works.md.

  1. Working WBP of the reconstruction's aligned (CTF-corrected) stack: the project's tilt.com without OFFSET, SHIFT,
     local alignments and the SIRT-like filter, XAXISTILT 0, about 730 nm thick - the geometry of the SIRT model.
  2. The lamella interior (lamella.find_interior) and, beyond each surface, the dense layer: its depth, extent and
     the depth where its cross-section is widest (find_layer, widest_depth, layer_stats).
  3. Grain masks: per surface, the grain footprint segmented in the flattened WBP slab at that depth (histogram
     valley, else dense cores), extruded over the layer's apparent depth. Only material outside the interior can
     be removed.
  4. Pass 1: SIRT (ASTRA, GPU) of q = -ln(p / view mean), constrained to the slab (interior plus the layers).
  5. Contamination c = low-passed pass-1 density inside the grain masks; its projections A c give each view's
     transmission T = exp(-A c).
  6. Cleaned stack p + (1 - T) m, with m the local grain-free mean: the grains' shadow is filled with what the rays
     would have carried without them. (Dividing by T instead weights the shadowed rays fully and adds noise.)
  7. The tomogram is rebuilt from the cleaned stack with the project's own tilt.com and trimvol command (same
     geometry, filter and positioning as <root>_rec.mrc, which is kept).

Outputs in the reconstruction folder: <root>_ali_redep.mrc (cleaned stack), <root>_redep_transmission.mrc,
<root>_rec_redep.mrc (tomogram), and a "redeposition" record in pyprep_recon.json.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
from scipy import ndimage

from . import imod, lamella
from .io import mrc
from .positioning import TiltCom, _com_args, run_tilt

WORK_THICKNESS_NM = 730.0     # working WBP (700 slices at bin 4, as validated)
ITERATIONS = 100              # pass-1 SIRT iterations
MARGIN_NM = 15.0              # slab = interior + each dense layer's extent + this margin
MEAN_SIGMA_PX = 30.0          # low-pass of the grain-free mean m that fills the shadows
SEARCH_NM = 250.0             # how far beyond each surface to look for a dense layer
CHUNK = 96                    # Y slices per GPU SIRT chunk
MAX_WORK_GB = 6.0             # working volume (float32) above this: use a coarser reconstruction


# ---------------------------------------------------------------------- availability and names
@lru_cache(maxsize=1)
def astra_status() -> tuple[bool, str]:
    """(usable, human-readable description) for the settings page."""
    try:
        import astra
    except Exception:                     # not installed, or its CUDA libraries fail to load
        return False, "ASTRA Toolbox not installed (pip install astra-toolbox, or pyprep[redep])"
    try:
        cuda = bool(astra.use_cuda())
    except Exception:
        cuda = False
    version = getattr(astra, "__version__", "?")
    if not cuda:
        return False, f"ASTRA {version} is installed but finds no CUDA GPU"
    return True, f"ASTRA {version} (CUDA)"


def stack_name(root: str) -> str:
    return f"{root}_ali_redep.mrc"


def transmission_name(root: str) -> str:
    return f"{root}_redep_transmission.mrc"


def tomogram_name(root: str) -> str:
    return f"{root}_rec_redep.mrc"


def _full_name(root: str) -> str:
    return f"{root}_full_rec_redep.mrc"


def _wbp_name(root: str) -> str:
    return f"{root}_redep_wbp.mrc"


# ---------------------------------------------------------------------- ASTRA (numpy [Y, Z, X] = ASTRA [slices, rows, cols])
def _geoms(astra, ny_c: int, nz: int, nx: int, angles_rad: np.ndarray):
    vg = astra.create_vol_geom(nz, nx, ny_c)                       # rows = Z, cols = X, slices = Y
    pg = astra.create_proj_geom("parallel3d", 1.0, 1.0, ny_c, nx, angles_rad)
    return vg, pg


def forward_project(vol_c: np.ndarray, angles_rad: np.ndarray) -> np.ndarray:
    """Projections [Y, views, X] of a [Y, Z, X] chunk."""
    import astra
    vg, pg = _geoms(astra, vol_c.shape[0], vol_c.shape[1], vol_c.shape[2], angles_rad)
    sid, proj = astra.create_sino3d_gpu(np.ascontiguousarray(vol_c, np.float32), pg, vg)
    astra.data3d.delete(sid)
    return proj


def sirt(q_c: np.ndarray, mask_c: np.ndarray, angles_rad: np.ndarray, iters: int) -> np.ndarray:
    """SIRT3D_CUDA of projections [Y, views, X], constrained to mask [Y, Z, X]."""
    import astra
    ny_c, _, nx = q_c.shape
    vg, pg = _geoms(astra, ny_c, mask_c.shape[1], nx, angles_rad)
    pid = astra.data3d.create("-sino", pg, np.ascontiguousarray(q_c, np.float32))
    rid = astra.data3d.create("-vol", vg, 0)
    mid = astra.data3d.create("-vol", vg, np.ascontiguousarray(mask_c, np.float32))
    cfg = astra.astra_dict("SIRT3D_CUDA")
    cfg.update(ProjectionDataId=pid, ReconstructionDataId=rid, option={"ReconstructionMaskId": mid})
    aid = astra.algorithm.create(cfg)
    astra.algorithm.run(aid, iters)
    out = astra.data3d.get(rid)
    astra.algorithm.delete(aid)
    astra.data3d.delete([pid, rid, mid])
    return out


def calibrate(wbp: np.ndarray, q: np.ndarray, angles_deg: np.ndarray, y0: int, n: int = 32) -> dict:
    """Angle sign and Z flip that make ASTRA's projections of IMOD's reconstruction match the data."""
    sub = wbp[y0:y0 + n]
    res = {}
    for sign in (1, -1):
        for flip in (False, True):
            v = sub[:, ::-1, :] if flip else sub
            f = forward_project(v - v.mean(), np.deg2rad(sign * angles_deg))
            d = q[y0:y0 + n]
            f = f - f.mean(axis=(0, 2), keepdims=True)
            d = d - d.mean(axis=(0, 2), keepdims=True)
            res[(sign, flip)] = float((f * d).sum() / np.sqrt((f * f).sum() * (d * d).sum()))
    (sign, flip), best = max(res.items(), key=lambda kv: abs(kv[1]))
    return {"angle_sign": sign, "flip_z": flip, "corr": best, "all": {f"{k[0]},{k[1]}": v for k, v in res.items()}}


# ---------------------------------------------------------------------- dense layers beyond the surfaces
def _classify(cov: float, largest: float, rule: str) -> str:
    if cov <= 0:
        return "none detected"
    if cov > 0.6 and largest > 0.5:
        return "continuous film"
    if largest < 0.2:
        return "discrete grains" + ("" if rule == "valley" else " (cores)")
    return "patchy / merging grains"


def find_layer(vol, zref, sign, thick, ys, xs, nm, sig_px) -> dict:
    """Dense layer beyond one surface (sign -1: above z_top, +1: below z_bot) in flattened coordinates."""
    search = int(SEARCH_NM * nm)
    inside = int(0.25 * thick)
    offs = np.arange(-search, inside + 1) if sign < 0 else np.arange(-inside, search + 1)
    prof = lamella.flat_profiles(vol, zref, offs, sig_px, ys, xs)
    sm = lambda a: ndimage.uniform_filter1d(np.nan_to_num(a, nan=np.nanmedian(a)), 5)  # noqa: E731
    p001, p05, p50 = sm(prof[:, 0]), sm(prof[:, 1]), sm(prof[:, 2])
    valid = np.isfinite(prof[:, 0])           # flattened slabs leave the volume near its top / bottom
    outside = (offs * sign >= 3) & valid
    i001 = int(np.argmin(np.where(outside, p001, np.inf)))
    i05 = int(np.argmin(np.where(outside, p05, np.inf)))
    r = int(round(0.5 * (offs[i001] + offs[i05])))
    # significance: how much further the dark tail reaches at the layer than inside the cell (10-25 % in)
    inner = valid & (offs * sign <= -0.10 * thick) & (offs * sign >= -0.25 * thick)
    ref = float(np.median(p50[inner] - p001[inner]))
    layer = float(p50[i001] - p001[i001])
    ratio = layer / max(ref, 1e-6)
    return {"r_vox": r, "r_p001_vox": int(offs[i001]), "r_p05_vox": int(offs[i05]), "tail_layer": layer,
            "tail_interior": ref, "dip_sigma": ratio, "significant": bool(ratio > 1.5)}


def widest_depth(vol, zref, r_dark, sign, ys, xs, nm, sig_px) -> tuple[int, list]:
    """Depth (9-slice flattened slabs from the surface to 45 nm past the darkest one) where the grains'
    cross-section is widest, bimodal 'valley' slabs preferred."""
    lo, hi = sorted((sign * 3, r_dark + sign * int(45 * nm)))
    scan = []
    for r in range(lo, hi + 1, 3):
        slab = lamella.flat_slab(vol, zref, r, half=4)[ys, xs]
        if not np.isfinite(slab).any():
            continue
        mask, seg = lamella.segment_layer(slab, sig_px, min_area_px=int(np.ceil((5 * nm) ** 2)))
        scan.append((r, seg["rule"], float(mask.mean())))
    valley = [s for s in scan if s[1] == "valley"]
    return (max(valley, key=lambda s: s[2])[0] if valley else r_dark), scan


def layer_stats(vol, zref, r, ys, xs, nm, sig_px, ez) -> dict:
    """Coverage, grains and apparent thickness of the dense layer at depth r beyond a surface."""
    slab = lamella.flat_slab(vol, zref, r, half=4)[ys, xs]
    mask, seg = lamella.segment_layer(slab, sig_px, min_area_px=int(np.ceil((5 * nm) ** 2)))
    eqd, largest = lamella.components(mask, 1 / nm)
    cov = float(mask.mean())
    rr = np.arange(r - int(45 * nm), r + int(45 * nm) + 1)  # covered columns still below the threshold, per depth
    frac = []
    for rz in rr:
        img = lamella.flat_slab(vol, zref, rz, half=0)[ys, xs]
        img = np.where(np.isfinite(img), img, np.nanmedian(img))
        lp = ndimage.gaussian_filter(img, sig_px)
        frac.append(float((lp[mask] < seg["threshold"]).mean()) if mask.any() else 0.0)
    frac = ndimage.uniform_filter1d(np.array(frac), 3)
    if mask.any() and frac.max() > 0:
        pk = int(np.argmax(frac))
        run = next(r_ for r_ in lamella.runs(frac >= 0.5 * frac[pk]) if r_[0] <= pk < r_[1])
        fwhm_nm = (run[1] - run[0]) / nm
    else:
        fwhm_nm = 0.0
    return {"depth_r_vox": int(r), "rule": seg["rule"], "coverage": cov, "n_grains": int(eqd.size),
            "largest_component_fraction": largest, "apparent_thickness_nm": fwhm_nm,
            "thickness_nm_wedge_corrected": fwhm_nm / ez if fwhm_nm else 0.0,
            "morphology": _classify(cov, largest, seg["rule"])}


def wbp_layer_masks(wbp, z_top, z_bot, ext_vox, nm, sig_px, ez, ys, xs) -> tuple[np.ndarray, dict]:
    """3D masks of the dense layers, per surface: the grain footprint segmented (full field) in the flattened 9-slice
    WBP slab where the cross-section is widest (bimodal 'valley' slabs preferred, else the darkest core slab),
    extruded over the layer's apparent depth extent (centre +- apparent thickness / 2, which includes the
    missing-wedge elongation)."""
    ny, nz, nx = wbp.shape
    zz = np.arange(nz, dtype=np.float32)[None, :, None]
    mask = np.zeros((ny, nz, nx), bool)
    rep = {}
    for name, zref, sign in (("upper", z_top, -1), ("lower", z_bot, +1)):
        scan = []
        for r in range(3, int(ext_vox[name]) + 1, 3):
            slab = lamella.flat_slab(wbp, zref, sign * r, half=4)
            if np.isfinite(slab).mean() < 0.9:
                continue
            m, info = lamella.segment_layer(slab, sig_px, min_area_px=int(np.ceil((5 * nm) ** 2)))
            scan.append((r, info["rule"], float(m[ys, xs].mean()), m))
        if not scan:
            rep[name] = {"rule": "none (layer outside the volume)"}
            continue
        valley = [s for s in scan if s[1] == "valley"]
        r_w, rule, cov, foot = max(valley or scan, key=lambda s: s[2])
        st = layer_stats(wbp, zref, sign * r_w, ys, xs, nm, sig_px, ez)
        half = max(5 * nm, 0.5 * st["apparent_thickness_nm"] * nm)
        zc = zref + sign * r_w
        mask |= foot[:, None, :] & (zz >= (zc - half)[:, None, :]) & (zz <= (zc + half)[:, None, :])
        rep[name] = {"widest_depth_nm": r_w / nm, "rule": rule, "coverage_at_widest": cov,
                     "half_extent_nm": half / nm, "apparent_thickness_nm": st["apparent_thickness_nm"]}
    return mask, rep


# ---------------------------------------------------------------------- the cleaned stack
def fill_shadows(p: np.ndarray, t: np.ndarray, sigma_px: float = MEAN_SIGMA_PX) -> np.ndarray:
    """Cleaned view p + (1 - T) m: the deficit left by the dense material (transmission T) filled with the local
    grain-free mean m (the view divided by T, low-passed). Noise-neutral, unlike dividing by T."""
    return p + (1.0 - t) * ndimage.gaussian_filter(p / np.maximum(t, 1e-6), sigma_px)


def _check_cancel(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise RuntimeError("cancelled")


def working_wbp(recon_dir: Path, root: str, thickness_px: int, log: Callable[[str], None] = print,
                cancel: threading.Event | None = None) -> Path:
    """Plain WBP of the aligned stack in the SIRT model's geometry (see step 1 of the module docstring)."""
    tc = TiltCom.read(recon_dir / "tilt.com")
    for key in ("OFFSET", "SHIFT", "LOCALFILE", "FakeSIRTiterations"):
        tc.remove(key)
    tc.set("InputProjections", f"{root}_ali.mrc")
    tc.set("OutputFile", _wbp_name(root))
    tc.set("THICKNESS", int(thickness_px))
    tc.set("XAXISTILT", 0)
    run_tilt(recon_dir, tc, log, cancel)
    return recon_dir / _wbp_name(root)


def rebuild_tomogram(recon_dir, root: str, log: Callable[[str], None] = print,
                     cancel: threading.Event | None = None) -> Path:
    """The tomogram of the cleaned stack, with the project's current tilt.com and trimvol command."""
    recon_dir = Path(recon_dir)
    tc = TiltCom.read(recon_dir / "tilt.com")
    tc.set("InputProjections", stack_name(root))
    tc.set("OutputFile", _full_name(root))
    run_tilt(recon_dir, tc, log, cancel)
    line = next((l.strip() for l in (recon_dir / "trimvol.com").read_text().splitlines()
                 if l.strip().startswith("$trimvol")), None)
    if line is None:
        raise RuntimeError("trimvol.com has no $trimvol command")
    args = _com_args(line)
    args[-2:] = [_full_name(root), tomogram_name(root)]
    r = subprocess.run(imod.imod_command("trimvol", args), cwd=str(recon_dir), env=imod._imod_env(),
                       capture_output=True, text=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if r.returncode != 0:
        raise RuntimeError("trimvol failed:\n" + "\n".join((r.stdout + r.stderr).splitlines()[-6:]))
    (recon_dir / _full_name(root)).unlink(missing_ok=True)
    log(f"Tomogram rebuilt from the cleaned stack: {tomogram_name(root)}")
    return recon_dir / tomogram_name(root)


def remove_redeposition(recon_dir, root: str, log: Callable[[str], None] = print,
                        cancel: threading.Event | None = None, iterations: int = ITERATIONS,
                        margin_nm: float = MARGIN_NM, mean_sigma_px: float = MEAN_SIGMA_PX,
                        thickness_nm: float = WORK_THICKNESS_NM) -> dict:
    """Clean the aligned stack of a reconstruction folder and rebuild its tomogram (module docstring); returns the
    record. Raises RuntimeError when ASTRA is unusable or no lamella interior is found."""
    ok, msg = astra_status()
    if not ok:
        raise RuntimeError(msg)
    t0 = time.perf_counter()
    recon_dir = Path(recon_dir)
    ali = recon_dir / f"{root}_ali.mrc"
    angles = np.array([float(a) for a in (recon_dir / f"{root}.tlt").read_text().split()])
    h = mrc.read_header(ali)
    apix = float(h.pixel_size or 1.0)
    nm = 10.0 / apix
    thick_px = int(round(thickness_nm * nm))
    work_gb = h.nx * h.ny * thick_px * 4 / 1e9
    if work_gb > MAX_WORK_GB:
        raise RuntimeError(f"the working volume would take {work_gb:.1f} GB: remove redeposition on a coarser "
                           f"reconstruction (pixel {apix:.1f} A; bin 4 of 2.6 A data is about 4 GB)")
    log(f"Redeposition removal ({msg}): working WBP {thick_px} slices ({thickness_nm:.0f} nm)")

    # 1-3: working WBP, interior and dense layers, grain masks
    wbp_path = working_wbp(recon_dir, root, thick_px, log, cancel)
    wbp = mrc.read_mrc(wbp_path).astype(np.float32)                    # [Y, Z, X]
    wbp_path.unlink(missing_ok=True)
    ny, nz, nx = wbp.shape
    p = mrc.read_mrc(ali).astype(np.float32)                           # [views, Y, X]
    view_mean = p.mean(axis=(1, 2), keepdims=True)
    q = np.ascontiguousarray(np.transpose(-np.log(np.clip(p, 1e-3, None) / view_mean), (1, 0, 2)))  # [Y, views, X]
    cal = calibrate(wbp, q, angles, ny // 2 - 16)
    ang = np.deg2rad(cal["angle_sign"] * angles)
    flip = cal["flip_z"]
    log(f"Projection geometry: angle sign {cal['angle_sign']:+d}, Z flip {flip} (correlation {cal['corr']:.3f})")
    try:
        lam = lamella.find_interior(wbp, apix)
    except RuntimeError as e:
        raise RuntimeError(f"no lamella interior found ({e})") from None
    z_top, z_bot = lam["z_top"], lam["z_bot"]
    thick = lam["thickness_vox_median"]
    ys, xs = slice(int(0.08 * ny), int(0.92 * ny)), slice(int(0.08 * nx), int(0.92 * nx))
    sig_px = 3.0 * nm
    ez = lamella.missing_wedge_elongation(0.5 * (angles.max() - angles.min()))
    ext = {}
    for w, zref, sign in (("upper", z_top, -1), ("lower", z_bot, +1)):
        det = find_layer(wbp, zref, sign, thick, ys, xs, nm, sig_px)
        r_wide, _ = widest_depth(wbp, zref, det["r_vox"], sign, ys, xs, nm, sig_px)
        st = layer_stats(wbp, zref, r_wide, ys, xs, nm, sig_px, ez)
        e = (max(abs(r_wide), abs(det["r_vox"])) / nm + st["apparent_thickness_nm"] / 2 + margin_nm) \
            if det["significant"] else margin_nm
        ext[w] = {"extent_nm": e, "significant": det["significant"], "dark_ratio": det["dip_sigma"],
                  "widest_nm": abs(r_wide) / nm, "darkest_nm": abs(det["r_vox"]) / nm,
                  "apparent_thickness_nm": st["apparent_thickness_nm"], "coverage_wbp": st["coverage"],
                  "morphology": st["morphology"]}
    log(f"Interior {thick / nm:.0f} nm; dense layers: " + "; ".join(
        f"{w} {v['morphology']}, {100 * v['coverage_wbp']:.0f} % coverage" + ("" if v["significant"] else " (weak)")
        for w, v in ext.items()))
    zz = np.arange(nz, dtype=np.float32)[None, :, None]
    lo = (z_top - ext["upper"]["extent_nm"] * nm)[:, None, :]
    hi = (z_bot + ext["lower"]["extent_nm"] * nm)[:, None, :]
    seg, masks = wbp_layer_masks(wbp, z_top, z_bot, {w: ext[w]["extent_nm"] * nm for w in ext}, nm, sig_px, ez, ys, xs)
    del wbp
    _check_cancel(cancel)

    # 4: pass 1, SIRT constrained to the slab
    x1 = np.zeros((ny, nz, nx), np.float32)
    for y0 in range(0, ny, CHUNK):
        y1 = min(ny, y0 + CHUNK)
        slab = (zz >= lo[y0:y1]) & (zz <= hi[y0:y1])
        x = sirt(q[y0:y1], slab[:, ::-1, :] if flip else slab, ang, iterations)
        x1[y0:y1] = x[:, ::-1, :] if flip else x
        _check_cancel(cancel)
    log(f"SIRT ({iterations} iterations) constrained to the interior and its layers: done")

    # 5-6: contamination, its projections, and the cleaned stack
    ac = np.zeros_like(q)
    for y0 in range(0, ny, CHUNK):
        y1 = min(ny, y0 + CHUNK)
        a = max(0, y0 - 4)
        xc = ndimage.gaussian_filter(x1[a:min(ny, y1 + 4)], 1.5)[y0 - a:][: y1 - y0]
        c = xc * seg[y0:y1]
        ac[y0:y1] = forward_project(c[:, ::-1, :] if flip else c, ang)
    del x1, q
    ac = np.transpose(ac, (1, 0, 2))                                   # [views, Y, X]
    t = np.exp(-ac).astype(np.float32)
    above = zz < z_top[:, None, :]
    coverage = {"upper": float((seg & above).any(axis=1)[ys, xs].mean()),
                "lower": float((seg & ~above).any(axis=1)[ys, xs].mean())}
    del seg, above
    label = mrc.make_label("pyPrep redeposition removal")
    with mrc.MrcStackWriter(recon_dir / transmission_name(root), h.nx, h.ny, h.nz, np.float32, apix, [label]) as wr:
        for i in range(h.nz):
            wr.write_section(i, t[i])
    with mrc.MrcStackWriter(recon_dir / stack_name(root), h.nx, h.ny, h.nz, np.float32, apix, [label]) as wr:
        for i in range(h.nz):
            wr.write_section(i, fill_shadows(p[i], t[i], mean_sigma_px).astype(np.float32))
    projection = {"max": float(ac.max()), "p99": float(np.percentile(ac, 99)),
                  "fraction_pixels_gt_0.05": float((ac > 0.05).mean())}
    del p, t, ac
    log(f"Cleaned stack {stack_name(root)}: grains cover {100 * coverage['upper']:.0f} % (upper) / "
        f"{100 * coverage['lower']:.0f} % (lower) of the field")
    _check_cancel(cancel)

    # 7: the tomogram of the cleaned stack
    tomo = rebuild_tomogram(recon_dir, root, log, cancel)
    return {"status": "complete", "root": root, "tomogram": str(tomo), "stack": str(recon_dir / stack_name(root)),
            "transmission": str(recon_dir / transmission_name(root)), "interior_nm": thick / nm,
            "locator_round": lam.get("global_round"), "layers": ext, "masks": masks, "coverage": coverage,
            "contamination_projection": projection, "calibration": cal,
            "parameters": {"iterations": iterations, "margin_nm": margin_nm, "mean_sigma_px": mean_sigma_px,
                           "work_thickness_nm": thickness_nm},
            "seconds": round(time.perf_counter() - t0, 1)}


def refresh(recon_dir, log: Callable[[str], None] = print) -> Path | None:
    """Rebuild the redeposition-removed tomogram after the reconstruction's tilt.com changed (positioning); the
    cleaned stack does not depend on it."""
    try:
        rec = json.loads((Path(recon_dir) / "pyprep_recon.json").read_text())
    except (OSError, ValueError):
        return None
    r = rec.get("redeposition") or {}
    if r.get("status") != "complete" or not Path(r.get("stack", "")).exists():
        return None
    return rebuild_tomogram(recon_dir, r["root"], log)
