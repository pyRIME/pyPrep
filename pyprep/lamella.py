"""Locate a FIB lamella in a thick tomogram: textured interior, surface planes, tilt, and dense surface layers.

Made for lamellae where automatic positioning (IMOD findsection / cryoposition) fails because dense surface
material (redeposition, Pt) dominates the density statistics: on the C. difficile session of 2026-10-13 it
centred a 100 nm slab on the redeposition grains (Position_11) or failed outright (Position_12).
pyPrep uses it to find the interior and the dense surface layers for redeposition removal (redep.py); it is the
locator of the cryo-ET pipeline this came from, tested on that session's 34 lamellae.
Volumes are tilt's native layout, numpy [Y, Z, X]; Y runs along the tilt axis.

Method
  1. Texture: band-pass (difference-of-Gaussians, ~2-8 nm at bin 4) energy of a block-binned copy. Cellular
     interior is strongly textured; vacuum, vitreous ice and amorphous FIB-damage layers are not. Candidate runs
     of high area-mean energy are classified (classify_runs): a run whose dark tail is redeposition-like (> 2.8 x
     the vacuum's) or whose texture comes from a minority of the area (median / mean < 0.5) is dense surface
     material, not cells. The longest remaining run is the interior (never first-to-last above threshold).
     A dense layer's streak texture can set the profile's levels so high that the cells beside it form no run
     (Positions 6_3, 9, 10, 14, 18, positioned by hand on 2026-10-21): every rejected dense layer is then masked
     and the profile re-thresholded (round 2). A faint run that fills part of the field is looked for in the
     textured part of it (round 3). A run whose texture sits in small blobs at low contrast (Position_4's flakes)
     is set aside as a surface layer; it is used only if nothing else passes and its density is not a flake
     layer's (round 4), since cells that fill the whole field have a uniform texture that looks the same.
     Tiles take the run that overlaps this global interior most.
  2. Per tile (~150 nm), the run's edges give z_top and z_bot. Only "full" tiles (run >= 0.75 x the median run)
     are used: tiles cutting a cell's edge or holding ice have short runs. A mid-plane and a thickness plane
     (z = a + b*x + c*y each, iterative MAD clipping) are fitted, so the surfaces stay nearly parallel (a wedge
     is allowed). The mid-plane's slope in X is a rotation about the tilt axis (angle offset), its slope in Y an
     X-axis tilt, its height at the centre a Z shift; the thickness plane gives the interior thickness.
  3. Dense layers: in coordinates flattened to the fitted surfaces, the 0.1 % darkest low-passed density per
     depth step marks dense material; its deepest dip beyond each surface (within a search distance) is the layer.
     In its XY slab the layer is segmented at the histogram valley when the slab is bimodal (grains and the
     background between them both common), otherwise as dense cores (darker than mode - 3 sigma_up).
"""
import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks


def bin_volume(vol, f):
    """Block mean by f in every axis (crops to a multiple of f)."""
    if f == 1:
        return np.asarray(vol, np.float32)
    ny, nz, nx = (s // f * f for s in vol.shape)
    v = np.asarray(vol[:ny, :nz, :nx], np.float32)
    return v.reshape(ny // f, f, nz // f, f, nx // f, f).mean(axis=(1, 3, 5))


def texture_energy(small, s1=1.0, s2=4.0, smooth=2.0):
    """Local band-pass energy (difference of Gaussians, squared, smoothed) of a binned volume."""
    dog = ndimage.gaussian_filter(small, s1) - ndimage.gaussian_filter(small, s2)
    return ndimage.gaussian_filter(dog * dog, smooth)


def runs(mask):
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))


def longest_run(profile, frac=0.5, base_pct=20, top_pct=95, smooth=5):
    """(z0, z1, contrast) of the longest run above base + frac * (top - base); None if no contrast."""
    p = ndimage.uniform_filter1d(np.asarray(profile, float), smooth)
    base, top = np.percentile(p, base_pct), np.percentile(p, top_pct)
    if top <= base * 1.05:
        return None
    rr = runs(p > base + frac * (top - base))
    z0, z1 = max(rr, key=lambda r: r[1] - r[0])
    return int(z0), int(z1), float(top / max(base, 1e-12))


MAX_DARK_RATIO = 2.8     # dark tail of a run / vacuum's: above this the run looks like redeposition, not cells
MIN_MEDIAN_RATIO = 0.5   # median / mean texture of a run: below this its texture comes from a minority of the area
SPARSE_MAX_NM = 100.0    # sparse runs thinner than this are dense material; a thicker sparse run (a cell filling part
                         # of the field, or a thick layer of large grains) must also pass the large-scale test (LSF_MIN)
MIN_RUN_CONTRAST = 1.5   # a run's mean texture / vacuum level below this is noise (matters only for profiles
                         # with little contrast, e.g. an empty field)
ROUND2_MIN_CONTRAST = 1.5  # round 2, after masking the dense layers: the round-1 floor (cells beside a layer
                           # 1.5-1.9 on the 2026-10-21 session; 2.5 lost them, and 1.5 kept all its automatic slabs)
MIN_RUN_NM = 20.0        # thinner runs are wiggles of the profile, not an interior
LSF_MIN = 0.2            # a sparse, thick run counts as a cell filling part of the field only if its texture
                         # forms large regions (validation: grain layers 0.04-0.10, partial cell 0.45)
LSF_SIGMA_NM = 50.0      # scale separating a cell-sized textured region from grain-sized blobs
BLOB_MAX_CONTRAST = 2.4  # a non-sparse run in small blobs is a flake layer only below this contrast (flake layers
                         # 1.8-2.2: Positions 4, 20_2; cells filling the field 2.6-3.8: Positions 11, 12_3)
LAST_RESORT_MAX_DARK = 1.45  # round 4 takes a blob run only if it is not dense (dark ratio: flake layers 1.52-1.71,
                             # Positions 4, 18, 20_2; cell bands 1.14-1.35, Positions 4, 6_3, 11, 12_3)
ROUND3_KEEP = 0.4        # textured-area fallback: the most textured 40 % of the field at the faint run's depths ...
ROUND3_MIN_AREA = 0.15   # ... which must cover at least this fraction of the field


def large_scale_fraction(emap, sigma_px):
    """Fraction of an XY texture-energy map's variance that survives a Gaussian low-pass of sigma_px: high when
    the texture forms large regions (cells), low when it sits in many small blobs (grains)."""
    e = emap - emap.mean()
    v = float((e * e).mean())
    if v <= 0:
        return 0.0
    g = ndimage.gaussian_filter(e, sigma_px)
    return float((g * g).mean() / v)


def textured_area_round(small, en, voxel_nm, info, cy, cx, frac=0.5, smooth=5):
    """Round 3, for a cell that fills only part of the field (Position_1: one cell over ~40 % of it).

    Only when every round-1 run failed on contrast alone ("faint", nothing else): the round-1 tests are repeated on
    profiles over the most textured part of the field at the depths of the strongest faint run (the area-mean
    energy there, smoothed on the LSF scale, top ROUND3_KEEP of the field), so ice beside the cell no longer
    dilutes it. The tests and limits are round 1's. Returns ((z0, z1, contrast) or None, per-run diagnostics)."""
    faint = [r for r in info if r.get("round") in (1, 2) and r.get("z") and r["why"].startswith("faint")
             and "," not in r["why"]]
    if not faint or any(r.get("accepted") for r in info):
        return None, []
    a0, b0 = max(faint, key=lambda r: (r["round"] == 2, r["contrast"]))["z"]
    dense = dense_depths(info, en.shape[1], 0, 0)          # dark / sparse / blob runs, padded
    ec = en[cy, :, cx]
    emap = ndimage.uniform_filter(ec[:, a0:b0, :].mean(axis=1), max(3, int(round(LSF_SIGMA_NM / voxel_nm))))
    mask = emap >= np.percentile(emap, 100 * (1 - ROUND3_KEEP))
    area = float(mask.mean())
    if area < ROUND3_MIN_AREA:
        return None, [{"z": None, "round": 3, "accepted": False, "why": f"textured area {area:.2f} too small"}]
    sub = ec.transpose(1, 0, 2)[:, mask]                       # (z, textured pixels)
    e_mean = ndimage.uniform_filter1d(sub.mean(axis=1), smooth)
    e_med = ndimage.uniform_filter1d(np.median(sub, axis=1), smooth)
    sig = max(0.5, 1.5 / voxel_nm)
    lp = ndimage.gaussian_filter(small[cy, :, cx][::2], (0, sig, sig)).transpose(1, 0, 2)[:, mask[::2]]
    dark = ndimage.uniform_filter1d(np.median(lp, axis=1) - np.percentile(lp, 0.1, axis=1), smooth)
    if dense.any():                                         # dense layers neither set the levels nor form runs
        fill = np.percentile(e_mean[~dense], 20)
        e_mean, e_med = np.where(dense, fill, e_mean), np.where(dense, fill, e_med)
    base, top = np.percentile(e_mean, 20), np.percentile(e_mean, 95)
    if top <= base * 1.05:
        return None, [{"z": None, "round": 3, "accepted": False, "why": "no texture contrast in the textured area"}]
    cand = runs(e_mean > base + frac * (top - base))
    inrun = np.zeros(len(e_mean), bool)
    for a, b in cand:
        inrun[a:b] = True
    outside = np.nonzero(~inrun)[0]
    quiet = outside[np.argsort(e_mean[outside])[: max(5, outside.size // 3)]] if outside.size >= 5 else None
    d_vac = float(np.median(dark[quiet])) if quiet is not None else float(np.percentile(dark, 10))
    info3, ok = [], []
    for a, b in cand:
        r_dark = float(np.median(dark[a:b]) / max(d_vac, 1e-6))
        r_med = float(e_med[a:b].mean() / max(e_mean[a:b].mean(), 1e-12))
        r_con = float(e_mean[a:b].mean() / max(base, 1e-12))
        why = []
        if r_con < MIN_RUN_CONTRAST:
            why.append(f"faint x{r_con:.2f}")
        if r_dark > MAX_DARK_RATIO:
            why.append(f"dark x{r_dark:.1f}")
        if (b - a) * voxel_nm < MIN_RUN_NM:
            why.append(f"too thin {(b - a) * voxel_nm:.0f} nm")
        if r_med < MIN_MEDIAN_RATIO:
            why.append(f"sparse {r_med:.2f} within the textured area")
        r_lsf = large_scale_fraction(ec[:, a:b, :].mean(axis=1), LSF_SIGMA_NM / voxel_nm)
        if r_lsf < LSF_MIN:
            why.append(f"blob texture (large-scale fraction {r_lsf:.2f})")
        info3.append({"z": [int(a), int(b)], "round": 3, "textured_area": area, "dark_ratio": r_dark,
                      "median_ratio": r_med, "contrast": r_con, "large_scale_fraction": r_lsf, "accepted": not why,
                      "why": ", ".join(why) or "accepted"})
        if not why:
            ok.append((b - a, float(e_mean[a:b].mean()), int(a), int(b)))
    if not ok:
        return None, info3
    _, _, a, b = max(ok)
    return (a, b, float(top / max(base, 1e-12))), info3


def blob_last_resort(info, e_mean, contrast):
    """Round 4, when rounds 1-3 accept nothing: the longest run that only the blob test rejected (rounds 1-2) is the
    interior after all, if it is not dense (dark ratio <= LAST_RESORT_MAX_DARK). The blob test cannot always tell a
    flake layer from cells filling the whole field: both have a uniform small-scale texture and so a low large-scale
    fraction (Position_6_3's cells under its dense layer: 0.07); their darkness differs (a flake layer used here put
    Position_20_2 on its flakes). The chosen run is marked accepted, round 4, for QC. Returns (z0, z1, contrast) or
    None."""
    blob = [r for r in info if r.get("z") and r.get("round") in (1, 2) and r["why"].startswith("blob texture")
            and "," not in r["why"] and r["dark_ratio"] <= LAST_RESORT_MAX_DARK]
    if not blob:
        return None
    r = max(blob, key=lambda q: (q["z"][1] - q["z"][0], float(e_mean[q["z"][0]:q["z"][1]].mean())))
    r.update(found_in_round=r["round"], round=4, accepted=True, why="last resort: " + r["why"])
    return r["z"][0], r["z"][1], contrast


def classify_runs(small, en, voxel_nm, frac=0.5, smooth=5):
    """Candidate textured runs in the global depth profile and which of them is the cellular interior.

    Candidates: runs of the area-MEAN band-pass energy above half-way between its 20th and 95th percentiles
    (the mean follows heterogeneous cells; the median would cut weakly textured parts away). Each run is then
    tested against the two signatures of dense surface material:
      * darkness - its dark tail (median - 0.1th percentile of 1.5 nm low-passed density) relative to the
        quietest depths (vacuum): redeposition is denser than anything biological;
      * sparseness - median / mean texture over the area: grains give strong texture on a minority of the area. A
        cell filling part of the field is sparse too: a sparse run is kept only if it is >= 100 nm thick AND its
        texture forms large regions (large_scale_fraction >= LSF_MIN; grain layers 0.04-0.10, cells 0.27-0.45).
    A run must also be clearly above the vacuum level (MIN_RUN_CONTRAST). The longest run that passes is the
    interior. If none passes and a run was rejected as dense material (dark, sparse, or blob texture), a second
    round runs on the profile with those runs masked (see below), then the textured-area round 3, then round 4
    (blob_last_resort); with no accepted run, the specimen is not located. Returns ((z0, z1, contrast) or None,
    per-run diagnostics)."""
    sny, snz, snx = en.shape
    cy, cx = slice(sny // 8, 7 * sny // 8), slice(snx // 8, 7 * snx // 8)
    e_mean = ndimage.uniform_filter1d(en[cy, :, cx].mean(axis=(0, 2)), smooth)
    e_med = ndimage.uniform_filter1d(np.median(en[cy, :, cx], axis=(0, 2)), smooth)
    sig = max(0.5, 1.5 / voxel_nm)
    lp = ndimage.gaussian_filter(small[cy, :, cx][::2], (0, sig, sig))
    dark = ndimage.uniform_filter1d(np.median(lp, axis=(0, 2)) - np.percentile(lp, 0.1, axis=(0, 2)), smooth)
    base, top = np.percentile(e_mean, 20), np.percentile(e_mean, 95)
    if top <= base * 1.05:
        return None, [{"z": None, "why": "no texture contrast"}]
    # Round 2 only if round 1 accepts nothing AND some run was rejected as dense material (dark, sparse, blob): a
    # dense layer's streak texture sets the 95th percentile and pushes the cells beside it below the half-level
    # (2026-10-21 session: cells 1.2-1.9 against levels 1.6-2.4). Those runs are masked, the profile re-thresholded,
    # and a sparse run is rejected at any thickness. Faint and thin runs are not masked (masking any rejected run
    # turned streak texture beside grains into a false interior on Position_11_2). Tested on all 34 positioning
    # volumes of the session (a test copy, 2026-10-22): found 5 of the 7 slabs Evan set by hand,
    # kept the 26 automatic ones, and found nothing in Position_11_2 (one faint cell in a corner).
    info, prof = [], e_mean
    for rnd in (1, 2):
        lb, lt = np.percentile(prof, 20), np.percentile(prof, 95)
        cand = runs(prof > lb + frac * (lt - lb))
        inrun = np.zeros(len(prof), bool)
        for a, b in cand:
            inrun[a:b] = True
        outside = np.nonzero(~inrun)[0]
        quiet = outside[np.argsort(e_mean[outside])[: max(5, outside.size // 3)]] if outside.size >= 5 else None
        d_vac = float(np.median(dark[quiet])) if quiet is not None else float(np.percentile(dark, 10))
        ok, dark_runs = [], []
        for a, b in cand:
            r_dark = float(np.median(dark[a:b]) / max(d_vac, 1e-6))
            r_med = float(e_med[a:b].mean() / max(e_mean[a:b].mean(), 1e-12))
            r_con = float(e_mean[a:b].mean() / max(base, 1e-12))
            r_lsf = large_scale_fraction(en[cy, a:b, cx].mean(axis=1), LSF_SIGMA_NM / voxel_nm)
            why = []
            if r_con < (ROUND2_MIN_CONTRAST if rnd == 2 else MIN_RUN_CONTRAST):
                why.append(f"faint x{r_con:.2f}")
            if r_dark > MAX_DARK_RATIO:
                why.append(f"dark x{r_dark:.1f}")
                dark_runs.append((a, b))
            if (b - a) * voxel_nm < MIN_RUN_NM:
                why.append(f"too thin {(b - a) * voxel_nm:.0f} nm")
            if r_med < MIN_MEDIAN_RATIO:
                if rnd == 2 or (b - a) * voxel_nm < SPARSE_MAX_NM:
                    why.append(f"sparse {r_med:.2f}" + ("" if rnd == 2 else f" and thin {(b - a) * voxel_nm:.0f} nm"))
                    if rnd == 1:
                        dark_runs.append((a, b))
                elif r_lsf < LSF_MIN:
                    why.append(f"sparse {r_med:.2f} in small blobs (large-scale fraction {r_lsf:.2f})")
                    dark_runs.append((a, b))
            elif r_lsf < LSF_MIN and r_con < BLOB_MAX_CONTRAST:
                # texture of small blobs filling the field at low contrast (a flake or grain layer on a surface:
                # Position_4, large-scale fraction 0.06; cells over part of the field 0.27-0.55): masked for round 2
                # like dark runs, but kept for round 4, since cells filling the whole field can look the same
                why.append(f"blob texture (large-scale fraction {r_lsf:.2f})")
                dark_runs.append((a, b))
            info.append({"z": [int(a), int(b)], "round": rnd, "dark_ratio": r_dark, "median_ratio": r_med,
                         "contrast": r_con, "large_scale_fraction": r_lsf, "accepted": not why,
                         "why": ", ".join(why) or "accepted"})
            if not why:
                ok.append((b - a, float(e_mean[a:b].mean()), int(a), int(b)))
        if ok:
            _, _, a, b = max(ok)
            return (a, b, float(lt / max(lb, 1e-12))), info
        if rnd == 2 or not dark_runs:
            g3, info3 = textured_area_round(small, en, voxel_nm, info, cy, cx, frac, smooth)
            if g3 is None:
                g3 = blob_last_resort(info, e_mean, float(top / max(base, 1e-12)))
            return g3, info + info3
        prof = e_mean.copy()
        for a, b in dark_runs:
            prof[max(0, a - smooth):b + smooth] = base
    return None, info


def dense_depths(runs_info, n, g0, g1, pad=5):
    """Bool per depth: the global runs rejected as dense material (dark or sparse; not those only faint or too thin),
    padded, never inside the accepted interior run [g0, g1)."""
    m = np.zeros(n, bool)
    for r in runs_info:
        if r.get("z") and not r["accepted"] and ("dark" in r["why"] or "sparse" in r["why"] or "blob" in r["why"]):
            m[max(0, r["z"][0] - pad):min(n, r["z"][1] + pad)] = True
    m[g0:g1] = False
    return m


def best_overlap_run(profile, g0, g1, exclude=None, frac=0.5, smooth=5):
    """In a tile's mean-energy profile, the run overlapping the global interior [g0, g1) most (>= half of the
    run inside it), so a tile cannot pick a grain layer that is longer than its interior run. ``exclude``
    (bool per depth) masks the global runs rejected as dense material, whose energy would otherwise set the
    tile's 95th percentile and push a weakly textured interior below the half-level."""
    p = ndimage.uniform_filter1d(np.asarray(profile, float), smooth)
    use = np.ones(p.size, bool) if exclude is None else ~exclude
    base, top = np.percentile(p[use], 20), np.percentile(p[use], 95)
    if top <= base * 1.05:
        return None
    p = np.where(use, p, base)          # depths of rejected (dense) global runs neither set the levels nor form runs
    best, best_ov = None, 0
    for a, b in runs(p > base + frac * (top - base)):
        ov = max(0, min(b, g1) - max(a, g0))
        if ov >= 0.5 * (b - a) and ov > best_ov:
            best, best_ov = (int(a), int(b)), ov
    if best is None:
        return None
    return best[0], best[1], float(top / max(base, 1e-12))


def _fit_plane(y, x, z, n_iter=6, clip=2.5):
    keep = np.ones(z.size, bool)
    coef = np.zeros(3)
    for _ in range(n_iter):
        a = np.stack([np.ones(keep.sum()), x[keep], y[keep]], axis=1)
        coef, *_ = np.linalg.lstsq(a, z[keep], rcond=None)
        res = z - (coef[0] + coef[1] * x + coef[2] * y)
        s = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep]))) + 1e-6
        new = np.abs(res) < clip * s
        if new.sum() < 4 or np.array_equal(new, keep):
            break
        keep = new
    rms = float(np.sqrt(np.mean((z[keep] - (coef[0] + coef[1] * x[keep] + coef[2] * y[keep])) ** 2)))
    return coef, keep, rms


def find_interior(vol, apix, down=2, tile_nm=150.0, min_contrast=1.15):
    """Textured interior of a lamella: per-tile band edges and robust surface planes (full-resolution voxels).

    Returns a dict with z_top / z_bot maps (ny, nx), plane coefficients (z = a + b*x + c*y, voxels), tile points,
    the global band, thickness, and positioning numbers (angle offset about Y and X-axis tilt in degrees from
    the mid-plane slopes, Z shift of the mid-plane from the volume centre in voxels)."""
    ny, nz, nx = vol.shape
    small = bin_volume(vol, down)
    en = texture_energy(small)
    sny, snz, snx = en.shape
    glob, runs_info = classify_runs(small, en, apix * down / 10)
    if glob is None:
        vnm = apix * down / 10
        raise RuntimeError("no cellular textured run in the depth profile: " + "; ".join(
            (f"{r['z'][0] * vnm:.0f}-{r['z'][1] * vnm:.0f} nm {r['why']}" if r.get("z") else r["why"]) for r in runs_info))
    g0, g1, gcon = glob
    glen = g1 - g0
    dense = dense_depths(runs_info, len(en[0, :, 0]), g0, g1)
    t = max(8, int(round(tile_nm / (apix / 10 * down))))
    pts = []
    for y0 in range(0, sny - t + 1, t // 2):
        for x0 in range(0, snx - t + 1, t // 2):
            r = best_overlap_run(en[y0:y0 + t, :, x0:x0 + t].mean(axis=(0, 2)), g0, g1, dense)
            if r is None or r[2] < min_contrast:
                continue
            z0, z1, con = r
            if not (0.5 * glen <= z1 - z0 <= 1.6 * glen):
                continue
            if not (g0 - 0.5 * glen <= 0.5 * (z0 + z1) <= g1 + 0.5 * glen):
                continue
            pts.append(((y0 + t / 2) * down, (x0 + t / 2) * down, (z0 + 0.5) * down, (z1 - 0.5) * down, con))
    if len(pts) < 6:
        raise RuntimeError(f"only {len(pts)} tiles with a textured interior")
    p = np.array(pts)
    # Fit a mid-plane and a thickness plane (a lamella's surfaces are nearly parallel) to the "full" tiles only:
    # tiles whose textured run spans most of the median depth. Tiles that cut the edge of a cell, or hold
    # ice with little texture, give short runs and would tilt independently fitted surfaces.
    length = p[:, 3] - p[:, 2]
    full = length >= 0.75 * np.median(length)
    if full.sum() < 6:
        full = np.ones(len(p), bool)
    q = p[full]
    yy, xx = q[:, 0] / ny - 0.5, q[:, 1] / nx - 0.5          # normalised coordinates, centre = 0
    cmid, kmid, rms_mid = _fit_plane(yy, xx, 0.5 * (q[:, 2] + q[:, 3]))
    cth, kth, rms_th = _fit_plane(yy, xx, q[:, 3] - q[:, 2])
    gy, gx = np.mgrid[0:ny, 0:nx].astype(np.float32)
    gy, gx = gy / ny - 0.5, gx / nx - 0.5
    zmid = cmid[0] + cmid[1] * gx + cmid[2] * gy
    th = np.clip(cth[0] + cth[1] * gx + cth[2] * gy, 0.5 * cth[0], 1.5 * cth[0])
    z_top, z_bot = zmid - th / 2, zmid + th / 2
    ctop, cbot = cmid - cth / 2, cmid + cth / 2
    mid = cmid
    return {
        "z_top": z_top, "z_bot": z_bot,
        "plane_top": ctop.tolist(), "plane_bot": cbot.tolist(),        # a, b, c with x, y normalised to [-0.5, 0.5]
        "plane_mid": cmid.tolist(), "plane_thickness": cth.tolist(),
        "tiles": p.tolist(), "tiles_full": int(full.sum()), "tiles_total": int(len(p)),
        "tiles_used_mid": int(kmid.sum()), "tiles_used_thickness": int(kth.sum()),
        "rms_mid_vox": rms_mid, "rms_thickness_vox": rms_th,
        # aliases kept for older callers: top = mid-plane fit, bot = thickness fit
        "tiles_used_top": int(kmid.sum()), "tiles_used_bot": int(kth.sum()), "rms_top_vox": rms_mid, "rms_bot_vox": rms_th,
        "global_band": [int(g0 * down), int(g1 * down)], "global_contrast": gcon,
        "global_round": next((r["round"] for r in runs_info if r.get("accepted")), None),
        "runs": [dict(r, z=[v * down for v in r["z"]]) if r.get("z") else r for r in runs_info],
        "thickness_vox_median": float(np.median(z_bot - z_top)),
        "thickness_vox_range": [float((z_bot - z_top).min()), float((z_bot - z_top).max())],
        # positioning numbers: slope per voxel of the mid-plane (normalised slopes divided by the extent)
        "angle_offset_deg": float(np.degrees(np.arctan(mid[1] / nx))),
        "x_axis_tilt_deg": float(np.degrees(np.arctan(mid[2] / ny))),
        "z_shift_vox": float(mid[0] - (nz - 1) / 2),
    }


def flat_values(vol, zmap, r):
    """Values at depth zmap + r in every column (nearest voxel; outside the volume -> nan)."""
    ny, nz, nx = vol.shape
    z = np.rint(zmap + r).astype(np.int64)
    ok = (z >= 0) & (z < nz)
    out = np.full(zmap.shape, np.nan, np.float32)
    yy, xx = np.nonzero(ok)
    out[ok] = vol[yy, z[ok], xx]
    return out


def flat_slab(vol, zmap, r, half=5):
    """Mean of the slices zmap + r - half .. zmap + r + half (flattened XY slab)."""
    acc = np.zeros(zmap.shape, np.float64)
    n = np.zeros(zmap.shape, np.float64)
    for d in range(-half, half + 1):
        v = flat_values(vol, zmap, r + d)
        ok = np.isfinite(v)
        acc[ok] += v[ok]
        n[ok] += 1
    return np.where(n > 0, acc / np.maximum(n, 1), np.nan).astype(np.float32)


def flat_profiles(vol, zmap, offsets, sigma_px, ys, xs, step=2):
    """Per depth offset: 0.1 / 5 / 50th percentiles of 2D low-passed density in flattened coordinates."""
    out = np.zeros((len(offsets), 3), np.float32)
    for i, r in enumerate(offsets):
        img = flat_slab(vol, zmap, r, half=1)[ys, xs]
        if not np.isfinite(img).any():
            out[i] = np.nan
            continue
        img = np.where(np.isfinite(img), img, np.nanmedian(img))
        lp = ndimage.gaussian_filter(img, sigma_px)[::step, ::step]
        out[i] = np.percentile(lp, [0.1, 5, 50])
    return out


def segment_layer(img, sigma_px, min_area_px=20):
    """Dense material in a flattened layer slab. Bimodal histogram -> threshold at the valley between the two
    main modes; otherwise dense cores darker than mode - 3 * sigma_up (sigma from the half above the mode)."""
    lp = ndimage.gaussian_filter(np.where(np.isfinite(img), img, np.nanmedian(img)), sigma_px)
    hist, edges = np.histogram(lp, bins=256)
    centres = 0.5 * (edges[1:] + edges[:-1])
    hs = ndimage.gaussian_filter1d(hist.astype(float), 3)
    pk, prop = find_peaks(hs, prominence=0.05 * hs.max())
    rule, thr = None, None
    if len(pk) >= 2:
        top2 = sorted(pk[np.argsort(prop["prominences"])[-2:]])
        lo, hi = top2
        v = lo + int(np.argmin(hs[lo:hi + 1]))
        if hs[v] < 0.8 * min(hs[lo], hs[hi]):
            rule, thr = "valley", float(centres[v])
    if rule is None:
        mode = float(centres[np.argmax(hs)])
        up = lp[lp > mode] - mode
        sig = 1.4826 * float(np.median(up)) if up.size else float(np.std(lp))
        rule, thr = "core (mode - 3 sigma_up)", mode - 3 * sig
    mask = lp < thr
    mask = ndimage.binary_opening(mask, np.ones((2, 2), bool))
    lab, n = ndimage.label(mask, structure=np.ones((3, 3), bool))
    if n:
        sizes = np.bincount(lab.ravel())
        small = sizes < min_area_px
        small[0] = False
        mask[small[lab]] = False
    return mask, {"rule": rule, "threshold": thr, "hist": hs.tolist(), "hist_x": centres.tolist()}


def components(mask, px_nm):
    lab, n = ndimage.label(mask, structure=np.ones((3, 3), bool))
    if not n:
        return np.array([]), 0.0
    sizes = np.bincount(lab.ravel())[1:]
    eqd = 2 * np.sqrt(sizes * px_nm ** 2 / np.pi)
    return eqd, float(sizes.max() / sizes.sum())


def missing_wedge_elongation(max_tilt_deg):
    """Radermacher's elongation factor along Z for a symmetric tilt range of +-max_tilt."""
    a = np.radians(max_tilt_deg)
    return float(np.sqrt((a + np.sin(a) * np.cos(a)) / (a - np.sin(a) * np.cos(a))))
