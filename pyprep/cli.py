"""Command-line interface: ``pyprep scan``, ``run``, ``ctf``, ``deconv``, ``redep``, ``gallery``, ``export``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .settings import ProcessingSettings


def add_ctf_args(p) -> None:
    p.add_argument("--cs", type=float, help="spherical aberration in mm (default 2.7)")
    p.add_argument("--amp-contrast", type=float, help="amplitude contrast (default 0.07)")
    p.add_argument("--ctf-range", type=float, nargs=2, metavar=("LOW_A", "HIGH_A"),
                   help="CTF fit range in Angstrom (default 30 8)")
    p.add_argument("--defocus-range", type=float, nargs=2, metavar=("MIN_UM", "MAX_UM"),
                   help="defocus search range in micrometres (default 0.5 12)")


def _collect(inputs, recursive, frames_dir, default_dose):
    from .tiltseries import find_mdocs, load_tilt_series
    mdocs = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            mdocs += find_mdocs(p, recursive)
        elif p.suffix.lower() == ".mdoc":
            mdocs.append(p)
        else:
            print(f"Skipping {p}: not a folder or .mdoc", file=sys.stderr)
    return [load_tilt_series(m, frames_dir, default_dose) for m in mdocs]


def cmd_scan(args) -> int:
    series = _collect(args.inputs, args.recursive, args.frames_dir, None)
    if not series:
        print("No tilt-series mdoc files found.")
        return 1
    from .io.frames import open_movie
    for s in series:
        print(s.summary())
        if s.usable:
            try:
                print("    movies: " + open_movie(s.usable[0].frame_path).describe())
            except Exception as e:
                print(f"    movies: could not open {s.usable[0].frame_path.name}: {e}")
        for t in s.missing:
            print(f"    missing: tilt {t.zvalue + 1:03d} ({t.angle:+.2f} deg) {t.section.get('SubFramePath', '')}")
    return 0


def cmd_run(args) -> int:
    from .pipeline import run_series

    settings = ProcessingSettings.load(args.settings) if args.settings else ProcessingSettings()
    o, m = settings.output, settings.motion
    if args.bin:
        o.bin_levels = args.bin
    if args.no_aligned:
        o.aligned = False
    if args.even_odd:
        o.even_odd = True
    if args.dose_weighted:
        o.dose_weighted = True
    if args.dtype:
        o.dtype = args.dtype
    if args.exclude:
        o.exclude_angles = args.exclude
    if args.align_bin:
        m.align_bin = args.align_bin
    if args.bfactor is not None:
        m.bfactor = args.bfactor
    if args.frames_dir:
        settings.frames_dir = args.frames_dir
    if args.cpu:
        settings.use_gpu = False
    if args.force:
        settings.skip_existing = False
    inp = settings.input
    if args.eer_fractions:
        inp.eer_fractions, inp.eer_group = args.eer_fractions, 0
    if args.eer_group:
        inp.eer_group = args.eer_group
    if args.eer_upsampling:
        inp.eer_upsampling = args.eer_upsampling
    if args.gain:
        inp.gain_path = args.gain
    if args.gain_mode:
        inp.gain_mode = args.gain_mode
    if args.gain_rotate is not None:
        inp.gain_rotate = args.gain_rotate
    if args.gain_flip:
        inp.gain_flip = args.gain_flip
    if args.invert_y:
        inp.invert_y = args.invert_y
    r = settings.recon
    if args.reconstruct:
        r.enabled = True
    if args.no_reconstruct:
        r.enabled = False
    if args.preset:
        r.preset = args.preset
    if args.recon_bin:
        r.bin = args.recon_bin
    if args.thickness:
        r.thickness_nm = args.thickness
    if args.fixed_thickness:
        r.positioning = "fixed"
    _apply_ctf_args(settings, args)
    if args.no_ctf_correct:
        r.ctf_correct = False
    if args.handedness:
        r.handedness = args.handedness
    if args.no_deconv:
        r.deconvolve = False
    if args.deconv_strength is not None:
        r.deconv_strength = args.deconv_strength
    if args.deconv_falloff is not None:
        r.deconv_falloff = args.deconv_falloff
    if args.remove_redeposition:
        from .redep import astra_status
        ok, msg = astra_status()
        if not ok:
            print(f"--remove-redeposition: {msg}")
            return 1
        r.remove_redeposition = True

    series = _collect(args.inputs, args.recursive, settings.frames_dir, settings.default_dose)
    if not series:
        print("No tilt-series mdoc files found.")
        return 1
    out_root = Path(args.output)
    try:
        import shutil
        from .pipeline import estimate_output_bytes, is_complete
        out_root.mkdir(parents=True, exist_ok=True)
        need = sum(estimate_output_bytes(s, settings) for s in series
                   if not (settings.skip_existing and is_complete(s, settings, out_root)))
        free = shutil.disk_usage(out_root).free
        if need > free:
            print(f"WARNING: this batch may need ~{need / 1e9:.1f} GB but only {free / 1e9:.1f} GB is free")
    except OSError:
        pass
    failures = 0
    for i, s in enumerate(series, 1):
        print(f"[{i}/{len(series)}] {s.summary()}")

        def progress(done, total, msg):
            print(f"\r    {msg:<40s}", end="", flush=True)

        def say(line):
            if not args.quiet:
                print("\r    " + line)

        try:
            res = run_series(s, settings, out_root, progress=progress, log_callback=say)
            print(f"\r    stacks: {res['stacks']}, tomogram: {res['recon'] or '-'} "
                  f"({res['seconds']} s) -> {out_root / s.name}")
            if res["stacks"] not in ("complete", "skipped") or res["recon"] in ("failed", "cancelled"):
                failures += 1
        except Exception as e:  # keep going with the rest of the batch
            failures += 1
            print(f"\r    FAILED: {e}")
    return 1 if failures else 0


def _apply_ctf_args(settings, args) -> None:
    c = settings.ctf
    if getattr(args, "no_ctf", False):
        c.enabled = False
    if args.cs is not None:
        c.cs_mm = args.cs
    if args.amp_contrast is not None:
        c.amplitude_contrast = args.amp_contrast
    if args.ctf_range:
        c.min_res_A, c.max_res_A = max(args.ctf_range), min(args.ctf_range)
    if args.defocus_range:
        c.defocus_min_um, c.defocus_max_um = min(args.defocus_range), max(args.defocus_range)


def cmd_ctf(args) -> int:
    from .pipeline import ensure_ctf, result_json_path
    settings = ProcessingSettings.load(args.settings) if args.settings else ProcessingSettings()
    if args.frames_dir:
        settings.frames_dir = args.frames_dir
    if args.cpu:
        settings.use_gpu = False
    _apply_ctf_args(settings, args)
    settings.ctf.enabled = True
    series = _collect(args.inputs, args.recursive, settings.frames_dir, settings.default_dose)
    out_root = Path(args.output)
    failures = 0
    for s in series:
        j = result_json_path(s, out_root / s.name)
        if not j.exists():
            print(f"{s.name}: no pyPrep results in {out_root / s.name}")
            failures += 1
            continue
        info = json.loads(j.read_text())
        info.pop("ctf", None)                        # re-estimate
        j.write_text(json.dumps(info, indent=1))
        rec = ensure_ctf(s, settings, out_root, log_callback=None if args.quiet else print)
        if rec is None:
            failures += 1
            continue
        print(f"{s.name}: defocus {rec['defocus_um']:.2f} um, handedness "
              f"{'IMOD' if rec['handedness'] > 0 else 'inverted'} -> {rec['defocus_file']}")
    return 1 if failures else 0


def cmd_deconv(args) -> int:
    from . import deconv, gpu
    from .io import mrc
    src = Path(args.input)
    pixel = args.pixel or mrc.read_header(src).pixel_size
    if not pixel:
        print("The MRC header has no pixel size; give --pixel")
        return 1
    p = deconv.DeconvParams(pixel_A=pixel, defocus_um=args.defocus, kv=args.kv, cs_mm=args.cs,
                            amplitude=args.amp_contrast, strength=args.strength, falloff=args.falloff,
                            highpass_nyquist=args.highpass, phase_flipped=args.phase_flipped)
    dev = gpu.select_device(not args.cpu)
    deconv.deconvolve_file(src, Path(args.output) if args.output else deconv.deconv_name(src), p, dev, print)
    return 0


def cmd_redep(args) -> int:
    """Redeposition removal in an existing reconstruction folder (``imod_bin<N>``); records it in its
    pyprep_recon.json like a run with --remove-redeposition."""
    import json
    from . import redep
    recon_dir = Path(args.recon_dir)
    root = args.root or next((p.stem for p in recon_dir.glob("*.edf")), None)
    if root is None or not (recon_dir / f"{root}_ali.mrc").exists():
        print(f"No aligned stack <root>_ali.mrc in {recon_dir}; give --root")
        return 1
    try:
        r = redep.remove_redeposition(recon_dir, root, print)
    except RuntimeError as e:
        print(f"Redeposition removal failed: {e}")
        return 1
    j = recon_dir / "pyprep_recon.json"
    try:
        rec = json.loads(j.read_text())
    except (OSError, ValueError):
        rec = {}
    rec["redeposition"], rec["redep_tomogram"] = r, r["tomogram"]
    j.write_text(json.dumps(rec, indent=1))
    print(f"{root}: {Path(r['tomogram']).name} ({r['seconds']:.0f} s)")
    return 0


def cmd_export(args) -> int:
    from .export import export_tiff
    out = export_tiff(args.input, args.output, bin_factor=args.bin, bits=16 if args.bits16 else 8,
                      progress=lambda d, n: print(f"\r  {d}/{n} sections", end="", flush=True))
    print(f"\r  wrote {out}")
    return 0


def cmd_gallery(args) -> int:
    from .thumbs import contact_sheet, gallery_entries, load_selection
    entries = gallery_entries(args.output)
    if not entries:
        print("No processed tilt series found in", args.output)
        return 1
    for e in entries:
        print(f"  {e['series']:30s} {e['source'] or 'no image':10s} {e['tilts']:3d} tilts"
              + (f", {e['flagged']} flagged" if e['flagged'] else ""))
    sheet = contact_sheet(entries, args.sheet or Path(args.output) / "gallery.png", tile=args.tile,
                          keep=load_selection(args.output))
    print("Contact sheet:", sheet)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pyprep", description="Cryo-ET tilt-series preparation")
    ap.add_argument("--version", action="version", version=f"pyPrep {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("inputs", nargs="+", help="mdoc files and/or folders containing them")
        p.add_argument("-r", "--recursive", action="store_true", help="search folders recursively")
        p.add_argument("--frames-dir", help="folder with the fraction files (default: next to the mdoc)")

    p = sub.add_parser("scan", help="list tilt series and check that all fraction files are present")
    common(p)
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser("run", help="motion-correct, build tilt-series stacks, optionally reconstruct")
    common(p)
    p.add_argument("-o", "--output", required=True, help="output folder (one subfolder per series)")
    p.add_argument("--settings", help="JSON settings file (e.g. saved from the GUI)")
    p.add_argument("--bin", type=int, nargs="+", help="output binning levels, e.g. --bin 1 4")
    p.add_argument("--no-aligned", action="store_true", help="skip the full aligned-sum stack")
    p.add_argument("--even-odd", action="store_true", help="also write even/odd half-sum stacks")
    p.add_argument("--dose-weighted", action="store_true", help="also write a dose-weighted stack")
    p.add_argument("--dtype", choices=["float32", "int16"])
    p.add_argument("--exclude", type=float, nargs="+", metavar="ANGLE", help="tilt angles to leave out")
    p.add_argument("--eer-fractions", type=int, help="EER: sum each tilt into this many fractions (default 10)")
    p.add_argument("--eer-group", type=int, help="EER: fixed number of EER frames per fraction instead")
    p.add_argument("--eer-upsampling", type=int, choices=[1, 2], help="EER: 1 = 4K, 2 = 8K super-resolution")
    p.add_argument("--gain", help="gain reference (EPU .gain, MRC or TIFF)")
    p.add_argument("--gain-mode", choices=["auto", "multiply", "divide"])
    p.add_argument("--gain-rotate", type=int, choices=[0, 90, 180, 270], help="rotate gain CCW (degrees)")
    p.add_argument("--gain-flip", choices=["none", "x", "y"], help="flip gain after rotation")
    p.add_argument("--invert-y", choices=["auto", "yes", "no"],
                   help="invert the aligned images in Y; auto (default): for Tomography 5 data, which stores "
                        "images upside down (without it, Tomo5 tomograms are mirror images)")
    p.add_argument("--align-bin", type=int, help="binning used for measuring shifts (default 4)")
    p.add_argument("--bfactor", type=float, help="B-factor for alignment in A^2 (default 500)")
    p.add_argument("--cpu", action="store_true", help="do not use the GPU")
    p.add_argument("--reconstruct", action="store_true", help="run IMOD batchruntomo after alignment (default)")
    p.add_argument("--no-reconstruct", action="store_true", help="stacks only, no IMOD reconstruction")
    p.add_argument("--preset", choices=["patch", "gold"], help="batchruntomo preset (default patch)")
    p.add_argument("--recon-bin", type=int, help="binning of the stack to reconstruct (default 4)")
    p.add_argument("--thickness", type=float, help="reconstruction (fallback) thickness in nm")
    p.add_argument("--fixed-thickness", action="store_true", help="skip IMOD positioning, use --thickness")
    p.add_argument("--no-ctf", action="store_true", help="skip per-tilt CTF estimation")
    add_ctf_args(p)
    p.add_argument("--no-ctf-correct", action="store_true", help="do not phase-flip (IMOD ctfphaseflip)")
    p.add_argument("--handedness", choices=["auto", "keep", "flip"],
                   help="tilt axis direction. auto (default): rotate it by 180 deg when the CTF defocus "
                        "gradient disagrees with it; keep: as recorded; flip: always rotate")
    p.add_argument("--no-deconv", action="store_true", help="do not write a deconvolved tomogram")
    p.add_argument("--deconv-strength", type=float, help="deconvolution strength (default 1.0)")
    p.add_argument("--deconv-falloff", type=float, help="deconvolution SNR falloff (default 1.0)")
    p.add_argument("--remove-redeposition", action="store_true",
                   help="also write a tomogram with the dense surface layers (FIB redeposition, Pt/GIS grains) "
                        "removed, <series>_rec_redep.mrc (needs ASTRA: pip install astra-toolbox)")
    p.add_argument("--force", action="store_true", help="reprocess series that are already complete")
    p.add_argument("-q", "--quiet", action="store_true")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("gallery", help="thumbnails of every processed series + a contact sheet (gallery.png)")
    p.add_argument("output", help="pyPrep output folder (the one containing the series folders)")
    p.add_argument("--sheet", help="contact sheet file (default: <output>/gallery.png)")
    p.add_argument("--tile", type=int, default=256, help="thumbnail size in the contact sheet (pixels)")
    p.set_defaults(func=cmd_gallery)

    p = sub.add_parser("export", help="save an MRC stack/tomogram as an ImageJ TIFF (8-bit, pixel size in nm)")
    p.add_argument("input", help="MRC file")
    p.add_argument("output", help="TIFF file to write")
    p.add_argument("--bin", type=int, default=1, help="block-bin by this factor")
    p.add_argument("--bits16", action="store_true", help="16-bit instead of 8-bit")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("ctf", help="(re-)estimate per-tilt CTF of processed series (needs the bin 1 stack)")
    common(p)
    p.add_argument("-o", "--output", required=True, help="pyPrep output folder used for 'run'")
    p.add_argument("--settings", help="JSON settings file")
    p.add_argument("--cpu", action="store_true", help="do not use the GPU")
    add_ctf_args(p)
    p.add_argument("-q", "--quiet", action="store_true")
    p.set_defaults(func=cmd_ctf)

    p = sub.add_parser("deconv", help="deconvolve a tomogram (Wiener-like filter with the CTF)")
    p.add_argument("input", help="tomogram (MRC)")
    p.add_argument("--defocus", type=float, required=True, help="defocus in micrometres (underfocus positive)")
    p.add_argument("-o", "--output", help="output MRC (default: <input>_deconv.mrc)")
    p.add_argument("--pixel", type=float, help="pixel size in A (default: from the MRC header)")
    p.add_argument("--kv", type=float, default=300.0)
    p.add_argument("--cs", type=float, default=2.7)
    p.add_argument("--amp-contrast", type=float, default=0.07)
    p.add_argument("--strength", type=float, default=1.0)
    p.add_argument("--falloff", type=float, default=1.0)
    p.add_argument("--highpass", type=float, default=0.02, help="high-pass cut-on, fraction of Nyquist")
    p.add_argument("--phase-flipped", action="store_true", help="the tilt series was CTF phase-flipped")
    p.add_argument("--cpu", action="store_true", help="do not use the GPU")
    p.set_defaults(func=cmd_deconv)

    p = sub.add_parser("redep", help="remove redeposition in an existing reconstruction folder (needs ASTRA)")
    p.add_argument("recon_dir", help="the series' reconstruction folder (imod_bin4)")
    p.add_argument("--root", help="root name of the etomo project (default: from its .edf)")
    p.set_defaults(func=cmd_redep)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
