# Command line

Everything the app does is available from a command prompt, for scripting or
for running on a processing computer without a display. Use the environment's
Python from the pyPrep folder:

```bat
cd path\to\pyPrep
env\python.exe -m pyprep <command> ...
```

## `pyprep scan` — check a session

Lists the tilt series found, their movie format, and any missing fraction files.

```bat
env\python.exe -m pyprep scan "D:\Sessions\2026-09-30"
env\python.exe -m pyprep scan "D:\Sessions\2026-09-30" --frames-dir "\\server\DoseFractions\2026-09-30"
```

```
Position_9_2: 47 tilts (-46.1 to 45.9 deg), 1 missing, pixel 3.300 A, 300 kV, tilt axis 86.2
    movies: MRC 4 frames 5760x4092
    missing: tilt 047 (-46.11 deg) \\192.168.12.2\DoseFractions\...\Position_9_2_047_-46.10_..._Fractions.mrc
```

| Option | |
|---|---|
| `inputs` | `.mdoc` files and/or folders containing them |
| `-r`, `--recursive` | also search subfolders |
| `--frames-dir DIR` | folder with the fraction files, if not next to the `.mdoc` |

## `pyprep run` — process tilt series

```bat
rem everything with defaults: bin 1 + bin 4 stacks, patch-tracking tomogram at bin 4
env\python.exe -m pyprep run "D:\Sessions\2026-09-30" -o "D:\Sessions\2026-09-30\pyPrep"

rem stacks only, int16, with even/odd half-sums for denoising
env\python.exe -m pyprep run "D:\Sessions\2026-09-30" -o out --no-reconstruct --dtype int16 --even-odd

rem gold-fiducial reconstruction with a fixed 250 nm thickness
env\python.exe -m pyprep run session -o out --preset gold --fixed-thickness --thickness 250

rem Falcon EER with the EPU gain reference, 12 fractions per tilt
env\python.exe -m pyprep run session -o out --gain "D:\Sessions\x\gain.gain" --eer-fractions 12

rem reuse settings saved from the app (File > Save settings), overriding one option
env\python.exe -m pyprep run session -o out --settings my_settings.json --force
```

Series whose outputs are already complete are skipped (use `--force` to redo).
A warning is printed if the output drive looks too small for the batch.
The exit code is 0 if every series succeeded, 1 otherwise.

### Options

**Input**

| Option | Default | |
|---|---|---|
| `inputs` | | `.mdoc` files and/or folders containing them |
| `-o`, `--output DIR` | required | output folder; one subfolder per tilt series |
| `-r`, `--recursive` | | search folders recursively |
| `--frames-dir DIR` | next to the mdoc | folder with the fraction files |
| `--settings FILE` | | JSON settings file (e.g. saved from the app); other options override it |
| `--exclude ANGLE ...` | | tilt angles to leave out (±0.5°) |

**Movies (EER, gain)**

| Option | Default | |
|---|---|---|
| `--eer-fractions N` | 10 | sum each tilt's EER frames into N fractions |
| `--eer-group N` | | instead: N EER frames per fraction |
| `--eer-upsampling {1,2}` | 1 | 1 = physical pixels (4K), 2 = 8K super-resolution, binned back |
| `--gain FILE` | | gain reference (EPU `.gain`, MRC or TIFF) |
| `--gain-mode {auto,multiply,divide}` | auto | auto = divide for `.gain`/EER, multiply otherwise |
| `--gain-rotate {0,90,180,270}` | 0 | rotate the reference counter-clockwise |
| `--gain-flip {none,x,y}` | none | flip after rotating (x = left-right, y = up-down) |
| `--invert-y {auto,yes,no}` | auto | invert the aligned images in Y; auto = for Tomography 5 data, which stores images upside down (otherwise its tomograms are mirror images) |

**Frame alignment**

| Option | Default | |
|---|---|---|
| `--align-bin N` | 4 | Fourier binning used to measure shifts |
| `--bfactor B` | 500 | B-factor (Å²) applied to the cross-correlation |
| `--cpu` | | run on the CPU instead of the GPU |

**Outputs**

| Option | Default | |
|---|---|---|
| `--bin N ...` | `1 4` | binning levels to write |
| `--no-aligned` | | do not write the aligned-sum stack |
| `--even-odd` | | also write even/odd half-sum stacks |
| `--dose-weighted` | | also write a dose-weighted stack |
| `--dtype {float32,int16}` | float32 | |

**Reconstruction (IMOD batchruntomo)**

| Option | Default | |
|---|---|---|
| `--reconstruct` / `--no-reconstruct` | reconstruct | |
| `--preset {patch,gold}` | patch | patch tracking or gold fiducials |
| `--recon-bin N` | 4 | which binned stack to reconstruct |
| `--thickness NM` | 200 | fallback thickness (auto positioning) or fixed thickness, in nm |
| `--fixed-thickness` | | skip IMOD positioning and use `--thickness` |

**CTF and deconvolution**

| Option | Default | |
|---|---|---|
| `--no-ctf` | | skip per-tilt CTF estimation |
| `--cs MM` | 2.7 | spherical aberration |
| `--amp-contrast A` | 0.07 | amplitude contrast |
| `--ctf-range LOW HIGH` | `30 8` | fit range in Å |
| `--defocus-range MIN MAX` | `0.5 12` | defocus search range in µm |
| `--no-ctf-correct` | | do not phase-flip in IMOD |
| `--handedness {auto,keep,flip}` | auto | direction of the tilt axis: auto = rotate it by 180° when the CTF defocus gradient disagrees with it |
| `--no-deconv` | | do not write the deconvolved tomogram |
| `--deconv-strength S` | 1.0 | deconvolution strength |
| `--deconv-falloff F` | 1.0 | SNR falloff |
| `--remove-redeposition` | | also write `<series>_rec_redep.mrc` with the dense surface layers removed (needs ASTRA: `pip install astra-toolbox`) |

**Other**

| Option | |
|---|---|
| `--force` | reprocess series that are already complete |
| `-q`, `--quiet` | print only progress and a summary per series |

## `pyprep ctf` — (re-)estimate the CTF of processed series

```bat
env\python.exe -m pyprep ctf "D:\Sessions\2026-09-30" -o "D:\Sessions\2026-09-30\pyPrep"
```

Re-estimates the per-tilt defocus from each series' bin 1 aligned stack, with
the same inputs as `run` and the CTF options above. Use it for series
processed before CTF estimation existed, or with a different fit range. It
writes `<series>.defocus` and `_ctf.npz` and updates `_pyprep.json`. To apply
the result, redo the reconstruction: *Redo reconstruction* on the Results page,
or `run --force`.

## `pyprep deconv` — deconvolve any tomogram

```bat
env\python.exe -m pyprep deconv "out\TS_01\imod_bin4\TS_01_rec.mrc" --defocus 2.5
env\python.exe -m pyprep deconv tomo.mrc --defocus 3.1 --pixel 10.2 --strength 0.8 --phase-flipped
```

Writes `<input>_deconv.mrc` (or `-o FILE`). The pixel size is read from the
MRC header unless `--pixel` is given. `--kv`, `--cs`, `--amp-contrast`,
`--strength`, `--falloff` and `--highpass` (fraction of Nyquist) set the
filter. `--phase-flipped` tells it the tilt series was CTF-corrected.

## `pyprep redep` — remove redeposition in a reconstruction folder

```bat
env\python.exe -m pyprep redep "out\TS_01\imod_bin4"
```

Runs redeposition removal (see [How it works](how-it-works.md#redeposition-removal))
on an existing reconstruction: it writes `<series>_ali_redep.mrc`,
`<series>_redep_transmission.mrc` and `<series>_rec_redep.mrc` and records them in
`pyprep_recon.json`, as `run --remove-redeposition` does. The root name comes
from the folder's `.edf`; `--root NAME` sets it. Needs the ASTRA Toolbox and a
CUDA GPU.

## `pyprep gallery` — thumbnails and contact sheet

```bat
env\python.exe -m pyprep gallery "D:\Sessions\2026-09-30\pyPrep"
```

Makes any missing thumbnails (average of the 10 central tomogram slices) and
writes `gallery.png`, a grid of all series with their names, in the output
folder. `--sheet FILE` chooses another file, `--tile N` the thumbnail size.

## `pyprep export` — TIFF for Fiji / segmentation

```bat
env\python.exe -m pyprep export "out\TS_01\imod_bin4\TS_01_rec.mrc" TS_01_rec.tif
env\python.exe -m pyprep export stack.mrc stack.tif --bin 2 --bits16
```

Writes an ImageJ TIFF scaled between the 0.5 and 99.5 percentiles, with the
pixel size in nm. `--bin N` block-bins, `--bits16` writes 16-bit.

## Settings file

Every option has an entry in the JSON settings file. *File > Save settings* in
the app writes one; this is the file with all defaults:

```json
{
  "input": {
    "eer_fractions": 10,
    "eer_group": 0,
    "eer_upsampling": 1,
    "gain_path": "",
    "gain_mode": "auto",
    "gain_rotate": 0,
    "gain_flip": "none"
  },
  "motion": {
    "align_bin": 4,
    "bfactor": 500.0,
    "max_iterations": 10,
    "tolerance": 0.1,
    "max_shift": 80.0,
    "mask_axes": true,
    "group": 1,
    "upsample": 20
  },
  "output": {
    "bin_levels": [1, 4],
    "aligned": true,
    "even_odd": false,
    "dose_weighted": false,
    "dtype": "float32",
    "exclude_angles": []
  },
  "recon": {
    "enabled": true,
    "preset": "patch",
    "bin": 4,
    "use_dose_weighted": false,
    "patch_size_nm": 400.0,
    "patch_overlap": 0.6,
    "gold_size_nm": 10.0,
    "gold_beads": 20,
    "erase_gold": true,
    "positioning": "auto",
    "positioning_thickness_nm": 330.0,
    "thickness_nm": 200.0,
    "sirt_like_iterations": 6,
    "remove_xrays": true,
    "cpus": 11,
    "use_gpu": false,
    "extra_directives": ""
  },
  "frames_dir": null,
  "use_gpu": true,
  "gpu_id": 0,
  "skip_existing": true,
  "default_dose": null
}
```

Missing entries take their default, so a settings file only needs the values
you want to change. Meanings are described in the [user guide](user-guide.md);
`max_shift` and `tolerance` are in full-resolution pixels, `upsample` is the
sub-pixel refinement factor of the shift search, and `extra_directives` holds
batchruntomo `key = value` lines separated by newlines (`\n`).

## Validation script

`scripts\validate_alignment.py` measures whether frame alignment improves a
dataset: it compares the Fourier ring correlation between even- and odd-frame
half-sums without and with pyPrep's shifts, for a sample of tilts.

```bat
env\python.exe scripts\validate_alignment.py "D:\Sessions\x\TS_01.mdoc" --tilts 8
```
