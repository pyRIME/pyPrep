# Output files

For every tilt series, pyPrep writes a folder `<output>\<series>\`:

```
<output>\
├── pyprep_settings_last_run.json       settings of the most recent batch
├── pyprep_selection.json               series ticked "keep" in the Gallery
├── gallery.png                         contact sheet (Gallery / `pyprep gallery`)
└── TS_01\
    ├── TS_01.mrc                       aligned sum, full resolution
    ├── TS_01.rawtlt                    tilt angles, one per line, in stack order
    ├── TS_01.mrc.mdoc                  metadata in stack order (angles, dose, tilt axis, pixel size)
    ├── TS_01_bin4.mrc                  aligned sum, Fourier-binned by 4
    ├── TS_01_bin4.rawtlt / .mrc.mdoc
    ├── TS_01_EVN*.mrc, TS_01_ODD*.mrc  even/odd half-sums        (if selected)
    ├── TS_01_DW*.mrc                   dose-weighted sums        (if selected)
    ├── TS_01.defocus                   per-tilt defocus for IMOD (CTF estimation)
    ├── TS_01_ctf.npz                   spectra and fitted CTF models (Results page plot)
    ├── TS_01_motion.csv                per-frame shifts of every tilt
    ├── TS_01_pyprep.json               full record: settings, per-tilt results, outputs, reconstruction
    ├── TS_01_pyprep.log                log of alignment and reconstruction
    ├── TS_01_thumb.png                 gallery thumbnail (10 central tomogram slices)
    └── imod_bin4\                      batchruntomo / etomo project (if reconstruction is on)
        ├── TS_01.edf                   open this in etomo
        ├── TS_01_rec.mrc               final tomogram (trimmed, rotated so Z is the slice axis)
        ├── TS_01_rec_deconv.mrc        deconvolved copy (if deconvolution is on)
        ├── TS_01_full_rec.mrc          untrimmed reconstruction
        ├── pyprep_batchruntomo.adoc    the directives used
        ├── pyprep_recon.json           reconstruction record
        └── ... the usual etomo files (.com, .log, .xf, .tlt, _ali.mrc, ...)
```

## Stacks (`.mrc`)

- One section per tilt, **sorted by tilt angle** (most negative first) — not in
  acquisition order. Missing and excluded tilts are left out.
- MRC2014, mode 2 (float32) or mode 1 (int16), pixel size in the header, no
  extended header. IMOD `header` and `extracttilts` read them directly.
- Values are the summed detector counts (for Tomo5 K3 fractions, ~32 units per
  electron; for EER, electrons, divided by the gain reference if one is given).
- Binned stacks are made by Fourier cropping (no aliasing); their pixel size is
  the full-resolution pixel size times the binning.

## `.rawtlt`

The tilt angles of the stack's sections, in stack order — what etomo uses as
the raw tilt angles.

## `.mrc.mdoc`

A SerialEM-style metadata file whose `[ZValue]` sections follow the **stack
order**. Each section keeps the original mdoc entries of that tilt (stage
position, defocus, `ExposureDose`, `PriorRecordDose`, `SubFramePath`, …) with
`PixelSpacing`, `Binning` and `MinMaxMean` updated, plus `AcquisitionOrder`
(the tilt's position in the original collection order). The title carries the
tilt-axis angle in the form IMOD reads (`Tilt axis angle = 86.20`).

Because it is named `<stack>.mrc.mdoc`, IMOD's `extracttilts` finds it
automatically as the stack's metadata. For dose weighting in etomo (Final
Aligned Stack > Filter / mtffilter), select this file as the dose file.

## `_motion.csv`

| column | |
|---|---|
| `z` | section in the stack |
| `acquisition` | position in the original collection order (0-based) |
| `angle` | tilt angle |
| `frame` | frame / fraction number |
| `dx_px`, `dy_px` | shift applied to that frame, in full-resolution (physical) pixels |

## `_pyprep.json`

Everything about the run: pyPrep version, settings, GPU, per-tilt results
(file, frame count, dose, prior dose, shifts, correlation scores, iterations,
drift in Å, mean counts, exposure time, saturated-pixel fraction, QC flags and
reasons), the QC summary (flagged tilts, intensity fit, saturation), the tilts
used (`used_tilts`, acquisition indices), outputs (path, binning, pixel size,
size), missing tilts and — after reconstruction — the reconstruction status and
tomogram path. The Results page
and the *skip completed steps* option read it.

## CTF (`.defocus`, `_ctf.npz`)

- `<series>.defocus`: the defocus of every view in IMOD's format (nm,
  underfocus positive, views numbered from 1), for ctfphaseflip or
  ctfplotter.
  - If the defocus gradient runs opposite to IMOD's convention, the file starts
    with the version-3 header `16 0 0. 0. 0 3`. Use `-invert` with IMOD's
    programs then.
  - Otherwise the first line ends with the version number `2`.
- `<series>_ctf.npz`: for each tilt, the flattened spectrum and the fitted
  model, as plotted on the Results page (numpy arrays `k`, `data`, `model`,
  `angles`, `defocus_um`).
- The `ctf` entry in `_pyprep.json` holds:
  - per-tilt defocus, fit score and fit resolution;
  - the series defocus (`defocus_um`, used for deconvolution);
  - the handedness and its confidence;
  - the specimen tilt offset (`tilt_offset_deg`).
- The `image_frame` entry in `_pyprep.json`:
  - whether the aligned images were inverted in Y (`invert_y`; Tomography 5
    data by default);
  - the tilt axis of the stacks as written (`tilt_axis`; 180° − the recorded
    one when inverted) and the one recorded in the mdoc (`recorded_tilt_axis`);
  - the reason.

  Frame shifts (`shifts_px`, `_motion.csv`) stay in raw-movie coordinates.
- The `handedness` entry in `_pyprep.json`:
  - whether the CTF check rotated the tilt axis by 180° (`flipped`);
  - the tilt axis used (`tilt_axis`), the stacks' axis before the check
    (`stack_tilt_axis`), the one recorded in the mdoc, and `invert_y`;
  - the reason.

  The stacks' `.mrc.mdoc` titles carry the axis used.

## Reconstruction folder (`imod_bin<N>`)

A complete etomo project created by batchruntomo from a copy of the binned
stack (named `<series>.mrc` inside the folder). You can open it in etomo and
change or redo any step. The final tomogram is `<series>_rec.mrc`. It is
trimmed and rotated around X so that sections are Z slices (IMOD's default for
batchruntomo).

With CTF correction, `<series>.defocus` is copied in, re-labelled with the
aligned tilt angles, and used by `ctfcorrection.com`. With deconvolution,
`<series>_rec_deconv.mrc` is written next to `<series>_rec.mrc` (float32,
same pixel size); its parameters are in `pyprep_recon.json` under `deconv`.

*Position tomogram* adds:

| File | Contents |
|---|---|
| `tilt.com.pyprep_orig` | `tilt.com` as batchruntomo wrote it (*Restore original* copies it back) |
| `pyprep_pos_ali_bin2.mrc` | aligned stack binned 2×, input for the trial tomogram |
| `pyprep_pos_trial.mrc` | the thick trial tomogram shown in the positioning window |

The two `pyprep_pos_*` files can be deleted at any time.
