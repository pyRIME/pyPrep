# How pyPrep works

pyPrep is written in Python. The numerical work runs on the GPU with PyTorch
(CUDA), or on the CPU with the same code. MRC files, mdoc files and the frame
alignment are implemented in pyPrep itself. EER decoding uses `imagecodecs`,
and reconstruction uses IMOD.

```
 .mdoc + fraction files ──► match tilts ──► read frames ──► gain ──► align frames ──► sum ──► stacks
   (Tomo5 / SerialEM)        (per tilt)     (MRC/TIFF/EER)          (GPU)                    sorted by angle
                                                                                              + .rawtlt + .mdoc
                                                                                                    │
                                                     per-tilt CTF (GPU) ◄──────── sums     │
                                                                  │ .defocus                 │
                                                                  ▼                          │
                                                          IMOD batchruntomo ◄────────────────┘ (bin 4)
                                                          (ctfphaseflip)
                                                                  │
                                                        tomogram ──► deconvolved tomogram
                                                                  │
                                              (optional, ASTRA)   └──► redeposition removed
```

## Reading the data

- **mdoc**: both dialects are parsed. The tilt-axis angle comes from SerialEM's
  title line (`Tilt axis angle = …`) or from Tomo5's per-section
  `RotationAngle`. Tomo5's `TiltAxisAngle` header value is the same axis for
  the upright image, measured from X (176.73° there is 86.73° from Y in IMOD's
  convention); etomo does not use it.
- **Image orientation**: Tomography 5 stores its images upside down. IMOD
  reports its own stacks as "inverted in Y" (FEI extended header); the fraction
  files carry no such header but are stored the same way. For Tomo5 series
  pyPrep reverses the rows of every aligned image and mirrors the recorded axis
  with them (*a* → 180° − *a*, since IMOD measures the axis from Y). Stacks, CTF,
  alignment and tomograms are then in the physical orientation and follow
  IMOD's conventions (see *Handedness* below). Frame shifts in `_motion.csv`
  and `_pyprep.json` stay in the coordinates of the raw movies.
- **Fraction files** are matched to tilts by the file name in `SubFramePath`.
  The match is case-insensitive, because Tomo5 writes `Fractions` in the mdoc
  and `fractions` on disk. Tomo5-style acquisition numbers are a fallback.
- **MRC mode 0**: the MRC2014 standard says signed bytes, but cameras and IMOD
  treat mode 0 as unsigned. pyPrep follows IMOD: Tomo5's 8-bit K3 fractions
  have values up to 255.
- **Dose history**: `PriorRecordDose` when present, otherwise accumulated in
  collection order.

## Frame alignment (MotionCor-style, written from scratch)

For each tilt:

1. Each frame is padded to an FFT-friendly size. It is then Fourier-cropped to
   the *alignment binning* (default 4). This is equivalent to anti-aliased
   binning.
2. **Iterative reference alignment.** Each frame is cross-correlated against
   the sum of *all other* frames at their current shifts. Leaving the frame out
   of its own reference avoids the zero-shift bias caused by its own shot noise.
   The correlation is weighted by a B-factor low-pass, `exp(-B q²/4)`.
3. **Detector fixed-pattern noise.** The kx = 0 and ky = 0 Fourier axes are
   excluded from the correlation. Row and column offsets of the camera are
   identical in every frame. On the K3 data used for development they carry
   about 100 times more power than neighbouring frequencies, and they would pin
   every shift to zero. On synthetic data with such a pattern, alignment with
   the axes masked is accurate to about 0.1 px. Without masking it is off by
   more than 5 px.
4. **Sub-pixel peaks** are found with a matrix-multiply upsampled DFT. The first
   pass covers ±1.5 px in 0.1 px steps, and a second pass refines to 0.005
   binned px.
5. Shifts are re-centred to zero mean and iterated until no shift changes by
   more than the tolerance.
6. **Summation.** The shifts are applied to the *full-resolution* frames as
   Fourier phase ramps, so there is no interpolation blur. The shifted spectra
   are accumulated into every requested output: the full sum, the even and odd
   half-sums, and the dose-weighted sum. Each is inverse-transformed at every
   requested binning by Fourier cropping.

On a 5760 x 4092 K3 tilt with 4 fractions, alignment takes about 0.15 s and all
sums about 0.5 s on a Quadro M4000. The shifts agree with IMOD `alignframes` to
within about 0.03 px.

## Tilt quality control

- **Intensity.** Transmitted intensity per second of exposure falls with the
  specimen thickness along the beam, `log I = a - b / cos(theta - theta0)`
  (Beer-Lambert, with `theta0` the specimen's own tilt). A robust fit over the
  series - `theta0` by grid search minimising a truncated squared error, so a
  fit cannot "win" by discarding tilts - gives each tilt's expected intensity;
  tilts more than max(15 %, 5 x the robust spread) below it are `dark`.
  Before processing this uses the mdoc's `MinMaxMean`, afterwards the measured
  sums. Intensity is normalised by exposure *time*, not the mdoc
  `ExposureDose`: Tomo5 derives that dose from the image counts, which would
  cancel exactly the darkening being detected.
- **Drift** above max(20 A, median + 8 MAD), **alignment score** below half the
  series median, and non-convergence are flagged after processing.
- **Saturation**: the fraction of fraction-file pixels at the integer maximum
  (255 for 8-bit) is reported per series.
- Stacks and tomograms record the tilts they contain (`used_tilts`); changing
  exclusions makes them out of date, so they are rebuilt on the next run.

## CTF estimation

`pyprep/ctf.py` was written for pyPrep. It borrows ideas from CTFFIND4 and
IMOD ctfplotter, but no code.

- **Spectra.** Each non-dose-weighted tilt sum is cut into 512-px tiles that
  overlap by half. The data are first Fourier-binned if the pixel is much
  smaller than the fit range needs. Each tile's power spectrum is rotationally
  averaged on the GPU. The kx = 0 and ky = 0 lines are left out, because
  detector fixed-pattern noise and tile-edge leakage sit there. Tiles much
  darker than the median (grid bars) are skipped.
- **Flattening.** In the fit range (30–8 Å) each tile's amplitude profile has
  a smooth background removed: it is projected onto the complement of a
  4th-order polynomial in *k*. It is then divided by a smooth envelope, so the
  weak high-resolution rings count as much as the strong low ones.
- **Tilt-aware fit.** A tile at distance *u* from the tilt axis sits at defocus
  *d₀ + s·u·tan(θ + δ)*. For each tilt, the central defocus *d₀* is the value
  that maximises the mean correlation between every tile's profile and its own
  model −cos 2(χ + w), where w = asin(amplitude contrast). This is a coarse
  50 nm grid over 0.5–12 µm, refined in 20 nm steps with a parabolic peak.
- **Handedness and specimen tilt.** The tilts beyond 20° fix two values for
  the whole series:
  - the direction *s* of the gradient;
  - the offset *δ* between the stage tilt and the actual specimen tilt (a
    pretilted lamella or a bent grid), by a joint search over ±15°.

  *s* = +1 is IMOD's convention: in the aligned stack, with the tilt axis
  vertical, the right side is more underfocused at positive tilt angles.
  *s* = −1 means the tilt axis is 180° off. By default pyPrep then rotates it
  (see *Handedness* below). With *Keep as recorded* it instead sets
  ctfphaseflip's `InvertTiltAngles`.
- **Fit resolution.** The tiles' profiles are rescaled in *k* to the central
  defocus and averaged. The fit resolution is where the local correlation with
  the model (±1 ring) drops below 0.3.

**Validation on the K3 test series** (46 tilts, 3.3 Å/px, target −2 µm):

- IMOD ctfplotter (auto-fit per view, `-invert`) and pyPrep agree to 36 nm RMS
  (mean difference −10 nm).
- Without `-invert`, ctfplotter's fits drift with tilt angle and fail on two
  views. pyPrep chose the inverted handedness in 26 of 26 high tilts. (These
  were Tomo5 images as stored, i.e. upside down; see *Handedness*.)
- pyPrep's specimen tilt offset was −7.6°. tiltalign independently found the
  same offset between the raw and aligned tilt angles.
- On synthetic images with a known gradient, defocus is recovered to within
  a few nm and the handedness and offset correctly.

**CTF correction in batchruntomo.** When CTF correction is on, pyPrep:

1. copies `<series>.defocus` into the IMOD project and sets `correctCTF`
   (with `InvertTiltAngles` when the handedness is inverted);
2. runs batchruntomo up to CTF plotting (step 9);
3. re-labels the defocus file with the tilt angles tiltalign refined
   (`<series>.tlt`), because ctfphaseflip matches defocus entries to views
   by angle;
4. continues from step 10.

**Handedness.** Two separate things decide whether a tomogram has the
specimen's hand.

1. *Image orientation.* A tomogram has the hand of the images it is built
   from. Images stored mirrored give a mirror-image tomogram, whatever the
   tilt geometry. Tomography 5 stores its images upside down, so pyPrep
   inverts them in Y (see *Reading the data*; setting *Image orientation*,
   stored as `image_frame` in `_pyprep.json`).
2. *Tilt-axis direction.* The CTF defocus gradient fixes the direction of
   the tilt axis relative to the images. If it disagrees with the recorded
   axis (at least 80 % of at least 4 tilts beyond 20°), the axis is 180° off.
   With *Handedness: Automatic*, pyPrep then passes the axis rotated by 180°
   to batchruntomo, so `InvertTiltAngles` is not needed. The decision (with
   its reason) is stored under `handedness` in `_pyprep.json`, and written into
   the stacks' mdoc titles.

The defocus gradient cannot tell the two apart. Images stored mirrored also
show an inverted gradient, and rotating the axis then makes the gradient
agree while the tomogram stays a mirror image. Earlier pyPrep versions took
every inverted gradient for an axis error. They rotated the axis of Tomo5 data
(e.g. 86.2° → −93.8° on the K3 test series), and their Tomo5 tomograms are
mirror images. The point-mirror relation reported for that series (0.95 after
a 180° rotation in XY) was a consistency check, not a test of the hand.

**Verification.** *C. difficile* CD1215 lamellae, Krios / K3 / Tomography
5.27, 10.48 Å/px, 70S ribosome reference EMD-50855 (a 2.3 Å map; the template
matches it as-is, CC 0.951 against 0.850 for any mirrored axis order):

- With the old rule (images as stored, axis rotated to −86.73°):
  - template matching found 775 ribosomes with the *mirrored* 70S and 7 with
    the correct one;
  - gold-standard subtomogram averaging of 765 particles, with a global search
    from both starting hands, converged on the mirrored 70S (final-map CC 0.217
    and 0.216, against 0.179 and 0.137). A synthetic control kept the correct
    hand.
- With the images inverted in Y:
  - the recorded axis, mirrored to 86.73°, agrees with the defocus gradient in
    27 of 27 high tilts, so the axis is not rotated;
  - the alignment residual is unchanged (0.32 nm);
  - the tomogram is the old one flipped in Y (voxel correlation 0.997; the best
    proper rotation reaches 0.855), i.e. it has the opposite hand: the
    specimen's.

## Deconvolution

`pyprep/deconv.py` applies a Wiener-like filter to the finished tomogram,
following the approach Warp introduced for tomograms. The filter is
*CTF(f) / (CTF(f)² + 1/SNR(f))*, where:

- SNR(f) = 10^(3·strength) · exp(−100·falloff·f / pixel) · (1 − cos(π·min(1, f/0.02))),
  with *f* as a fraction of Nyquist;
- the CTF is evaluated at the series' defocus, which is the median over
  well-fitted tilts within ±30°;
- |CTF| is used when the stack was phase-flipped.

The CTF sign keeps low-resolution contrast, so protein stays dark. The filter
is radially symmetric in 3D and applied with one FFT of the volume on the GPU,
with a fallback to the CPU when memory runs out. The output is scaled to the
input's mean and standard deviation. After *Position tomogram* rebuilds a
tomogram, the deconvolved copy is re-made with the same parameters.

## Redeposition removal

FIB lamellae carry dense material on their surfaces: redeposited grains and the
Pt/GIS protective layer. Single-axis tilting smears it through the volume, so
it casts shading and blotches into the cell. `pyprep/redep.py` (optional; needs
the ASTRA Toolbox with a CUDA GPU) removes it from the tilt series:

1. A plain WBP of the reconstruction's aligned, phase-flipped stack, about
   730 nm thick, in the projection geometry (no offset, shift or X-axis tilt).
2. `pyprep/lamella.py` locates the lamella interior from its band-pass texture.
   Beyond each surface, the dense layer's depth, extent and coverage are
   measured in coordinates flattened to the fitted surfaces.
3. Grain masks: per surface, the grain footprint is segmented in the flattened
   slab where the layer's cross-section is widest (the valley of a bimodal
   histogram, else dense cores) and extruded over the layer's apparent depth.
   Only material outside the interior can be removed.
4. SIRT (ASTRA, GPU, 100 iterations) of *q* = −ln(*p* / view mean),
   constrained to the slab: the interior plus the layers.
5. The SIRT density inside the grain masks, projected, gives each view's
   transmission *T* = exp(−*A c*).
6. The cleaned view is *p* + (1 − *T*) *m*, with *m* the local grain-free mean
   (*p* / *T*, low-passed). The grains' shadow is filled with what the rays
   would have carried without them; dividing by *T* instead would weight the
   shadowed rays, which carry fewer electrons, as fully as the others and add
   noise.
7. The tomogram is rebuilt from the cleaned stack with the project's own
   `tilt.com` and `trimvol` (same geometry, filter and positioning), as
   `<series>_rec_redep.mrc`. The normal tomogram is kept.

On C. difficile lamellae with heavy redeposition it removed the shading under
the grains at 50–400 nm (54–81 % of the power above 100 nm next to the layers)
and changed nothing below about 12 nm: the even/odd FSC is unchanged. It is a
cleaner tomogram for viewing and segmentation; template matching and averaging
gain little. The working volumes need about 10 GB of memory at bin 4 (2.6 Å
data); finer reconstructions are refused. When no lamella interior is found,
the step is skipped and logged. After *Position tomogram*, the
redeposition-removed tomogram is rebuilt from the same cleaned stack.

## Dose weighting

This is the optional `_DW` stack. Each frame is weighted by
`exp(-N / (2 Nc(q)))`, following Grant & Grigorieff (2015), with
`Nc(q) = 0.245 q^-1.665 + 2.81` at 300 kV (scaled by 0.8 at 200 kV). Here `N`
is the dose accumulated up to the middle of that frame, *including all earlier
tilts*. The weighted frames are summed without renormalisation, as for
tomography exposure filtering: later tilts keep their low frequencies but lose
high-frequency content.

## EER (Falcon)

- An EER file is a TIFF with one page per detector frame. Each page holds a
  run-length-coded electron bitstream: skip codes of 7 or 8 bits, or the widths
  given in tags 65007–65009, followed by 2+2 sub-pixel bits. This covers codecs
  65000, 65001 and 65002.
- Pages are decoded by `imagecodecs.eer_decode`, the decoder `tifffile` uses.
  It adds the electrons of each page directly into a 16-bit fraction image, and
  several fractions are decoded in parallel. This takes about 2.3 ms per 4K
  frame.
- Frames are grouped into fractions by count or by frames per fraction, with
  the remainder spread evenly so no frames are lost.
- **Super-resolution.** With 2x rendering, the sub-pixel bits place each
  electron on an 8K grid. All user settings (binning, shift limits) stay in
  physical pixels. Alignment runs at 8K with the binning doubled, the output is
  Fourier-binned back to the physical pixel size, and shifts are reported in
  physical pixels.
- **Dose per fraction.** It is summed from per-frame doses when the mdoc lists
  them. Otherwise the tilt's dose is split in proportion to the frames in each
  fraction.
- The EER support was validated against synthetic EER files. A test-side
  encoder was checked against `imagecodecs`. Both independent implementations
  use the same sub-pixel convention, with each field XOR-offset by half its
  range. It has not yet been checked on real Falcon data.

## Gain reference

The reference is oriented by the rotation and flip settings. It is expanded by
pixel replication for super-resolution rendering, then turned into a per-pixel
multiplier: `1/gain` for *divide* (EPU `.gain`), or `gain` for *multiply*.
Pixels whose gain is zero, which marks defects, become zero. The multiplier is
applied to every frame on the GPU before alignment.

## Stacks for etomo

Tilts are written in **tilt-angle order**, not collection order. Each stack
gets:

- a `.rawtlt` file with the tilt angles;
- a `.mrc.mdoc` whose sections are re-ordered to match the stack, keeping each
  tilt's original metadata (dose, prior dose, defocus, …);
- the pixel size in the MRC header.

IMOD `extracttilts` reads the angles from the mdoc, and etomo reads the
tilt-axis angle from it.

## Reconstruction (IMOD batchruntomo)

pyPrep copies the binned stack, its `.rawtlt` and its `.mdoc` into
`<series>\imod_bin<N>\`. It writes a directive file from the chosen preset and
runs IMOD's `batchruntomo.cmd`. The result is a standard etomo project.

**Patch tracking preset (default).** Patch tracking is `trackingMethod = 1`.
Patches are 400 nm with 0.6 overlap (303 px at 13.2 Å/px); these values are
scaled from an etomo project tuned by hand at 3.3 Å/px. Fine alignment solves
rotation for every view, tilt angles in groups, and magnification for every
view, with robust fitting and no local alignments.

**Gold preset.** It uses the bead size (10 nm by default), `autofidseed` with
20 target beads, bead tracking, cryo positioning that accounts for the beads,
and gold erasing using the fiducial model.

**Both presets:**

- They use IMOD's `cryoSample.adoc` system template.
- They remove X-rays, then position the tomogram. Cryo positioning is used
  with a fallback thickness; *fixed thickness* skips it.
- They reconstruct by weighted back-projection with a SIRT-like filter
  (6 iterations by default).
- They trim the volume and rotate it around X.

Extra directives entered on the Reconstruction page override the preset.

IMOD's scripts run with the `python` that etomo uses. pyPrep removes its own
environment from the `PATH` it passes to IMOD, so a different Python version
cannot interfere. The batchruntomo output streams to the log, and cancelling
stops the whole IMOD process tree.

## Manual tomogram positioning

This replaces etomo's *tomopitch* step (`pyprep/positioning.py`).

**Trial tomogram.** The aligned stack (`<series>_ali.mrc`) is binned 2× on the
GPU. IMOD `tilt` then reconstructs it with the project's own `tilt.com`
parameters, a larger THICKNESS and `IMAGEBINNED 2`. THICKNESS stays in unbinned
pixels.

**Views.** Averaging the trial along Y or X shows only the smooth
reconstruction background. So each Y slice is high-pass filtered (the slice
minus a Gaussian blur with σ = 6 px), and the squared result is summed:

- over Y to give the XZ view;
- over X to give the YZ view.

The square root of these sums is displayed. Specimen detail is bright; ice or
vacuum is dark. *Auto* places the lines where the row-averaged energy rises 30%
of the way from the background to the peak.

**Geometry.** The reconstruction is stored with rows = Z (row 0 at the bottom),
columns = X and sections = Y. For a specimen with:

- mid-plane slope angle *a* in XZ and *b* in YZ, and
- centre *c* pixels above the middle of the volume,

the specimen is made flat and centred by:

- adding *a* to `OFFSET` (the tilt-angle offset);
- adding *b* to `XAXISTILT`;
- adding −*c*·cos *a*·cos *b* to the Z component of `SHIFT`;
- setting `THICKNESS` = specimen thickness · cos *a* · cos *b* + 2 × margin.

These signs and factors were measured, not assumed. A synthetic cloud of dots
was tilted by 10° and −6° and raised by 25 px, then projected and reconstructed
with IMOD `tilt`. Fitting planes to the reconstructed dots gave the slopes and
centre. After the correction, the fitted slopes were −0.8° and −0.2° and the
centre was 0.6 px from the middle.

**Rebuild.** The original `tilt.com` is kept once as `tilt.com.pyprep_orig`.
The new parameters are written into `tilt.com`, which stays a valid etomo
command file. `tilt` is re-run, then the `trimvol` command from batchruntomo's
`trimvol.com` rotates the volume to Z slices again. Each rebuild is noted under
`positioning` in `pyprep_recon.json`. A new batchruntomo run (*Redo
reconstruction*) removes the backup and the trial files.

## Responsiveness

GPU processing, reading and writing run on worker threads:

- the next tilt is read while the current one is aligned;
- sections are written while the next tilt is processed;
- stacks for the Results page are loaded in the background.

The window stays responsive throughout; the longest pause measured during a
full batch was 77 ms.

## References

- Zheng et al. (2017) MotionCor2. *Nat Methods* 14:331.
- Grant & Grigorieff (2015) Measuring the optimal exposure for single particle
  cryo-EM using a 2.6 Å reconstruction of rotavirus VP6. *eLife* 4:e06980.
- Guizar-Sicairos, Thurman & Fienup (2008) Efficient subpixel image
  registration algorithms. *Opt Lett* 33:156.
- Kremer, Mastronarde & McIntosh (1996) Computer visualization of
  three-dimensional image data using IMOD. *J Struct Biol* 116:71.
- Mastronarde & Held (2017) Automated tilt series alignment and tomographic
  reconstruction in IMOD. *J Struct Biol* 197:102.
- Rohou & Grigorieff (2015) CTFFIND4: Fast and accurate defocus estimation
  from electron micrographs. *J Struct Biol* 192:216.
- Xiong, Morphew, Mastronarde & McIntosh (2009) CTF determination and
  correction for low dose tomographic tilt series. *J Struct Biol* 168:378.
- Tegunov & Cramer (2019) Real-time cryo-electron microscopy data preprocessing
  with Warp. *Nat Methods* 16:1146.
