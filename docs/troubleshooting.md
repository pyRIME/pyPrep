# Troubleshooting

## pyPrep does not start

- **Nothing happens when I double-click `pyPrep.bat`**: a splash screen should
  appear within a few seconds. If it never does, run from a command prompt to
  see errors:
  ```bat
  env\python.exe -m pyprep.gui.launcher
  ```
- **"pyPrep could not start"** message: the details are in
  `%LOCALAPPDATA%\pyPrep\startup_error.log`.
- **pyPrep closes suddenly, without a message**: the Python stack of every
  thread at the moment of the crash is in `%LOCALAPPDATA%\pyPrep\crash.log`
  (the last block, after the latest "pyPrep ... started" line). Include it
  when reporting the problem.
- **"pyPrep is not installed yet"**: run `install.bat` first
  ([Installation](installation.md)).
- **Slow first start (~30 s)**: Python loads PyTorch and Qt from disk; later
  starts are faster. The splash screen shows progress.

## GPU not detected

The install check or the status bar says no GPU / CPU mode.

1. Check that the NVIDIA driver works: run `nvidia-smi` in a command prompt.
2. The default PyTorch build (CUDA 11.8) needs driver 452.39 or newer. Very new
   GPUs (RTX 50-series) need `install.bat cu128` and a driver ≥ 570.
3. Test PyTorch directly:
   ```bat
   env\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   ```
   A version ending in `+cpu` means the CPU-only PyTorch was installed — rerun
   `install.bat` (it installs the CUDA build first).

pyPrep still works on the CPU, only more slowly.

## "Not Responding" / the window seems frozen

pyPrep keeps the window responsive during processing and loading. If Windows
reports it as not responding, it is usually because the computer is out of
memory (full-resolution stacks of large detectors need several GB) or a disk
is very slow. Check the Log page and Windows Task Manager. Processing itself
continues in the background; the batch ends with a green or red *Finished*
message in the Batch panel.

## Tilts are missing

The Tilt series page shows tilts as `missing` when the file named in the mdoc's
`SubFramePath` is not found.

- Set the **Fractions folder** to where the fraction files are (e.g. the
  `DoseFractions` share), or copy them next to the `.mdoc`.
- `pyprep scan <folder>` lists the missing files by name.
- Missing tilts are simply left out of the stack; the `.rawtlt` and `.mdoc`
  match the tilts that were used.

## "gain reference ... does not match frames"

The gain reference has a different size or orientation from the frames.
If pyPrep says it *looks transposed*, set the gain rotation to 90° (or 270°)
on the Frame alignment page. Use **Test on one tilt**: with the right
orientation the aligned image shows no detector pattern.

For Tomo5 K3 fractions, leave the gain reference empty — they are already
gain-normalized.

## Tilts are flagged by QC

Hover over the QC entry for the reason. A `dark` tilt is usually blocked
(grid bar, lamella edge) or contaminated — check it in 3dmod and exclude it.
A few flags at the highest tilts of a thick specimen can be normal. Flags are
advisory; nothing is excluded unless you untick the tilt or press
**Exclude flagged tilts**.

## "fraction files are saturated"

8-bit fraction files (a Tomo5 option) clip pixels that receive more than about
8 electrons in one fraction. A few percent is common at low tilt. It cannot be
undone in processing; to avoid it, save fractions as 16-bit or use more
fractions per tilt.

## Frame alignment finds almost no drift

Many tilt series move only a few Å during each exposure; alignment then makes
little difference, which is expected. The *Drift per tilt* plot shows the
measured drift. To check whether alignment helps a given dataset, run
`scripts\validate_alignment.py` ([Command line](command-line.md#validation-script)).

If shifts look implausibly large or erratic for very low-dose fractions, raise
the B-factor (1000–2000 Å²), keep alignment binning at 4, or for EER use fewer,
larger fractions.

## Reconstruction problems

- **"IMOD not found"**: install IMOD for Windows and make sure `IMOD_DIR` is
  set (the IMOD installer does this; log out and in again afterwards). The
  Reconstruction page shows what pyPrep found.
- **"no python on PATH"**: IMOD's scripts need Python; install it as described
  in IMOD's installation instructions and check that `etomo` runs.
- **"Tomogram positioning did not work; proceeding if possible"**: IMOD could
  not find the specimen slab (common for sparse samples); the fallback thickness
  is used. Afterwards, use *Position tomogram…* on the Results page to set the
  thickness, tilt and centre by eye. Or choose *Fixed thickness* on the
  Reconstruction page to skip the attempt.
- **Tomogram is tilted, off-centre or cuts off part of the specimen**: use
  *Position tomogram…* on the Results page.
- **Positioning window: the specimen is not visible or fills the whole view**:
  make a new trial with a larger *Trial thickness*.
- **Reconstruction failed**: the IMOD messages are in the Log page and in
  `<series>_pyprep.log`; the directives used are in
  `imod_bin<N>\pyprep_batchruntomo.adoc`. Open the project in etomo to see which
  step failed and continue from there.
- **Poor alignment with patch tracking**: try different patch sizes (Advanced
  directives or the patch size setting), or the gold preset if the sample has
  fiducials.

## CTF estimation

- **"CTF not estimated - it needs the bin 1 aligned stack"**: this series was
  processed before CTF estimation existed, without a bin 1 stack. Tick bin 1
  and re-run it (the stacks are redone), or run `pyprep ctf` once a bin 1 stack
  exists.
- **Defocus per tilt looks noisy or has red points**: check those tilts'
  *CTF fit* plot. Thick ice, drift and contamination weaken the rings. The fit
  range can be narrowed (for example 30 to 12 Å) on the Frame alignment page.
- **"direction of the defocus gradient is uncertain"**: the series has few
  tilts beyond 20° or weak rings. pyPrep then uses the majority vote, and CTF
  correction at bin 4 is barely affected by it.
- **New tomograms look rotated by 180° compared with older ones**: the
  handedness was corrected (Reconstruction page, *Handedness*). The new ones
  have the correct handedness. Choose *Keep as recorded* to get the old
  orientation back.
- **Phase-plate data**: phase shifts are not fitted yet. Turn CTF correction
  off for such data.

## Disk space

A full-resolution float32 stack of a K3 tilt series is ~4.3 GB. To save space:
choose *int16* (half the size), untick *bin 1* if you only need binned stacks,
and leave even/odd and dose-weighted stacks off unless needed.

## Re-running

With *Skip steps whose outputs are already complete* on, finished stacks and
tomograms are not recomputed — delete the series' output folder, untick that
option, or use `--force` on the command line to redo them.

## Reporting a problem

Please include the series' `<name>_pyprep.log` and `<name>_pyprep.json`, the
pyPrep version (Help > About), and what you expected. Open an issue at
<https://github.com/EvanKrystofiak/pyPrep/issues>.
