# Installation

pyPrep runs on **Windows 10 or 11 (64-bit)**. It installs into its own Python
environment inside the pyPrep folder, so it does not touch any other Python on
the computer.

## Requirements

| | Minimum | Recommended / tested |
|---|---|---|
| Operating system | Windows 10 64-bit | Windows 10 / 11 |
| GPU | NVIDIA with CUDA support; optional (CPU mode is 10-50x slower) | 8 GB NVIDIA GPU (tested: Quadro M4000, driver 471.41) |
| NVIDIA driver | 452.39 or newer (CUDA 11.8 build) | the newest driver for your card |
| RAM | 16 GB | 32-64 GB |
| Disk | ~6 GB for the software | fast local disk for outputs (see below) |
| Python | installed automatically (Miniconda needed) | — |
| IMOD for Windows | only for reconstruction | IMOD 4.11 or newer |

Output size for one K3 tilt series (5760 x 4092, ~46 tilts): full-resolution
float32 stack ~4.3 GB (int16 halves it), bin-4 stack ~0.27 GB, bin-4 IMOD
project ~3 GB.

## 1. Install Miniconda

Download and install **Miniconda for Windows** from
<https://docs.conda.io/en/latest/miniconda.html> (the default "just me" install
is fine). pyPrep only uses conda to create its private Python 3.11; you do not
need to use conda yourself afterwards.

## 2. Get pyPrep

Either clone with git:

```bat
git clone https://github.com/EvanKrystofiak/pyPrep.git
```

or download the ZIP from the GitHub page (**Code > Download ZIP**) and extract
it. Put it on a drive with a few GB free (the environment is ~5 GB).

## 3. Run the installer

Double-click **`install.bat`** in the pyPrep folder, or run it from a command
prompt:

```bat
cd path\to\pyPrep
install.bat
```

It will:

1. find Miniconda/Anaconda,
2. create a Python 3.11 environment in `pyPrep\env`,
3. install PyTorch with CUDA (about 2.5 GB download),
4. install pyPrep and its other dependencies (NumPy, SciPy, PySide6, pyqtgraph,
   tifffile, imagecodecs, pytest),
5. create **pyPrep shortcuts** on the Desktop and in the Start menu,
6. print the GPU it found.

The last line should read something like:

```
pyPrep 0.1.0 | PyTorch 2.7.1+cu118 | GPU: Quadro M4000
```

If it says `GPU: NOT AVAILABLE`, see [Troubleshooting](troubleshooting.md#gpu-not-detected).

### Choosing the PyTorch/CUDA build

The default build (CUDA 11.8) works with NVIDIA drivers from 452.39 on and with
GPUs from Maxwell (GTX 900 / Quadro M-series) to Ada (RTX 40-series). Pass a
different build to the installer if needed:

| Your GPU / driver | Command |
|---|---|
| Most GPUs, older drivers (default) | `install.bat` |
| Newer driver (2023+), want CUDA 12 | `install.bat cu126` |
| RTX 50-series (Blackwell) | `install.bat cu128` (driver 570 or newer) |
| No NVIDIA GPU | `install.bat cpu` |

Re-running `install.bat` on an existing installation just updates the packages.

## 4. Install IMOD (for reconstruction)

Frame alignment and stack building need nothing else. To let pyPrep run
**batchruntomo** and produce tomograms, install **IMOD for Windows** from
<https://bio3d.colorado.edu/imod/> following its instructions, including the
Python that IMOD's scripts need. Check it works by opening a command prompt and
running `etomo`.

pyPrep finds IMOD through the `IMOD_DIR` environment variable (set by the IMOD
installer) or at `C:\Program Files\IMOD`. It runs IMOD's scripts with the same
`python` that etomo uses, never with pyPrep's own environment. The
**Reconstruction** page shows whether IMOD was found and which Python it will use.

### Optional: redeposition removal

Redeposition removal needs the [ASTRA Toolbox](https://astra-toolbox.com) and a
CUDA GPU. Install it into pyPrep's environment:

```bat
env\python.exe -m pip install astra-toolbox
```

or run `set PYPREP_REDEP=1` before `install.bat`, which installs pyPrep with its
`redep` extra. ASTRA brings its own CUDA runtime from pip. Restart pyPrep; the
**Reconstruction** page then enables *Also write a tomogram with the redeposition
removed* and shows the ASTRA version under it. ASTRA is licensed under the GPLv3
and is not part of pyPrep.

## 5. Start pyPrep

Use the **pyPrep** shortcut on the Desktop or in the Start menu (or
double-click `pyPrep.bat`). A splash screen appears within a couple of
seconds while the libraries load (the first start after installing or rebooting
can take ~30 s on a hard disk).

To recreate the shortcuts (e.g. after moving the folder):
`env\python.exe scripts\make_shortcut.py`. To skip them during installation,
run `set PYPREP_NO_SHORTCUT=1` before `install.bat`.

## Checking the installation (optional)

```bat
env\python.exe -m pytest
```

All tests should pass (they use synthetic data and take well under a minute).

## Updating

```bat
git pull
install.bat
```

(or download the new ZIP, copy your `env` folder into it, and run `install.bat`).

## Uninstalling

Delete the pyPrep folder and the two **pyPrep** shortcuts (Desktop, Start menu).
pyPrep also stores window layout and last-used settings in the Windows registry
(`HKEY_CURRENT_USER\Software\pyPrep`), saved presets in `%APPDATA%\pyPrep`,
and its startup error and crash logs in `%LOCALAPPDATA%\pyPrep`.

## Manual installation

If you prefer to set things up yourself (any conda or venv works, Python 3.11+):

```bat
conda create -y --prefix env python=3.11
env\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
env\python.exe -m pip install -e ".[test]"
```

Install PyTorch **before** pyPrep: `pip install` of a package that lists `torch`
as a dependency would otherwise fetch the CPU-only PyTorch from PyPI.
