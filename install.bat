@echo off
rem ---------------------------------------------------------------------------
rem  pyPrep installer for Windows
rem
rem  Creates a self-contained Python environment in the "env" folder next to this
rem  file, installs PyTorch with CUDA support and pyPrep, then checks the GPU.
rem
rem  Usage:   install.bat            (CUDA 11.8 build - works with NVIDIA drivers >= 452)
rem           install.bat cu126      (CUDA 12.6 build - needs driver >= 528)
rem           install.bat cu128      (CUDA 12.8 build - needed for RTX 50-series, driver >= 570)
rem           install.bat cpu        (no GPU)
rem  Requires Miniconda or Anaconda: https://docs.conda.io/en/latest/miniconda.html
rem ---------------------------------------------------------------------------
setlocal EnableExtensions
cd /d "%~dp0"

set "TORCH_VARIANT=%~1"
if "%TORCH_VARIANT%"=="" set "TORCH_VARIANT=cu118"
set "TORCH_VERSION=2.7.1"

rem ---- find conda ------------------------------------------------------------
set "CONDA="
for %%p in (
  "%CONDA_EXE%"
  "%USERPROFILE%\miniconda3\Scripts\conda.exe"
  "%USERPROFILE%\anaconda3\Scripts\conda.exe"
  "%LOCALAPPDATA%\miniconda3\Scripts\conda.exe"
  "%ProgramData%\Miniconda3\Scripts\conda.exe"
  "%ProgramData%\Anaconda3\Scripts\conda.exe"
) do if not defined CONDA if exist "%%~p" set "CONDA=%%~p"
if not defined CONDA for /f "delims=" %%i in ('where conda 2^>nul') do if not defined CONDA set "CONDA=%%i"
if not defined CONDA (
  echo.
  echo  Could not find conda. Install Miniconda first:
  echo    https://docs.conda.io/en/latest/miniconda.html
  echo  then run install.bat again.
  goto :fail
)
echo Using conda: %CONDA%

rem ---- environment -----------------------------------------------------------
if exist "%~dp0env\python.exe" (
  echo Environment already exists in "%~dp0env" - updating packages.
) else (
  echo Creating Python 3.11 environment in "%~dp0env" ...
  call "%CONDA%" create -y --prefix "%~dp0env" python=3.11 || goto :fail
)
set "PY=%~dp0env\python.exe"
"%PY%" -m pip install --upgrade pip || goto :fail

rem ---- PyTorch (GPU build first, so pip does not pull the CPU-only wheel) -----
echo Installing PyTorch %TORCH_VERSION% (%TORCH_VARIANT%) - about 2.5 GB for CUDA builds ...
"%PY%" -m pip install torch==%TORCH_VERSION% --index-url https://download.pytorch.org/whl/%TORCH_VARIANT% || goto :fail

rem ---- pyPrep and its other dependencies ---------------------------------------
rem      set PYPREP_REDEP=1 to add the ASTRA Toolbox (GPLv3) for redeposition removal
set "EXTRAS=test"
if "%PYPREP_REDEP%"=="1" set "EXTRAS=test,redep"
"%PY%" -m pip install -e ".[%EXTRAS%]" || goto :fail

rem ---- shortcuts (Desktop + Start menu); set PYPREP_NO_SHORTCUT=1 to skip -----
if not "%PYPREP_NO_SHORTCUT%"=="1" "%PY%" "%~dp0scripts\make_shortcut.py"

rem ---- check -------------------------------------------------------------------
echo.
"%PY%" -c "import torch, pyprep, imagecodecs, PySide6; ok = torch.cuda.is_available(); print('pyPrep', pyprep.__version__, '| PyTorch', torch.__version__, '| GPU:', torch.cuda.get_device_name(0) if ok else 'NOT AVAILABLE - pyPrep will run on the CPU (slow)')" || goto :fail
echo.
echo  Installation complete. Start pyPrep from the desktop shortcut or pyPrep.bat
echo  (reconstruction also needs IMOD for Windows: https://bio3d.colorado.edu/imod/)
echo.
if not "%PYPREP_NOPAUSE%"=="1" pause
exit /b 0

:fail
echo.
echo  Installation FAILED - see the messages above.
if not "%PYPREP_NOPAUSE%"=="1" pause
exit /b 1
