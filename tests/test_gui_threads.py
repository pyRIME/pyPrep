"""GUI worker threads: closing the positioning dialog while IMOD runs, and the crash log.

Each check runs in a child process (offscreen Qt), so a native crash fails the test
instead of taking pytest down."""

import os
import subprocess
import sys

import pytest

pytest.importorskip("PySide6")

# Closing the dialog during Apply used to leave the worker thread holding the last reference
# to it: the dialog was then destroyed on the worker thread while the GUI thread was still
# delivering its queued log signals - an access violation in PySide.
CLOSE_DURING_APPLY = r'''
import sys, threading, time
from pathlib import Path
from PySide6.QtCore import QCoreApplication, QEvent, Qt, QTimer
from PySide6.QtWidgets import QApplication, QMainWindow
from pyprep import positioning as P
from pyprep.gui import positioning_dialog as D

def slow_apply(recon_dir, root, corr, margin_px, log=print, cancel=None):
    for i in range(2000):                       # IMOD tilt/trimvol output, queued to the dialog
        log(f"tilt line {i}")
    time.sleep(0.1)
    return {"THICKNESS": corr.thickness_px + 2 * margin_px}

P.apply_positioning = slow_apply
D.PositioningDialog._correction = lambda self: P.Correction(0.0, 0.0, 0.0, 100)
recon_dir = Path(sys.argv[1])
ROUNDS = 10
app = QApplication([])
win = QMainWindow()
stats = {"refused": 0, "closed": 0}
destroyed_on = []                               # thread that deleted each dialog's C++ object

def drive():                                    # in the open dialog: Apply, then keep pressing Close
    dlg = QApplication.activeModalWidget()
    dlg.destroyed.connect(lambda *_: destroyed_on.append(threading.current_thread().name),
                          Qt.ConnectionType.DirectConnection)
    dlg.views = {"pixel_size": 10.0, "scale": 1.0}
    dlg.apply()

    def press_close():
        dlg.reject()
        if dlg.isVisible():
            stats["refused"] += 1
            QTimer.singleShot(20, press_close)
    QTimer.singleShot(0, press_close)

def open_positioning():                         # as MainWindow.open_positioning
    QTimer.singleShot(0, drive)
    D.open_dialog(recon_dir, "TS", win)
    stats["closed"] += 1
    QTimer.singleShot(0, open_positioning if stats["closed"] < ROUNDS else app.quit)

QTimer.singleShot(0, open_positioning)
app.exec()
time.sleep(0.5)                                 # let any worker thread still running finish
QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
print(f"OK refused={stats['refused']} closed={stats['closed']} destroyed={len(destroyed_on)} "
      f"destroyed_off_gui_thread={sum(t != 'MainThread' for t in destroyed_on)}", flush=True)
'''


def _child(code, *args, env=None, timeout=180):
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", **(env or {}))
    return subprocess.run([sys.executable, "-c", code, *map(str, args)], capture_output=True, text=True,
                          env=env, timeout=timeout)


def test_positioning_dialog_survives_close_during_apply(tmp_path):
    recon_dir = tmp_path / "TS" / "imod_bin4"
    recon_dir.mkdir(parents=True)
    r = _child(CLOSE_DURING_APPLY, recon_dir)
    assert r.returncode == 0, f"exit code {r.returncode:#x}\n{r.stdout}\n{r.stderr[-3000:]}"
    line = next(l for l in r.stdout.splitlines() if l.startswith("OK"))
    stats = dict(kv.split("=") for kv in line.split()[1:])
    assert int(stats["closed"]) == 10
    assert int(stats["refused"]) > 0            # Close was refused while Apply was running
    assert int(stats["destroyed"]) == 10        # closed dialogs are destroyed, not kept by the main window
    assert int(stats["destroyed_off_gui_thread"]) == 0   # ... and on the GUI thread


def test_crash_log_records_python_stack(tmp_path):
    code = ("from pyprep.gui import launcher\n"
            "print(launcher.enable_crash_log(), flush=True)\n"
            "import faulthandler\n"
            "def rebuild_tomogram():\n"
            "    faulthandler._sigsegv()\n"
            "rebuild_tomogram()\n")
    r = _child(code, env={"LOCALAPPDATA": str(tmp_path)}, timeout=120)
    assert r.returncode != 0                    # the child really crashed
    log = (tmp_path / "pyPrep" / "crash.log").read_text(encoding="utf-8", errors="replace")
    assert "pyPrep" in log and "started" in log
    assert "rebuild_tomogram" in log            # the Python stack of the crashing thread
