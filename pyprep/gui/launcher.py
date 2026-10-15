"""Fast-start entry point: show a splash screen, then load the heavy libraries.

Importing PyTorch, SciPy and pyqtgraph from a cold hard drive takes tens of
seconds, and pyPrep runs without a console, so without this the user sees
nothing at all until the main window appears.  Only PySide6 is imported before
the splash is on screen; everything else is loaded step by step behind it.
Startup errors are shown in a message box and written to a log file.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QGuiApplication, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QApplication, QMessageBox, QSplashScreen

from .. import __version__
from . import theme

W, H = 560, 300

STEPS = [
    ("Loading NumPy and SciPy", lambda: (__import__("numpy"), __import__("scipy.fft"))),
    ("Loading PyTorch", lambda: __import__("torch")),
    ("Checking the GPU", lambda: __import__("pyprep.gpu", fromlist=["list_gpus"]).list_gpus()),
    ("Loading the image viewer", lambda: __import__("pyqtgraph")),
    ("Loading pyPrep", lambda: __import__("pyprep.gui.app", fromlist=["MainWindow"])),
]


class Splash(QSplashScreen):
    """pyFIB-palette splash: brand, subtitle, status line and a step progress bar."""

    def __init__(self):
        dpr = QGuiApplication.primaryScreen().devicePixelRatio() if QGuiApplication.primaryScreen() else 1.0
        pm = QPixmap(int(W * dpr), int(H * dpr))
        pm.setDevicePixelRatio(dpr)
        pm.fill(Qt.transparent)
        super().__init__(pm)
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)   # keep the rounded corners clean
        self.status = "Starting…"
        self.fraction = 0.0

    def set_step(self, text: str, fraction: float):
        self.status, self.fraction = text, max(0.0, min(1.0, fraction))
        self.repaint()
        QApplication.processEvents()

    def drawContents(self, p: QPainter):
        p.setRenderHint(QPainter.Antialiasing)
        # card
        p.setPen(QPen(QColor(theme.BORDER), 1))
        p.setBrush(QColor(theme.BG))
        p.drawRoundedRect(QRectF(0.5, 0.5, W - 1, H - 1), 14, 14)
        # accent stripe
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(theme.ACCENT))
        p.drawRoundedRect(QRectF(32, 44, 6, 74), 3, 3)
        # brand + subtitle
        f = QFont("Segoe UI")
        f.setPixelSize(52)
        f.setWeight(QFont.Weight.ExtraBold)
        p.setFont(f)
        p.setPen(QColor(theme.TEXT))
        p.drawText(QRectF(54, 34, W - 80, 64), Qt.AlignLeft | Qt.AlignVCenter, "pyPrep")
        f.setPixelSize(17)
        f.setWeight(QFont.Weight.Normal)
        p.setFont(f)
        p.setPen(QColor(theme.TEXT_DIM))
        p.drawText(QRectF(56, 96, W - 80, 26), Qt.AlignLeft | Qt.AlignVCenter,
                   "cryo-ET tilt-series preparation")
        f.setPixelSize(13)
        p.setFont(f)
        p.drawText(QRectF(56, 124, W - 80, 20), Qt.AlignLeft | Qt.AlignVCenter,
                   f"version {__version__}  ·  frame alignment · etomo stacks · IMOD reconstruction")
        # status + progress bar
        f.setPixelSize(14)
        p.setFont(f)
        p.setPen(QColor(theme.TEXT))
        p.drawText(QRectF(32, H - 84, W - 64, 22), Qt.AlignLeft | Qt.AlignVCenter, self.status)
        bar = QRectF(32, H - 52, W - 64, 10)
        p.setPen(QPen(QColor(theme.BORDER), 1))
        p.setBrush(QColor(theme.BG_ALT))
        p.drawRoundedRect(bar, 5, 5)
        if self.fraction > 0:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(theme.ACCENT))
            p.drawRoundedRect(QRectF(bar.x(), bar.y(), max(10.0, bar.width() * self.fraction), bar.height()), 5, 5)


def _error_log_path() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "pyPrep"
    base.mkdir(parents=True, exist_ok=True)
    return base / "startup_error.log"


_crash_log = None      # kept open for the life of the process: faulthandler writes to its descriptor


def enable_crash_log() -> Path | None:
    """A native crash (access violation in Qt, PySide or an extension) ends the process with no
    Python traceback; faulthandler then writes the Python stack of every thread to crash.log."""
    global _crash_log
    import faulthandler
    try:
        path = _error_log_path().with_name("crash.log")
        mode = "a" if path.exists() and path.stat().st_size < 1_000_000 else "w"
        _crash_log = open(path, mode, encoding="utf-8")
        _crash_log.write(f"--- pyPrep {__version__} started {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        _crash_log.flush()
        faulthandler.enable(file=_crash_log, all_threads=True)
        return path
    except OSError:
        return None


def set_app_identity(app) -> None:
    """Window/taskbar icon; on Windows also an AppUserModelID so the taskbar shows
    pyPrep's icon instead of Python's."""
    from PySide6.QtGui import QIcon
    icon = Path(__file__).with_name("pyprep.ico")
    if icon.exists():
        app.setWindowIcon(QIcon(str(icon)))
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("pyPrep.pyPrep")
        except Exception:
            pass


def main() -> int:
    t0 = time.perf_counter()
    enable_crash_log()
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("pyPrep")
    set_app_identity(app)
    splash = Splash()
    splash.show()
    splash.set_step("Starting…", 0.03)
    try:
        for i, (text, load) in enumerate(STEPS):
            splash.set_step(text + "…", (i + 0.3) / (len(STEPS) + 1))
            load()
        splash.set_step("Building the interface…", len(STEPS) / (len(STEPS) + 1))
        theme.apply_theme(app)
        from .app import MainWindow   # already imported above; this is instant
        w = MainWindow()
        splash.set_step(f"Ready ({time.perf_counter() - t0:.0f} s)", 1.0)
        w.show()
        splash.finish(w)
    except Exception:
        tb = traceback.format_exc()
        log = _error_log_path()
        log.write_text(tb, encoding="utf-8")
        splash.close()
        QMessageBox.critical(None, "pyPrep could not start",
                             f"{tb.strip().splitlines()[-1]}\n\nFull details were written to:\n{log}")
        return 1
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
