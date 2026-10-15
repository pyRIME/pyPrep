"""Interactive tomogram positioning: mark the specimen's top and bottom, rebuild the tomogram."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (QDialog, QDoubleSpinBox, QHBoxLayout, QLabel, QMessageBox, QPushButton,
                               QVBoxLayout)

from .. import positioning as P
from . import theme
from .theme import card, dim_label, section_label, title_label


class _Bridge(QObject):
    done = Signal(str, object)            # task name, result or Exception
    log = Signal(str)


# Run on the worker thread: plain values only, no widgets.
def _refresh_deconv(recon_dir: Path, log) -> None:
    """The deconvolved tomogram must follow the rebuilt one."""
    from .. import deconv
    try:
        deconv.refresh(recon_dir, log=log)
    except Exception as e:
        log(f"Deconvolution not updated: {e}")


def _record(recon_dir: Path, geom, corr, margin_nm) -> None:
    """Note the positioning in pyprep_recon.json and refresh the gallery thumbnail."""
    j = recon_dir / "pyprep_recon.json"
    try:
        rec = json.loads(j.read_text(encoding="utf-8"))
        rec["positioning"] = None if geom is None else {"tilt_com": geom, "margin_nm": margin_nm,
                                                       "specimen_px": corr.thickness_px}
        j.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    except (OSError, ValueError):
        pass
    try:
        from ..thumbs import series_thumbnail
        series_thumbnail(recon_dir.parent, force=True)
    except Exception:
        pass


def open_dialog(recon_dir, root: str, parent, log=None, on_updated=None) -> None:
    """Show the positioning dialog modally, then destroy it on the GUI thread.

    Without the explicit delete the main window (its parent) keeps every closed dialog alive."""
    dlg = PositioningDialog(recon_dir, root, log=log, parent=parent)
    if on_updated is not None:
        dlg.tomogram_updated.connect(on_updated)
    dlg.exec()
    dlg.deleteLater()


class PositioningDialog(QDialog):
    tomogram_updated = Signal(str)        # recon_dir

    def __init__(self, recon_dir, root: str, log=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Position tomogram - {root}")
        self.resize(1400, 860)
        self.recon_dir, self.root = Path(recon_dir), root
        self._log_cb = log
        self.views = None
        self._busy = False
        self._worker: threading.Thread | None = None
        self._bridge = _Bridge(self)
        self._bridge.done.connect(self._on_done)
        self._bridge.log.connect(self._log)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 16, 20, 16)
        lay.addWidget(title_label("Position tomogram"))
        lay.addWidget(dim_label(
            "The views show where the trial tomogram has fine detail (bright = specimen), in XZ (summed over Y) "
            "and YZ (summed over X). Drag the teal line to the top of the specimen and the amber line to its "
            "bottom in both views - include material you want to keep. pyPrep then levels the specimen, "
            "centres it and sets the thickness, and rebuilds the tomogram with IMOD tilt."))

        row = QHBoxLayout()
        row.addWidget(QLabel("Trial thickness:"))
        self.trial_nm = QDoubleSpinBox()
        self.trial_nm.setRange(100, 5000)
        self.trial_nm.setSingleStep(100)
        self.trial_nm.setDecimals(0)
        self.trial_nm.setSuffix(" nm")
        self.trial_nm.setValue(600)
        self.trial_nm.setToolTip("Should be comfortably thicker than the specimen.")
        row.addWidget(self.trial_nm)
        self.btn_trial = QPushButton("Make trial tomogram")
        self.btn_trial.clicked.connect(self.make_trial)
        row.addWidget(self.btn_trial)
        row.addSpacing(20)
        self.status = dim_label("", wrap=False)
        row.addWidget(self.status, 1)
        lay.addLayout(row)

        views = QHBoxLayout()
        self.plots = {}
        for key, title in (("xz", "XZ  (x across, z up)"), ("yz", "YZ  (y across, z up)")):
            c = card()
            cl = QVBoxLayout(c)
            cl.setContentsMargins(8, 8, 8, 8)
            cl.addWidget(section_label(title))
            pw = pg.PlotWidget()
            pw.setAspectLocked(True)
            pw.hideAxis("left")
            pw.hideAxis("bottom")
            img = pg.ImageItem()
            pw.addItem(img)
            top = pg.LineSegmentROI([(0, 1), (1, 1)], pen=pg.mkPen(theme.ACCENT_Hi, width=3))
            bot = pg.LineSegmentROI([(0, 0), (1, 0)], pen=pg.mkPen(theme.WARNING, width=3))
            for roi in (top, bot):
                pw.addItem(roi)
                roi.sigRegionChanged.connect(self._update_readout)
            cl.addWidget(pw)
            views.addWidget(c, 1)
            self.plots[key] = {"widget": pw, "image": img, "top": top, "bot": bot}
        lay.addLayout(views, 1)

        res = card()
        rl = QHBoxLayout(res)
        rl.setContentsMargins(14, 10, 14, 10)
        self.readout = QLabel("Make or load a trial tomogram to start.")
        self.readout.setTextFormat(Qt.RichText)
        rl.addWidget(self.readout, 1)
        rl.addWidget(QLabel("Margin each side:"))
        self.margin = QDoubleSpinBox()
        self.margin.setRange(0, 500)
        self.margin.setDecimals(0)
        self.margin.setSuffix(" nm")
        self.margin.setValue(20)
        self.margin.valueChanged.connect(self._update_readout)
        rl.addWidget(self.margin)
        lay.addWidget(res)

        buttons = QHBoxLayout()
        self.btn_auto = QPushButton("Auto")
        self.btn_auto.setToolTip("Place the lines where the detail energy rises above the background")
        self.btn_auto.clicked.connect(self.auto_lines)
        self.btn_restore = QPushButton("Restore original")
        self.btn_restore.setToolTip("Go back to the positioning batchruntomo chose (tilt.com.pyprep_orig)")
        self.btn_restore.clicked.connect(self.restore)
        self.btn_apply = QPushButton("Rebuild tomogram with these boundaries")
        self.btn_apply.setObjectName("startButton")
        self.btn_apply.clicked.connect(self.apply)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        buttons.addWidget(self.btn_auto)
        buttons.addWidget(self.btn_restore)
        buttons.addStretch(1)
        buttons.addWidget(self.btn_apply)
        buttons.addWidget(close)
        lay.addLayout(buttons)
        self._set_enabled()

        trial = self.recon_dir / P.TRIAL_NAME
        if trial.exists() and trial.stat().st_mtime >= (self.recon_dir / "tilt.com").stat().st_mtime:
            self._run("load", lambda: P.trial_views(trial))
        else:
            self.status.setText("No trial tomogram yet - press Make trial tomogram (about 15 s).")

    # ------------------------------------------------------------------ helpers
    def _pixel_nm(self) -> float:
        if self.views:
            return self.views["pixel_size"] / 10 / self.views["scale"]
        from ..io import mrc
        return (mrc.read_header(self.recon_dir / f"{self.root}_ali.mrc").pixel_size or 10.0) / 10

    def _log(self, text: str):
        self.status.setText(text.splitlines()[0][:140])
        if self._log_cb:
            self._log_cb(text)

    def _set_enabled(self):
        ready = self.views is not None and not self._busy
        self.btn_trial.setEnabled(not self._busy)
        for b in (self.btn_auto, self.btn_apply):
            b.setEnabled(ready)
        self.btn_restore.setEnabled(not self._busy and (self.recon_dir / "tilt.com.pyprep_orig").exists())

    def _run(self, name, fn):
        """Run ``fn`` on a worker thread; its result (or exception) arrives in ``_on_done``.

        ``fn`` must not hold the dialog or any widget - read widget values before starting.
        Together with ``reject`` refusing to close while a worker runs, this keeps the dialog
        from being destroyed on the worker thread (a Qt use-after-free that crashed pyPrep)."""
        self._busy = True
        self._set_enabled()
        bridge = self._bridge

        def work():
            try:
                bridge.done.emit(name, fn())
            except Exception as e:
                bridge.done.emit(name, e)

        self._worker = threading.Thread(target=work, name=f"pyprep-positioning-{name}", daemon=True)
        self._worker.start()

    def reject(self):
        """Close button, Esc and the window's X - refused while IMOD is still working."""
        if self._busy:
            self.status.setText("Still working - wait until it has finished before closing.")
            return
        if self._worker is not None:
            self._worker.join(5)                 # has delivered its result and is exiting
        super().reject()

    # ------------------------------------------------------------------ actions
    def make_trial(self):
        thick_px = int(round(self.trial_nm.value() / self._pixel_nm()))
        self._log(f"Making trial tomogram ({self.trial_nm.value():.0f} nm)…")
        log, recon_dir, root = self._bridge.log.emit, self.recon_dir, self.root
        self._run("trial", lambda: P.trial_views(P.make_trial(recon_dir, root, thick_px, log)))

    def auto_lines(self):
        if not self.views:
            return
        for key in ("xz", "yz"):
            v = self.views[key]
            lo, hi = P.auto_boundaries(v)
            w = v.shape[1]
            self._set_line(self.plots[key]["top"], (0, hi), (w, hi))
            self._set_line(self.plots[key]["bot"], (0, lo), (w, lo))
        self._update_readout()

    @staticmethod
    def _set_line(roi, p1, p2):
        roi.blockSignals(True)
        h = roi.getHandles()
        roi.movePoint(h[0], pg.Point(*p1), finish=False, coords="parent")
        roi.movePoint(h[1], pg.Point(*p2), finish=False, coords="parent")
        roi.blockSignals(False)

    @staticmethod
    def _points(roi):
        return tuple((p.x(), p.y()) for p in (roi.mapToParent(h.pos()) for h in roi.getHandles()))

    def _correction(self):
        pl = self.plots
        return P.correction(self._points(pl["xz"]["top"]), self._points(pl["xz"]["bot"]),
                            self._points(pl["yz"]["top"]), self._points(pl["yz"]["bot"]),
                            self.views["shape"], self.views["scale"])

    def _update_readout(self, *_):
        if not self.views:
            return
        c = self._correction()
        nm = self._pixel_nm()
        final = c.thickness_px * nm + 2 * self.margin.value()
        self.readout.setText(
            f"Specimen <b>{c.thickness_px * nm:.0f} nm</b> thick &nbsp;·&nbsp; tilt correction "
            f"<b>{c.offset_add:+.2f}°</b> &nbsp;·&nbsp; X-axis tilt <b>{c.xtilt_add:+.2f}°</b> &nbsp;·&nbsp; "
            f"Z shift <b>{c.shift_add * nm:+.0f} nm</b> &nbsp;&nbsp;→&nbsp; tomogram <b>{final:.0f} nm</b> "
            f"({int(round(final / nm))} slices)")

    def apply(self):
        if not self.views:
            return
        c = self._correction()
        if c.thickness_px < 4:
            QMessageBox.warning(self, "pyPrep", "The top and bottom lines are too close together.")
            return
        margin_nm = self.margin.value()
        margin_px = int(round(margin_nm / self._pixel_nm()))
        log, recon_dir, root = self._bridge.log.emit, self.recon_dir, self.root
        self._log("Rebuilding the tomogram…")

        def work():
            geom = P.apply_positioning(recon_dir, root, c, margin_px, log)
            _refresh_deconv(recon_dir, log)
            _record(recon_dir, geom, c, margin_nm)
            return geom
        self._run("apply", work)

    def restore(self):
        if QMessageBox.question(self, "pyPrep", "Rebuild the tomogram with the original batchruntomo "
                                                "positioning?") != QMessageBox.Yes:
            return
        log, recon_dir = self._bridge.log.emit, self.recon_dir
        self._log("Restoring the original positioning…")

        def work():
            P.restore_original(recon_dir, log)
            _refresh_deconv(recon_dir, log)
            _record(recon_dir, None, None, None)
        self._run("restore", work)

    def _on_done(self, name, result):
        self._busy = False
        if isinstance(result, Exception):
            self._set_enabled()
            self._log(f"{name} failed: {result}")
            QMessageBox.warning(self, "pyPrep", f"{name.capitalize()} failed:\n{result}")
            return
        if name in ("trial", "load"):
            self.views = result
            for key in ("xz", "yz"):
                v = result[key]
                lo, hi = np.percentile(v, (1, 99.5))
                self.plots[key]["image"].setImage(v, levels=(lo, hi))     # row-major: x across, z up
            self.auto_lines()
            for key in ("xz", "yz"):
                self.plots[key]["widget"].autoRange()
            self._log("Trial tomogram loaded - adjust the lines, then rebuild.")
        elif name in ("apply", "restore"):
            self._log("Tomogram rebuilt." if name == "apply" else "Original positioning restored.")
            self.tomogram_updated.emit(str(self.recon_dir))
        self._set_enabled()
