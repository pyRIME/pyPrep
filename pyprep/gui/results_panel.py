"""Results page: browse stacks and tomograms, motion plots, open in IMOD."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QMessageBox, QProgressBar, QPushButton,
                               QSplitter, QVBoxLayout, QWidget)

from ..io import mrc
from . import theme
from .theme import card, dim_label, title_label

pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)
DISPLAY_MAX = 1100   # longest display edge; bigger data are block-averaged for viewing


def imod_program(name: str) -> str | None:
    """Locate an IMOD executable (3dmod, etomo) via IMOD_DIR or PATH."""
    imod = os.environ.get("IMOD_DIR")
    if imod:
        for ext in (".exe", ".cmd", ""):
            p = Path(imod) / "bin" / f"{name}{ext}"
            if p.exists():
                return str(p)
    return shutil.which(name)


def load_display_stack(path: Path, progress=None, cancel: threading.Event | None = None) -> np.ndarray | None:
    """Whole stack/volume, block-averaged so the longest edge is <= DISPLAY_MAX (display only).

    Reads one section at a time with ordinary file reads (no memory map, so the
    file is never locked against a batch rewriting it).  Runs on a background
    thread; returns None if ``cancel`` is set.
    """
    h = mrc.read_header(path)
    step = max(1, int(np.ceil(max(h.nx, h.ny) / DISPLAY_MAX)))
    out = []
    for z in range(h.nz):
        if cancel is not None and cancel.is_set():
            return None
        sec = mrc.read_sections(path, z, 1, header=h)[0]
        if step > 1:
            ny, nx = (h.ny // step) * step, (h.nx // step) * step
            sec = sec[:ny, :nx].reshape(ny // step, step, nx // step, step).mean((1, 3), dtype=np.float32)
        out.append(sec.astype(np.float32, copy=False))
        if progress is not None:
            progress(z + 1, h.nz)
    return np.stack(out)


class _LoadBridge(QObject):
    """Carries results from loader threads back to the GUI thread (queued signals)."""
    loaded = Signal(int, object)              # request id, (kind, path, volume) or Exception
    progress = Signal(int, int, int)          # request id, sections done, total
    exported = Signal(object)                 # output path, or Exception


def robust_levels(img: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(img[..., ::4, ::4], (0.5, 99.5))
    return float(lo), float(hi if hi > lo else lo + 1)


class ResultsPanel(QWidget):
    position_requested = Signal(str, str)     # recon_dir, root
    redo_requested = Signal(str)              # series name

    def __init__(self, parent=None):
        super().__init__(parent)
        self.out_dir: Path | None = None
        self.record: dict | None = None
        self._angles: list = []
        self._shifts: list = []
        self._showing_tilts = True
        self._request = 0                        # id of the newest display load
        self._load_cancel: threading.Event | None = None
        self._bridge = _LoadBridge(self)
        self._bridge.loaded.connect(self._on_loaded)
        self._bridge.progress.connect(self._on_load_progress)
        self._bridge.exported.connect(self._on_exported)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(22, 18, 22, 18)
        lay.setSpacing(10)
        lay.addWidget(title_label("Results"))
        self.title = dim_label("Select a processed tilt series on the Tilt series page, or use "
                               "'Test on one tilt'.")
        self.title.setTextFormat(Qt.RichText)
        lay.addWidget(self.title)

        top = QHBoxLayout()
        self.stack_combo = QComboBox()
        self.stack_combo.setMinimumWidth(320)
        self.stack_combo.currentIndexChanged.connect(self._show_selected)
        top.addWidget(QLabel("Show:"))
        top.addWidget(self.stack_combo, 1)
        self.btn_3dmod = QPushButton("Open in 3dmod")
        self.btn_3dmod.clicked.connect(self._open_3dmod)
        self.btn_etomo = QPushButton("Open in etomo")
        self.btn_etomo.clicked.connect(self._open_etomo)
        self.btn_folder = QPushButton("Open folder")
        self.btn_folder.clicked.connect(self._open_folder)
        self.btn_tiff = QPushButton("Export TIFF…")
        self.btn_tiff.setToolTip("Save the shown stack/tomogram as an 8-bit ImageJ TIFF (pixel size in nm),\n"
                                 "e.g. for Fiji or segmentation tools.")
        self.btn_tiff.clicked.connect(self._export_tiff)
        self.btn_position = QPushButton("Position tomogram…")
        self.btn_position.setToolTip("Mark the top and bottom of the specimen and rebuild the tomogram\n"
                                     "level, centred and at the right thickness.")
        self.btn_position.clicked.connect(self._request_position)
        self.btn_redo = QPushButton("Redo reconstruction")
        self.btn_redo.setToolTip("Run batchruntomo again for this series with the current Reconstruction settings.")
        self.btn_redo.clicked.connect(self._request_redo)
        for b in (self.btn_position, self.btn_redo, self.btn_3dmod, self.btn_etomo, self.btn_tiff, self.btn_folder):
            top.addWidget(b)
        lay.addLayout(top)
        self.loading = QProgressBar()
        self.loading.setMaximumHeight(22)
        self.loading.hide()
        lay.addWidget(self.loading)

        split = QSplitter(Qt.Vertical)
        view_card = card()
        vl = QVBoxLayout(view_card)
        vl.setContentsMargins(6, 6, 6, 6)
        self.image = pg.ImageView()
        self.image.ui.roiBtn.hide()
        self.image.ui.menuBtn.hide()
        self.image.sigTimeChanged.connect(self._time_changed)
        vl.addWidget(self.image)
        split.addWidget(view_card)

        plots = QWidget()
        pl = QHBoxLayout(plots)
        pl.setContentsMargins(0, 0, 0, 0)
        accent = pg.mkColor(theme.ACCENT)
        self.drift_plot = pg.PlotWidget()
        self.drift_plot.setLabel("bottom", "tilt angle (°)")
        self.drift_plot.showGrid(x=True, y=True, alpha=0.25)
        self.drift_scatter = pg.ScatterPlotItem(size=8, brush=pg.mkBrush(accent), pen=pg.mkPen(None))
        self.drift_scatter.sigClicked.connect(self._drift_clicked)
        self.drift_plot.addItem(self.drift_scatter)
        self.expected_curve = pg.PlotDataItem(pen=pg.mkPen(theme.TEXT_DIM, width=1, style=Qt.DashLine))
        self.drift_plot.addItem(self.expected_curve)
        self.drift_marker = pg.InfiniteLine(angle=90, pen=pg.mkPen(theme.WARNING, width=1))
        self.drift_plot.addItem(self.drift_marker)
        self.ctf_data_curve = pg.PlotDataItem(pen=pg.mkPen(theme.TEXT_DIM, width=1))
        self.ctf_model_curve = pg.PlotDataItem(pen=pg.mkPen(theme.ACCENT_Hi, width=2))
        self.drift_plot.addItem(self.ctf_data_curve)
        self.drift_plot.addItem(self.ctf_model_curve)
        self._ctf, self._ctf_curves, self._ctf_index = None, None, 0
        self.plot_kind = QComboBox()
        self.plot_kind.addItem("Drift per tilt", "drift")
        self.plot_kind.addItem("Intensity per tilt (QC)", "intensity")
        self.plot_kind.addItem("Alignment score per tilt", "score")
        self.plot_kind.addItem("Defocus per tilt (CTF)", "defocus")
        self.plot_kind.addItem("CTF fit of the current tilt", "ctffit")
        self.plot_kind.setToolTip("Red points are tilts flagged by quality control.")
        self.plot_kind.currentIndexChanged.connect(self._draw_series_plot)
        self.traj_plot = pg.PlotWidget(title="Frame trajectory")
        self.traj_plot.setLabel("bottom", "x shift (Å)")
        self.traj_plot.setLabel("left", "y shift (Å)")
        self.traj_plot.setAspectLocked(True)
        self.traj_plot.showGrid(x=True, y=True, alpha=0.25)
        for p in (self.drift_plot, self.traj_plot):
            for side in ("left", "bottom"):
                p.getAxis(side).enableAutoSIPrefix(False)   # keep plain Å / degrees, no x0.001 scaling
            c = card()
            cl = QVBoxLayout(c)
            cl.setContentsMargins(6, 6, 6, 6)
            if p is self.drift_plot:
                cl.addWidget(self.plot_kind)
            cl.addWidget(p)
            pl.addWidget(c)
        split.addWidget(plots)
        split.setSizes([620, 240])
        lay.addWidget(split, 1)
        self._set_buttons()

    # ------------------------------------------------------------------ series results
    def clear(self, message: str = "No results yet for this tilt series."):
        if self._load_cancel is not None:
            self._load_cancel.set()
        self._request += 1
        self.loading.hide()
        self.out_dir, self.record = None, None
        self.stack_combo.clear()
        self.image.clear()
        self.drift_scatter.clear()
        self.traj_plot.clear()
        self.title.setText(message)
        self._set_buttons()

    def load_series(self, out_dir: Path, prefer: str = "stack") -> bool:
        """Show results for a series output folder; returns False if there are none.

        ``prefer`` picks what to display first: "stack" (binned tilt series) or "tomo".
        """
        out_dir = Path(out_dir)
        j = next(iter(sorted(out_dir.glob("*_pyprep.json"))), None) if out_dir.is_dir() else None
        try:
            rec = json.loads(j.read_text()) if j else None
        except (OSError, ValueError):
            rec = None
        if rec is None:
            self.clear()
            return False
        self.out_dir, self.record = out_dir, rec
        tilts = rec.get("tilts", [])
        self._angles = [t["angle"] for t in tilts]
        self._shifts = [np.asarray(t["shifts_px"]) for t in tilts]
        self._tilt_records = tilts
        self._qc_fit = (rec.get("qc") or {}).get("intensity_fit")
        self._ctf = rec.get("ctf")
        self._ctf_curves = None
        npz = out_dir / f"{rec.get('series', '')}_ctf.npz"
        if self._ctf and npz.exists():
            try:
                with np.load(npz) as z:
                    self._ctf_curves = (z["k"], z["data"], z["model"])
            except (OSError, ValueError, KeyError):
                pass
        self._ctf_index = int(np.argmin(np.abs(self._angles))) if self._angles else 0
        self._draw_series_plot()
        self.stack_combo.blockSignals(True)
        self.stack_combo.clear()
        recon = rec.get("reconstruction") or {}
        if not recon:
            # Reconstructions run outside the batch still leave their record on disk.
            for j in sorted(out_dir.glob("imod_bin*/pyprep_recon.json")):
                try:
                    recon = json.loads(j.read_text())
                    rec["reconstruction"] = recon
                except (OSError, ValueError):
                    pass
        deconv = recon.get("deconvolved")
        if deconv and Path(deconv).exists():
            self.stack_combo.addItem(f"Tomogram, deconvolved   {Path(deconv).name}   ({Path(deconv).parent.name})",
                                     ("tomo", deconv))
        tomo = recon.get("tomogram")
        if tomo and Path(tomo).exists():
            self.stack_combo.addItem(f"Tomogram   {Path(tomo).name}   ({Path(tomo).parent.name})", ("tomo", tomo))
        for o in sorted(rec.get("outputs", []), key=lambda o: (-o["bin"], o["kind"] != "sum")):
            p = Path(o["path"])
            if not p.exists():
                p = out_dir / p.name
            nx, ny, nz = o["size"]
            self.stack_combo.addItem(f"Tilt series   {p.name}   ({nx} x {ny} x {nz}, {o['pixel_size']:.2f} A/px)",
                                     ("stack", str(p)))
        self.stack_combo.blockSignals(False)
        status = rec.get("status", "?")
        parts = [f"<b>{rec['series']}</b>: stacks {status}, {len(tilts)} tilts"]
        if rec.get("missing"):
            parts.append(f"<span style='color:{theme.WARNING}'>{len(rec['missing'])} missing</span>")
        if status == "complete":
            parts.append(f"{rec.get('seconds', 0):.0f} s on {rec.get('device', '')}")
        if recon:
            color = theme.OK if recon.get("status") == "complete" else theme.DANGER
            parts.append(f"<span style='color:{color}'>tomogram {recon.get('status')}</span>"
                         + (f" ({recon.get('preset')}, {recon.get('seconds', 0):.0f} s)" if recon.get("seconds") else ""))
        if self._ctf:
            parts.append(f"defocus {self._ctf.get('defocus_um', 0):.2f} µm")
        if (rec.get("image_frame") or {}).get("invert_y"):
            parts.append("images inverted in Y (Tomography 5)")
        if (rec.get("handedness") or {}).get("flipped"):
            parts.append(f"tilt axis rotated 180° by the CTF check ({rec['handedness']['tilt_axis']:.1f}°)")
        if rec.get("error"):
            parts.append(f"<span style='color:{theme.DANGER}'>{rec['error']}</span>")
        self.title.setText(" &nbsp;·&nbsp; ".join(parts))
        self._set_buttons()
        if self.stack_combo.count():
            # Binned tilt series for a quick first look, or the tomogram after a batch.
            idx = next((i for i in range(self.stack_combo.count())
                        if self.stack_combo.itemData(i)[0] == prefer), None)
            if idx is None:
                idx = next((i for i in range(self.stack_combo.count())
                            if self.stack_combo.itemData(i)[0] == "stack"), 0)
            self.stack_combo.setCurrentIndex(idx)
            self._show_selected()
        return True

    def _show_selected(self):
        """Load the selected stack/tomogram on a background thread (never block the window)."""
        data = self.stack_combo.currentData()
        if not data or not Path(data[1]).exists():
            return
        kind, path = data
        if self._load_cancel is not None:
            self._load_cancel.set()                  # a newer choice supersedes a running load
        self._request += 1
        req, cancel = self._request, threading.Event()
        self._load_cancel = cancel
        self.loading.setRange(0, 0)
        self.loading.setFormat(f"Loading {Path(path).name}…")
        self.loading.show()
        bridge = self._bridge

        def work():
            try:
                vol = load_display_stack(Path(path), lambda d, n: bridge.progress.emit(req, d, n), cancel)
                if vol is not None:
                    bridge.loaded.emit(req, (kind, path, vol))
            except Exception as e:  # reported on the GUI thread
                try:
                    bridge.loaded.emit(req, e)
                except RuntimeError:                 # window closed while loading
                    pass

        threading.Thread(target=work, name="pyprep-display-load", daemon=True).start()

    def _on_load_progress(self, req, done, total):
        if req == -1:                                # TIFF export
            self.loading.setRange(0, total)
            self.loading.setValue(done)
            self.loading.setFormat("Exporting… %v / %m sections")
            return
        if req == self._request:
            self.loading.setRange(0, total)
            self.loading.setValue(done)
            self.loading.setFormat(f"Loading… %v / %m sections")

    def _on_loaded(self, req, payload):
        if req != self._request:
            return                                   # superseded by a newer selection
        self.loading.hide()
        self._load_cancel = None
        if isinstance(payload, Exception):
            self.title.setText(self.title.text() + f"<br><span style='color:{theme.DANGER}'>"
                               f"Could not display: {payload}</span>")
            return
        kind, path, vol = payload
        self._showing_tilts = kind == "stack"
        xvals = (np.asarray(self._angles, dtype=float)
                 if self._showing_tilts and len(self._angles) == len(vol) else None)
        mid = len(vol) // 2
        self.image.setImage(vol, xvals=xvals, autoLevels=False, levels=robust_levels(vol[mid]))
        self.image.setCurrentIndex(mid)
        self._time_changed(mid, None)
        self._set_buttons()

    def _time_changed(self, ind, _time):
        if not self._showing_tilts or not self._shifts or ind >= len(self._shifts):
            return
        s = self._shifts[ind] * self._pixel_size()
        self.traj_plot.clear()
        accent = theme.ACCENT
        self.traj_plot.plot(s[:, 1], s[:, 0], pen=pg.mkPen(accent, width=2), symbol="o", symbolSize=7,
                            symbolBrush=accent, symbolPen=None)
        if len(s):
            self.traj_plot.plot([s[0, 1]], [s[0, 0]], pen=None, symbol="s", symbolSize=10,
                                symbolBrush=theme.WARNING, symbolPen=None)
        self.traj_plot.setTitle(f"Frame trajectory at {self._angles[ind]:+.1f} deg (square = first frame)")
        self.drift_marker.setValue(self._angles[ind])
        self._ctf_index = ind
        if self.plot_kind.currentData() == "ctffit":
            self._draw_series_plot()

    def _draw_series_plot(self):
        """Drift, QC intensity or alignment score against tilt angle; flagged tilts in red."""
        tilts = getattr(self, "_tilt_records", None) or []
        kind = self.plot_kind.currentData()
        self.expected_curve.setData([], [])
        self.ctf_data_curve.setData([], [])
        self.ctf_model_curve.setData([], [])
        self.drift_plot.setTitle(None)
        fit_view = kind == "ctffit"
        self.drift_marker.setVisible(not fit_view)
        self.drift_scatter.setVisible(not fit_view)
        self.drift_plot.setLabel("bottom", "spatial frequency (1/Å)" if fit_view else "tilt angle (°)")
        if not tilts:
            return
        if fit_view:
            self._draw_ctf_fit()
            return
        ang = np.array([t["angle"] for t in tilts])
        ctf_tilts = (self._ctf or {}).get("tilts") or []
        if kind == "intensity" and all("mean_counts" in t for t in tilts):
            y = np.array([t["mean_counts"] / t.get("exposure", 1.0) for t in tilts])
            label = "counts per second"
            fit = self._qc_fit
            if fit:
                xs = np.linspace(ang.min(), ang.max(), 200)
                self.expected_curve.setData(xs, np.exp(fit["a"] - fit["b"] / np.cos(np.radians(xs - fit["offset"]))))
        elif kind == "score":
            y = np.array([float(np.mean(t["scores"])) for t in tilts])
            label = "alignment score"
        elif kind == "defocus":
            if len(ctf_tilts) != len(tilts):
                self.drift_scatter.setData([], [])
                self.drift_plot.setLabel("left", "no CTF estimate for this series")
                return
            y = np.array([c["defocus_um"] for c in ctf_tilts])
            res = np.array([c["resolution_A"] for c in ctf_tilts])
            poor = res > 1.5 * np.nanmedian(res)
            brushes = [pg.mkBrush(theme.DANGER if b else theme.ACCENT) for b in poor]
            self.drift_scatter.setData(ang, y, brush=brushes)
            self.drift_plot.setLabel("left", "defocus (µm)")
            self.drift_plot.setTitle("red: CTF fit resolution much worse than the series median")
            return
        else:
            y = np.array([t["drift_A"] for t in tilts])
            label = "total drift (Å)"
        bad = [any(f in ("dark", "drift", "low score") for f in t.get("flags", [])) for t in tilts]
        brushes = [pg.mkBrush(theme.DANGER if b else theme.ACCENT) for b in bad]
        self.drift_scatter.setData(ang, y, brush=brushes)
        self.drift_plot.setLabel("left", label)

    def _draw_ctf_fit(self):
        """Tile-averaged, background-flattened spectrum of one tilt with the fitted CTF model."""
        if self._ctf_curves is None:
            self.drift_plot.setLabel("left", "no CTF estimate for this series")
            return
        k, data, model = self._ctf_curves
        i = min(max(self._ctf_index, 0), len(data) - 1)
        self.ctf_data_curve.setData(k, data[i])
        self.ctf_model_curve.setData(k, model[i])
        self.drift_plot.setLabel("left", "flattened amplitude")
        t = ((self._ctf or {}).get("tilts") or [{}] * len(data))[i]
        if t:
            self.drift_plot.setTitle(f"{t['angle']:+.1f}°: defocus {t['defocus_um']:.2f} µm, "
                                     f"fit to {t['resolution_A']:.1f} Å (teal = model)")
        self.drift_plot.autoRange()

    def _drift_clicked(self, _item, points, *args):
        if points is not None and len(points) and self._showing_tilts:
            idx = int(np.argmin([abs(a - points[0].pos().x()) for a in self._angles]))
            self.image.setCurrentIndex(idx)

    def _pixel_size(self) -> float:
        rec = self.record or {}
        for o in rec.get("outputs", []):
            return o["pixel_size"] / o["bin"]
        return rec.get("pixel_size", 1.0)

    # ------------------------------------------------------------------ single-tilt preview
    def show_preview(self, res: dict, pixel_size: float):
        if self._load_cancel is not None:
            self._load_cancel.set()
        self._request += 1                           # a pending stack load must not overwrite this
        self.loading.hide()
        self.out_dir, self.record = None, {"pixel_size": pixel_size}
        self.stack_combo.clear()
        stack = np.stack([res["unaligned"], res["aligned"]])
        self._showing_tilts = True
        self._angles = [res["tilt"].angle] * 2
        self._shifts = [res["shifts"], res["shifts"]]
        self.image.setImage(stack, xvals=np.array([0.0, 1.0]), autoLevels=False,
                            levels=robust_levels(res["aligned"]))
        self.image.setCurrentIndex(1)
        self._tilt_records = []
        self.expected_curve.setData([], [])
        self.drift_scatter.setData([res["tilt"].angle], [res["drift"]])
        self._time_changed(1, None)
        conv = "converged" if res["converged"] else "NOT converged"
        self.title.setText(
            f"<b>Preview</b> of tilt {res['tilt'].zvalue + 1:03d} ({res['tilt'].angle:+.2f} deg), "
            f"bin {res['bin']}: drift {res['drift']:.2f} A, {res['iterations']} iterations ({conv}), "
            f"mean score {np.mean(res['scores']):.3f}, {res['seconds']:.2f} s. &nbsp;Slider: 0 = unaligned, "
            f"1 = aligned. Nothing was written to disk."
            + (f"<br>{res['movie']}" if res.get("movie") else ""))
        self._set_buttons()

    # ------------------------------------------------------------------ IMOD / folders
    def _edf(self) -> Path | None:
        recon = (self.record or {}).get("reconstruction") or {}
        d = recon.get("recon_dir")
        if d and Path(d).is_dir():
            edfs = sorted(Path(d).glob("*.edf"))
            return edfs[0] if edfs else None
        return None

    def _export_tiff(self):
        from PySide6.QtWidgets import QFileDialog
        data = self.stack_combo.currentData()
        if not data:
            return
        src = Path(data[1])
        dst, _ = QFileDialog.getSaveFileName(self, "Export TIFF", str(src.with_suffix(".tif")), "TIFF (*.tif)")
        if not dst:
            return
        self.btn_tiff.setEnabled(False)
        self.loading.setRange(0, 0)
        self.loading.setFormat(f"Exporting {Path(dst).name}…")
        self.loading.show()
        bridge = self._bridge

        def work():
            from ..export import export_tiff
            try:
                bridge.exported.emit(export_tiff(src, dst, progress=lambda d, n: bridge.progress.emit(-1, d, n)))
            except Exception as e:
                bridge.exported.emit(e)

        threading.Thread(target=work, name="pyprep-tiff-export", daemon=True).start()

    def _on_exported(self, result):
        self.loading.hide()
        self._set_buttons()
        if isinstance(result, Exception):
            QMessageBox.warning(self, "pyPrep", f"TIFF export failed:\n{result}")
        elif result is not None:
            self.title.setText(self.title.text() + f"<br>Exported {Path(result).name}")

    def _request_position(self):
        recon = (self.record or {}).get("reconstruction") or {}
        d = recon.get("recon_dir")
        if d and (Path(d) / "tilt.com").exists():
            self.position_requested.emit(d, self.record["series"])

    def _request_redo(self):
        if self.record and self.record.get("series"):
            self.redo_requested.emit(self.record["series"])

    def _set_buttons(self):
        data = self.stack_combo.currentData()
        recon = (self.record or {}).get("reconstruction") or {}
        can_position = bool(recon.get("recon_dir")) and (Path(recon["recon_dir"]) / "tilt.com").exists()
        self.btn_position.setEnabled(can_position)
        self.btn_redo.setEnabled(bool((self.record or {}).get("series")) and self.out_dir is not None)
        self.btn_3dmod.setEnabled(bool(data) and imod_program("3dmod") is not None)
        self.btn_tiff.setEnabled(bool(data))
        self.btn_etomo.setEnabled(self._edf() is not None and imod_program("etomo") is not None)
        self.btn_folder.setEnabled(self.out_dir is not None)

    def _launch(self, exe, args, cwd):
        try:
            subprocess.Popen([exe, *args], cwd=str(cwd))
        except OSError as e:
            QMessageBox.warning(self, "pyPrep", f"Could not start {Path(exe).name}:\n{e}")

    def _open_3dmod(self):
        exe = imod_program("3dmod")
        data = self.stack_combo.currentData()
        if exe and data:
            self._launch(exe, [data[1]], Path(data[1]).parent)

    def _open_etomo(self):
        exe, edf = imod_program("etomo"), self._edf()
        if exe and edf:
            self._launch(exe, [edf.name], edf.parent)

    def _open_folder(self):
        if self.out_dir:
            os.startfile(str(self.out_dir))
