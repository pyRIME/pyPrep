"""pyPrep main window, laid out like pyFIB: top bar, left navigation, stacked pages."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from PySide6.QtCore import QSettings, Qt, QThread
from PySide6.QtGui import QAction, QBrush, QColor, QIcon
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QDialog, QFileDialog,
                               QSystemTrayIcon,
                               QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox,
                               QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSplitter,
                               QStackedWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from .. import __version__, gpu, imod, qc
from ..pipeline import estimate_output_bytes, is_complete, recon_complete
from ..settings import ProcessingSettings
from ..tiltseries import TiltSeries, find_mdocs, load_tilt_series
from . import theme
from .gallery_panel import GalleryPanel
from .results_panel import ResultsPanel
from .settings_panel import SettingsForms
from .theme import card, dim_label, section_label, title_label
from .workers import BatchWorker, PreviewWorker

APP_TITLE = "pyPrep"
NAV = [
    ("series", "Tilt series"),
    ("alignment", "Frame alignment"),
    ("outputs", "Outputs"),
    ("recon", "Reconstruction"),
    ("results", "Results"),
    ("gallery", "Gallery"),
    ("log", "Log"),
]
STATUS_COLORS = {"done": theme.OK, "running": theme.ACCENT_Hi, "failed": theme.DANGER,
                 "cancelled": theme.WARNING, "queued": theme.TEXT_DIM, "ready": theme.TEXT_DIM,
                 "-": theme.TEXT_DIM, "no frames": theme.DANGER}
COL_NAME, COL_TILTS, COL_MISSING, COL_QC, COL_RANGE, COL_PIXEL, COL_STACKS, COL_TOMO, COL_TIME = range(9)
T_USE, T_ACQ, T_ANGLE, T_DOSE, T_PRIOR, T_QC, T_FILE = range(7)


def _item(text, align_right=False) -> QTableWidgetItem:
    it = QTableWidgetItem(str(text))
    it.setFlags(it.flags() & ~Qt.ItemIsEditable)
    if align_right:
        it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
    return it


def _scroll(widget: QWidget) -> QScrollArea:
    sa = QScrollArea()
    sa.setWidget(widget)
    sa.setWidgetResizable(True)
    sa.setFrameShape(QScrollArea.NoFrame)
    return sa


class PathEdit(QWidget):
    """Line edit + Browse button for a folder."""

    def __init__(self, caption: str, placeholder: str, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self.caption = caption
        self.edit = QLineEdit()
        self.edit.setPlaceholderText(placeholder)
        btn = QPushButton("Browse…")
        btn.clicked.connect(self._browse)
        lay.addWidget(self.edit, 1)
        lay.addWidget(btn)

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, self.caption, self.edit.text() or "")
        if d:
            self.edit.setText(str(Path(d)))

    def path(self) -> Path | None:
        t = self.edit.text().strip().strip('"')
        return Path(t) if t else None


class TopBar(QWidget):
    """Shared across pages (like pyFIB's ArmBar): session + output folders."""

    def __init__(self, on_scan, parent=None):
        super().__init__(parent)
        self.setObjectName("topBar")
        self.setAttribute(Qt.WA_StyledBackground, True)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(10)
        lay.addWidget(QLabel("Session:"))
        self.session = PathEdit("Session folder", "folder with the tilt-series .mdoc files")
        lay.addWidget(self.session, 3)
        lay.addSpacing(10)
        lay.addWidget(QLabel("Output:"))
        self.output = PathEdit("Output folder", "where stacks and tomograms go")
        lay.addWidget(self.output, 3)
        lay.addSpacing(10)
        self.recursive = QCheckBox("Subfolders")
        self.recursive.setToolTip("Also search subfolders of the session folder for .mdoc files")
        lay.addWidget(self.recursive)
        self.scan_btn = QPushButton("Find tilt series")
        self.scan_btn.setObjectName("startButton")
        self.scan_btn.clicked.connect(on_scan)
        lay.addWidget(self.scan_btn)


class LogPage(QWidget):
    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(22, 18, 22, 18)
        lay.addWidget(title_label("Log"))
        lay.addWidget(dim_label("Everything pyPrep and IMOD report. Each series also keeps its own "
                                "<name>_pyprep.log next to its outputs."))
        self.view = QPlainTextEdit()
        self.view.setReadOnly(True)
        self.view.setMaximumBlockCount(50000)
        self.view.setStyleSheet(f"font-family: Consolas, 'Courier New', monospace; font-size: 13px; "
                                f"background: {theme.CANVAS_BG};")
        lay.addWidget(self.view, 1)
        row = QHBoxLayout()
        row.addStretch(1)
        b = QPushButton("Clear")
        b.clicked.connect(self.view.clear)
        row.addWidget(b)
        lay.addLayout(row)

    def append(self, text: str):
        self.view.appendPlainText(text.rstrip())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_TITLE} {__version__}")
        self.qs = QSettings("pyPrep", "pyPrep")
        self.series: list[TiltSeries] = []
        self._thread: QThread | None = None
        self._worker = None
        self._on_done = None
        self._preview_series = None
        self._batch: dict | None = None
        self._prerun: dict = {}
        self._tray = None

        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.top = TopBar(self.scan)
        outer.addWidget(self.top)

        content = QWidget()
        root = QHBoxLayout(content)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        outer.addWidget(content, 1)

        # ---------------- pages
        self.forms = SettingsForms(self)
        self.forms.changed.connect(self._settings_changed)
        self.forms.preview_requested.connect(self.preview)
        self.forms.show_directives_requested.connect(self.show_directives)
        self.results = ResultsPanel()
        self.results.position_requested.connect(self.open_positioning)
        self.results.redo_requested.connect(self.redo_reconstruction)
        self.gallery = GalleryPanel()
        self.gallery.open_series.connect(self._open_series_from_gallery)
        self.log_page = LogPage()
        self._pages = {
            "series": self._build_series_page(),
            "alignment": _scroll(self.forms.alignment_page),
            "outputs": _scroll(self.forms.outputs_page),
            "recon": _scroll(self.forms.recon_page),
            "results": self.results,
            "gallery": self.gallery,
            "log": self.log_page,
        }
        self.stack = QStackedWidget()
        for key, _ in NAV:
            self.stack.addWidget(self._pages[key])

        root.addWidget(self._build_nav())
        root.addWidget(self.stack, 1)

        self._build_menu()
        devs = gpu.list_gpus()
        ok, imod_msg = imod.imod_status()
        self.statusBar().addPermanentWidget(QLabel(
            (f"GPU: {devs[0]}" if devs else "No CUDA GPU - CPU mode") + "   ·   "
            + (f"IMOD {imod.imod_version()}" if ok else "IMOD not found") + "  "))
        self.statusBar().showMessage("Ready")
        self._restore()
        self.show_page("series")

    # ------------------------------------------------------------------ building
    def _build_nav(self) -> QWidget:
        nav = QWidget()
        nav.setObjectName("navBar")
        nav.setAttribute(Qt.WA_StyledBackground, True)
        nav.setFixedWidth(230)
        lay = QVBoxLayout(nav)
        lay.setContentsMargins(12, 16, 12, 14)
        lay.setSpacing(6)
        brand = QLabel(APP_TITLE)
        brand.setStyleSheet(f"font-size: 28px; font-weight: 800; color: {theme.TEXT};")
        lay.addWidget(brand)
        sub = dim_label("cryo-ET tilt-series prep")
        lay.addWidget(sub)
        lay.addSpacing(12)
        self._nav_buttons: dict[str, QPushButton] = {}
        for key, label in NAV:
            b = QPushButton(label)
            b.setObjectName("toolButton")
            b.setCheckable(True)
            b.setMinimumHeight(44)
            b.clicked.connect(lambda _=False, k=key: self.show_page(k))
            lay.addWidget(b)
            self._nav_buttons[key] = b
        lay.addStretch(1)

        run = card()
        rl = QVBoxLayout(run)
        rl.setContentsMargins(10, 10, 10, 10)
        rl.setSpacing(6)
        rl.addWidget(section_label("Batch"))
        self.run_msg = dim_label("Idle")
        rl.addWidget(self.run_msg)
        self.overall = QProgressBar()
        self.overall.setRange(0, 1)
        self.overall.setValue(0)
        self.overall.setFormat("no batch running")
        self.current = QProgressBar()
        self.current.setFormat("%p%")
        rl.addWidget(self.overall)
        rl.addWidget(self.current)
        self.btn_start = QPushButton("Start")
        self.btn_start.setObjectName("startButton")
        self.btn_start.setToolTip("Process every checked tilt series with the current settings")
        self.btn_start.clicked.connect(self.start)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setObjectName("cancelButton")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel)
        rl.addWidget(self.btn_start)
        rl.addWidget(self.btn_cancel)
        lay.addWidget(run)
        return nav

    def _build_series_page(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(22, 18, 22, 18)
        lay.setSpacing(10)
        lay.addWidget(title_label("Tilt series"))
        lay.addWidget(dim_label(
            "pyPrep finds every tilt-series .mdoc in the session folder and matches each tilt to its "
            "fraction file. Check the series to process; uncheck single tilts below to leave them out."))
        row = QHBoxLayout()
        row.addWidget(QLabel("Fractions folder:"))
        self.frames_path = PathEdit("Fractions folder", "optional - only if the fraction files are not next to the mdoc")
        row.addWidget(self.frames_path, 1)
        lay.addLayout(row)

        split = QSplitter(Qt.Vertical)
        c = card()
        cl = QVBoxLayout(c)
        cl.setContentsMargins(12, 12, 12, 12)
        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels(["Tilt series", "Tilts", "Missing", "QC", "Range (deg)",
                                              "Pixel (A)", "Stacks", "Tomogram", "Time"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().hide()
        self.table.setShowGrid(False)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(COL_NAME, QHeaderView.Stretch)
        for col in range(1, 9):
            hh.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._series_selected)
        cl.addWidget(self.table, 1)
        row = QHBoxLayout()
        for text, state in (("Check all", True), ("Uncheck all", False)):
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, s=state: self._check_all(s))
            row.addWidget(b)
        row.addStretch()
        self.series_count = dim_label("", wrap=False)
        row.addWidget(self.series_count)
        cl.addLayout(row)
        split.addWidget(c)

        c = card()
        cl = QVBoxLayout(c)
        cl.setContentsMargins(12, 12, 12, 12)
        self.tilts_title = section_label("Tilts")
        cl.addWidget(self.tilts_title)
        self.tilts_label = dim_label("Select a tilt series.")
        cl.addWidget(self.tilts_label)
        self.tilt_table = QTableWidget(0, 7)
        self.tilt_table.setHorizontalHeaderLabels(["Use", "Acq #", "Angle", "Dose", "Prior dose", "QC",
                                                   "Fraction file"])
        self.tilt_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tilt_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tilt_table.setAlternatingRowColors(True)
        self.tilt_table.setShowGrid(False)
        self.tilt_table.verticalHeader().hide()
        hh = self.tilt_table.horizontalHeader()
        for col in range(T_FILE):
            hh.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(T_FILE, QHeaderView.Stretch)
        self.tilt_table.itemChanged.connect(self._tilt_toggled)
        cl.addWidget(self.tilt_table, 1)
        row = QHBoxLayout()
        row.addWidget(dim_label("Select a tilt, then test the alignment settings on it.", wrap=False))
        row.addStretch()
        self.btn_exclude = QPushButton("Exclude flagged tilts")
        self.btn_exclude.setToolTip("Untick tilts flagged as dark, drifting or poorly aligned\n"
                                    "(hover over a QC entry to see why it was flagged).")
        self.btn_exclude.clicked.connect(self.exclude_flagged)
        row.addWidget(self.btn_exclude)
        self.btn_preview = QPushButton("Test on one tilt")
        self.btn_preview.setObjectName("toolButton")
        self.btn_preview.clicked.connect(self.preview)
        row.addWidget(self.btn_preview)
        cl.addLayout(row)
        split.addWidget(c)
        split.setSizes([320, 420])
        lay.addWidget(split, 1)
        return page

    def _build_menu(self):
        m = self.menuBar().addMenu("&File")
        for text, slot in (("Load settings…", self.forms.load_dialog), ("Save settings…", self.forms.save_dialog)):
            a = QAction(text, self)
            a.triggered.connect(slot)
            m.addAction(a)
        m.addSeparator()
        a = QAction("Quit", self)
        a.triggered.connect(self.close)
        m.addAction(a)
        h = self.menuBar().addMenu("&Help")
        a = QAction("About pyPrep", self)
        a.triggered.connect(lambda: QMessageBox.about(
            self, "pyPrep", f"<b>pyPrep {__version__}</b><br>Cryo-ET tilt-series preparation: GPU frame "
                            f"alignment, etomo-ready stacks and IMOD batchruntomo reconstruction.<br><br>"
                            f"Compute: {gpu.device_summary(gpu.select_device())}<br>{imod.imod_status()[1]}"))
        h.addAction(a)

    # ------------------------------------------------------------------ navigation
    def show_page(self, key: str):
        if key not in self._pages:
            return
        self.stack.setCurrentWidget(self._pages[key])
        for k, b in self._nav_buttons.items():
            b.setChecked(k == key)
        if key == "gallery":
            self.gallery.set_output(self.top.output.path())

    def open_positioning(self, recon_dir: str, root: str):
        from .positioning_dialog import open_dialog
        open_dialog(recon_dir, root, self, log=self.append_log, on_updated=self._tomogram_updated)

    def _tomogram_updated(self, recon_dir: str):
        self.results.load_series(Path(recon_dir).parent, prefer="tomo")
        self.gallery.refresh()

    def redo_reconstruction(self, name: str):
        row = next((r for r, ts in enumerate(self.series) if ts.name == name), None)
        if row is None:
            QMessageBox.information(self, "pyPrep", f"{name} is not in the current session list - "
                                                    "use Find tilt series first.")
            return
        settings = self.forms.get_settings()
        if not settings.recon.enabled:
            QMessageBox.information(self, "pyPrep", "Reconstruction is turned off on the Reconstruction page.")
            return
        if QMessageBox.question(self, "pyPrep", f"Run batchruntomo again for {name}? The current tomogram and "
                                                "any manual positioning will be replaced.") != QMessageBox.Yes:
            return
        self._start_jobs([(row, self.series[row])], settings, force_recon=True)

    def _open_series_from_gallery(self, name: str):
        for r, ts in enumerate(self.series):
            if ts.name == name:
                self.table.selectRow(r)
                break
        out_root = self.top.output.path()
        if out_root is not None:
            self.results.load_series(out_root / name, prefer="tomo")
        self.show_page("results")

    # ------------------------------------------------------------------ persistence
    def _restore(self):
        self.top.session.edit.setText(self.qs.value("input_dir", ""))
        self.frames_path.edit.setText(self.qs.value("frames_dir", ""))
        self.top.output.edit.setText(self.qs.value("output_dir", ""))
        self.top.recursive.setChecked(self.qs.value("recursive", "false") == "true")
        raw = self.qs.value("settings", "")
        if raw:
            try:
                self.forms.set_settings(ProcessingSettings.from_dict(json.loads(raw)))
            except (ValueError, TypeError, KeyError):
                pass
        geo = self.qs.value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)
        else:
            self.resize(1560, 980)

    def _save_state(self):
        self.qs.setValue("input_dir", self.top.session.edit.text())
        self.qs.setValue("frames_dir", self.frames_path.edit.text())
        self.qs.setValue("output_dir", self.top.output.edit.text())
        self.qs.setValue("recursive", "true" if self.top.recursive.isChecked() else "false")
        self.qs.setValue("settings", json.dumps(self.forms.get_settings().to_dict()))
        self.qs.setValue("geometry", self.saveGeometry())

    def closeEvent(self, ev):
        if self._thread is not None and self._thread.isRunning():
            if QMessageBox.question(self, "pyPrep", "Processing is running. Cancel it and quit?") != QMessageBox.Yes:
                ev.ignore()
                return
            self.cancel()
            self._thread.quit()
            self._thread.wait(60000)
        self._save_state()
        ev.accept()

    # ------------------------------------------------------------------ scanning
    def scan(self):
        folder = self.top.session.path()
        if not folder or not folder.is_dir():
            QMessageBox.warning(self, "pyPrep", "Choose the session folder that contains the .mdoc files.")
            return
        if not self.top.output.path():
            self.top.output.edit.setText(str(folder / "pyPrep"))
        frames_dir = self.frames_path.path()
        s = self.forms.get_settings()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.series = []
            for m in find_mdocs(folder, self.top.recursive.isChecked()):
                try:
                    self.series.append(load_tilt_series(m, frames_dir, s.default_dose))
                except Exception as e:
                    self.append_log(f"Could not read {m}: {e}")
        finally:
            QApplication.restoreOverrideCursor()
        self._prerun = {ts.name: qc.prerun_flags(ts) for ts in self.series}
        self._fill_table()
        self.append_log(f"Found {len(self.series)} tilt series in {folder}")
        for ts in self.series:
            n = sum(f.excludable for f in self._prerun[ts.name].values())
            if n:
                self.append_log(f"  {ts.name}: {n} tilt(s) flagged from the mdoc (see the QC column)")
        for ts in self.series:
            if ts.missing:
                self.append_log(f"  {ts.name}: {len(ts.missing)} fraction file(s) missing - those tilts will be skipped")
        self.statusBar().showMessage(f"Found {len(self.series)} tilt series", 5000)
        if self.series:
            self.table.selectRow(0)
        self.show_page("series")
        self._save_state()

    def _fill_table(self):
        self.table.setRowCount(len(self.series))
        for r, ts in enumerate(self.series):
            name = _item(ts.name)
            name.setFlags(name.flags() | Qt.ItemIsUserCheckable)
            name.setCheckState(Qt.Checked if ts.usable else Qt.Unchecked)
            self.table.setItem(r, COL_NAME, name)
            self.table.setItem(r, COL_TILTS, _item(len(ts.tilts), True))
            miss = _item(len(ts.missing), True)
            if ts.missing:
                miss.setForeground(QBrush(QColor(theme.WARNING)))
            self.table.setItem(r, COL_MISSING, miss)
            angles = [t.angle for t in ts.tilts]
            self.table.setItem(r, COL_RANGE, _item(f"{min(angles):+.0f} to {max(angles):+.0f}" if angles else "-"))
            self.table.setItem(r, COL_PIXEL, _item(f"{ts.pixel_size:.3f}", True))
            self.table.setItem(r, COL_TIME, _item("", True))
            self._update_qc_cell(r)
        self._refresh_status()
        self.series_count.setText(f"{len(self.series)} tilt series")

    def _refresh_status(self):
        out_root = self.top.output.path()
        s = self.forms.get_settings()
        for r, ts in enumerate(self.series):
            if not ts.usable:
                self._set_cell(r, COL_STACKS, "no frames")
                self._set_cell(r, COL_TOMO, "-")
                continue
            done = out_root is not None and is_complete(ts, s, out_root)
            self._set_cell(r, COL_STACKS, "done" if done else "ready")
            if not s.recon.enabled:
                self._set_cell(r, COL_TOMO, "-")
            else:
                tomo = out_root is not None and recon_complete(ts, s, out_root)
                self._set_cell(r, COL_TOMO, "done" if tomo else "ready")

    # ------------------------------------------------------------------ QC
    def _series_record(self, ts) -> dict | None:
        out_root = self.top.output.path()
        if out_root is None:
            return None
        j = out_root / ts.name / f"{ts.name}_pyprep.json"
        try:
            return json.loads(j.read_text()) if j.exists() else None
        except (OSError, ValueError):
            return None

    def _tilt_qc(self, ts) -> tuple[dict, float | None]:
        """{acquisition index: (flags, details)} from the mdoc, overridden by processing results;
        and the series' saturation fraction if measured."""
        out = {z: (list(f.flags), list(f.detail)) for z, f in self._prerun.get(ts.name, {}).items()}
        rec = self._series_record(ts)
        sat = None
        if rec and rec.get("status") == "complete":
            for t in rec.get("tilts", []):
                if "flags" in t:
                    out[t["acquisition"]] = (list(t["flags"]), list(t.get("flag_detail", [])))
            sat = (rec.get("qc") or {}).get("saturation")
        return out, sat

    def _update_qc_cell(self, row: int):
        ts = self.series[row]
        flags, sat = self._tilt_qc(ts)
        active = {t.zvalue for t in ts.tilts if not t.missing and not t.excluded}
        n = sum(any(f in qc.EXCLUDABLE for f in fl) for z, (fl, _) in flags.items() if z in active)
        parts = [f"{n} flagged" if n else ("ok" if flags else "-")]
        saturated = sat is not None and sat > qc.SATURATION_WARN
        if saturated:
            parts.append(f"sat {100 * sat:.1f}%")
        it = _item("  ·  ".join(parts))
        it.setForeground(QBrush(QColor(theme.WARNING if (n or saturated) else
                                       (theme.OK if flags else theme.TEXT_DIM))))
        tips = []
        if n:
            tips.append(f"{n} tilt(s) look dark, drifted or poorly aligned - see the QC column of the tilt list")
        if saturated:
            tips.append(f"{100 * sat:.1f}% of fraction pixels are clipped at the 8-bit maximum")
        it.setToolTip("\n".join(tips) or "no problems found")
        self.table.setItem(row, COL_QC, it)

    def exclude_flagged(self):
        r, ts = self._current_series()
        if ts is None:
            return
        flags, _ = self._tilt_qc(ts)
        hit = [t for t in ts.tilts if not t.missing and not t.excluded
               and any(f in qc.EXCLUDABLE for f in flags.get(t.zvalue, ([], []))[0])]
        for t in hit:
            t.excluded = True
        angles = ", ".join(f"{t.angle:+.1f}" for t in sorted(hit, key=lambda t: t.angle))
        self.append_log(f"{ts.name}: excluded {len(hit)} flagged tilt(s)" + (f": {angles}" if hit else ""))
        self._fill_tilt_table(ts)
        self._update_qc_cell(r)
        self._refresh_status()

    def _set_cell(self, row: int, col: int, status: str, extra: str = ""):
        it = _item(status + (f" {extra}" if extra else ""))
        it.setForeground(QBrush(QColor(STATUS_COLORS.get(status, theme.TEXT_DIM))))
        self.table.setItem(row, col, it)

    def _check_all(self, state: bool):
        for r in range(self.table.rowCount()):
            self.table.item(r, COL_NAME).setCheckState(Qt.Checked if state else Qt.Unchecked)

    def _settings_changed(self):
        if self.series and self._thread is None:
            self._refresh_status()

    # ------------------------------------------------------------------ selection
    def _current_series(self):
        sm = self.table.selectionModel()
        rows = sm.selectedRows() if sm else []
        if not rows:
            return None, None
        r = rows[0].row()
        return r, self.series[r]

    def _series_selected(self):
        r, ts = self._current_series()
        if ts is None:
            return
        self._fill_tilt_table(ts)
        out_root = self.top.output.path()
        if out_root is not None:
            self.results.load_series(out_root / ts.name)
        else:
            self.results.clear()

    def _fill_tilt_table(self, ts):
        flags, _ = self._tilt_qc(ts)
        self.tilt_table.blockSignals(True)
        self.tilt_table.setRowCount(len(ts.tilts))
        for i, t in enumerate(sorted(ts.tilts, key=lambda t: t.angle)):
            use = _item("")
            if t.missing:
                use.setFlags(Qt.ItemIsEnabled)
                use.setText("missing")
                use.setForeground(QBrush(QColor(theme.WARNING)))
            else:
                use.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable)
                use.setCheckState(Qt.Unchecked if t.excluded else Qt.Checked)
            use.setData(Qt.UserRole, t.zvalue)
            self.tilt_table.setItem(i, T_USE, use)
            self.tilt_table.setItem(i, T_ACQ, _item(f"{t.zvalue + 1:03d}", True))
            self.tilt_table.setItem(i, T_ANGLE, _item(f"{t.angle:+.2f}", True))
            self.tilt_table.setItem(i, T_DOSE, _item(f"{t.exposure_dose:.2f}", True))
            self.tilt_table.setItem(i, T_PRIOR, _item(f"{t.prior_dose:.2f}", True))
            fl, detail = flags.get(t.zvalue, ([], []))
            q = _item(", ".join(fl) if fl else ("ok" if flags else ""))
            if fl:
                q.setForeground(QBrush(QColor(theme.WARNING if any(f in qc.EXCLUDABLE for f in fl)
                                              else theme.TEXT_DIM)))
                q.setToolTip("\n".join(detail) or ", ".join(fl))
            else:
                q.setForeground(QBrush(QColor(theme.OK)))
            self.tilt_table.setItem(i, T_QC, q)
            f = _item(t.frame_path.name if t.frame_path else (t.section.get("SubFramePath") or "?"))
            if t.missing:
                f.setForeground(QBrush(QColor(theme.WARNING)))
            self.tilt_table.setItem(i, T_FILE, f)
        self.tilt_table.blockSignals(False)
        self.tilts_title.setText(f"Tilts in {ts.name}")
        self.tilts_label.setText(ts.summary() + (f"  ·  fractions from {ts.frames_dir}" if ts.frames_dir else ""))

    def _tilt_toggled(self, item: QTableWidgetItem):
        if item.column() != 0:
            return
        _, ts = self._current_series()
        if ts is None:
            return
        z = item.data(Qt.UserRole)
        for t in ts.tilts:
            if t.zvalue == z and not t.missing:
                t.excluded = item.checkState() != Qt.Checked
        r, _ = self._current_series()
        self._update_qc_cell(r)
        self._refresh_status()

    # ------------------------------------------------------------------ preview / directives
    def preview(self):
        r, ts = self._current_series()
        if ts is None or not ts.usable:
            QMessageBox.information(self, "pyPrep", "Select a tilt series with fraction files first "
                                                    "(Tilt series page).")
            return
        tilt = None
        rows = self.tilt_table.selectionModel().selectedRows()
        if rows:
            z = self.tilt_table.item(rows[0].row(), 0).data(Qt.UserRole)
            tilt = next((t for t in ts.tilts if t.zvalue == z and not t.missing), None)
        if tilt is None:
            tilt = min(ts.usable, key=lambda t: abs(t.angle))
        self._set_busy(True, preview=True)
        self.statusBar().showMessage(f"Aligning {ts.name} tilt {tilt.zvalue + 1:03d} ({tilt.angle:+.2f} deg)…")
        self._preview_series = ts
        worker = PreviewWorker(ts, tilt, self.forms.get_settings())
        # Signals from worker threads must go to methods of this QObject (not lambdas),
        # so Qt queues them onto the GUI thread.
        worker.result.connect(self._on_preview_result)
        worker.error.connect(self._on_preview_error)
        self._run_in_thread(worker, on_done=self._preview_done)

    def _on_preview_result(self, res):
        self.results.show_preview(res, self._preview_series.pixel_size)
        self.show_page("results")
        self.append_log(f"Preview: drift {res['drift']:.2f} A in {res['seconds']:.2f} s")

    def _on_preview_error(self, msg):
        self.append_log("Preview failed: " + msg)
        QMessageBox.warning(self, "pyPrep", "Preview failed:\n" + msg.splitlines()[0])

    def _preview_done(self):
        self._set_busy(False)
        self.statusBar().showMessage("Ready")

    def show_directives(self):
        _, ts = self._current_series()
        s = self.forms.get_settings()
        pix = (ts.pixel_size if ts else 1.0) * s.recon.bin
        axis = (ts.tilt_axis if ts and ts.tilt_axis is not None else 0.0)
        dirs = imod.build_directives(pix, axis, ts.voltage if ts else 300, s.recon)
        text = (f"# {'Series ' + ts.name if ts else 'No series selected - generic values'}: "
                f"bin {s.recon.bin} stack, {pix:.2f} A/px, tilt axis {axis:.2f}\n"
                + "\n".join(f"{k} = {v}" for k, v in dirs.items()))
        dlg = QDialog(self)
        dlg.setWindowTitle("batchruntomo directives")
        dlg.resize(820, 620)
        v = QVBoxLayout(dlg)
        ed = QPlainTextEdit(text)
        ed.setReadOnly(True)
        ed.setStyleSheet("font-family: Consolas, monospace; font-size: 13px;")
        v.addWidget(ed)
        b = QPushButton("Close")
        b.clicked.connect(dlg.accept)
        v.addWidget(b, 0, Qt.AlignRight)
        dlg.exec()

    # ------------------------------------------------------------------ batch
    def start(self):
        err = self.forms.validate()
        if err:
            QMessageBox.warning(self, "pyPrep", err)
            return
        out_root = self.top.output.path()
        if out_root is None:
            QMessageBox.warning(self, "pyPrep", "Choose an output folder in the top bar.")
            return
        jobs = [(r, ts) for r, ts in enumerate(self.series)
                if self.table.item(r, COL_NAME).checkState() == Qt.Checked and ts.usable]
        if not jobs:
            QMessageBox.information(self, "pyPrep", "No checked tilt series with fraction files. "
                                                    "Use 'Find tilt series' first.")
            return
        settings = self.forms.get_settings()
        settings.frames_dir = str(self.frames_path.path()) if self.frames_path.path() else None
        out_root.mkdir(parents=True, exist_ok=True)
        if not self._enough_disk(jobs, settings, out_root):
            return
        settings.save(out_root / "pyprep_settings_last_run.json")
        self._save_state()
        for r, _ in jobs:
            self._set_cell(r, COL_STACKS, "queued")
            if settings.recon.enabled:
                self._set_cell(r, COL_TOMO, "queued")
        self.overall.setRange(0, len(jobs))
        self.overall.setValue(0)
        self.overall.setFormat("%v / %m series")
        self.current.setValue(0)
        self.run_msg.setStyleSheet("")
        self._batch = {"t0": time.monotonic(), "jobs": len(jobs), "failed": [], "last_row": None,
                       "cancelled": False}
        self._set_busy(True)
        self.show_page("log")
        self._launch_batch(jobs, settings, out_root)

    def _start_jobs(self, jobs, settings, force_recon=False):
        """Start a batch outside the normal Start button (e.g. redo one reconstruction)."""
        out_root = self.top.output.path()
        if out_root is None:
            return
        settings.frames_dir = str(self.frames_path.path()) if self.frames_path.path() else None
        for r, _ in jobs:
            self._set_cell(r, COL_TOMO, "queued")
        self.overall.setRange(0, len(jobs))
        self.overall.setValue(0)
        self.overall.setFormat("%v / %m series")
        self.current.setValue(0)
        self.run_msg.setStyleSheet("")
        self._batch = {"t0": time.monotonic(), "jobs": len(jobs), "failed": [], "last_row": None,
                       "cancelled": False}
        self._set_busy(True)
        self.show_page("log")
        self._launch_batch(jobs, settings, out_root, force_recon)

    def _launch_batch(self, jobs, settings, out_root, force_recon=False):
        worker = BatchWorker(jobs, settings, out_root, force_recon)
        worker.series_started.connect(self._on_series_started)
        worker.series_progress.connect(self._on_progress)
        worker.series_finished.connect(self._on_series_finished)
        worker.log.connect(self.append_log)
        self._run_in_thread(worker, on_done=self._batch_done)

    def _enough_disk(self, jobs, settings, out_root) -> bool:
        """Warn before starting if the output drive looks too small for the batch."""
        import shutil
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            need = sum(estimate_output_bytes(ts, settings) for _, ts in jobs
                       if not (settings.skip_existing and is_complete(ts, settings, out_root)))
            free = shutil.disk_usage(out_root).free
        finally:
            QApplication.restoreOverrideCursor()
        self.append_log(f"Estimated output size {need / 1e9:.1f} GB; {free / 1e9:.1f} GB free on {out_root.anchor}")
        if need <= free:
            return True
        return QMessageBox.question(
            self, "pyPrep - disk space",
            f"This batch may need about {need / 1e9:.1f} GB, but only {free / 1e9:.1f} GB is free on "
            f"{out_root.anchor}.\n\nChoose int16 output, fewer binning levels, or another output folder "
            f"to save space.\n\nStart anyway?") == QMessageBox.Yes

    def _notify(self, text: str, ok: bool):
        """Windows notification (useful for overnight batches), plus the taskbar flash."""
        QApplication.alert(self)
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return
        if self._tray is None:
            self._tray = QSystemTrayIcon(QApplication.windowIcon(), self)
            self._tray.setToolTip("pyPrep")
            self._tray.messageClicked.connect(lambda: (self.showNormal(), self.activateWindow()))
        self._tray.show()
        self._tray.showMessage("pyPrep", text, QSystemTrayIcon.Information if ok else QSystemTrayIcon.Warning,
                               15000)

    def cancel(self):
        if self._worker is not None and hasattr(self._worker, "cancel"):
            self._worker.cancel()
            if self._batch:
                self._batch["cancelled"] = True
            self.append_log("Cancelling…")
            self.run_msg.setText("Cancelling…")
            self.btn_cancel.setEnabled(False)

    def _on_series_started(self, row):
        self._set_cell(row, COL_STACKS, "running")
        self.run_msg.setText(self.series[row].name)

    def _on_progress(self, row, done, total, msg):
        name = self.series[row].name
        self.run_msg.setText(f"{name}\n{msg}")
        recon_phase = msg.startswith("IMOD") or msg.startswith("Tomogram")
        if done >= 0:
            self.current.setRange(0, max(total, 1))
            self.current.setValue(done)
        col = COL_TOMO if recon_phase else COL_STACKS
        if recon_phase:
            self._set_cell(row, COL_STACKS, "done")
        self._set_cell(row, col, "running", "" if recon_phase or done < 0 else f"{done}/{total}")
        self.statusBar().showMessage(f"{name}: {msg}")

    def _on_series_finished(self, row, res):
        def label(status):
            return {"complete": "done", "skipped": "done", None: "-"}.get(status, status)
        self._set_cell(row, COL_STACKS, label(res.get("stacks")))
        self._set_cell(row, COL_TOMO, label(res.get("recon")))
        secs = res.get("seconds") or 0
        self.table.setItem(row, COL_TIME, _item(f"{secs / 60:.1f} min" if secs >= 90 else f"{secs:.0f} s", True))
        self.overall.setValue(self.overall.value() + 1)
        if self._batch is not None:
            self._batch["last_row"] = row
            if res.get("stacks") not in ("complete", "skipped") or res.get("recon") in ("failed", "cancelled"):
                self._batch["failed"].append(self.series[row].name)
        self._update_qc_cell(row)
        r, ts = self._current_series()
        if r == row and self.top.output.path():
            self._fill_tilt_table(ts)
            self.results.load_series(self.top.output.path() / ts.name, prefer="tomo")

    def _batch_done(self):
        """Make the end of a batch unmistakable: summary, taskbar flash, results on screen."""
        self._set_busy(False)
        b = self._batch or {"t0": time.monotonic(), "jobs": 0, "failed": [], "last_row": None, "cancelled": False}
        self._batch = None
        secs = time.monotonic() - b["t0"]
        took = f"{secs:.0f} s" if secs < 90 else f"{secs / 60:.1f} min"
        n, failed = b["jobs"], b["failed"]
        if b["cancelled"]:
            text, color = f"Cancelled after {took}", theme.WARNING
        elif failed:
            text, color = f"Finished in {took} - {len(failed)} of {n} series had problems", theme.DANGER
        else:
            text, color = f"Finished {n} series in {took}", theme.OK
        self.run_msg.setText(text)
        self.run_msg.setStyleSheet(f"color: {color}; font-weight: 700;")
        self.append_log(f"===== Batch finished: {text}" + (f" ({', '.join(failed)})" if failed else "") + " =====")
        self.statusBar().showMessage(text)
        self._notify(text, ok=not failed and not b["cancelled"])
        if self.top.output.path():
            self.gallery.out_root = self.top.output.path()
            self.gallery.refresh()
        if not b["cancelled"] and b["last_row"] is not None and self.top.output.path():
            if n > 1:
                self.show_page("gallery")         # several series: pick the good ones at a glance
            else:
                r, _ = self._current_series()
                row = r if r is not None else b["last_row"]
                if r is None:
                    self.table.selectRow(row)
                self.results.load_series(self.top.output.path() / self.series[row].name, prefer="tomo")
                self.show_page("results")

    def _set_busy(self, busy: bool, preview: bool = False):
        self.btn_start.setEnabled(not busy)
        self.btn_cancel.setEnabled(busy and not preview)
        self.top.scan_btn.setEnabled(not busy)
        self.btn_preview.setEnabled(not busy)
        self.btn_exclude.setEnabled(not busy)
        self.forms.preview_button_alignment.setEnabled(not busy)
        self.forms.set_enabled(not busy)
        if not busy:
            self.current.setFormat("%p%")

    def _run_in_thread(self, worker, on_done=None):
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_thread_finished)
        self._thread, self._worker, self._on_done = thread, worker, on_done
        thread.start()

    def _on_thread_finished(self):
        on_done = self._on_done
        self._thread, self._worker, self._on_done = None, None, None
        if on_done:
            on_done()

    def append_log(self, text: str):
        self.log_page.append(text)


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    theme.apply_theme(app)
    from .launcher import set_app_identity
    set_app_identity(app)
    w = MainWindow()
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
