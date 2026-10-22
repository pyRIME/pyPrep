"""Settings forms, split across the Frame alignment, Outputs and Reconstruction pages.

One :class:`SettingsForms` object owns every settings widget so the whole
configuration can be read or restored at once; the main window places its
page widgets (``alignment_page``, ``outputs_page``, ``recon_page``).
"""

from __future__ import annotations

import os

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
                               QPlainTextEdit, QPushButton, QSpinBox, QStackedWidget, QVBoxLayout,
                               QWidget)

from .. import gpu, imod, redep
from ..ctf import CtfSettings
from ..imod import ReconSettings
from ..motion import MotionSettings
from ..settings import (InputSettings, OutputSettings, ProcessingSettings, delete_preset, list_presets,
                        load_preset, save_preset)
from . import theme
from .theme import card, dim_label, section_label, title_label

BIN_CHOICES = (1, 2, 4, 8)


FIELD_WIDTH = 200


def _dspin(lo, hi, step, decimals, suffix="", tip=""):
    w = QDoubleSpinBox()
    w.setMinimumWidth(FIELD_WIDTH)
    w.setRange(lo, hi)
    w.setSingleStep(step)
    w.setDecimals(decimals)
    if suffix:
        w.setSuffix(suffix)
    w.setToolTip(tip)
    return w


def _ispin(lo, hi, tip="", suffix=""):
    w = QSpinBox()
    w.setMinimumWidth(FIELD_WIDTH)
    w.setRange(lo, hi)
    w.setToolTip(tip)
    if suffix:
        w.setSuffix(suffix)
    return w


def _card_with_form(title: str | None = None, note: str | None = None):
    """A pyFIB-style card containing an optional section title, note and a form."""
    c = card()
    v = QVBoxLayout(c)
    v.setContentsMargins(16, 14, 16, 14)
    v.setSpacing(8)
    if title:
        v.addWidget(section_label(title))
    if note:
        v.addWidget(dim_label(note))
    f = QFormLayout()
    f.setHorizontalSpacing(18)
    f.setVerticalSpacing(10)
    f.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)
    v.addLayout(f)
    return c, v, f


def _page(title: str, intro: str) -> tuple[QWidget, QVBoxLayout]:
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.setContentsMargins(22, 18, 22, 18)
    lay.setSpacing(12)
    lay.addWidget(title_label(title))
    lay.addWidget(dim_label(intro))
    return w, lay


class SettingsForms(QObject):
    changed = Signal()
    preview_requested = Signal()
    show_directives_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.alignment_page = self._build_alignment()
        self.outputs_page = self._build_outputs()
        self.recon_page = self._build_recon()
        self.set_settings(ProcessingSettings())
        for page in (self.alignment_page, self.outputs_page, self.recon_page):
            for w in page.findChildren(QSpinBox) + page.findChildren(QDoubleSpinBox):
                w.valueChanged.connect(self.changed)
            for w in page.findChildren(QCheckBox):
                w.toggled.connect(self.changed)
            for w in page.findChildren(QComboBox):
                w.currentIndexChanged.connect(self.changed)
        self.exclude.textChanged.connect(self.changed)
        self.gain_path.textChanged.connect(self.changed)

    # ------------------------------------------------------------------ pages
    def _build_alignment(self) -> QWidget:
        page, lay = _page(
            "Frame alignment",
            "Each tilt's dose fractions are aligned MotionCor-style: frames are Fourier-binned, "
            "cross-correlated against the sum of all other frames, and the shifts refined until "
            "they converge. Shifts are then applied at full resolution as Fourier phase ramps.")
        c, v, f = _card_with_form(
            "Input frames", "MRC/TIFF fraction files are used as saved. Falcon EER movies hold hundreds of "
                            "detector frames per tilt; they are summed into fractions before alignment.")
        eer_row = QHBoxLayout()
        self.eer_mode = QComboBox()
        self.eer_mode.addItem("Number of fractions", "fractions")
        self.eer_mode.addItem("EER frames per fraction", "group")
        self.eer_mode.setMinimumWidth(240)
        self.eer_value = _ispin(1, 5000, "Fractions per tilt, or EER frames per fraction.\n"
                                         "Each fraction should hold enough dose to align (~0.2-0.5 e/A²).")
        self._eer_vals = {"fractions": 10, "group": 50}   # remembered value per grouping mode
        self._eer_prev = "fractions"
        self.eer_mode.currentIndexChanged.connect(self._eer_mode_changed)
        eer_row.addWidget(self.eer_mode)
        eer_row.addWidget(self.eer_value)
        eer_row.addStretch()
        f.addRow("EER grouping", eer_row)
        self.eer_up = QComboBox()
        self.eer_up.addItem("Physical pixels (4K)", 1)
        self.eer_up.addItem("2x super-resolution (8K), binned back", 2)
        self.eer_up.setMinimumWidth(320)
        self.eer_up.setToolTip("8K rendering uses the EER sub-pixel positions; frames are aligned at 8K and\n"
                               "Fourier-binned to the physical pixel size, reducing aliasing. ~4x slower.")
        f.addRow("EER rendering", self.eer_up)
        gain_row = QHBoxLayout()
        self.gain_path = QLineEdit()
        self.gain_path.setPlaceholderText("none - EPU .gain for EER; not needed for Tomo5 K3 fractions")
        self.gain_path.setMinimumWidth(420)
        gb = QPushButton("Browse…")
        gb.clicked.connect(self._browse_gain)
        gain_row.addWidget(self.gain_path, 1)
        gain_row.addWidget(gb)
        f.addRow("Gain reference", gain_row)
        g_row = QHBoxLayout()
        self.gain_mode = QComboBox()
        self.gain_mode.addItem("Auto (divide .gain / EER)", "auto")
        self.gain_mode.addItem("Multiply", "multiply")
        self.gain_mode.addItem("Divide", "divide")
        self.gain_mode.setMinimumWidth(240)
        self.gain_mode.setToolTip("EPU .gain files are detector gains: frames are divided by them.\n"
                                  "SerialEM/DigitalMicrograph references are multiplied.")
        self.gain_rotate = QComboBox()
        for deg in (0, 90, 180, 270):
            self.gain_rotate.addItem(f"rotate {deg}°", deg)
        self.gain_flip = QComboBox()
        for label, key in (("no flip", "none"), ("flip left-right", "x"), ("flip up-down", "y")):
            self.gain_flip.addItem(label, key)
        for w in (self.gain_mode, self.gain_rotate, self.gain_flip):
            g_row.addWidget(w)
        g_row.addStretch()
        f.addRow("Gain handling", g_row)
        self.invert_y = QComboBox()
        self.invert_y.setMinimumWidth(320)
        self.invert_y.addItem("Automatic (invert Tomography 5 data in Y)", "auto")
        self.invert_y.addItem("Invert in Y", "yes")
        self.invert_y.addItem("As stored", "no")
        self.invert_y.setToolTip("Tomography 5 stores images upside down (IMOD reports its stacks as 'inverted in Y').\n"
                                 "Used as stored, the tomograms are mirror images of the specimen. Automatic inverts\n"
                                 "the aligned images of Tomo5 series in Y and negates the recorded tilt axis, so the\n"
                                 "stacks, the CTF and the tomograms follow IMOD's conventions and the hand is right.")
        f.addRow("Image orientation", self.invert_y)
        lay.addWidget(c)

        c, v, f = _card_with_form("Alignment")
        self.align_bin = _ispin(1, 16, "Frames are Fourier-binned by this factor when measuring shifts.\n"
                                       "4 suits low-dose tomography fractions, whose shared signal is\n"
                                       "mostly coarser than ~30 A. Shifts are still applied at full resolution.")
        self.bfactor = _dspin(0, 10000, 100, 0, " A²", "Low-pass (B-factor) applied to the cross-correlation.\n"
                                                         "Higher = smoother, more robust for noisy frames.")
        self.max_shift = _dspin(1, 2000, 10, 0, " px", "Largest frame shift searched for (full-resolution pixels).")
        self.iterations = _ispin(1, 100, "Maximum refinement iterations per tilt.")
        self.tolerance = _dspin(0.01, 5, 0.05, 2, " px", "Stop when no shift changes by more than this.")
        self.group = _ispin(1, 50, "Sum this many consecutive frames before aligning (for very noisy movies).\n"
                                   "Per-frame shifts are interpolated. Keep 1 for 4-fraction tilts.")
        self.mask_axes = QCheckBox("Ignore detector fixed-pattern noise (Fourier axes)")
        self.mask_axes.setToolTip("Row/column offsets of the camera are identical in every frame and pull\n"
                                  "all shifts towards zero. Leave on unless you know the frames are clean.")
        f.addRow("Alignment binning", self.align_bin)
        f.addRow("B-factor", self.bfactor)
        f.addRow("Max shift", self.max_shift)
        f.addRow("Max iterations", self.iterations)
        f.addRow("Tolerance", self.tolerance)
        f.addRow("Group frames", self.group)
        f.addRow("", self.mask_axes)
        lay.addWidget(c)

        c, v, f = _card_with_form(
            "CTF estimation", "Defocus of every tilt, measured on the GPU from its aligned sum while the series "
                              "is processed (tilt-aware: the defocus gradient across tilted images is fitted, "
                              "including its direction). Used for CTF correction and deconvolution on the "
                              "Reconstruction page, and written as <series>.defocus for IMOD.")
        self.ctf_enabled = QCheckBox("Estimate the CTF of every tilt")
        self.ctf_enabled.setStyleSheet("font-weight: 700;")
        self.ctf_cs = _dspin(0, 10, 0.1, 2, " mm", "Spherical aberration of the objective lens (Krios: 2.7).")
        self.ctf_amp = _dspin(0, 0.5, 0.01, 3, "", "Amplitude contrast (0.07 for cryo samples).")
        rng = QHBoxLayout()
        self.ctf_lo = _dspin(10, 200, 1, 0, " Å", "Low-resolution end of the fitted range.")
        self.ctf_hi = _dspin(3, 50, 0.5, 1, " Å", "High-resolution end (limited to 2.2 x the pixel size).")
        for w in (self.ctf_lo, self.ctf_hi):
            w.setMinimumWidth(120)
        rng.addWidget(self.ctf_lo)
        rng.addWidget(QLabel("to"))
        rng.addWidget(self.ctf_hi)
        rng.addStretch()
        drng = QHBoxLayout()
        self.def_min = _dspin(0.1, 20, 0.1, 1, " µm", "Smallest defocus searched (underfocus).")
        self.def_max = _dspin(0.5, 30, 0.5, 1, " µm", "Largest defocus searched (underfocus).")
        for w in (self.def_min, self.def_max):
            w.setMinimumWidth(120)
        drng.addWidget(self.def_min)
        drng.addWidget(QLabel("to"))
        drng.addWidget(self.def_max)
        drng.addStretch()
        v.insertWidget(2, self.ctf_enabled)
        f.addRow("Spherical aberration", self.ctf_cs)
        f.addRow("Amplitude contrast", self.ctf_amp)
        f.addRow("Fit range", rng)
        f.addRow("Defocus search", drng)
        lay.addWidget(c)

        c, v, f = _card_with_form(
            "Try it", "Align one tilt of the selected series with these settings and compare the "
                      "unaligned and aligned sums on the Results page. Nothing is written to disk. "
                      "Pick the tilt on the Tilt series page (default: the one nearest 0 deg).")
        b = QPushButton("Test on one tilt")
        b.setObjectName("toolButton")
        b.clicked.connect(self.preview_requested)
        self.preview_button_alignment = b
        v.addWidget(b)
        lay.addWidget(c)
        lay.addStretch(1)
        return page

    def _build_outputs(self) -> QWidget:
        page, lay = _page(
            "Outputs",
            "One set of stacks per tilt series, sorted by tilt angle, each with a matching "
            ".rawtlt and .mrc.mdoc (tilt angles, tilt axis, pixel size and dose) that etomo reads directly.")
        c, v, f = _card_with_form("Stacks")
        self.out_aligned = QCheckBox("Aligned sum   (<name>.mrc)")
        self.out_evenodd = QCheckBox("Even / odd half-sums   (_EVN / _ODD, for denoising)")
        self.out_dw = QCheckBox("Dose-weighted sum   (_DW - skip etomo's own dose weighting on it)")
        for w in (self.out_aligned, self.out_evenodd, self.out_dw):
            f.addRow("", w)
        row = QHBoxLayout()
        row.setSpacing(22)
        self.bin_boxes = {}
        for b in BIN_CHOICES:
            cb = QCheckBox(f"bin {b}")
            cb.setToolTip("Full resolution" if b == 1 else f"Fourier-binned by {b} (_bin{b}.mrc)")
            self.bin_boxes[b] = cb
            row.addWidget(cb)
        row.addStretch()
        f.addRow("Binning levels", row)
        self.dtype = QComboBox()
        self.dtype.setMinimumWidth(320)
        self.dtype.addItems(["float32", "int16"])
        self.dtype.setToolTip("int16 halves file size; lossless in practice for counting-camera data.")
        f.addRow("Data type", self.dtype)
        self.exclude = QLineEdit()
        self.exclude.setMinimumWidth(420)
        self.exclude.setPlaceholderText("e.g. -46.1, 45.9   (applies to every series)")
        self.exclude.setToolTip("Tilt angles to leave out of every stack (within 0.5 deg).\n"
                                "Per-series exclusions can be set on the Tilt series page.")
        f.addRow("Exclude angles", self.exclude)
        lay.addWidget(c)

        c, v, f = _card_with_form("Processing")
        self.device = QComboBox()
        self.device.setMinimumWidth(320)
        for i, name in enumerate(gpu.list_gpus()):
            self.device.addItem(f"GPU {i}: {name}", ("gpu", i))
        self.device.addItem("CPU (slow)", ("cpu", 0))
        self.skip_existing = QCheckBox("Skip steps whose outputs are already complete")
        self.default_dose = _dspin(0, 100, 0.1, 2, " e/A²", "Dose per tilt to assume when the mdoc has no\n"
                                                           "ExposureDose (0 = not set). Only used for dose weighting.")
        f.addRow("Compute on", self.device)
        f.addRow("", self.skip_existing)
        f.addRow("Fallback dose per tilt", self.default_dose)
        lay.addWidget(c)

        c, v, f = _card_with_form("Presets and settings files",
                                  "A preset stores the settings of all pages under a name, e.g. one per "
                                  "microscope or camera. Settings files can also be given to the command line.")
        prow = QHBoxLayout()
        self.preset_combo = QComboBox()
        self.preset_combo.setMinimumWidth(320)
        prow.addWidget(QLabel("Preset:"))
        prow.addWidget(self.preset_combo)
        for text, slot in (("Apply", self._apply_preset), ("Save as preset…", self._save_preset),
                           ("Delete", self._delete_preset)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            prow.addWidget(b)
        prow.addStretch()
        v.addLayout(prow)
        self._refresh_presets()
        row = QHBoxLayout()
        for text, slot in (("Load settings...", self.load_dialog), ("Save settings...", self.save_dialog),
                           ("Reset to defaults", lambda: self.set_settings(ProcessingSettings()))):
            b = QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)
        row.addStretch()
        v.addLayout(row)
        lay.addWidget(c)
        lay.addStretch(1)
        return page

    def _build_recon(self) -> QWidget:
        page, lay = _page(
            "Reconstruction",
            "After frame alignment pyPrep can run IMOD's batchruntomo on the binned stack of each "
            "series. The result is a normal etomo project in <series>/imod_bin<N>/ that you can "
            "open in etomo to inspect or redo any step.")
        ok, msg = imod.imod_status()
        self.imod_label = QLabel(("IMOD found: " if ok else "") + msg)
        self.imod_label.setWordWrap(True)
        self.imod_label.setStyleSheet(f"color: {theme.OK if ok else theme.DANGER};")
        lay.addWidget(self.imod_label)

        c, v, f = _card_with_form()
        self.recon_enabled = QCheckBox("Reconstruct each series with IMOD batchruntomo after frame alignment")
        self.recon_enabled.setStyleSheet("font-weight: 700;")
        v.addWidget(self.recon_enabled)
        lay.addWidget(c)

        c, v, f = _card_with_form("Alignment preset")
        row = QHBoxLayout()
        self.preset_group = QButtonGroup(self)
        self.preset_group.setExclusive(True)
        self.preset_buttons = {}
        for key, label in imod.PRESETS.items():
            b = QPushButton(label)
            b.setObjectName("toolButton")
            b.setCheckable(True)
            b.setMinimumWidth(260)
            self.preset_group.addButton(b)
            self.preset_buttons[key] = b
            row.addWidget(b)
        row.addStretch()
        v.addLayout(row)
        self.preset_stack = QStackedWidget()
        self.preset_stack.setObjectName("cardBody")
        # patch tracking parameters
        pw = QWidget()
        pw.setObjectName("cardBody")
        pf = QFormLayout(pw)
        pf.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)
        pf.setContentsMargins(0, 6, 0, 0)
        pf.addRow(dim_label("Tracks overlapping image patches through the series - no fiducials needed. "
                            "Defaults match the reference Position_9_2 etomo project (1200 px patches at "
                            "3.3 A, overlap 0.6)."))
        self.patch_size = _dspin(50, 3000, 50, 0, " nm", "Edge length of each tracked patch.")
        self.patch_overlap = _dspin(0, 0.9, 0.05, 2, "", "Fractional overlap between neighbouring patches.")
        pf.addRow("Patch size", self.patch_size)
        pf.addRow("Patch overlap", self.patch_overlap)
        self.preset_stack.addWidget(pw)
        # gold parameters
        gw = QWidget()
        gw.setObjectName("cardBody")
        gf = QFormLayout(gw)
        gf.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)
        gf.setContentsMargins(0, 6, 0, 0)
        gf.addRow(dim_label("Seeds and tracks gold beads automatically (autofidseed + beadtrack), "
                            "as in the lab's Linux batchruntomo script, and erases them from the stack."))
        self.gold_size = _dspin(1, 50, 1, 1, " nm", "Gold bead diameter.")
        self.gold_beads = _ispin(3, 500, "Number of beads autofidseed tries to pick.")
        self.erase_gold = QCheckBox("Erase gold from the aligned stack")
        gf.addRow("Bead size", self.gold_size)
        gf.addRow("Beads to track", self.gold_beads)
        gf.addRow("", self.erase_gold)
        self.preset_stack.addWidget(gw)
        v.addWidget(self.preset_stack)
        self.preset_group.buttonToggled.connect(self._preset_toggled)
        lay.addWidget(c)

        c, v, f = _card_with_form("Tomogram")
        self.recon_bin = QComboBox()
        self.recon_bin.setMinimumWidth(320)
        for b in BIN_CHOICES:
            self.recon_bin.addItem(f"bin {b}", b)
        self.recon_bin.setToolTip("Which pyPrep stack to reconstruct. It is written automatically\n"
                                  "even if that binning is not ticked on the Outputs page.")
        self.recon_source = QComboBox()
        self.recon_source.setMinimumWidth(320)
        self.recon_source.addItem("Aligned sum", False)
        self.recon_source.addItem("Dose-weighted sum", True)
        self.positioning = QComboBox()
        self.positioning.setMinimumWidth(320)
        self.positioning.addItem("Automatic (IMOD cryo positioning)", "auto")
        self.positioning.addItem("Fixed thickness (skip positioning)", "fixed")
        self.positioning.setToolTip("Automatic finds the specimen slab and sets thickness and pitch.\n"
                                    "Sparse samples can defeat it; the thickness below is then used.")
        self.pos_thickness = _dspin(20, 5000, 10, 0, " nm", "Thickness of the trial tomogram used for positioning.")
        self.thickness = _dspin(20, 5000, 10, 0, " nm")
        self.thickness_label = QLabel("Fallback thickness")
        self.handedness = QComboBox()
        self.handedness.setMinimumWidth(320)
        self.handedness.addItem("Automatic (from the CTF defocus gradient)", "auto")
        self.handedness.addItem("Keep as recorded", "keep")
        self.handedness.addItem("Always flip", "flip")
        self.handedness.setToolTip("Direction of the tilt axis. Automatic: if the defocus gradient measured by CTF\n"
                                   "estimation disagrees with the recorded axis, the axis is rotated by 180 deg\n"
                                   "(a 180 deg error in the recorded axis; the stacks' .mdoc files get the corrected\n"
                                   "axis as well). Images stored mirrored are handled by 'Image orientation'.")
        self.sirt = _ispin(0, 100, "SIRT-like radial filter equivalent to this many SIRT iterations\n"
                                   "(better low-resolution contrast). 0 = plain weighted back-projection.")
        self.remove_xrays = QCheckBox("Remove X-rays / hot pixels (ccderaser)")
        self.cpus = _ispin(1, os.cpu_count() or 1, "CPU cores IMOD may use.")
        self.imod_gpu = QCheckBox("Use GPU for back-projection (IMOD tilt)")
        f.addRow("Stack binning", self.recon_bin)
        f.addRow("Stack", self.recon_source)
        f.addRow("Positioning", self.positioning)
        self.pos_thickness_label = QLabel("Positioning thickness")
        f.addRow(self.pos_thickness_label, self.pos_thickness)
        f.addRow(self.thickness_label, self.thickness)
        f.addRow("Handedness", self.handedness)
        f.addRow("SIRT-like filter", self.sirt)
        f.addRow("CPU cores", self.cpus)
        f.addRow("", self.remove_xrays)
        f.addRow("", self.imod_gpu)
        self.positioning.currentIndexChanged.connect(self._positioning_changed)
        lay.addWidget(c)

        c, v, f = _card_with_form(
            "CTF correction and deconvolution",
            "Both use the defocus pyPrep measured for each tilt (Frame alignment page, CTF estimation).")
        self.ctf_correct = QCheckBox("Correct the CTF (IMOD ctfphaseflip with pyPrep's per-tilt defocus)")
        self.ctf_correct.setToolTip("Phase-flips the aligned stack before back-projection. At bin 4 it changes\n"
                                    "little unless the defocus is large; it matters for bin 1-2 tomograms.")
        self.deconvolve = QCheckBox("Also write a deconvolved tomogram   (<series>_rec_deconv.mrc)")
        self.deconvolve.setToolTip("Wiener-like filter with the CTF: stronger low-resolution contrast and less\n"
                                   "noise, for viewing and segmentation. The normal tomogram is kept.")
        self.deconv_strength = _dspin(0.1, 3, 0.1, 2, "", "Higher = stronger contrast boost (assumed SNR 10^(3 x strength)).")
        self.deconv_falloff = _dspin(0.1, 5, 0.1, 2, "", "Higher = SNR assumed to fall faster with resolution (smoother).")
        v.insertWidget(2, self.ctf_correct)
        v.insertWidget(3, self.deconvolve)
        f.addRow("Deconvolution strength", self.deconv_strength)
        f.addRow("SNR falloff", self.deconv_falloff)
        self.deconvolve.toggled.connect(lambda on: (self.deconv_strength.setEnabled(on),
                                                    self.deconv_falloff.setEnabled(on)))
        lay.addWidget(c)

        c, v, f = _card_with_form(
            "Redeposition removal",
            "Takes dense surface material (FIB redeposition, Pt/GIS grains) out of the aligned tilt series by "
            "SIRT constrained to the lamella, then rebuilds the tomogram. Needs the ASTRA Toolbox and a CUDA GPU.")
        astra_ok, astra_msg = redep.astra_status()
        self.remove_redep = QCheckBox("Also write a tomogram with the redeposition removed   (<series>_rec_redep.mrc)")
        self.remove_redep.setEnabled(astra_ok)
        self.remove_redep.setToolTip(
            "Finds the lamella and the dense layers beyond its surfaces, models them by SIRT (GPU) and fills their\n"
            "shadow in every view. Removes the shading and blotches they cast into the interior; fine detail\n"
            "(below ~12 nm) is unchanged. The normal tomogram is kept. Needs about 10 GB of memory at bin 4."
            if astra_ok else f"Unavailable: {astra_msg}.\nInstall it into pyPrep's environment, then restart pyPrep.")
        v.insertWidget(2, self.remove_redep)
        f.addRow("ASTRA", dim_label(astra_msg))
        lay.addWidget(c)

        c, v, f = _card_with_form("Advanced", "Extra batchruntomo directives, one 'key = value' per line. "
                                              "They override the preset (see IMOD's directives.csv).")
        self.extra = QPlainTextEdit()
        self.extra.setPlaceholderText("comparam.align.tiltalign.LocalAlignments = 1")
        self.extra.setMaximumHeight(90)
        self.extra.textChanged.connect(self.changed)
        v.addWidget(self.extra)
        b = QPushButton("Show directives for the selected series")
        b.clicked.connect(self.show_directives_requested)
        row = QHBoxLayout()
        row.addWidget(b)
        row.addStretch()
        v.addLayout(row)
        lay.addWidget(c)
        lay.addStretch(1)
        return page

    def _eer_mode_changed(self):
        self._eer_vals[self._eer_prev] = self.eer_value.value()
        self._eer_prev = self.eer_mode.currentData()
        self.eer_value.setValue(self._eer_vals[self._eer_prev])

    def _browse_gain(self):
        path, _ = QFileDialog.getOpenFileName(None, "Gain reference", self.gain_path.text() or "",
                                              "Gain references (*.gain *.mrc *.tif *.tiff);;All files (*)")
        if path:
            self.gain_path.setText(path)

    def _preset_toggled(self, button, checked):
        if checked:
            keys = list(self.preset_buttons)
            idx = keys.index(next(k for k, b in self.preset_buttons.items() if b is button))
            self.preset_stack.setCurrentIndex(idx)
            self.changed.emit()

    def _positioning_changed(self):
        auto = self.positioning.currentData() == "auto"
        self.pos_thickness.setEnabled(auto)
        self.pos_thickness_label.setEnabled(auto)
        self.thickness_label.setText("Fallback thickness" if auto else "Thickness")
        self.thickness.setToolTip("Used if automatic positioning fails." if auto
                                  else "Reconstruction thickness.")

    # ------------------------------------------------------------------ get / set
    def get_settings(self) -> ProcessingSettings:
        m = MotionSettings(align_bin=self.align_bin.value(), bfactor=self.bfactor.value(),
                           max_iterations=self.iterations.value(), tolerance=self.tolerance.value(),
                           max_shift=self.max_shift.value(), mask_axes=self.mask_axes.isChecked(),
                           group=self.group.value())
        excl = []
        for tok in self.exclude.text().replace(";", ",").split(","):
            try:
                excl.append(float(tok.strip()))
            except ValueError:
                pass
        o = OutputSettings(bin_levels=[b for b, cb in self.bin_boxes.items() if cb.isChecked()],
                           aligned=self.out_aligned.isChecked(), even_odd=self.out_evenodd.isChecked(),
                           dose_weighted=self.out_dw.isChecked(), dtype=self.dtype.currentText(),
                           exclude_angles=excl)
        preset = next((k for k, b in self.preset_buttons.items() if b.isChecked()), "patch")
        r = ReconSettings(enabled=self.recon_enabled.isChecked(), preset=preset,
                          bin=int(self.recon_bin.currentData()),
                          use_dose_weighted=bool(self.recon_source.currentData()),
                          patch_size_nm=self.patch_size.value(), patch_overlap=self.patch_overlap.value(),
                          gold_size_nm=self.gold_size.value(), gold_beads=self.gold_beads.value(),
                          erase_gold=self.erase_gold.isChecked(),
                          positioning=self.positioning.currentData(),
                          positioning_thickness_nm=self.pos_thickness.value(),
                          thickness_nm=self.thickness.value(), sirt_like_iterations=self.sirt.value(),
                          remove_xrays=self.remove_xrays.isChecked(), cpus=self.cpus.value(),
                          use_gpu=self.imod_gpu.isChecked(), handedness=self.handedness.currentData(),
                          ctf_correct=self.ctf_correct.isChecked(),
                          deconvolve=self.deconvolve.isChecked(), deconv_strength=self.deconv_strength.value(),
                          deconv_falloff=self.deconv_falloff.value(),
                          remove_redeposition=self.remove_redep.isEnabled() and self.remove_redep.isChecked(),
                          extra_directives=self.extra.toPlainText())
        c = CtfSettings(enabled=self.ctf_enabled.isChecked(), cs_mm=self.ctf_cs.value(),
                        amplitude_contrast=self.ctf_amp.value(),
                        min_res_A=max(self.ctf_lo.value(), self.ctf_hi.value()),
                        max_res_A=min(self.ctf_lo.value(), self.ctf_hi.value()),
                        defocus_min_um=min(self.def_min.value(), self.def_max.value()),
                        defocus_max_um=max(self.def_min.value(), self.def_max.value()))
        by_group = self.eer_mode.currentData() == "group"
        i = InputSettings(eer_fractions=self._eer_vals["fractions"] if by_group else self.eer_value.value(),
                          eer_group=self.eer_value.value() if by_group else 0,
                          eer_upsampling=int(self.eer_up.currentData()),
                          gain_path=self.gain_path.text().strip().strip('"'),
                          gain_mode=self.gain_mode.currentData(), gain_rotate=int(self.gain_rotate.currentData()),
                          gain_flip=self.gain_flip.currentData(), invert_y=self.invert_y.currentData())
        kind, idx = self.device.currentData()
        dose = self.default_dose.value()
        return ProcessingSettings(input=i, motion=m, output=o, recon=r, ctf=c, use_gpu=(kind == "gpu"), gpu_id=idx,
                                  skip_existing=self.skip_existing.isChecked(),
                                  default_dose=dose if dose > 0 else None)

    def set_settings(self, s: ProcessingSettings) -> None:
        m, o, r, i = s.motion, s.output, s.recon, s.input
        self._eer_vals = {"fractions": int(i.eer_fractions), "group": int(i.eer_group) or 50}
        self._eer_prev = "group" if i.eer_group else "fractions"
        self.eer_mode.blockSignals(True)
        self.eer_mode.setCurrentIndex(1 if i.eer_group else 0)
        self.eer_mode.blockSignals(False)
        self.eer_value.setValue(self._eer_vals[self._eer_prev])
        self.eer_up.setCurrentIndex(max(0, self.eer_up.findData(int(i.eer_upsampling))))
        self.gain_path.setText(i.gain_path or "")
        self.gain_mode.setCurrentIndex(max(0, self.gain_mode.findData(i.gain_mode)))
        self.gain_rotate.setCurrentIndex(max(0, self.gain_rotate.findData(int(i.gain_rotate))))
        self.gain_flip.setCurrentIndex(max(0, self.gain_flip.findData(i.gain_flip)))
        self.invert_y.setCurrentIndex(max(0, self.invert_y.findData(i.invert_y)))
        self.align_bin.setValue(int(m.align_bin))
        self.bfactor.setValue(m.bfactor)
        self.max_shift.setValue(m.max_shift)
        self.iterations.setValue(m.max_iterations)
        self.tolerance.setValue(m.tolerance)
        self.group.setValue(m.group)
        self.mask_axes.setChecked(m.mask_axes)
        self.out_aligned.setChecked(o.aligned)
        self.out_evenodd.setChecked(o.even_odd)
        self.out_dw.setChecked(o.dose_weighted)
        for b, cb in self.bin_boxes.items():
            cb.setChecked(b in o.bin_levels)
        self.dtype.setCurrentText(o.dtype)
        self.exclude.setText(", ".join(f"{a:g}" for a in o.exclude_angles))
        for i in range(self.device.count()):
            kind, idx = self.device.itemData(i)
            if (kind == "gpu") == s.use_gpu and (kind == "cpu" or idx == s.gpu_id):
                self.device.setCurrentIndex(i)
                break
        self.skip_existing.setChecked(s.skip_existing)
        self.default_dose.setValue(s.default_dose or 0.0)
        self.recon_enabled.setChecked(r.enabled)
        self.preset_buttons.get(r.preset, self.preset_buttons["patch"]).setChecked(True)
        self.recon_bin.setCurrentIndex(max(0, self.recon_bin.findData(int(r.bin))))
        self.recon_source.setCurrentIndex(1 if r.use_dose_weighted else 0)
        self.patch_size.setValue(r.patch_size_nm)
        self.patch_overlap.setValue(r.patch_overlap)
        self.gold_size.setValue(r.gold_size_nm)
        self.gold_beads.setValue(r.gold_beads)
        self.erase_gold.setChecked(r.erase_gold)
        self.positioning.setCurrentIndex(max(0, self.positioning.findData(r.positioning)))
        self.pos_thickness.setValue(r.positioning_thickness_nm)
        self.thickness.setValue(r.thickness_nm)
        self.sirt.setValue(r.sirt_like_iterations)
        self.remove_xrays.setChecked(r.remove_xrays)
        self.cpus.setValue(min(r.cpus, self.cpus.maximum()))
        self.imod_gpu.setChecked(r.use_gpu)
        self.extra.setPlainText(r.extra_directives)
        self.handedness.setCurrentIndex(max(0, self.handedness.findData(r.handedness)))
        self.ctf_correct.setChecked(r.ctf_correct)
        self.deconvolve.setChecked(r.deconvolve)
        self.deconv_strength.setValue(r.deconv_strength)
        self.deconv_falloff.setValue(r.deconv_falloff)
        self.deconv_strength.setEnabled(r.deconvolve)
        self.deconv_falloff.setEnabled(r.deconvolve)
        self.remove_redep.setChecked(bool(r.remove_redeposition) and self.remove_redep.isEnabled())
        c = s.ctf
        self.ctf_enabled.setChecked(c.enabled)
        self.ctf_cs.setValue(c.cs_mm)
        self.ctf_amp.setValue(c.amplitude_contrast)
        self.ctf_lo.setValue(c.min_res_A)
        self.ctf_hi.setValue(c.max_res_A)
        self.def_min.setValue(c.defocus_min_um)
        self.def_max.setValue(c.defocus_max_um)
        self._positioning_changed()

    def validate(self) -> str | None:
        s = self.get_settings()
        if not s.output.stack_kinds() and not s.recon.enabled:
            return "Select at least one output stack type (Outputs page)."
        if s.output.stack_kinds() and not s.output.bin_levels:
            return "Select at least one binning level (Outputs page)."
        if s.recon.enabled and not imod.imod_status()[0]:
            return "Reconstruction is enabled but IMOD was not found:\n" + imod.imod_status()[1]
        return None

    def set_enabled(self, enabled: bool):
        for p in (self.alignment_page, self.outputs_page, self.recon_page):
            p.setEnabled(enabled)

    def _refresh_presets(self, select: str | None = None):
        self.preset_combo.clear()
        self.preset_combo.addItems(list_presets())
        if select:
            self.preset_combo.setCurrentText(select)

    def _apply_preset(self):
        name = self.preset_combo.currentText()
        if name:
            try:
                self.set_settings(load_preset(name))
            except Exception as e:
                QMessageBox.warning(None, "pyPrep", f"Could not load preset '{name}':\n{e}")

    def _save_preset(self):
        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(None, "Save preset", "Preset name (e.g. microscope and camera):",
                                        text=self.preset_combo.currentText())
        if ok and name.strip():
            save_preset(name.strip(), self.get_settings())
            self._refresh_presets(select=name.strip())

    def _delete_preset(self):
        name = self.preset_combo.currentText()
        if name and QMessageBox.question(None, "pyPrep", f"Delete preset '{name}'?") == QMessageBox.Yes:
            delete_preset(name)
            self._refresh_presets()

    def load_dialog(self):
        path, _ = QFileDialog.getOpenFileName(None, "Load settings", "", "pyPrep settings (*.json)")
        if path:
            try:
                self.set_settings(ProcessingSettings.load(path))
            except Exception as e:
                QMessageBox.warning(None, "pyPrep", f"Could not load settings:\n{e}")

    def save_dialog(self):
        path, _ = QFileDialog.getSaveFileName(None, "Save settings", "pyprep_settings.json",
                                              "pyPrep settings (*.json)")
        if path:
            self.get_settings().save(path)
