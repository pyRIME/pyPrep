"""Processing settings, shared by the CLI and GUI and saved with every output as JSON."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path

from .ctf import CtfSettings
from .imod import ReconSettings
from .motion import MotionSettings


@dataclass
class InputSettings:
    """How movies are read: EER fractionation/rendering and the gain reference."""
    eer_fractions: int = 10         # group each tilt's EER frames into this many fractions
    eer_group: int = 0              # > 0: fixed EER frames per fraction instead of eer_fractions
    eer_upsampling: int = 1         # 1 = physical pixels (4K), 2 = 8K super-resolution rendering
    gain_path: str = ""             # gain reference: EPU .gain (TIFF), MRC or TIFF; "" = none
    gain_mode: str = "auto"         # auto (divide for .gain/EER, multiply otherwise) | multiply | divide
    gain_rotate: int = 0            # rotate the gain reference counter-clockwise by 0/90/180/270 deg
    gain_flip: str = "none"         # then flip it: none | x (left-right) | y (up-down)
    invert_y: str = "auto"          # invert the aligned images in Y: auto (Tomography 5 data) | yes | no


@dataclass
class OutputSettings:
    bin_levels: list = field(default_factory=lambda: [1, 4])
    aligned: bool = True            # motion-corrected sum of all frames
    even_odd: bool = False          # half-sums of alternate frames (denoising training)
    dose_weighted: bool = False     # exposure-filtered sum (skip etomo's dose weighting then)
    dtype: str = "float32"          # "float32" or "int16"
    exclude_angles: list = field(default_factory=list)   # tilt angles (deg) to leave out

    def stack_kinds(self) -> list[str]:
        kinds = []
        if self.aligned:
            kinds.append("sum")
        if self.even_odd:
            kinds += ["even", "odd"]
        if self.dose_weighted:
            kinds.append("dw")
        return kinds


@dataclass
class ProcessingSettings:
    input: InputSettings = field(default_factory=InputSettings)
    motion: MotionSettings = field(default_factory=MotionSettings)
    output: OutputSettings = field(default_factory=OutputSettings)
    recon: ReconSettings = field(default_factory=ReconSettings)
    ctf: CtfSettings = field(default_factory=CtfSettings)
    frames_dir: str | None = None   # where fraction files live, if not next to the mdoc
    use_gpu: bool = True
    gpu_id: int = 0
    skip_existing: bool = True      # skip series whose outputs are already complete
    default_dose: float | None = None   # e/A^2 per tilt if the mdoc has none

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ProcessingSettings":
        def build(klass, data):
            names = {f.name for f in fields(klass)}
            return klass(**{k: v for k, v in (data or {}).items() if k in names})
        top = {k: v for k, v in d.items() if k not in ("input", "motion", "output", "recon", "ctf")}
        s = build(cls, top)
        s.input = build(InputSettings, d.get("input"))
        s.motion = build(MotionSettings, d.get("motion"))
        s.output = build(OutputSettings, d.get("output"))
        s.recon = build(ReconSettings, d.get("recon"))
        s.ctf = build(CtfSettings, d.get("ctf"))
        return s

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path) -> "ProcessingSettings":
        return cls.from_dict(json.loads(Path(path).read_text()))


# ---------------------------------------------------------------------- presets
def presets_dir() -> Path:
    import os
    base = Path(os.environ.get("APPDATA", Path.home())) / "pyPrep" / "presets"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _preset_file(name: str) -> Path:
    safe = "".join(c if c.isalnum() or c in " -_()." else "_" for c in name).strip() or "preset"
    return presets_dir() / f"{safe}.json"


def list_presets() -> list[str]:
    """Saved presets (names), plus the built-in ones on first use."""
    d = presets_dir()
    if not any(d.glob("*.json")):
        ProcessingSettings().save(_preset_file("Tomo5 K3 fractions (default)"))
        eer = ProcessingSettings()
        eer.input.eer_fractions = 10
        eer.input.gain_mode = "auto"
        eer.save(_preset_file("Falcon EER (set the gain reference)"))
    return sorted(p.stem for p in d.glob("*.json"))


def save_preset(name: str, settings: ProcessingSettings) -> Path:
    path = _preset_file(name)
    settings.save(path)
    return path


def load_preset(name: str) -> ProcessingSettings:
    return ProcessingSettings.load(_preset_file(name))


def delete_preset(name: str) -> None:
    _preset_file(name).unlink(missing_ok=True)
