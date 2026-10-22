"""Redeposition removal: the shadow fill, ASTRA availability, the option on the Reconstruction page, and (with ASTRA
and a CUDA GPU) its projector and constrained SIRT."""

import numpy as np
import pytest

from pyprep import redep
from test_gui_threads import _child

ASTRA_OK = redep.astra_status()[0]


def test_fill_shadows_changes_nothing_without_a_shadow():
    p = 1.0 + 0.1 * np.random.default_rng(0).standard_normal((64, 64)).astype(np.float32)
    assert np.allclose(redep.fill_shadows(p, np.ones_like(p)), p)


def test_fill_shadows_restores_the_mean_under_a_grain_without_amplifying_noise():
    rng = np.random.default_rng(1)
    clean = 1.0 + 0.05 * rng.standard_normal((128, 128))
    t = np.ones_like(clean)
    t[48:80, 48:80] = 0.6                                  # a grain absorbing 40 %
    p = clean * t
    out = redep.fill_shadows(p, t, sigma_px=8)
    inside = (slice(54, 74), slice(54, 74))
    assert abs(p[inside].mean() - 1.0) > 0.3
    assert abs(out[inside].mean() - 1.0) < 0.03
    assert out[inside].std() < 1.2 * p[inside].std()      # dividing by T would give 1 / 0.6 x the noise


def test_astra_status_is_reported():
    ok, msg = redep.astra_status()
    assert isinstance(ok, bool) and msg


OPTION = r'''
import sys
from PySide6.QtWidgets import QApplication
from pyprep import redep
available = sys.argv[1] == "yes"
redep.astra_status = lambda: (available, "ASTRA test (CUDA)" if available else "ASTRA Toolbox not installed")
from pyprep.gui.settings_panel import SettingsForms
app = QApplication([])
forms = SettingsForms()
s = forms.get_settings()
s.recon.remove_redeposition = True
forms.set_settings(s)
print("OK", forms.remove_redep.isEnabled(), forms.get_settings().recon.remove_redeposition, flush=True)
'''


@pytest.mark.parametrize("available", ["yes", "no"])
def test_option_is_greyed_out_without_astra(available):
    pytest.importorskip("PySide6")
    r = _child(OPTION, available)
    assert r.returncode == 0, f"{r.stdout}\n{r.stderr[-3000:]}"
    line = next(l for l in r.stdout.splitlines() if l.startswith("OK"))
    enabled, chosen = line.split()[1:]
    want = "True" if available == "yes" else "False"
    assert enabled == want                    # greyed out without ASTRA ...
    assert chosen == want                     # ... and a saved setting cannot switch it on


@pytest.mark.skipif(not ASTRA_OK, reason="needs the ASTRA Toolbox with a CUDA GPU")
def test_constrained_sirt_recovers_a_dense_block():
    vol = np.zeros((8, 64, 64), np.float32)                # [Y, Z, X]
    vol[:, 20:28, 16:48] = 1.0
    block = vol > 0
    ang = np.deg2rad(np.arange(-60.0, 61.0, 3.0))          # a missing wedge, as in a tilt series
    proj = redep.forward_project(vol, ang)                 # [Y, views, X]
    assert proj.shape == (8, len(ang), 64)
    support = np.zeros(vol.shape, bool)
    support[:, 12:36, :] = True
    rec = redep.sirt(proj, support, ang, 150)
    assert np.allclose(rec[~support], 0)                   # nothing outside the support
    reproj = redep.forward_project(rec, ang)
    assert np.abs(reproj - proj).mean() < 0.05 * np.abs(proj).mean()     # it fits the data
    assert rec[block].mean() > 4 * rec[support & ~block].mean()          # the block stands out
    free = redep.sirt(proj, np.ones(vol.shape, bool), ang, 150)
    corr = lambda r: np.corrcoef(r[support], vol[support])[0, 1]  # noqa: E731
    assert corr(rec) > corr(free)                          # the support constraint helps against the wedge
