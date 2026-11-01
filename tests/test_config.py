import math

import pytest

from bdnmc_dune import config

CFG = config.load()
NOT_DUNE = ("user", "convention", "argon-40")


def test_every_number_has_a_unit_and_a_source():
    keys = set(CFG.data["sources"])
    qs = CFG.quantities()
    assert len(qs) > 30
    for path, q in qs:
        assert set(q) == {"value", "unit", "source"}, path
        src = q["source"]
        assert src.strip(), path
        assert any(k in src for k in keys) or src.startswith(NOT_DUNE), (path, src)


def test_active_volume_matches_the_published_masses():
    # 7 m x 3 m x 5 m of LAr at 1.396 g/cm^3 against TDR Vol. II Table 8.2 (147 t total, 67.2 t fiducial)
    assert CFG.active_mass_t == pytest.approx(CFG.value("nd_lar.total_mass"), rel=0.01)
    fx, fy, fz = CFG.value("nd_lar.fiducial_size")
    assert fx * fy * fz * CFG.value("lar.density") == pytest.approx(CFG.value("nd_lar.fiducial_mass"), rel=0.01)


def test_modules_tile_the_active_volume():
    mx, my, mz = CFG.value("nd_lar.module_active_size")
    w, h, l = CFG.active_size
    assert CFG.value("nd_lar.modules_transverse") * mx == w
    assert CFG.value("nd_lar.modules_along_beam") * mz == l
    assert my == h


def test_argon_numbers():
    assert CFG.lar_number_density == pytest.approx(1.396 / 39.948 * 6.02214076e23)
    assert CFG.lar_number_density * 18 == pytest.approx(3.788e23, rel=1e-3)  # electrons per cm^3
    assert CFG.lar_nucleus_mass == pytest.approx(37.21, abs=0.01)
    assert CFG.slope_rad == pytest.approx(math.radians(5.3))
    assert CFG.pot == 1.1e21


def test_overrides(tmp_path):
    f = tmp_path / "o.toml"
    f.write_text("[beam]\nslope_at_nd = 5.8\n[frame]\ngdml_offset = { value = [0.0, -2.68, 13.0], unit = \"m\", "
                 "source = \"user: my GDML\" }\n")
    cfg = config.load(f)
    assert cfg.value("beam.slope_at_nd") == 5.8 and cfg.source("beam.slope_at_nd") == "user override"
    assert cfg.value("frame.gdml_offset") == [0.0, -2.68, 13.0]
    assert CFG.value("beam.slope_at_nd") == 5.3  # the packaged default is untouched
    bad = tmp_path / "bad.toml"
    bad.write_text("[beam]\nslope = 1\n")
    with pytest.raises(KeyError):
        config.load(bad)
