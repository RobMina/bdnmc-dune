import math

import numpy as np
import pytest

from bdnmc_dune import config, gdml, testgeom

# Two boxes of "Active" in a "Module", the module rotated and shifted in a "Hall", the hall shifted in the world:
# the GeGeDe pattern (define-section positions and rotations, physvols by reference, mixed units).
NESTED = """<?xml version="1.0"?>
<gdml>
  <define>
    <position name="a0" unit="cm" x="-25" y="0" z="0"/>
    <position name="a1" unit="cm" x="25" y="0" z="0"/>
    <position name="mod_pos" unit="m" x="1" y="2" z="3"/>
    <rotation name="mod_rot" unit="degree" x="0" y="90" z="0"/>
    <position name="hall_pos" unit="mm" x="-500" y="100" z="7000"/>
    <rotation name="identity" x="0" y="0" z="0"/>
  </define>
  <materials/>
  <solids>
    <box name="World_box" lunit="m" x="40" y="40" z="40"/>
    <box name="Hall_box" lunit="m" x="20" y="20" z="20"/>
    <box name="Mod_box" lunit="cm" x="100" y="300" z="100"/>
    <box name="Act_box" lunit="mm" x="400" y="3000" z="1000"/>
    <union name="U"><first ref="Hall_box"/><second ref="Mod_box"/><position name="u_pos" x="1" y="0" z="0"/></union>
  </solids>
  <structure>
    <volume name="Active">
      <materialref ref="LAr"/><solidref ref="Act_box"/>
      <auxiliary auxtype="SensDet" auxvalue="Active"/>
    </volume>
    <volume name="Module">
      <materialref ref="LAr"/><solidref ref="Mod_box"/>
      <physvol name="act0"><volumeref ref="Active"/><positionref ref="a0"/><rotationref ref="identity"/></physvol>
      <physvol name="act1"><volumeref ref="Active"/><positionref ref="a1"/><rotationref ref="identity"/></physvol>
    </volume>
    <volume name="Hall">
      <materialref ref="Air"/><solidref ref="Hall_box"/>
      <physvol name="mod"><volumeref ref="Module"/><positionref ref="mod_pos"/><rotationref ref="mod_rot"/></physvol>
      <physvol name="mod_b"><volumeref ref="Module"/>
        <position name="inline_pos" unit="m" x="-3" y="0" z="0"/></physvol>
    </volume>
    <volume name="World">
      <materialref ref="Vac"/><solidref ref="World_box"/>
      <physvol name="hall"><volumeref ref="Hall"/><positionref ref="hall_pos"/><rotationref ref="identity"/></physvol>
      <auxiliary auxtype="git_tag" auxvalue="test"/>
    </volume>
  </structure>
  <setup name="Default" version="1.0"><world ref="World"/></setup>
</gdml>
"""


@pytest.fixture
def nested(tmp_path):
    p = tmp_path / "nested.gdml"
    p.write_text(NESTED)
    return gdml.load(p)


def test_rotation_follows_geant4():
    # Geant4 builds the matrix as rotateX, rotateY, rotateZ (each from the left) and places the daughter with its
    # inverse: a GDML rotation of +90 deg about y turns the daughter's +x axis into the mother's +z.
    M = gdml.rotation_matrix(0.0, math.pi / 2, 0.0)
    np.testing.assert_allclose(M.T @ [1, 0, 0], [0, 0, 1], atol=1e-15)
    a, b, c = 0.3, -0.7, 1.1
    np.testing.assert_allclose(gdml.rotation_matrix(a, b, c),
                               gdml._rz(c) @ gdml._ry(b) @ gdml._rx(a), atol=1e-15)


def test_nested_placements(nested):
    inst = nested.instances("Active")
    assert len(inst) == 4
    hall = np.array([-500.0, 100.0, 7000.0])
    R = gdml.rotation_matrix(0.0, math.pi / 2, 0.0).T
    want = {
        ("hall", "mod", "act0"): hall + [1000, 2000, 3000] + R @ [-250, 0, 0],
        ("hall", "mod", "act1"): hall + [1000, 2000, 3000] + R @ [250, 0, 0],
        ("hall", "mod_b", "act0"): hall + [-3000 - 250, 0, 0],
        ("hall", "mod_b", "act1"): hall + [-3000 + 250, 0, 0],
    }
    got = {path: t for path, _, t in inst}
    assert set(got) == set(want)
    for k in want:
        np.testing.assert_allclose(got[k], want[k], atol=1e-9)
    # the rotated module's active boxes lie along z, 250 mm either side of the module centre
    np.testing.assert_allclose(got[("hall", "mod", "act1")][2] - got[("hall", "mod", "act0")][2], 500.0, atol=1e-9)


def test_union_box_and_report(nested):
    low, high = gdml.union_box(nested, "Active")
    # rotated module: active boxes 400 (x) x 3000 (y) x 1000 (z) become 1000 x 3000 x 400 at z = 10000 +- 250
    np.testing.assert_allclose(low, [-3950.0, -1400.0, 6500.0], atol=1e-9)
    np.testing.assert_allclose(high, [1000.0, 3600.0, 10450.0], atol=1e-9)
    rep = gdml.frame_report(nested, active="Active", landmarks=["Module"])
    assert rep["active"]["placements"] == 4
    assert rep["active"]["max_abs_rotation_minus_identity"] == pytest.approx(1.0)
    assert rep["active"]["determinants"] == [1.0]
    assert rep["sensitive_detectors"] == {"Active": {"volumes": 1, "first": ["Active"]}}
    assert len(rep["active"]["distinct_rotations"]) == 2  # the turned module and the unturned one
    assert rep["world_aux"] == {"git_tag": "test"}
    assert rep["world_box_size_m"] == pytest.approx([40, 40, 40])
    np.testing.assert_allclose(rep["active"]["union_centre_m"], (low + high) / 2e3)
    assert rep["active"]["summed_volume_m3"] == pytest.approx(4 * 0.4 * 3.0 * 1.0)
    assert len(rep["landmarks_m"]["Module"]) == 2


def test_test_geometry_is_detector_centred(tmp_path):
    p = tmp_path / "box.gdml"
    p.write_text(testgeom.box_gdml())
    rep = gdml.frame_report(gdml.load(p), active="volLArActive")
    a = rep["active"]
    assert a["placements"] == 1
    assert a["union_centre_m"] == pytest.approx([0.0, 0.0, 0.0])
    cfg = config.load()
    assert a["union_size_m"] == pytest.approx([cfg.value(f"nd_lar.active_{k}") for k in ("width", "height", "length")])
    assert rep["sensitive_detectors"] == {testgeom.SD_NAME: {"volumes": 1, "first": ["volLArActive"]}}


def test_unsupported(tmp_path):
    p = tmp_path / "bad.gdml"
    p.write_text(NESTED.replace('<rotationref ref="mod_rot"/>', '<rotationref ref="mod_rot"/><scale name="s" x="-1" '
                                'y="1" z="1"/>'))
    with pytest.raises(ValueError, match="scale"):
        gdml.load(p)


def test_cli_gdml_frame(tmp_path, capsys):
    import json
    from bdnmc_dune import cli
    p = tmp_path / "nested.gdml"
    p.write_text(NESTED)
    cli.main(["gdml-frame", str(p), "--active", "Active", "--landmark", "Hall", "--out", str(tmp_path / "r.json")])
    rep = json.loads((tmp_path / "r.json").read_text())
    assert json.loads(capsys.readouterr().out) == rep
    assert len(rep["gdml_sha256"]) == 64 and rep["active"]["placements"] == 4
    assert rep["landmarks_m"]["Hall"] == [{"origin": [-0.5, 0.1, 7.0], "max_abs_rotation_minus_identity": 0.0}]
