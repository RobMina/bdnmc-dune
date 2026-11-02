import math

import numpy as np
import pytest

from bdnmc_dune import cards, config, geometry

CFG = config.load()


def test_beam_points_down_by_the_slope_in_the_hall():
    s = CFG.slope_rad
    R = geometry.beam_to_hall(s)
    np.testing.assert_allclose(R @ [0, 0, 1], [0, -math.sin(s), math.cos(s)], atol=1e-15)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-15)
    assert np.linalg.det(R) == pytest.approx(1.0)


def test_cuboid_axes_are_the_hall_axes():
    s = CFG.slope_rad
    axes = geometry.cuboid_axes(*geometry.euler_angles(s))
    # rows: the hall axes in the beam frame = the rows of R (v_hall = R v_beam)
    np.testing.assert_allclose(axes, geometry.beam_to_hall(s), atol=1e-15)


def test_cuboid_axes_match_the_port(tmp_path):
    from bdnmc_jax import detector
    for angles in [(-0.0925, 0.0, 0.0), (0.3, -0.2, 0.7), (0.0, 1.1, -0.4)]:
        det = detector.make_cuboid(0.0, 0.0, 0.0, 2.0, 4.0, 6.0, *angles)
        faces = np.asarray(det.faces[:3]) / np.array([[1.0], [2.0], [3.0]])
        np.testing.assert_allclose(geometry.cuboid_axes(*angles), faces, atol=1e-14)


@pytest.mark.parametrize("x", [0.0, 6.0, -12.0, 24.0])
def test_placement(x):
    p = geometry.placement(CFG, x)
    assert (p["x-position"], p["y-position"], p["z-position"]) == pytest.approx((x, 0.0, 574.0))
    assert (p["width"], p["height"], p["length"]) == (7.0, 3.0, 5.0)
    assert p["det-phi"] == pytest.approx(-math.radians(5.3)) and p["det-theta"] == 0 and p["det-psi"] == 0


def test_vertical_offset_is_vertical_in_the_hall():
    c = geometry.center_beam_frame(574.0, 0.0, 1.0, CFG.slope_rad)
    R = geometry.beam_to_hall(CFG.slope_rad)
    np.testing.assert_allclose(R @ (c - [0, 0, 574.0]), [0, 1, 0], atol=1e-14)


@pytest.mark.parametrize("x", [0.0, 18.0])
@pytest.mark.parametrize("g", [(0.0, 0.0, 0.0), (0.5, -2.68, 13.0)])
def test_frame_from_card_matches_the_config(tmp_path, x, g):
    text, run, _ = cards.card_text(CFG, "pi0", 0.01, off_axis_x=x)
    path = tmp_path / f"{run}.dat"
    path.write_text(text)
    o1, r1 = geometry.frame_from_card(path, g)
    o2, r2 = geometry.frame(geometry.center_beam_frame(574.0, x, 0.0, CFG.slope_rad),
                            geometry.beam_to_hall(CFG.slope_rad), g)
    np.testing.assert_allclose(r1, r2, atol=1e-10)
    np.testing.assert_allclose(o1, o2, atol=1e-8)
    # write_hepevt's transform rotation @ (r - origin): the detector centre lands on g, the hall axes stay axes
    centre = np.array([x, 0.0, 574.0])
    np.testing.assert_allclose(r1 @ (centre - o1), g, atol=1e-8)
    R = geometry.beam_to_hall(CFG.slope_rad)
    for i in range(3):
        np.testing.assert_allclose(r1 @ (centre + R[i] - o1) - np.asarray(g), np.eye(3)[i], atol=1e-8)


def test_test_geometry_is_the_active_volume_in_air(tmp_path):
    import xml.etree.ElementTree as ET
    from bdnmc_dune import cli, testgeom
    out = tmp_path / "box.gdml"
    cli.main(["testgeom", "--out", str(out)])
    root = ET.parse(out).getroot()
    boxes = {b.get("name"): [float(b.get(k)) for k in "xyz"] for b in root.iter("box")}
    assert boxes["LArActive_box"] == [7.0, 3.0, 5.0]
    vol = {v.get("name"): v for v in root.iter("volume")}
    assert vol["volLArActive"].find("materialref").get("ref") == "G4_lAr"
    aux = vol["volLArActive"].find("auxiliary")
    assert (aux.get("auxtype"), aux.get("auxvalue")) == ("SegmentDetector", testgeom.SD_NAME)
    assert root.find("setup/world").get("ref") == "volWorld"
    with pytest.raises(ValueError):
        testgeom.box_gdml(CFG, world_m=(5.0, 20.0, 20.0))
