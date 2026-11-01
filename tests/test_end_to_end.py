"""A few 2^16-trial blocks of a real ND-LAr card through the JAX port on the CPU, to edep-sim HEPEVT input."""
import json
import math

import numpy as np
import pytest

from bdnmc_dune import cards, cli, config, geometry, hepevt

CFG = config.load()
BLOCK = 1 << 16
HALF_CM = np.array(CFG.active_size) * 100 / 2


@pytest.fixture(scope="module", params=[0.0, 12.0])
def run(request, tmp_path_factory):
    from bdnmc_jax import pipeline
    d = tmp_path_factory.mktemp(f"x{request.param:g}")
    text, run, _ = cards.card_text(CFG, "pi0", 0.005, off_axis_x=request.param, seed=11)
    card = d / f"{run}.dat"
    card.write_text(text)
    res = pipeline.events(card, 2 * BLOCK, batch=2)
    info = hepevt.write(res, d / "events.hepevt")
    return request.param, card, res, info, d


def test_events_are_edepsim_readable_one_vertex_each(run):
    x, card, res, info, d = run
    events = hepevt.read(info["path"])
    assert res["accepted"] > 100
    assert len(events) == info["edepsim_events"] == res["accepted"]  # one interaction point per edep-sim event
    assert info["bdnmc_events"] <= res["accepted"]
    for head, parts in events:
        assert len(parts) == 2
        el, dm = parts
        assert el[:2] == ["1", "11"] and dm[:2] == ["2", "18"]  # only the electron is tracked
        assert el[11:15] == dm[11:15]
    index = (d / "events.hepevt.index.csv").read_text().splitlines()
    assert index[0] == "edepsim_event,bdnmc_event" and len(index) == len(events) + 1
    bdnmc = [int(l.split(",")[1]) for l in index[1:]]
    assert bdnmc == sorted(bdnmc) and len(set(bdnmc)) == info["bdnmc_events"]


def test_vertices_are_inside_ndlar_in_the_detector_frame(run):
    x, card, res, info, d = run
    v = np.array([[float(t) for t in p[11:15]] for _, parts in hepevt.read(info["path"]) for p in parts[:1]])
    assert (np.abs(v[:, :3]) <= HALF_CM + 1e-6).all()
    # spread over the volume, not collapsed (frame or units wrong would show here)
    assert (np.ptp(v[:, :3], axis=0) > HALF_CM).all()
    # times in ns: the DM's time of flight from the target, ~ 574 m / c
    tof = 574.0 / 299792458.0 * 1e9
    assert np.median(v[:, 3]) == pytest.approx(tof, rel=0.01)


def test_hepevt_values_are_the_ports_rows_transformed(run):
    x, card, res, info, d = run
    from bdnmc_jax import output
    rows, _ = output.accepted_events(res)
    c = res["candidates"]
    origin, rot = geometry.frame_from_card(card)
    parts = [p for _, ps in hepevt.read(info["path"]) for p in ps]
    el = np.array([[float(t) for t in p] for p in parts[0::2]])
    dm = np.array([[float(t) for t in p] for p in parts[1::2]])
    vtx = c["vertex"][rows]
    np.testing.assert_allclose(el[:, 11:14], (vtx[:, :3] - origin) @ rot.T * 100, rtol=1e-8, atol=1e-5)
    np.testing.assert_allclose(el[:, 14], vtx[:, 3] * 1e9, rtol=1e-8)
    np.testing.assert_allclose(el[:, 6:9], c["electron"][rows, 1:] @ rot.T, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(el[:, 9], c["electron"][rows, 0], rtol=1e-9)
    np.testing.assert_allclose(dm[:, 6:9], c["recoil_dm"][rows, 1:] @ rot.T, rtol=1e-8, atol=1e-12)
    # GeV: |p|^2 + m^2 = E^2 for the electron
    np.testing.assert_allclose(np.sum(el[:, 6:9] ** 2, axis=1) + el[:, 10] ** 2, el[:, 9] ** 2, rtol=1e-6)


def test_dm_comes_down_the_beam_in_the_hall_frame(run):
    x, card, res, info, d = run
    from bdnmc_jax import output
    rows, _ = output.accepted_events(res)
    _, rot = geometry.frame_from_card(card)
    p = res["candidates"]["dm"][rows, 1:] @ rot.T
    u = p / np.linalg.norm(p, axis=1)[:, None]
    s = CFG.slope_rad
    # the DM direction ~ from the target to the detector; on axis, the beam's (0, -sin s, cos s)
    to_det = geometry.beam_to_hall(s) @ (np.array([x, 0.0, 574.0]) / math.hypot(x, 574.0))
    np.testing.assert_allclose(u.mean(axis=0), to_det, atol=0.005)


def test_gdml_offset_shifts_the_vertices(run):
    x, card, res, info, d = run
    g = (0.25, -2.68, 13.0)
    info2 = hepevt.write(res, d / "shifted.hepevt", gdml_offset=g)
    a = np.array([[float(t) for t in p[11:14]] for _, ps in hepevt.read(info["path"]) for p in ps])
    b = np.array([[float(t) for t in p[11:14]] for _, ps in hepevt.read(info2["path"]) for p in ps])
    np.testing.assert_allclose(b - a, np.tile(np.array(g) * 100, (len(a), 1)), atol=1e-5)


def test_cli_run(run, tmp_path):
    x, card, res, info, d = run
    if x:
        pytest.skip("once is enough")
    cli.main(["run", str(card), "--trials", str(2 * BLOCK), "--batch", "2", "--hepevt", str(tmp_path / "e.hepevt"),
              "--macro", str(tmp_path / "gen.mac"), "--summary", str(tmp_path / "s.json")])
    s = json.loads((tmp_path / "s.json").read_text())
    assert s["accepted"] == res["accepted"] and s["se"] == res["se"]
    assert s["hepevt"]["edepsim_events"] == res["accepted"]
    assert (tmp_path / "e.hepevt").read_text() == (d / "events.hepevt").read_text()
    assert f"/generator/kinematics/hepevt/input {(tmp_path / 'e.hepevt').resolve()}" in (tmp_path / "gen.mac").read_text()
