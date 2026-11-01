import json

import numpy as np
import pytest

from bdnmc_dune import cards, config, geometry

CFG = config.load()
POINTS = [("pi0", 0.005), ("pi0", 0.03), ("pi0", 0.06), ("eta", 0.01), ("eta", 0.15), ("brem", 0.02),
          ("brem", 0.25)]


def _write(tmp_path, channel, m, x=0.0):
    text, run, mv = cards.card_text(CFG, channel, m, off_axis_x=x, seed=7)
    path = tmp_path / f"{run}.dat"
    path.write_text(text)
    return path, mv


def test_cards_obey_the_card_format(tmp_path):
    from bdnmc_jax.card import split_string
    for ch, m in POINTS:
        path, _ = _write(tmp_path, ch, m)
        for line in path.read_text().splitlines():
            assert line and line == line.strip() and "\t" not in line, line
            if line.startswith("#"):
                continue
            assert "#" not in line, line
            status, key, val = split_string(line)
            # every value is one token, so stock's split_string reads exactly it
            assert status == 1 and val == line.split(" ", 1)[1] and " " not in val, line
        text = path.read_text()
        for key in ("samplesize", "max_trials", "seed", "burn_max", "burn_timeout"):
            v = next(l.split()[1] for l in text.splitlines() if l.startswith(key + " "))
            assert v.isdigit(), (key, v)
        assert text.index("detector cuboid") < text.index("material Argon")


@pytest.mark.parametrize("x", [0.0, 24.0])
@pytest.mark.parametrize("ch,m", POINTS)
def test_port_parses_and_builds_the_cards(tmp_path, ch, m, x):
    from bdnmc_jax import branching, card as cardmod, detector, pipeline
    path, mv = _write(tmp_path, ch, m, x)
    card = cardmod.parse_card(path)
    det, sc, chans = pipeline.setup(card)
    assert isinstance(det, detector.Cuboid)
    assert det.center == pytest.approx((x, 0.0, 574.0))
    axes = np.asarray(det.faces[:3]) / (np.asarray(det.face_dist)[:, None] / 2)
    np.testing.assert_allclose(axes, geometry.beam_to_hall(CFG.slope_rad), atol=1e-9)
    assert card.e_density == pytest.approx(18 * CFG.lar_number_density)
    assert card.mass_dm == m and card.mass_v == pytest.approx(mv) and card.alpha_d == 0.5
    assert card.pot == 1.1e21 and card.beam_energy == 120.0
    (chan,) = chans
    if ch == "pi0":
        br, _ = branching.pi0_channel(card.mass_v, card.mass_dm, card.epsilon, card.alpha_d)
        assert chan.vnum == pytest.approx(br * 1.1e21 * 4.5)
        assert card.target_mass_number == 12
    elif ch == "eta":
        br, _ = branching.eta_channel(card.mass_v, card.mass_dm, card.epsilon, card.alpha_d)
        assert chan.vnum == pytest.approx(br * 1.1e21 * 0.5)  # DUNE's c_eta ~ 0.5 per POT
    else:
        assert chan.vnum > 0


def test_closed_points_and_prism_range(tmp_path):
    with pytest.raises(ValueError):
        cards.card_text(CFG, "pi0", 0.07)  # 2 m_chi > m_pi0
    with pytest.raises(ValueError):
        cards.card_text(CFG, "brem", 0.05, mass_v=0.08)
    with pytest.raises(ValueError):
        cards.card_text(CFG, "pi0", 0.01, off_axis_x=31.0)
    assert cards.card_text(CFG, "pi0", 0.01, off_axis_x=31.0, allow_beyond_prism=True)[0]


def test_mass_scan(tmp_path):
    from bdnmc_jax import card as cardmod
    man = cards.write_scan(CFG, tmp_path, channels=("pi0", "eta"), masses_dm=(0.005, 0.1, 0.3),
                           off_axis=CFG.value("prism.scan_positions"), seed0=100)
    # pi0: 0.005 open, 0.1 and 0.3 closed; eta: 0.005 and 0.1 open, 0.3 closed; 5 positions each
    assert len(man["cards"]) == 3 * 5 and len(man["skipped"]) == 3
    assert [c["seed"] for c in man["cards"]] == list(range(100, 115))
    assert json.loads((tmp_path / "manifest.json").read_text()) == man
    for c in man["cards"]:
        card = cardmod.parse_card(c["path"])
        assert card.seed == c["seed"]
        assert cardmod.stod(card.keymap["x-position"]) == c["off_axis_m"]
