"""BdNMC parameter cards for ND-LAr: one card per (channel, dark-matter mass, off-axis position).

The cards follow the fork's generic 120 GeV cuboid cards (gpu/phase0/cards/he_pi0_cuboid_m005.dat and
he_brem_cuboid_m005.dat) key for key, with ND-LAr's placement, size and liquid-argon material, LBNF's beam and
target, and DUNE's meson multiplicities (sources in data/nd_lar.toml). They obey the card format's pitfalls
(docs/parameter-card.md in the fork): no comment after a value, no indentation, integers written in full, the
detector keys before the material block. Stock BdNMC reads the same cards (the in-process mode).

Channels (meson production: BdNMC's own BMPT parametrisation at the LBNF beam energy on carbon, a stand-in until
g4lbnf meson lists replace it; see the README):
- `pi0`: pi0 -> gamma V, BMPT pi0 (`bmpt`), normalised to DUNE's pi0 per POT;
- `eta`: eta -> gamma V, BMPT K0 shape for eta (`bpmt_k0`, as stock spells it), normalised to DUNE's eta per POT;
- `brem`: proton bremsstrahlung (`V_decay` with `proton_brem`), the generic cards' z and pT window.
"""
from dataclasses import dataclass, asdict
import json
from pathlib import Path

from . import __version__, geometry

# Parent masses (GeV) for the kinematic limit 2 m_chi < m_meson; the port uses the same values (bdnmc_jax.constants).
MESON_MASS = {"pi0": 0.1349766, "eta": 0.547862}
CHANNELS = ("pi0", "eta", "brem")


def _g(x):
    return "%.10g" % x


@dataclass
class CardInfo:
    path: str
    channel: str
    mass_dm: float
    mass_v: float
    off_axis_m: float
    seed: int
    run: str


def kinematically_open(channel, mass_dm, mass_v):
    """Whether the port can run the point: 2 m_chi below the meson mass (meson channels; above m_V the decay is
    off-shell, which the port handles), or below m_V (brem)."""
    if channel == "brem":
        return 2 * mass_dm < mass_v
    return 2 * mass_dm < MESON_MASS[channel]


def card_text(cfg, channel, mass_dm, mass_v=None, off_axis_x=0.0, seed=1, run=None, allow_beyond_prism=False):
    """The card for one point. `mass_v` defaults to generator.mv_over_mchi x mass_dm; `off_axis_x` is the
    horizontal DUNE-PRISM displacement (m), limited to prism.max_off_axis unless `allow_beyond_prism`."""
    if channel not in CHANNELS:
        raise ValueError(f"channel {channel!r} not in {CHANNELS}")
    if mass_v is None:
        mass_v = round(cfg.value("generator.mv_over_mchi") * mass_dm, 12)
    if not kinematically_open(channel, mass_dm, mass_v):
        raise ValueError(f"{channel} closed for m_chi = {mass_dm} GeV, m_V = {mass_v} GeV")
    if abs(off_axis_x) > cfg.value("prism.max_off_axis") and not allow_beyond_prism:
        raise ValueError(f"off-axis {off_axis_x} m beyond DUNE-PRISM's {cfg.value('prism.max_off_axis')} m")
    if int(seed) != seed or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    run = run or f"ndlar_{channel}_mchi{_g(mass_dm)}_mv{_g(mass_v)}_x{_g(off_axis_x)}"
    if " " in run:
        raise ValueError("run name must be one token")
    det = geometry.placement(cfg, off_axis_x)
    g = cfg.value
    lines = [
        f"# bdnmc-dune {__version__}: DUNE ND-LAr, channel {channel}, off-axis {_g(off_axis_x)} m.",
        "# Numbers and their sources: bdnmc_dune/data/nd_lar.toml (published DUNE/LBNF documents, PDG).",
        "model Dark_Photon",
        f"epsilon {_g(g('generator.epsilon'))}",
        f"dark_matter_mass {_g(mass_dm)}",
        f"dark_photon_mass {_g(mass_v)}",
        f"alpha_D {_g(g('generator.alpha_d'))}",
        f"POT {_g(cfg.pot)}",
    ]
    if channel != "brem":
        lines.append(f"pi0_per_POT {_g(g('beam.pi0_per_pot'))}")
    lines += [
        "samplesize 1000000000",
        "max_trials 20000000",
        f"seed {int(seed)}",
        "burn_max 1000",
        "burn_timeout 20000",
        f"beam_energy {_g(g('beam.proton_energy'))}",
        f"run {run}",
    ]
    if channel != "brem":
        lines += [f"p_num_target {int(g('beam.target_proton_number'))}",
                  f"n_num_target {int(g('beam.target_neutron_number'))}"]
    if channel == "pi0":
        lines += ["production_channel pi0_decay", "production_distribution bmpt"]
    elif channel == "eta":
        lines += ["production_channel eta_decay", "production_distribution bpmt_k0",
                  f"meson_per_pi0 {_g(g('beam.eta_per_pot') / g('beam.pi0_per_pot'))}"]
    else:
        lines += ["production_channel V_decay", "production_distribution proton_brem",
                  f"zmin {_g(g('generator.brem_zmin'))}", f"zmax {_g(g('generator.brem_zmax'))}",
                  f"ptmax {_g(g('generator.brem_ptmax'))}"]
    lines += [
        "signal_channel NCE_electron",
        "output_file events.dat",
        "summary_file summary.dat",
        "output_mode comprehensive",
        f"max_scatter_energy {_g(g('generator.max_scatter_energy'))}",
        f"min_scatter_energy {_g(g('generator.min_scatter_energy'))}",
        "detector cuboid",
    ]
    lines += [f"{k} {_g(v)}" for k, v in det.items()]
    lines += [
        "material Argon",
        f"number_density {_g(cfg.lar_number_density)}",
        f"proton_number {int(g('lar.atomic_number'))}",
        f"neutron_number {int(g('lar.neutron_number'))}",
        f"electron_number {int(g('lar.atomic_number'))}",
        f"mass {_g(cfg.lar_nucleus_mass)}",
    ]
    return "\n".join(lines) + "\n", run, mass_v


def write_scan(cfg, outdir, channels=("pi0",), masses_dm=(0.005,), off_axis=(0.0,), mass_v=None, seed0=20261031,
               allow_beyond_prism=False):
    """Cards for every (channel, mass, off-axis position) open to the port, in `outdir`, plus `manifest.json`
    (one entry per card; skipped points with the reason). Seeds are seed0, seed0 + 1, ... in manifest order.
    Returns the manifest."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cards, skipped, seed = [], [], seed0
    for ch in channels:
        for m in masses_dm:
            mv = mass_v if mass_v is not None else round(cfg.value("generator.mv_over_mchi") * m, 12)
            if not kinematically_open(ch, m, mv):
                skipped.append({"channel": ch, "mass_dm": m, "mass_v": mv, "reason": "kinematically closed"})
                continue
            for x in off_axis:
                text, run, mv = card_text(cfg, ch, m, mv, x, seed, allow_beyond_prism=allow_beyond_prism)
                path = outdir / f"{run}.dat"
                path.write_text(text)
                cards.append(asdict(CardInfo(str(path), ch, m, mv, x, seed, run)))
                seed += 1
    manifest = {"generator": f"bdnmc-dune {__version__}", "cards": cards, "skipped": skipped}
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest
