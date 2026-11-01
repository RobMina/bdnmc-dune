"""BdNMC events as edep-sim HEPEVT input, written by the fork's `bdnmc_jax.output.write_hepevt`.

edep-sim's HEPEVT reader (src/kinem/EDepSimHEPEVTKinematicsGenerator.cc, edep-sim 4.4.0 = master a3abbcf, read on
2026-10-31) with `/generator/kinematics/hepevt/flavor marley`:
- header line with 2 tokens: `<event id> <number of particles>` (the id is read, not used);
- particle lines with exactly 15 tokens: `status pdg mother1 mother2 daughter1 daughter2 px py pz E m x y z t`;
- momenta and masses in GeV, vertex x y z in cm, t in ns (the reader multiplies by Geant4's GeV, cm, ns);
- only status 1 is tracked; other lines are skipped before their vertex is read;
- one G4PrimaryVertex per header, placed at the last status-1 line's x y z t.

`write_hepevt` writes exactly this layout, so the units are set to match (`length_unit="cm"`, `time_unit="ns"`),
with the recoil electron as status 1 (tracked) and the recoil DM as status 2, PDG 18 (skipped). A BdNMC trial can
have two accepted DM, each with its own interaction point, and edep-sim would put both electrons at the second one,
so `write` asks write_hepevt for one event per interaction (`per_interaction=True`) and keeps the BdNMC event number
in an index file. `split_vertices` does the same for a HEPEVT file written with events grouped.
"""
from pathlib import Path

import numpy as np

from . import geometry

# What edep-sim's marley-flavour reader expects (verified from its source, see the module docstring).
EDEPSIM_FLAVOR = "marley"
EDEPSIM_UNITS = {"momentum": "GeV", "energy": "GeV", "mass": "GeV", "length": "cm", "time": "ns"}
WRITE_HEPEVT_UNITS = {"length_unit": "cm", "time_unit": "ns"}  # the same, in write_hepevt's arguments
ELECTRON_STATUS, DM_STATUS, DM_PDG = 1, 2, 18


def read(path):
    """The events of a HEPEVT file the way edep-sim's marley reader walks it: a list of (header tokens, list of
    particle token lists). Raises on what edep-sim would reject (header of another length, lines that are not 15
    tokens, a truncated event)."""
    lines = [l.split("#", 1)[0].split() for l in Path(path).read_text().splitlines()]
    lines = [t for t in lines if t]
    events, i = [], 0
    while i < len(lines):
        head = lines[i]
        if len(head) != 2:
            raise ValueError(f"header {head}: the marley flavour expects '<event id> <number of particles>'")
        n = int(head[1])
        parts = lines[i + 1:i + 1 + n]
        if len(parts) != n:
            raise ValueError(f"event {head[0]} truncated")
        bad = [p for p in parts if len(p) != 15]
        if bad:
            raise ValueError(f"event {head[0]}: particle line with {len(bad[0])} tokens, the marley flavour needs 15")
        events.append((head, parts))
        i += 1 + n
    return events


def split_vertices(src, dst, index_path=None):
    """Rewrite `src` with one event per interaction point (x y z t of the particle lines) into `dst`, numbered from
    0. `index_path` (default dst + '.index.csv') maps each output event to the input event index and the vertex's
    order in it. Returns (number of input events, number of output events)."""
    dst = Path(dst)
    index_path = Path(index_path) if index_path else dst.with_name(dst.name + ".index.csv")
    events = read(src)
    out, index = [], ["edepsim_event,hepevt_event,vertex_in_event"]
    for head, parts in events:
        groups = {}
        for p in parts:
            groups.setdefault(tuple(p[11:15]), []).append(p)
        for k, plist in enumerate(groups.values()):
            out.append(f"{len(index) - 1} {len(plist)}\n" + "".join(" ".join(p) + "\n" for p in plist))
            index.append(f"{len(index) - 1},{head[0]},{k}")
    tmp = dst.with_name(dst.name + ".tmp")
    tmp.write_text("".join(out))
    tmp.replace(dst)
    index_path.write_text("\n".join(index) + "\n")
    return len(events), len(index) - 1


def write(res, path, gdml_offset=(0.0, 0.0, 0.0), split=True):
    """Write a `pipeline.events` (or `pipeline.finish`) result as edep-sim HEPEVT input in ND-LAr's detector frame
    (geometry.frame_from_card on the result's card, plus the simulation geometry's offset, m). Returns a dict with
    the frame and the event counts.
    With `split` (the default) each interaction point is its own event and `<path>.index.csv` maps edep-sim events to
    BdNMC events; without it the DM of one BdNMC event share an event."""
    from bdnmc_jax import output
    res = output.collect(res)
    origin, rotation = geometry.frame_from_card(res["card"], gdml_offset)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_out = output.write_hepevt(res, path, origin=origin, rotation=rotation, dm_pdg=DM_PDG, dm_status=DM_STATUS,
                                per_interaction=split, **WRITE_HEPEVT_UNITS)
    _, event = output.accepted_events(res)
    n_bdnmc = len(np.unique(event))
    if split:
        index = path.with_name(path.name + ".index.csv")
        index.write_text("edepsim_event,bdnmc_event\n" + "".join(f"{i},{e}\n" for i, e in enumerate(event)))
    return {"path": str(path), "bdnmc_events": int(n_bdnmc), "edepsim_events": int(n_out),
            "origin_m": np.asarray(origin).tolist(), "rotation": np.asarray(rotation).tolist(),
            "gdml_offset_m": list(map(float, gdml_offset)), "units": EDEPSIM_UNITS, "flavor": EDEPSIM_FLAVOR}
