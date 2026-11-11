"""Check an edep-sim output against the HEPEVT file it was run on (run in the image, where PyROOT and edep-sim's
I/O library are): each event's primary vertex, time and electron momentum against the HEPEVT event of the same
index, and the energy the electron deposits in the segment detectors against its kinetic energy, in total and per
detector. With --active VOLUME it also finds the geometry volume of every primary vertex in the geometry edep-sim
stores in its output (EDepSimGeometry, ROOT's import of the GDML it ran), and with --frame REPORT.json (from
`python -m bdnmc_dune gdml-frame`) it checks that the world-frame centre ROOT gives each placement of VOLUME that a
vertex lands in is one of the report's (bdnmc_dune.gdml's reading of the same GDML).

Usage: python3 check_edepsim.py <events.edepsim.root> <events.hepevt> [--events N] [--active volTPCActive]
                                [--frame ndlar_frame.json]
"""
import argparse
from collections import Counter
import json

import numpy as np


def read_hepevt(path, n):
    """The first n events: (vertex cm, time ns, electron p GeV, electron E GeV) of each event's status-1 line."""
    out = []
    with open(path) as f:
        while len(out) < n:
            head = f.readline().split()
            if not head:
                break
            lines = [f.readline().split() for _ in range(int(head[1]))]
            e = [ln for ln in lines if ln[0] == "1" and ln[1] == "11"]
            if len(e) != 1:
                raise ValueError(f"event {head[0]}: {len(e)} tracked electrons")
            v = [float(x) for x in e[0]]
            out.append((v[11:14], v[14], v[6:9], v[9]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root")
    ap.add_argument("hepevt")
    ap.add_argument("--events", type=int, default=0)
    ap.add_argument("--active", help="geometry volume the vertices should be in (e.g. volTPCActive)")
    ap.add_argument("--frame", help="gdml-frame report (JSON) to compare the placements' centres with")
    a = ap.parse_args()

    import ROOT
    ROOT.gROOT.SetBatch(True)
    ROOT.gSystem.Load("libedepsim_io")
    f = ROOT.TFile.Open(a.root)
    tree = f.Get("EDepSimEvents")
    n = tree.GetEntries() if a.events <= 0 else min(a.events, tree.GetEntries())
    ref = read_hepevt(a.hepevt, n)
    if len(ref) < n:
        raise SystemExit(f"{a.hepevt}: only {len(ref)} events for {n} edep-sim events")

    geo = f.Get("EDepSimGeometry") if a.active else None
    if a.active and not geo:
        raise SystemExit(f"{a.root}: no EDepSimGeometry")
    if geo:
        # ROOT's length unit in this geometry: the world box's half-length against the GDML's (in the report), or cm
        top = geo.GetTopVolume().GetShape()
        per_cm = 1.0
        if a.frame:
            rep = json.load(open(a.frame))
            per_cm = top.GetDX() / (rep["world_box_size_m"][0] * 50.0)
            centres = np.asarray(rep["active_centres_m"])
        print(f"geometry: top volume {geo.GetTopVolume().GetName()}, half x {top.GetDX():g} "
              f"({per_cm:g} geometry units per cm)")
    vol_of, centre_dev, hit_tpcs, in_active = Counter(), [], set(), []

    dpos, dt, dp, frac, ke, n_sd = [], [], [], [], [], set()
    dep_sd = Counter()
    for i in range(n):
        tree.GetEntry(i)
        ev = tree.Event
        if ev.Primaries.size() != 1 or ev.Primaries[0].Particles.size() != 1:
            raise SystemExit(f"event {i}: {ev.Primaries.size()} vertices, expected 1 with 1 particle")
        vtx = ev.Primaries[0]
        part = vtx.Particles[0]
        x_cm, t_ns, p_gev, e_gev = ref[i]
        pos = vtx.GetPosition()  # mm, ns
        dpos.append(np.abs(np.array([pos.X(), pos.Y(), pos.Z()]) / 10.0 - x_cm).max())
        dt.append(abs(pos.T() - t_ns))
        mom = part.GetMomentum()  # MeV
        if part.GetPDGCode() != 11:
            raise SystemExit(f"event {i}: primary PDG {part.GetPDGCode()}")
        dp.append(np.abs(np.array([mom.X(), mom.Y(), mom.Z()]) / 1e3 - p_gev).max() / np.linalg.norm(p_gev))
        k = (e_gev - 0.51099895e-3) * 1e3
        dep = 0.0
        for sd in ev.SegmentDetectors:
            n_sd.add(str(sd.first))
            d = sum(s.GetEnergyDeposit() for s in sd.second)
            dep_sd[str(sd.first)] += d
            dep += d
        if geo:
            node = geo.FindNode(*(c * per_cm for c in x_cm))
            # edep-sim's ROOT geometry names volumes after Geant4's physical volumes: <GDML volume>_PV
            name = node.GetVolume().GetName() if node else "(outside)"
            vol_of[name] += 1
            in_active.append(name in (a.active, a.active + "_PV"))
            if a.frame and in_active[-1]:
                m = geo.GetCurrentMatrix().GetTranslation()
                c = np.array([m[0], m[1], m[2]]) / per_cm / 100.0  # m
                centre_dev.append(np.abs(centres - c).max(axis=1).min())
                hit_tpcs.add(tuple(np.round(c, 4)))
        ke.append(k)
        frac.append(dep / k)

    dpos, dt, dp, frac, ke = map(np.asarray, (dpos, dt, dp, frac, ke))
    print(f"{n} events; segment detectors {sorted(n_sd)}")
    print(f"vertex   max |edep-sim - HEPEVT| = {dpos.max():.3g} cm; time {dt.max():.3g} ns")
    print(f"electron max relative momentum difference = {dp.max():.3g}")
    print(f"electron kinetic energy: median {np.median(ke):.1f} MeV, range {ke.min():.1f}-{ke.max():.1f} MeV")
    q = np.quantile(frac, [0.1, 0.5, 0.9])
    print(f"deposited / kinetic energy: 10/50/90 % = {q[0]:.3f} / {q[1]:.3f} / {q[2]:.3f}; "
          f"above 1.001: {(frac > 1.001).sum()}")
    tot = ke.sum()
    for name, d in sorted(dep_sd.items(), key=lambda kv: -kv[1]):
        print(f"  {name}: {d / tot:.4f} of the summed kinetic energy ({d / 1e3:.3g} GeV)")
    if geo:
        m = np.asarray(in_active)
        for label, sel in ((f"in {a.active}", m), (f"not in {a.active}", ~m)):
            if sel.any():
                q = np.quantile(frac[sel], [0.1, 0.5, 0.9])
                print(f"  vertex {label} ({sel.sum()}): deposited / kinetic 10/50/90 % = "
                      f"{q[0]:.3f} / {q[1]:.3f} / {q[2]:.3f}")
        print(f"primary vertices by geometry volume ({n} events):")
        for name, k in vol_of.most_common():
            print(f"  {name}: {k} ({k / n:.4f})")
        if a.frame:
            dev = np.asarray(centre_dev)
            print(f"{a.active} placements with a vertex: {len(hit_tpcs)} of {len(centres)}; ROOT's centre vs the "
                  f"gdml-frame report: max {dev.max() if dev.size else float('nan'):.3g} m")


if __name__ == "__main__":
    main()
