"""Check an edep-sim output against the HEPEVT file it was run on (run in the image, where PyROOT and edep-sim's
I/O library are): each event's primary vertex, time and electron momentum against the HEPEVT event of the same
index, and the energy the electron deposits in the segment detectors against its kinetic energy.

Usage: python3 check_edepsim.py <events.edepsim.root> <events.hepevt> [--events N]
"""
import argparse

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

    dpos, dt, dp, frac, ke, n_sd = [], [], [], [], [], set()
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
            dep += sum(s.GetEnergyDeposit() for s in sd.second)
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


if __name__ == "__main__":
    main()
