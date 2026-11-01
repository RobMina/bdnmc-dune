"""The edep-sim macro that reads a bdnmc-dune HEPEVT file.

Generator commands as in edep-sim's own HEPEVT tests (validate/fast-tests/110TestHEPEvtFlavors.sh) and DUNE's
ND_Production edep-sim macros (run-edep-sim/macros/dune-nd.mac), with:
- `flavor marley`: the 15-column layout write_hepevt writes (the default flavour, pythia, has 8);
- `position free` and `time free`: keep the vertex and time from the file. edep-sim's defaults are position `free`
  and time `fixed`, and the fixed time generator forces the vertex time (EDepSimFixedTimeGenerator::ForceTime
  returns true), so `time free` is required to keep BdNMC's times;
- one vertex per event (`count fixed 1`): `hepevt.split_vertices` gives one interaction point per event.

Detector and physics settings (`/edep/...`, e.g. ND_Production's `/edep/hitSeparation volTPCActive -1 mm` and
thresholds) depend on the geometry and go in a separate macro run before this one; that macro ends with
`/edep/update`. Without one, run edep-sim with `-u` (it then applies `/edep/update` before the first macro).
"""


def generator_macro(hepevt_path, verbose=0):
    return "\n".join([
        "# bdnmc-dune: BdNMC recoil electrons from a HEPEVT file (marley flavour: GeV, cm, ns).",
        f"/generator/kinematics/hepevt/input {hepevt_path}",
        "/generator/kinematics/hepevt/flavor marley",
        f"/generator/kinematics/hepevt/verbose {int(verbose)}",
        "/generator/kinematics/set hepevt",
        "/generator/position/set free",
        "/generator/time/set free",
        "/generator/count/fixed/number 1",
        "/generator/count/set fixed",
        "/generator/add",
    ]) + "\n"
