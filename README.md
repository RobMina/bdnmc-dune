# bdnmc-dune

Light-dark-matter event generation with [BdNMC](https://github.com/RobMina/BdNMC) for the DUNE Near Detector,
feeding the ND simulation (edep-sim). BdNMC itself (the stock C++ generator and its validated JAX GPU port) stays
experiment-independent in its own repository; this package adds what DUNE needs around it: the detector's
placement relative to the LBNF target, parameter cards, the detector frame for the events, the edep-sim input, and
container and job recipes.

**Public material only.** Every number is from a published DUNE or LBNF document or the PDG, with its source next
to it in [`bdnmc_dune/data/nd_lar.toml`](bdnmc_dune/data/nd_lar.toml). No collaboration-internal geometry, fluxes or
code. Inputs that are not public are configuration fields with neutral defaults (below).

## Two ways to run

DUNE's software runs on current Linux, so one AlmaLinux 9 image can hold both BdNMC and the ND simulation (the JAX
port needs glibc >= 2.27; AL9 has 2.34).

| Mode | Generation | Simulation | Status |
|---|---|---|---|
| **JAX port, same image** (`jobs/run_bdnmc_edepsim.sh`, `containers/bdnmc-dune.def`) | `bdnmc_jax.pipeline.events` on a GPU (`--nv`) or CPU; `bdnmc_jax.output.write_hepevt` in ND-LAr's frame | edep-sim's HEPEVT kinematics generator in the same image, same job | implemented and tested up to the HEPEVT file; image and edep-sim stage written, not run |
| **stock libBdNMC in process** | stock BdNMC from the fork's `shared_library` branch (`libBdNMC`, C API `src/BdNMC_capi.h`) | an edep-sim external kinematics generator (`/generator/kinematics/external/loadGenerator <lib> CreateKinematicsGenerator`, a subclass of `EDepSim::VKinematicsGenerator`, as in edep-sim's `examples/ExternalKinematics`) that calls `bdnmc_configure(card)` / `bdnmc_run` and turns each event's `electron` into a `G4PrimaryVertex` in the same frame (`bdnmc_dune.geometry`) | design only: not written |

The JAX mode gives the GPU port's speed (and its mass scans without recompilation); the in-process mode needs no
intermediate file. Both read the same cards.

## Decisions still to take

1. **Which detector first.** ND-LAr is implemented (it is the dark-matter target in DUNE's own study, TDR Vol. II
   Sec. 8.8.2). ND-GAr, TMS (the Phase I muon spectrometer, which post-dates the ND CDR: no CDR numbers) and SAND
   are not; for ND-GAr the ND CDR gives only outside dimensions (12.8 m x 8.5 m x 10 m, p. 8-253).
2. **LBNF meson production.** No public LBNF meson lists are known. The default is BdNMC's own parametrisations at
   120 GeV on carbon (graphite target): BMPT for pi0, BMPT's K0 shape for eta, and proton bremsstrahlung, normalised
   to DUNE's Pythia multiplicities (4.5 pi0 and 0.5 eta per POT, TDR Vol. II p. 8-271). This is a stand-in, to be
   replaced by g4lbnf meson lists (`production_distribution particle_list`, which the port supports) once there
   are lists that can be used.
3. **DUNE-PRISM.** Off-axis position is a scan axis of the card generator (`--off-axis`, or `prism` for the
   positions 0, 6, 12, 18, 24 m of DUNE's LDM study); the range is limited to the 30.5 m of the ND CDR. Which
   positions to produce, and with what exposure each, is open.
4. **Numbers to settle** (all configuration fields):
   - the beam slope at the ND: 5.3 deg (ND CDR p. 8-250, used) against the 5.8 deg primary-beam bend of the LBNF
     CDR (Vol. 3 p. 6-99);
   - the 574 m: measured to which point of ND-LAr (front face or centre; the centre is used) and from where (the
     target, or the start of horn 1 as in TDR Vol. II p. 8-244), worth +-2.5 m;
   - the generator-level cuts (`min_scatter_energy` 35 MeV etc., the fork's generic values, not DUNE thresholds),
     `alpha_D` = 0.5 (DUNE's study), `epsilon` and `m_V/m_chi` = 3 (benchmarks);
   - `frame.gdml_offset`: where ND-LAr's active-volume centre sits in the GDML the simulation uses (not a published
     number; read it from the geometry).

## Status

First version (2026-10-31).
- **Real and tested on a laptop CPU:** the configuration with sources; ND-LAr's placement and the card generator
  (cards parsed by the port's `parse_card` and built by `pipeline.setup` for pi0, eta and brem, on and off axis);
  the detector frame; the HEPEVT writer through `write_hepevt` with edep-sim's units; a few 2^16-trial blocks end
  to end, every vertex inside ND-LAr's active volume in the detector frame; the job script's control flow; the
  job spec.
- **Written, not run:** the AL9 image (`containers/bdnmc-dune.def`), the edep-sim stage of the job, the edep-sim
  macro (checked against edep-sim's source, not run in edep-sim).
- **Placeholders:** the hep-endpoints submission (`bdnmc_dune.jobspec.submit`, marked integration point; raises);
  the in-process mode (design above); detectors other than ND-LAr; real LBNF meson lists.

## The edep-sim input

From edep-sim's source (`src/kinem/EDepSimHEPEVTKinematicsGenerator.cc`, edep-sim 4.4.0 = master `a3abbcf`, read on
2026-10-31):
- **verified from the code:** the `marley` flavour (`/generator/kinematics/hepevt/flavor marley`) reads a 2-token
  header `<event id> <number of particles>` and 15-token particle lines
  `status pdg mother1 mother2 daughter1 daughter2 px py pz E m x y z t`, which is `write_hepevt`'s layout; momenta
  and masses in GeV, the vertex in cm and ns, so `write_hepevt` is called with `length_unit="cm"`,
  `time_unit="ns"`; only status-1 lines are tracked (the recoil DM is written as status 2, PDG 18, and skipped);
  all particles of one header share one vertex, placed at the last tracked line's position;
- **verified from the code:** edep-sim's default time generator (`fixed`) forces the vertex time, so the macro sets
  `/generator/time/set free` (and `position free`) to keep BdNMC's vertex and time;
- **found while checking:** `write_hepevt` puts the DM of one BdNMC trial in one event, and a trial often has two
  accepted DM (about 30 % of the events in a test run), each with its own interaction point. edep-sim would put
  both electrons at the second one, so `bdnmc_dune.hepevt.split_vertices` writes one event per interaction point,
  with an index file back to BdNMC's events;
- **not verified:** an actual edep-sim run on these files; the detector and physics macro settings for the DUNE ND
  geometry (DUNE's ND_Production uses e.g. `/edep/hitSeparation volTPCActive -1 mm` in
  `run-edep-sim/macros/dune-nd.mac`; pass such settings as `EDEP_PRE_MAC`).

Vertex times are BdNMC's: seconds since production at the target, written in ns (about 1.9 us at 574 m).

## Use

```bash
pip install -e '.[test]'        # pulls bdnmc-jax from RobMina/BdNMC (branch gpu-port)
```

```bash
python -m bdnmc_dune cards --channels pi0,eta,brem --mchi 0.005,0.01,0.05 --off-axis prism --out cards/
python -m bdnmc_dune run cards/ndlar_pi0_mchi0.005_mv0.015_x0.dat --trials 16777216 --batch 128 \
    --hepevt out/events.hepevt --macro out/generator.mac --summary out/summary.json [--hdf5 out/events.h5]
python -m bdnmc_dune frame --off-axis 12             # origin and rotation for write_hepevt
```

One whole job (generation and edep-sim), in the image:

```bash
IMAGE=bdnmc-dune.sif GPU=1 CARD=cards/<card>.dat GEOMETRY=<nd.gdml> OUTDIR=out bash jobs/run_bdnmc_edepsim.sh
```

For hep-endpoints, `python -m bdnmc_dune jobspec <card> --handle <name> --geometry <gdml at the site> --image <sif
at the site> --gpus 1 --script job.sh` prints the job (resources and settings) and writes a self-contained
`job.sh` with the card embedded; `bdnmc_dune.jobspec.submit` is where hependpoints will submit it.

Overrides: `--config my.toml` with, for example, `[frame] gdml_offset = [0.0, -2.68, 13.0]` or
`[beam] slope_at_nd = 5.8`. Precision: float32 by default (the port's default); `BDNMC_JAX_PRECISION=float64` (or
`PRECISION=float64` for the job script) for float64.

## Published numbers used

All in `bdnmc_dune/data/nd_lar.toml` with page references: ND CDR = arXiv:2103.13910; TDR Vol. II =
arXiv:2002.03005; LBNF CDR Vol. 3 = arXiv:1601.05823; PDG liquid argon.

| Quantity | Value | Source |
|---|---|---|
| ND-LAr active volume | 7 m (x) x 3 m (y) x 5 m (z, nearly along the beam); 7 x 5 modules of 1 x 1 x 3 m | ND CDR pp. 2-37, 2-69, 2-75; TDR Vol. II Table 8.2 |
| mass: total, fiducial | 147 t; 67.2 t (6 x 2 x 4 m) | TDR Vol. II Table 8.2 |
| distance from the target | 574 m | TDR Vol. II Table 8.2; ND CDR p. 1-3 |
| beam through the active-volume centre (on axis) | vertical offset 0 | ND CDR p. 8-250, Fig. 8.5 |
| beam slope at the ND | 5.3 deg below the cavern horizontal | ND CDR p. 8-250 |
| DUNE-PRISM range; LDM-study positions | 30.5 m; 0-24 m in 6 m steps | ND CDR p. 2-39; TDR Vol. II p. 8-272 |
| beam | 120 GeV, 1.2 MW, 1.1e21 POT/yr | TDR Vol. II Table 8.1; ND CDR p. 1-4 |
| target | graphite (C: Z = 6, N = 6), 2.2 m | TDR Vol. II p. 4-60 |
| pi0, eta per POT | 4.5, 0.5 (Pythia) | TDR Vol. II p. 8-271 |
| alpha_D | 0.5 | TDR Vol. II p. 8-271 |
| liquid argon | 1.396 g/cm^3, Z = 18, A = 39.948 (N = 22: argon-40) | PDG |

## Layout

| Path | Role |
|---|---|
| `bdnmc_dune/data/nd_lar.toml` | the configuration: numbers, units, sources |
| `bdnmc_dune/config.py` | loads it, merges overrides, derived numbers (LAr density, POT) |
| `bdnmc_dune/geometry.py` | beam and hall frames, ND-LAr's placement and Euler angles, the detector frame from a card |
| `bdnmc_dune/cards.py` | card generator and mass scans (pi0, eta, brem; off-axis positions) |
| `bdnmc_dune/hepevt.py`, `edepsim.py` | edep-sim HEPEVT input via `write_hepevt`, one vertex per event; the edep-sim macro |
| `bdnmc_dune/jobspec.py` | the job for hep-endpoints (script + resources); the integration point |
| `bdnmc_dune/cli.py` | `python -m bdnmc_dune {cards,run,frame,macro,jobspec}` |
| `bdnmc_dune/data/run_bdnmc_edepsim.sh` (linked from `jobs/`) | the two-stage job script |
| `containers/` | the AL9 Apptainer recipe and build instructions |
| `tests/` | pytest, laptop CPU only |

## Tests

```bash
python -m pytest -q
```

They need `bdnmc_jax` (installed, or `BDNMC_JAX_SRC=<BdNMC>/gpu/src`), run on a CPU without DUNE software, and
include a few 2^16-trial blocks of a real ND-LAr card (under a minute on a laptop).
