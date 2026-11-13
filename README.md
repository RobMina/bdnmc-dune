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
| **JAX port, same image** (`jobs/run_bdnmc_edepsim.sh`, `containers/bdnmc-dune.def`) | `bdnmc_jax.pipeline.events` on a GPU (`--nv`) or CPU; `bdnmc_jax.output.write_hepevt` in ND-LAr's frame | edep-sim's HEPEVT kinematics generator in the same image, same job | implemented; run end to end on a GPU node (Status) |
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
     CDR (Vol. 3 p. 6-99). DUNE's ND production uses the second: its GENIE flux configuration
     (`DUNE/ND_Production`, `run-genie/flux/GNuMIFlux.xml`, param set `DUNEND`) turns the beam about x by 0.101 rad
     (5.79 deg) in the ND hall geometry's frame;
   - the beam's height in the hall: the ND CDR puts it through the active volume's centre (used). The same flux
     configuration ties the beam to the point (0, 0.054, 6.66) m of its geometry, "the center of LArActive" in a 2020
     geometry; in the current hall geometry (below) the active centre is at (0, -0.669, 6.658) m, so either that
     configuration is not the one used with it, or DUNE's beam passes 0.72 m above the active centre there;
   - the 574 m: measured to which point of ND-LAr (front face or centre; the centre is used) and from where (the
     target, or the start of horn 1 as in TDR Vol. II p. 8-244), worth +-2.5 m. The same flux configuration puts
     that 2020 active centre 562.12 m along the beam from the origin of the beam simulation's coordinates;
   - the generator-level cuts (`min_scatter_energy` 35 MeV etc., the fork's generic values, not DUNE thresholds),
     `alpha_D` = 0.5 (DUNE's study), `epsilon` and `m_V/m_chi` = 3 (benchmarks);
   - `frame.gdml_offset`: where ND-LAr's active-volume centre sits in the GDML the simulation uses. Not a published
     number: `python -m bdnmc_dune gdml-frame <gdml>` reads it from the geometry. Done for DUNE's current ND hall
     geometry (The DUNE ND geometry, below), packaged as `bdnmc_dune/data/frames/nd_hall_v2026.03.06.toml`; the
     default stays 0 (the detector-centred frame), and any other GDML needs its own reading.

## Status

First version (2026-10-31).
- **Real and tested on a laptop CPU:** the configuration with sources; ND-LAr's placement and the card generator
  (cards parsed by the port's `parse_card` and built by `pipeline.setup` for pi0, eta and brem, on and off axis);
  the detector frame; the HEPEVT writer through `write_hepevt` with edep-sim's units; a few 2^16-trial blocks end
  to end, every vertex inside ND-LAr's active volume in the detector frame; the job script's control flow; the
  job spec.
- **Run end to end (2026-11-01):** one job submitted through hep-endpoints (`slurm_ssh`) to a Slurm cluster, on
  one NVIDIA V100 (32 GB) with 4 CPU cores, in the AL9 image (Geant4 11.4.0, edep-sim `a3abbcf`), with the
  test geometry (`testgeom`: ND-LAr's active volume as one argon box). The pi0 card at m_chi = 10 MeV, on axis:
  2^24 trials in 20.6 s (float32), 602037 edep-sim events; edep-sim on the first 2000 in 191 s. Checked with
  `jobs/check_edepsim.py`: every primary vertex equals its HEPEVT vertex to 3e-14 cm, the times are equal, the
  electron momenta agree to 1e-9 (relative); the median electron deposits 99.1 % of its kinetic energy in the
  box (10 % quantile 47 %: showers leaving the bare box). edep-sim's default output keeps every trajectory,
  about 0.44 MB per event at these energies.
- **Run end to end in DUNE's ND hall geometry (2026-11-11):** the same card, hardware and image, with
  `nd_hall_with_lar_tms_sand_stt1_v2026.03.06.gdml` (ND-LAr, TMS, SAND; The DUNE ND geometry, below), its frame
  override and `data/edep_dune_nd.mac`. Generation: 2^24 trials in 24.7 s, the same 602037 events (same seed). edep-sim: the
  first 2000 in 257 s, after about 40 s reading the geometry (1.6 million nodes); 0.42 MB per event. Checked with
  `jobs/check_edepsim.py --active volTPCActive --frame ...`:
  - vertices equal the HEPEVT vertices to 1e-13 cm, the times are equal, the momenta agree to 1e-9;
  - 89.7 % of the vertices are in ND-LAr's active argon (`volTPCActive`). That is what a uniform spread in BdNMC's
    7 x 3 x 5 m box gives: the 70 active boxes fill 93.9 of its 105 m^3 (0.894). The rest are in the modules'
    other volumes (argon between modules 4.9 %, light detectors 1.8 %, field cages 1.6 %, backplates, pixel
    planes and boards 2 %). None is outside ND-LAr's modules;
  - all 70 active boxes get vertices, and ROOT's centre for each (from the geometry edep-sim stores) equals
    `gdml-frame`'s to 2e-16 m;
  - deposits: the segment detectors hit are `volTPCActive` (73.95 % of the summed electron kinetic energy) and
    `volTMS` (0.83 %), none in SAND. Per electron (deposit / kinetic energy), the 10/50/90 % quantiles are
    0.294 / 0.906 / 0.983 overall, 0.474 / 0.913 / 0.984 for vertices in active argon and 0.075 / 0.430 / 0.962
    for the others. The electrons have a median kinetic energy of 576 MeV, ranging from 34.5 MeV to 22 GeV.
- **Placeholders:** the in-process mode (design above); detectors other than ND-LAr; real LBNF meson lists.

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
  both electrons at the second one, so `bdnmc_dune.hepevt.write` asks the fork's `write_hepevt` for one event
  per interaction point (`per_interaction=True`), with an index file back to BdNMC's events;
- **verified in edep-sim** (Status): vertices, times and momenta arrive unchanged, one primary per event;
- **used in edep-sim** (Status): the detector and physics settings of DUNE's ND production
  (`run-edep-sim/macros/dune-nd.mac` of `DUNE/ND_Production`), as `bdnmc_dune/data/edep_dune_nd.mac`: the
  ionisation model, zero thresholds, and `/edep/hitSeparation volTPCActive -1 mm` (one trajectory per hit segment
  in the active argon). Left out: the generator commands, and `/edep/db/set/requireEventsWithHits`, which would drop
  events without hits and break the one-to-one match with the HEPEVT file. The line `volTPCActive is not found in /`
  that `/edep/hitSeparation` prints is harmless: the command creates the segment detector by name before the GDML
  is read, and the GDML's `SensDet` auxiliary then gets the same one;
- **from the code** (`app/edepSim.cc`): `-C` turns edep-sim's geometry overlap check off (it is on by default); the
  job script passes it, as ND_Production's does.

Vertex times are BdNMC's: seconds since production at the target, written in ns (about 1.9 us at 574 m).

## The DUNE ND geometry

**Source.** DUNE's ND hall geometries are made with `github.com/DUNE/dunendggd` (GeGeDe; its `setup.py` declares
GPLv2, and the repository has no licence file). DUNE's ND production repository, `github.com/DUNE/ND_Production`
(Apache-2.0), distributes the generated GDML files. Nothing from dunendggd is in this repository, and no GDML is
committed. The geometry used is ND_Production's `geometry/nd_hall_with_lar_tms_sand_stt1_v2026.03.06.gdml` (73 MB;
dunendggd tag `v2026.03.06`, commit `9802970`, recorded in the world volume's auxiliaries), fetched at a fixed
commit:

```bash
curl -sSfLO https://raw.githubusercontent.com/DUNE/ND_Production/d91b4c14b771e70ed73f959a156761983ca8e126/geometry/nd_hall_with_lar_tms_sand_stt1_v2026.03.06.gdml
```

```bash
sha256sum nd_hall_with_lar_tms_sand_stt1_v2026.03.06.gdml   # d25940daa282eebbf1ef091b862329094006cab4e01fe0751a92bef9b1dd0a42
```

To make it from dunendggd instead (not tried here): `pip install` GeGeDe and dunendggd into a venv, then `make` the target
`nd_hall_with_lar_tms_sand_stt1.gdml` of its `Makefile`. The other targets (no SAND, PRISM shifts) give other
files, and each needs its own `gdml-frame` reading.

**Frame.** `python -m bdnmc_dune gdml-frame <gdml> --landmark volTMS --landmark volSAND --landmark craneRail1_lv`
(`bdnmc_dune/gdml.py`, standard-library XML, Geant4's transform convention; 12 s on one core) gives:
- ND-LAr's active argon: 70 placements of the box `volTPCActive` (0.468 x 2.996 x 0.957 m; 7 modules across, 2
  drift volumes each, 5 along the beam), the segment detector `volTPCActive`. Their union is 6.942 x 2.996 x 4.957 m
  and is centred at **(0.000, -0.669, 6.658) m** in the world frame, which is `frame.gdml_offset`
  (`bdnmc_dune/data/frames/nd_hall_v2026.03.06.toml`). The boxes add up to 93.9 m^3 (131 t of argon) against the
  105 m^3 (147 t) of the published 7 x 3 x 5 m;
- axes: every volume from the world down to ND-LAr's modules is placed without rotation. The drift volumes are
  either unrotated or turned 180 deg about y, which leaves their union unchanged. +z points downstream: ND-LAr, TMS
  and SAND are at z = 6.66, 14.86 and 23.91 m, in the order the ND CDR gives along the beam. +y points up: the crane
  rails and the elevator block are at y = +5.6 m, against +1.0 m for the hall's centre. So the world frame is the
  hall frame of `geometry.py`, and no rotation is needed. The GDML has no beam in it. ND_Production's flux
  configuration (Decisions, item 4) rotates the beam about x only, so the beam's horizontal projection is the
  world z;
- the other segment detectors: TMS scintillator (`volTMS`), SAND's calorimeter (`EMCalSci`), straws (`Straw`) and
  GRAIN argon (`LArHit`).

The frame of the run in Status is checked a second way: ROOT's import of the same GDML gives the same 70 centres.

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

Through hep-endpoints (RobMina/hep-endpoints; the site settings are its `HEPEP_*` variables, see its README):

```bash
python -m bdnmc_dune jobspec <card> --handle <name> --geometry <gdml at the site> --image <sif at the site> --gpus 1 --submit slurm_ssh
```

`--preamble` adds the site's shell setup before the job (for example `--preamble "module load apptainer"` where
Apptainer is a module); `--edep-events N` simulates only the first N events of the HEPEVT file. Without a DUNE
geometry, `python -m bdnmc_dune testgeom --out box.gdml` writes a smoke-test GDML (ND-LAr's active volume as one
argon box, a segment detector, in the detector-centred frame). After the job, inside the image:

```bash
python3 jobs/check_edepsim.py <out>/events.edepsim.root <out>/events.hepevt
```

With a DUNE ND geometry, give the frame and the detector macro too, and check where the vertices land (The DUNE ND
geometry, above):

```bash
python -m bdnmc_dune jobspec <card> --handle <name> --geometry <site>/nd_hall_with_lar_tms_sand_stt1_v2026.03.06.gdml \
    --config bdnmc_dune/data/frames/nd_hall_v2026.03.06.toml --embed-pre-mac bdnmc_dune/data/edep_dune_nd.mac \
    --image <sif at the site> --gpus 1 --edep-events 2000 --submit slurm_ssh
```

```bash
python3 jobs/check_edepsim.py <out>/events.edepsim.root <out>/events.hepevt --active volTPCActive --frame ndlar_frame.json
```

```bash
python -m bdnmc_dune job status <name> --endpoint slurm_ssh
```

```bash
python -m bdnmc_dune job fetch <name> --endpoint slurm_ssh --dest results
```

The job writes to the endpoint's output directory (`$BDNMC_OUT`), which `fetch` copies to `results/<name>/`;
resubmitting the same job under the same handle returns the same record, and a changed one is refused.
`--script job.sh` (without `--submit`) writes the self-contained job script instead, for a site hep-endpoints does
not cover.

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

## Published studies to compare with

Light-dark-matter studies for the DUNE ND that this package's outputs can be compared with (checked against arXiv
on 2026-11-11). None of them uses BdNMC: they take pi0 and eta from Pythia (4.5 and 0.5 per POT, the same
normalisation as here) with their own acceptance Monte Carlo, and quote no signal counts in the text, so numbers
come from their figures. No comparison has been made yet.

| Paper | Simulation and benchmark | What to compare |
|---|---|---|
| V. De Romeri, K. J. Kelly, P. A. N. Machado, *Hunting On- and Off-Axis for Light Dark Matter with DUNE-PRISM*, arXiv:1903.10505, PRD 100, 095010 (2019). The basis of DUNE's own study | Pythia8 pi0 and eta from the primary interaction only; no kaons, no proton brem. ND at 574 m, 75 t fiducial; off axis 0-36 m; E_e θ_e² < 2 MeV rad². **Scalar** DM at α_D = 0.1 (fermion DM in App. E: M_χ = 20 MeV, M_A' = 3 M_χ) | Fig. 3: signal events per year against off-axis distance at (M_A', M_DM) = (90, 30) MeV, α_D ε⁴ = 1e-15, one year per position (the `--off-axis prism` cards, rescaled in ε⁴ α_D). Fig. 4/9: normalised electron-energy spectra. Fig. 8: E_e θ_e² |
| DUNE Collaboration, *Prospects for Beyond the Standard Model Physics Searches at DUNE*, arXiv:2008.12769, EPJC 81 (2021) 322, Sec. 8.1 (also TDR Vol. II Sec. 8.8.2) | pi0 and eta (Pythia), proton brem stated to be subdominant; α_D = 0.5, M_V = 3 M_χ; 3.5 yr ν + 3.5 yr ν̄; DUNE-PRISM 0-24 m in 6 m steps; backgrounds ν_μ-e and ν_e CCQE (10 % normalisation, 1 % per bin) | Fig. 19, the 90 % CL sensitivity in Y against mass, on axis and with DUNE-PRISM: needs a background model, so it checks a full sensitivity chain rather than the generator |
| M. Breitbach, L. Buonocore, C. Frugiuele, J. Kopp, L. Mittnacht, *Searching for Physics Beyond the Standard Model in an Off-Axis DUNE Near Detector*, arXiv:2102.03383, JHEP 01 (2022) 048 | independent DUNE-PRISM study of DM scattering and heavy-neutral-lepton decays | finds that for DM coupling to leptons on-axis data give marginally stronger limits than off-axis: a cross-check of the off-axis trend |
| G. Krnjaic, N. Toro et al., *Snowmass Whitepaper: Dark Matter Production at Intensity-Frontier Experiments*, arXiv:2207.00597 | review | context for Y-against-mass plots and thermal targets |
| P. deNiverville, C.-Y. Chen, M. Pospelov, A. Ritz, arXiv:1609.01770, PRD 95, 035006 (2017) | the BdNMC paper | the generator itself |

Differences to expect against 1903.10505, besides the scalar/fermion model:
- **Parent angular distributions:** BdNMC's (BMPT for pi0, BMPT's K0 shape for eta) differ from Pythia's, at the same normalisation.
- **Proton brem:** it is left out there, but is a channel here.
- **Detector:** its 75 t fiducial volume and 3 x 4 x 5 m size, against the TDR's 67.2 t (6 x 2 x 4 m) used here.

## Layout

| Path | Role |
|---|---|
| `bdnmc_dune/data/nd_lar.toml` | the configuration: numbers, units, sources |
| `bdnmc_dune/config.py` | loads it, merges overrides, derived numbers (LAr density, POT) |
| `bdnmc_dune/geometry.py` | beam and hall frames, ND-LAr's placement and Euler angles, the detector frame from a card |
| `bdnmc_dune/cards.py` | card generator and mass scans (pi0, eta, brem; off-axis positions) |
| `bdnmc_dune/hepevt.py`, `edepsim.py` | edep-sim HEPEVT input via `write_hepevt`, one vertex per event; the edep-sim macro |
| `bdnmc_dune/jobspec.py` | the job for hep-endpoints (script + resources), its endpoint and submission |
| `bdnmc_dune/cli.py` | `python -m bdnmc_dune {cards,run,frame,macro,jobspec,job,testgeom,gdml-frame}` |
| `bdnmc_dune/testgeom.py` | the smoke-test GDML (one argon box) |
| `bdnmc_dune/gdml.py` | placements in a GDML (Geant4's transform convention): `frame.gdml_offset` and the axes check (`gdml-frame`) |
| `bdnmc_dune/data/edep_dune_nd.mac` | edep-sim detector and physics settings for DUNE's ND hall geometries (from ND_Production, Apache-2.0) |
| `bdnmc_dune/data/frames/` | `frame.gdml_offset` overrides for named DUNE geometries, with how each was read |
| `bdnmc_dune/data/run_bdnmc_edepsim.sh` (linked from `jobs/`) | the two-stage job script |
| `jobs/check_edepsim.py` | edep-sim output against its HEPEVT input (vertices, times, momenta, deposits per detector; the volume of each vertex) |
| `containers/` | the AL9 Apptainer recipe and build instructions |
| `tests/` | pytest, laptop CPU only |

## Tests

```bash
python -m pytest -q
```

They need `bdnmc_jax` (installed, or `BDNMC_JAX_SRC=<BdNMC>/gpu/src`), run on a CPU without DUNE software, and
include a few 2^16-trial blocks of a real ND-LAr card (under a minute on a laptop).

CI (`.github/workflows/ci.yml`) runs them on every push and pull request and weekly, against the head of the
port's `gpu-port` branch, and checks that the job script and the container recipe's `%post` parse.
