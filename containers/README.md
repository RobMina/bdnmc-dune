# Containers

One AlmaLinux 9 image holds both stages of a job: BdNMC's JAX port (with `bdnmc_dune`) and the ND simulation layer
(ROOT, Geant4 with GDML, edep-sim). The user builds every image.

| Recipe | Contents | Status |
|---|---|---|
| `bdnmc-dune.def` | AL9 (glibc 2.34); Python 3.12 venv with `jax[cuda12]==0.12.0.1`, numpy, h5py, `bdnmc_jax` (fork `RobMina/BdNMC`, branch `gpu-port`) and `bdnmc_dune` (this checkout); ROOT 6.36.04 (no OpenGL), Geant4 11.4.0 (GDML, data sets), edep-sim 4.4.0 (no event display), built from source | built (2026-11-01) and run end to end on a V100 (top-level README, Status) |

**Environment.** Apptainer applies `%environment` with its own shell interpreter, which cannot parse ROOT's
`thisroot.sh` (it prints `thisroot.sh:214:13: parameter expansion requires a literal` and stops there). Images
built before this was fixed source `thisroot.sh` and `geant4.sh` there and so lack the ROOT and Geant4 library
paths: `edep-sim` fails with `libG4Tree.so: cannot open shared object file`. The recipe now writes plain exports to
`/opt/environment.sh`, and `%test` checks `edep-sim`'s libraries and PyROOT in that environment. The job script
sources `/opt/environment.sh` in bash inside the image, so it also works with the older images.

## Build

From the repository root (the recipe copies this checkout into the image):

```bash
apptainer build bdnmc-dune.sif containers/bdnmc-dune.def
```

Build arguments (defaults in the recipe's `%arguments`), for example:

```bash
apptainer build --build-arg BDNMC_REF=<commit> --build-arg JOBS=32 bdnmc-dune.sif containers/bdnmc-dune.def
```

- `BDNMC_REF`: fork commit or branch (needs `output.write_hepevt` with `per_interaction`, commit `89ccf7a` or later); recorded in
  `/opt/BdNMC/COMMIT`.
- `EDEPSIM_REF`: edep-sim tag or commit; recorded in `/opt/EDEPSIM_COMMIT`.
- `HEPEP_REF`: the hep-endpoints commit (submission; standard-library only), the one `pyproject.toml` pins.
- `ROOT_VERSION`, `GEANT4_VERSION`: edep-sim's README says it was tested with Geant4 11.4 and ROOT 6.36.
- `PYTHON`: the AL9 Python for the venv (`python3.12`, from AppStream). If pip finds no `jax==0.12.0.1` wheel for it,
  use a newer Python (the fork's tests ran on 3.14).
- `JOBS`: parallel compile jobs. ROOT and Geant4 take most of the build time (of order an hour or more on 8 cores),
  and the Geant4 data sets add several GB.

`%test` (run by `apptainer build`, or `apptainer test bdnmc-dune.sif`) imports `bdnmc_jax` and `bdnmc_dune`, finds
`edep-sim` and checks that its libraries resolve, imports PyROOT, and runs this repository's tests.

## Run

```bash
apptainer exec --nv bdnmc-dune.sif python3 -m bdnmc_dune run card.dat --trials 16777216 --batch 128 \
    --hepevt out/events.hepevt --macro out/generator.mac --summary out/summary.json
apptainer exec bdnmc-dune.sif edep-sim -C -u -g nd.gdml -o out/events.edepsim.root -e <events> out/generator.mac
```

or both stages with `jobs/run_bdnmc_edepsim.sh` (`IMAGE=bdnmc-dune.sif`). `--nv` only for a GPU; without it JAX
runs on the CPU. `apptainer run bdnmc-dune.sif <args>` is `python3 -m bdnmc_dune <args>`.

## What was checked, and what was not

- **Checked (2026-10-31):** edep-sim's build requirements (GDML in Geant4, C++ standard matching ROOT) from its
  README and its example external-kinematics CMakeLists; the HEPEVT reader's input format and units from its source
  (`bdnmc_dune/hepevt.py`).
- **Checked:** DUNE's ND production (`github.com/DUNE/ND_Production`, Apache-2.0) runs edep-sim in Docker Hub images
  `mjkramer/sim2x2` (`run-edep-sim/run_edep_sim.sh` defaults to `sim2x2:ndlar011`; the newest tag is `ndlar012`).
  Their image metadata say **Fedora 37**, built in layers `build_base.sh`, `build_root.sh`, `build_genie.sh`,
  `build_geant4.sh`, `build_edepsim.sh`, ...; the build scripts themselves were not found in a public repository.
- **Checked (2026-11-01):** the recipe builds (BdNMC `28c56f3`); its `%test` passes on a V100 node; both job
  stages run in it (top-level README, Status).
- **Not checked:** the exact ROOT and Geant4 configure options DUNE uses; whether the
  `mjkramer/sim2x2` images would run the JAX port (Fedora 37 ships Python 3.11).

## Alternative: start from DUNE's ND production image

To get exactly the edep-sim build DUNE's ND production uses, the JAX layer can go on top of that image instead
(Fedora 37, glibc 2.36). Not built or tested:

```
Bootstrap: docker
From: mjkramer/sim2x2:ndlar012

%files
    . /opt/bdnmc-dune

%post
    dnf -y install git python3 python3-pip && dnf clean all
    python3 -m venv /opt/venv
    /opt/venv/bin/pip install --no-cache-dir "jax[cuda12]==0.12.0.1" numpy h5py pytest
    git clone https://github.com/RobMina/BdNMC.git /opt/BdNMC && git -C /opt/BdNMC checkout gpu-port
    /opt/venv/bin/pip install --no-cache-dir --no-deps /opt/BdNMC/gpu /opt/bdnmc-dune
```

The image's own environment (`/opt/environment`, loaded through `BASH_ENV`) sets up ROOT, Geant4 and edep-sim; put
`/opt/venv/bin` first on `PATH` in `%environment`.
