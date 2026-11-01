#!/usr/bin/env bash
# One bdnmc-dune job: BdNMC events from one card with the JAX port, as edep-sim HEPEVT input in ND-LAr's frame,
# then edep-sim on the user's ND geometry. Both stages run in the single AL9 image (containers/bdnmc-dune.def), or
# on the host when the job already runs inside it.
#
# Environment (inputs):
#   CARD          BdNMC card (python -m bdnmc_dune cards ...)                                   required
#   GEOMETRY      GDML of the ND simulation geometry, as seen by the job                       required unless SKIP_EDEPSIM=1
#   OUTDIR        output directory                                                             default: out
#   TRIALS        BdNMC trials (blocks of 65536)                                               default: 16777216
#   BATCH         2^16-trial blocks per device call (64-256 on an A100)                        default: 8
#   PRECISION     float32 (port default) or float64                                            default: float32
#   GPU           1: run the generation on an NVIDIA GPU (apptainer --nv)                      default: 0
#   IMAGE         Apptainer image (.sif); empty: run on the host / inside the image already    default: empty
#   GDML_OFFSET   x,y,z (m) of ND-LAr's active-volume centre in the GDML world frame           default: from the config
#   EDEP_PRE_MAC  edep-sim macro with the /edep/ detector and physics settings, ending in
#                 /edep/update (empty: edep-sim -u, defaults)                                  default: empty
#   SKIP_EDEPSIM  1: stop after the HEPEVT file                                                default: 0
#
# Outputs in OUTDIR: events.hepevt (+ .index.csv), summary.json, generator.mac, events.edepsim.root, logs.
#
# The file lives in bdnmc_dune/data/ (packaged); jobs/run_bdnmc_edepsim.sh links to it.
# hep-endpoints: bdnmc_dune.jobspec.JobSpec renders this script with the inputs set (and the card embedded) for
# hependpoints' endpoints (Slurm over SSH, Fermilab EAF, FNAL jobsub); see bdnmc_dune/jobspec.py.
set -euo pipefail

: "${CARD:?set CARD to a BdNMC card}"
OUTDIR=${OUTDIR:-out}
TRIALS=${TRIALS:-16777216}
BATCH=${BATCH:-8}
PRECISION=${PRECISION:-float32}
GPU=${GPU:-0}
IMAGE=${IMAGE:-}
GDML_OFFSET=${GDML_OFFSET:-}
EDEP_PRE_MAC=${EDEP_PRE_MAC:-}
SKIP_EDEPSIM=${SKIP_EDEPSIM:-0}
mkdir -p "$OUTDIR"

nv=()
[[ "$GPU" == 1 ]] && nv=(--nv)

in_image() {
    if [[ -n "$IMAGE" && -z "${APPTAINER_CONTAINER:-}" ]]; then
        apptainer exec ${nv[@]+"${nv[@]}"} --env "BDNMC_JAX_PRECISION=$PRECISION" "$IMAGE" "$@"
    else
        BDNMC_JAX_PRECISION=$PRECISION "$@"
    fi
}

offset=()
[[ -n "$GDML_OFFSET" ]] && offset=(--gdml-offset "$GDML_OFFSET")

# Stage 1: generation (JAX port) -> HEPEVT in ND-LAr's frame (cm, ns), one interaction point per event.
in_image python3 -m bdnmc_dune run "$CARD" --trials "$TRIALS" --batch "$BATCH" ${offset[@]+"${offset[@]}"} \
    --hepevt "$OUTDIR/events.hepevt" --summary "$OUTDIR/summary.json" --macro "$OUTDIR/generator.mac" \
    2>&1 | tee "$OUTDIR/bdnmc.log"

[[ "$SKIP_EDEPSIM" == 1 ]] && exit 0
: "${GEOMETRY:?set GEOMETRY to the ND GDML (or SKIP_EDEPSIM=1)}"

n=$(in_image python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["hepevt"]["edepsim_events"])' \
    "$OUTDIR/summary.json")
if [[ "$n" == 0 ]]; then
    echo "no accepted events: edep-sim not run" | tee "$OUTDIR/edepsim.log"
    exit 0
fi

# Stage 2: edep-sim (-C validates the geometry; -u applies /edep/update when no detector macro is given).
macros=("$OUTDIR/generator.mac")
update=(-u)
if [[ -n "$EDEP_PRE_MAC" ]]; then
    macros=("$EDEP_PRE_MAC" "$OUTDIR/generator.mac")
    update=()
fi
in_image edep-sim -C ${update[@]+"${update[@]}"} -g "$GEOMETRY" -o "$OUTDIR/events.edepsim.root" -e "$n" "${macros[@]}" \
    2>&1 | tee "$OUTDIR/edepsim.log"
