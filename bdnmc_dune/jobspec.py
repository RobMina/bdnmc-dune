"""A bdnmc-dune job as a self-contained script plus resources, submitted through the experiment-neutral
`hep-endpoints` package (RobMina/hep-endpoints, import name `hependpoints`).

hep-endpoints moves a job script to a site, runs it and brings its output directory back (Slurm over SSH, Fermilab
EAF, FNAL grid through jobsub); the site settings are its `HEPEP_*` variables or keyword arguments. This module:
- `JobSpec.script()`: jobs/run_bdnmc_edepsim.sh with its inputs set and the card embedded, so the job needs only
  the image and the geometry at the site; by default it writes to the endpoint's output directory (`$BDNMC_OUT`);
- `JobSpec.resources()`: cores, memory, GPUs, wall time (the fields of hependpoints.Resources);
- `endpoint(name, **kw)`, `submit(spec, endpoint)`: an endpoint with bdnmc-dune's job environment (variables
  `BDNMC_*`, job names `bdnmcdune_<handle>`), and the submission. Status, fetch and cancel are the endpoint's own.
"""
from dataclasses import asdict, dataclass, field
from importlib import resources as ilr
import json
from pathlib import Path
import re
import shlex

HANDLE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
VAR_PREFIX, JOB_PREFIX = "BDNMC", "bdnmcdune"
OUT_VAR = f"{VAR_PREFIX}_OUT"


@dataclass
class JobSpec:
    handle: str  # the caller's name for the job (hependpoints: idempotent per handle)
    card_text: str  # the BdNMC card, embedded in the script
    geometry: str = ""  # ND GDML path at the site ("" with skip_edepsim)
    image: str = ""  # Apptainer image at the site ("" when the job already runs inside it)
    trials: int = 1 << 24
    batch: int = 8
    precision: str = "float32"
    gpus: int = 0
    gpu_type: str = ""
    cores: int = 1
    mem_gb: int = 8
    minutes: int = 120
    outdir: str = ""  # "" = the endpoint's output directory ($BDNMC_OUT), or ./out outside an endpoint
    gdml_offset: tuple = (0.0, 0.0, 0.0)
    edep_pre_mac: str = ""
    skip_edepsim: bool = False
    extra_env: dict = field(default_factory=dict)

    def __post_init__(self):
        if not HANDLE.match(self.handle):
            raise ValueError(f"handle {self.handle!r}: letters, digits, '_', '.', '-'")
        if self.precision not in ("float32", "float64"):
            raise ValueError("precision: float32 or float64")
        if not self.skip_edepsim and not self.geometry:
            raise ValueError("geometry (the ND GDML at the site) is required unless skip_edepsim")
        if "BDNMC_CARD_EOF" in self.card_text:
            raise ValueError("card text contains the heredoc delimiter")

    def env(self):
        """The job script's inputs (jobs/run_bdnmc_edepsim.sh)."""
        e = {"TRIALS": str(int(self.trials)),
             "BATCH": str(int(self.batch)), "PRECISION": self.precision, "GPU": "1" if self.gpus else "0",
             "IMAGE": self.image, "GEOMETRY": self.geometry, "EDEP_PRE_MAC": self.edep_pre_mac,
             "GDML_OFFSET": ",".join(repr(float(v)) for v in self.gdml_offset),
             "SKIP_EDEPSIM": "1" if self.skip_edepsim else "0"}
        return e | {k: str(v) for k, v in self.extra_env.items()}

    def script(self):
        body = ilr.files("bdnmc_dune").joinpath("data", "run_bdnmc_edepsim.sh").read_text()
        body = body.split("\n", 1)[1]  # drop the shebang; the prelude has its own
        env = self.env()
        prelude = ["#!/usr/bin/env bash", f"# bdnmc-dune job {self.handle}", "set -euo pipefail"]
        out = shlex.quote(self.outdir) if self.outdir else f'"${{{OUT_VAR}:-$PWD/out}}"'
        prelude += [f"export OUTDIR={out}", 'export CARD="$OUTDIR/card.dat"']
        prelude += [f"export {k}={shlex.quote(v)}" for k, v in env.items()]
        prelude += ['mkdir -p "$OUTDIR"', "cat > \"$CARD\" <<'BDNMC_CARD_EOF'", self.card_text.rstrip("\n"),
                    "BDNMC_CARD_EOF"]
        return "\n".join(prelude) + "\n" + body

    def resources(self):
        """Keyword arguments for hependpoints.Resources."""
        return {"cores": self.cores, "mem_gb": self.mem_gb, "gpus": self.gpus, "gpu_type": self.gpu_type,
                "minutes": self.minutes}

    def to_json(self):
        return json.dumps(asdict(self) | {"resources": self.resources()}, indent=1)

    @classmethod
    def from_card(cls, handle, card_path, **kw):
        return cls(handle=handle, card_text=Path(card_path).read_text(), **kw)


def environment():
    """bdnmc-dune's hependpoints.JobEnvironment: jobs see $BDNMC_JOB_DIR and $BDNMC_OUT; job names bdnmcdune_<handle>."""
    from hependpoints.environment import JobEnvironment
    return JobEnvironment(var_prefix=VAR_PREFIX, job_prefix=JOB_PREFIX)


def endpoint(name, **kw):
    """A hependpoints endpoint ("slurm_ssh", "eaf", "jobsub" or one an experiment registered) with bdnmc-dune's job
    environment; its site settings come from `kw` or its HEPEP_* variables."""
    import hependpoints
    kw.setdefault("environment", environment())
    return hependpoints.get(name, **kw)


def submit(spec, ep, **endpoint_kw):
    """Submit `spec` to `ep` (an endpoint, or a name for `endpoint`). Idempotent per handle: resubmitting the same
    spec returns the same record; a changed spec under the same handle is refused. The image and the geometry must
    already be at the site (paths in the spec); the endpoint's fetch brings $BDNMC_OUT back."""
    import hependpoints
    if isinstance(ep, str):
        ep = endpoint(ep, **endpoint_kw)
    return ep.submit(spec.handle, spec.script(), hependpoints.Resources(**spec.resources()))
