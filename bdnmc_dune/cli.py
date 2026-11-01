"""python -m bdnmc_dune <command>

  cards    write a mass scan of ND-LAr cards (channels x masses x off-axis positions) and manifest.json
  run      run one card through the JAX port; write edep-sim HEPEVT input in ND-LAr's frame (+ HDF5, summary,
           edep-sim macro)
  frame    print the detector frame (origin, rotation) of a card or of an off-axis position
  macro    print or write the edep-sim generator macro for a HEPEVT file
  jobspec  build the hep-endpoints job (self-contained script + resources) for one card; --submit ENDPOINT sends it
  job      status, fetch or cancel a submitted job
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def _floats(s):
    return [float(x) for x in s.split(",") if x.strip()]


def _offset(s):
    v = _floats(s)
    if len(v) != 3:
        raise argparse.ArgumentTypeError("x,y,z in metres")
    return v


def cmd_cards(a):
    from . import cards, config
    cfg = config.load(*a.config)
    off = cfg.value("prism.scan_positions") if a.off_axis == "prism" else _floats(a.off_axis)
    man = cards.write_scan(cfg, a.out, channels=a.channels.split(","), masses_dm=_floats(a.mchi), off_axis=off,
                           mass_v=a.mv, seed0=a.seed0, allow_beyond_prism=a.allow_beyond_prism)
    print(f"{len(man['cards'])} cards in {a.out} ({len(man['skipped'])} points kinematically closed)")


def cmd_run(a):
    from . import config, edepsim, hepevt
    cfg = config.load(*a.config)
    gdml = a.gdml_offset if a.gdml_offset is not None else cfg.value("frame.gdml_offset")
    import jax
    import bdnmc_jax
    from bdnmc_jax import output, pipeline
    t0 = time.perf_counter()
    res = pipeline.events(a.card, a.trials, batch=a.batch, parent_sampler=a.parent_sampler)
    dt = time.perf_counter() - t0
    print(f"backend {jax.default_backend()}, precision {bdnmc_jax.PRECISION}, {res['trials']} trials in {dt:.1f} s")
    print(f"DM reaching ND-LAr: {res['n_hits']}; accepted: {res['accepted']}")
    print(f"Predicted number of signal events (unweighted, two-pass) = {res['se']:.6g}; "
          f"weighted = {res['se_weighted']:.6g}")
    summary = {"card": str(a.card), "card_sha256": hashlib.sha256(Path(a.card).read_bytes()).hexdigest(),
               "seed": res["seed"], "n_trials": res["n_trials"], "trials": res["trials"], "n_hits": res["n_hits"],
               "accepted": res["accepted"], "se": res["se"], "se_weighted": res["se_weighted"], "pmax": res["pmax"],
               "precision": res["precision"], "parent_sampler": res["parent_sampler"],
               "backend": jax.default_backend(), "seconds": dt}
    if a.hepevt:
        info = hepevt.write(res, a.hepevt, gdml_offset=gdml, split=not a.no_split)
        summary["hepevt"] = info
        print(f"{info['edepsim_events']} edep-sim events ({info['bdnmc_events']} BdNMC events) in {a.hepevt}")
        if a.macro:
            Path(a.macro).write_text(edepsim.generator_macro(Path(a.hepevt).resolve()))
    if a.hdf5:
        summary["hdf5"] = {"path": str(a.hdf5), "rows": output.write_hdf5(res, a.hdf5)}
    if a.summary:
        Path(a.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(a.summary).write_text(json.dumps(summary, indent=1) + "\n")


def cmd_frame(a):
    import numpy as np
    from . import config, geometry
    cfg = config.load(*a.config)
    gdml = a.gdml_offset if a.gdml_offset is not None else cfg.value("frame.gdml_offset")
    if a.card:
        origin, rot = geometry.frame_from_card(a.card, gdml)
    else:
        p = geometry.placement(cfg, a.off_axis)
        origin, rot = geometry.frame([p["x-position"], p["y-position"], p["z-position"]],
                                     geometry.beam_to_hall(cfg.slope_rad), gdml)
    print(json.dumps({"origin_m": np.asarray(origin).tolist(), "rotation": np.asarray(rot).tolist()}, indent=1))


def cmd_macro(a):
    from . import edepsim
    text = edepsim.generator_macro(a.hepevt)
    if a.out:
        Path(a.out).write_text(text)
    else:
        sys.stdout.write(text)


def cmd_jobspec(a):
    from . import config, jobspec
    cfg = config.load(*a.config)
    spec = jobspec.JobSpec.from_card(
        a.handle, a.card, geometry=a.geometry, image=a.image, trials=a.trials, batch=a.batch,
        precision=a.precision, gpus=a.gpus, gpu_type=a.gpu_type, cores=a.cores, mem_gb=a.mem_gb,
        minutes=a.minutes, outdir=a.outdir, edep_pre_mac=a.edep_pre_mac, skip_edepsim=a.skip_edepsim,
        gdml_offset=tuple(a.gdml_offset if a.gdml_offset is not None else cfg.value("frame.gdml_offset")))
    if a.script:
        Path(a.script).write_text(spec.script())
        Path(a.script).chmod(0o755)
    print(spec.to_json())
    if a.submit:
        rec = jobspec.submit(spec, a.submit)
        print(f"submitted {spec.handle} to {a.submit}: job {rec.job_id}")


def cmd_job(a):
    from . import jobspec
    ep = jobspec.endpoint(a.endpoint)
    if a.action == "status":
        st = ep.status(a.handle)
        print(f"{a.handle}: {st.state} {st.message}".rstrip())
    elif a.action == "fetch":
        for f in ep.fetch(a.handle, a.dest):
            print(f)
    else:
        print(ep.cancel(a.handle))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m bdnmc_dune", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--config", action="append", default=[], help="TOML override file(s) over nd_lar.toml")

    p = sub.add_parser("cards", help="mass scan of ND-LAr cards")
    common(p)
    p.add_argument("--channels", default="pi0", help="comma list of pi0, eta, brem")
    p.add_argument("--mchi", default="0.001,0.003,0.005,0.01,0.02,0.03,0.04,0.06,0.1,0.15,0.2,0.25",
                   help="dark-matter masses, GeV (comma list); closed points are skipped")
    p.add_argument("--mv", type=float, help="fixed dark-photon mass, GeV (default: generator.mv_over_mchi x m_chi)")
    p.add_argument("--off-axis", default="0", help="comma list of off-axis displacements (m), or 'prism' for the "
                   "DUNE LDM study's positions (prism.scan_positions)")
    p.add_argument("--allow-beyond-prism", action="store_true")
    p.add_argument("--seed0", type=int, default=20261031)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_cards)

    p = sub.add_parser("run", help="one card through the JAX port")
    common(p)
    p.add_argument("card", type=Path)
    p.add_argument("--trials", type=int, default=1 << 24)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--parent-sampler", default="stock", choices=("stock", "envelope"))
    p.add_argument("--gdml-offset", type=_offset, help="x,y,z (m) of ND-LAr's centre in the GDML world")
    p.add_argument("--hepevt", type=Path)
    p.add_argument("--no-split", action="store_true", help="keep write_hepevt's grouping (several vertices per "
                   "event; edep-sim would put them all at the last one)")
    p.add_argument("--macro", type=Path, help="edep-sim generator macro for the HEPEVT file")
    p.add_argument("--hdf5", type=Path)
    p.add_argument("--summary", type=Path)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("frame", help="detector frame for write_hepevt")
    common(p)
    p.add_argument("--card", type=Path)
    p.add_argument("--off-axis", type=float, default=0.0)
    p.add_argument("--gdml-offset", type=_offset)
    p.set_defaults(func=cmd_frame)

    p = sub.add_parser("macro", help="edep-sim generator macro")
    p.add_argument("hepevt")
    p.add_argument("--out", type=Path)
    p.set_defaults(func=cmd_macro)

    p = sub.add_parser("jobspec", help="hep-endpoints job for one card")
    common(p)
    p.add_argument("card", type=Path)
    p.add_argument("--handle", required=True)
    p.add_argument("--geometry", default="")
    p.add_argument("--image", default="")
    p.add_argument("--trials", type=int, default=1 << 24)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--precision", default="float32", choices=("float32", "float64"))
    p.add_argument("--gpus", type=int, default=0)
    p.add_argument("--gpu-type", default="")
    p.add_argument("--cores", type=int, default=1)
    p.add_argument("--mem-gb", type=int, default=8)
    p.add_argument("--minutes", type=int, default=120)
    p.add_argument("--outdir", default="", help="output directory in the job (default: the endpoint's, $BDNMC_OUT)")
    p.add_argument("--gdml-offset", type=_offset)
    p.add_argument("--edep-pre-mac", default="")
    p.add_argument("--skip-edepsim", action="store_true")
    p.add_argument("--script", type=Path, help="also write the rendered job script here")
    p.add_argument("--submit", metavar="ENDPOINT", help="submit through hep-endpoints (slurm_ssh, eaf, jobsub; site "
                   "settings from the HEPEP_* variables)")
    p.set_defaults(func=cmd_jobspec)

    p = sub.add_parser("job", help="status, fetch or cancel a submitted job (hep-endpoints)")
    p.add_argument("action", choices=("status", "fetch", "cancel"))
    p.add_argument("handle")
    p.add_argument("--endpoint", required=True)
    p.add_argument("--dest", type=Path, default=Path("."), help="fetch: copy the job's output under DEST/<handle>/")
    p.set_defaults(func=cmd_job)

    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
