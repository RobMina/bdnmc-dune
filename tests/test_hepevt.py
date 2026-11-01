import subprocess

import pytest

from bdnmc_dune import edepsim, hepevt, jobspec

E1 = "1 11 0 0 0 0 0.1 0.2 0.3 0.4 0.000511 1 2 3 1900"
D1 = "2 18 0 0 0 0 0.1 0.2 0.3 0.4 0.005 1 2 3 1900"
E2 = "1 11 0 0 0 0 0.1 0.2 0.3 0.4 0.000511 4 5 6 1901"
D2 = "2 18 0 0 0 0 0.1 0.2 0.3 0.4 0.005 4 5 6 1901"


def test_units_match_edepsim():
    from bdnmc_jax import output
    # edep-sim's marley reader: GeV for momenta and masses, cm and ns for the vertex
    assert hepevt.EDEPSIM_UNITS == {"momentum": "GeV", "energy": "GeV", "mass": "GeV", "length": "cm", "time": "ns"}
    assert output.LENGTH_UNITS[hepevt.WRITE_HEPEVT_UNITS["length_unit"]] == 100.0  # per metre: BdNMC is in m
    assert output.TIME_UNITS[hepevt.WRITE_HEPEVT_UNITS["time_unit"]] == 1e9  # per second: BdNMC is in s


def test_split_vertices(tmp_path):
    src = tmp_path / "in.hepevt"
    src.write_text(f"0 4\n{E1}\n{D1}\n{E2}\n{D2}\n1 2\n{E1}\n{D1}\n")
    dst = tmp_path / "out.hepevt"
    assert hepevt.split_vertices(src, dst) == (2, 3)
    ev = hepevt.read(dst)
    assert [h for h, _ in ev] == [["0", "2"], ["1", "2"], ["2", "2"]]
    assert [p[1] for _, parts in ev for p in parts] == ["11", "18"] * 3
    assert ev[1][1][0][11:15] == ["4", "5", "6", "1901"]
    assert (tmp_path / "out.hepevt.index.csv").read_text().splitlines() == [
        "edepsim_event,hepevt_event,vertex_in_event", "0,0,0", "1,0,1", "2,1,0"]


def test_reader_refuses_what_edepsim_refuses(tmp_path):
    f = tmp_path / "x.hepevt"
    f.write_text(f"0 0 2\n{E1}\n{D1}\n")  # a 3-token header is the multi-vertex form, not ours
    with pytest.raises(ValueError):
        hepevt.read(f)
    f.write_text("0 1\n1 11 0 0 0.1 0.2 0.3 0.000511\n")  # pythia flavour: 8 tokens
    with pytest.raises(ValueError):
        hepevt.read(f)
    f.write_text(f"0 3\n{E1}\n{D1}\n")
    with pytest.raises(ValueError):
        hepevt.read(f)


def test_macro():
    m = edepsim.generator_macro("/data/events.hepevt").splitlines()
    for cmd in ("/generator/kinematics/hepevt/input /data/events.hepevt", "/generator/kinematics/hepevt/flavor marley",
                "/generator/kinematics/set hepevt", "/generator/position/set free", "/generator/time/set free",
                "/generator/count/fixed/number 1", "/generator/count/set fixed"):
        assert cmd in m
    assert m[-1] == "/generator/add"
    assert m.index("/generator/kinematics/hepevt/flavor marley") < m.index("/generator/kinematics/set hepevt")


def test_jobspec(tmp_path):
    card = "model Dark_Photon\nseed 3\n"
    spec = jobspec.JobSpec("ndlar-pi0-m005", card, geometry="/cvmfs/x/nd.gdml", image="/site/bdnmc-dune.sif",
                           gpus=1, gpu_type="a100", minutes=30, gdml_offset=(0.0, -2.68, 13.0))
    s = spec.script()
    assert s.startswith("#!/usr/bin/env bash\n") and s.count("#!/") == 1
    assert "<<'BDNMC_CARD_EOF'\nmodel Dark_Photon\nseed 3\nBDNMC_CARD_EOF\n" in s
    assert "export GPU=1" in s and "export GDML_OFFSET=0.0,-2.68,13.0" in s and "python3 -m bdnmc_dune run" in s
    f = tmp_path / "job.sh"
    f.write_text(s)
    subprocess.run(["bash", "-n", str(f)], check=True)
    assert spec.resources() == {"cores": 1, "mem_gb": 8, "gpus": 1, "gpu_type": "a100", "minutes": 30}
    with pytest.raises(ValueError):
        jobspec.JobSpec("bad/handle", card, geometry="g")
    with pytest.raises(ValueError):
        jobspec.JobSpec("ok", card)  # no geometry, edep-sim not skipped
    assert 'export OUTDIR="${BDNMC_OUT:-$PWD/out}"' in s  # the endpoint's output directory by default


def test_slurm_job_file_has_the_dune_environment(tmp_path):
    # hep-endpoints renders the complete Slurm job offline: its header exports $BDNMC_OUT, which the script writes to
    from hependpoints import Resources
    spec = jobspec.JobSpec("ndlar-pi0-m005", "seed 3\n", geometry="/site/nd.gdml", gpus=1, gpu_type="a100")
    ep = jobspec.endpoint("slurm_ssh", host="login.example", account="acct", workdir="/scratch/jobs",
                          state_dir=tmp_path / "state")
    body = ep.body(spec.handle, spec.script(), Resources(**spec.resources()))
    assert "#SBATCH -J bdnmcdune_ndlar-pi0-m005" in body and "#SBATCH --gres=gpu:a100:1" in body
    assert "export BDNMC_JOB_DIR=/scratch/jobs/ndlar-pi0-m005 BDNMC_OUT=/scratch/jobs/ndlar-pi0-m005/out" in body
    f = tmp_path / "job.sh"
    f.write_text(body)
    subprocess.run(["bash", "-n", str(f)], check=True)


def test_submit_runs_the_job_through_a_registered_endpoint(tmp_path):
    # a local endpoint registered with hependpoints: it runs the script in bash with $BDNMC_OUT set, and a stub
    # python3 records the generation call (no apptainer, no edep-sim)
    import hependpoints
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "python3").write_text("#!/bin/sh\necho \"$@\" > \"$OUTDIR/called\"\n")
    (bindir / "python3").chmod(0o755)
    seen = {}

    class Local:
        def __init__(self, environment):
            self.env = environment

        def submit(self, handle, script, resources):
            out = tmp_path / handle / "out"
            env = {"PATH": f"{bindir}:/usr/bin:/bin", self.env.out_var: str(out)}
            subprocess.run(["bash", "-c", script], env=env, check=True, cwd=tmp_path)
            seen.update(handle=handle, resources=resources, out=out)
            return hependpoints.JobRecord(handle=handle, spec_sha=hependpoints.spec_sha(script, resources),
                                          job_id="local-1", remote_dir=str(out.parent), submitted_at=0.0)

    hependpoints.register("local-test", Local)
    spec = jobspec.JobSpec("h1", "model Dark_Photon\nseed 3\n", skip_edepsim=True, minutes=5)
    rec = jobspec.submit(spec, "local-test")
    assert rec.job_id == "local-1" and seen["resources"] == hependpoints.Resources(1, 8, 0, "", 5)
    assert (seen["out"] / "card.dat").read_text() == "model Dark_Photon\nseed 3\n"
    called = (seen["out"] / "called").read_text().split()
    assert called[:4] == ["-m", "bdnmc_dune", "run", str(seen["out"] / "card.dat")]


def test_job_script_runs_generation_only(tmp_path):
    # the script's control flow without apptainer or edep-sim: a stub python3 on PATH records the call
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "python3").write_text("#!/bin/sh\necho \"$@\" > \"$OUTDIR/called\"\n")
    (bindir / "python3").chmod(0o755)
    from importlib import resources
    script = resources.files("bdnmc_dune").joinpath("data", "run_bdnmc_edepsim.sh")
    env = {"PATH": f"{bindir}:/usr/bin:/bin", "CARD": "c.dat", "OUTDIR": str(tmp_path / "out"), "SKIP_EDEPSIM": "1",
           "GDML_OFFSET": "0,1,2"}
    subprocess.run(["bash", str(script)], env=env, check=True, cwd=tmp_path)
    called = (tmp_path / "out" / "called").read_text().split()
    assert called[:4] == ["-m", "bdnmc_dune", "run", "c.dat"] and "--gdml-offset" in called and "0,1,2" in called
