"""Tests run on a laptop CPU: no GPU, no DUNE software. bdnmc_jax comes from the installed package or from
BDNMC_JAX_SRC (a checkout's gpu/src). Like the fork's suite, they run in float64 unless BDNMC_JAX_PRECISION is set.
"""
import os
from pathlib import Path
import sys

os.environ.setdefault("BDNMC_JAX_PRECISION", "float64")
if os.environ.get("BDNMC_JAX_SRC"):
    sys.path.insert(0, os.environ["BDNMC_JAX_SRC"])
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
