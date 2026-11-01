"""bdnmc-dune: BdNMC light-dark-matter events for the DUNE Near Detector (ND-LAr first).

- `config`: published ND-LAr and LBNF numbers with their sources (data/nd_lar.toml), and user overrides;
- `geometry`: ND-LAr's placement in BdNMC's beam frame and the detector frame for HEPEVT output;
- `cards`: BdNMC parameter cards for ND-LAr (mass scans, DUNE-PRISM off-axis positions);
- `hepevt`, `edepsim`: the JAX port's events as edep-sim HEPEVT input, and the edep-sim macro;
- `jobspec`: a job as a self-contained script and resources for hep-endpoints;
- `cli`: `python -m bdnmc_dune {cards,run,frame,macro,jobspec}`.
"""
__version__ = "0.1.0"
