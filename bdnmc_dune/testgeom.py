"""A minimal GDML for running edep-sim without DUNE's geometry: ND-LAr's active volume as one liquid-argon box,
centred at the world origin (the detector-centred frame, frame.gdml_offset = 0), in a box of air.

It is a stand-in for a smoke test of the generation -> edep-sim chain, not a detector model: no cryostat, no
modules, no readout, no hall. The box is a SegmentDetector (edep-sim's GDML auxiliary), so edep-sim records energy
deposits in it. Materials are Geant4's NIST materials by name (G4_lAr, G4_AIR), which the GDML reader builds.
"""
from . import config

SD_NAME = "LArActive"


def box_gdml(cfg=None, world_m=(20.0, 20.0, 20.0)):
    """The GDML text: the active volume's size from the configuration (nd_lar.active_*), in a world of `world_m`."""
    cfg = cfg or config.load()
    x, y, z = (cfg.value(f"nd_lar.active_{k}") for k in ("width", "height", "length"))
    wx, wy, wz = world_m
    if not (wx > x and wy > y and wz > z):
        raise ValueError("the world must be larger than the active volume")
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!-- bdnmc-dune test geometry: ND-LAr's active volume ({x} x {y} x {z} m) as one liquid-argon box in air.
     Not a detector model (bdnmc_dune/testgeom.py). -->
<gdml xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
      xsi:noNamespaceSchemaLocation="http://service-spi.web.cern.ch/service-spi/app/releases/GDML/schema/gdml.xsd">
  <define/>
  <materials/>
  <solids>
    <box name="World_box" x="{wx}" y="{wy}" z="{wz}" lunit="m"/>
    <box name="LArActive_box" x="{x}" y="{y}" z="{z}" lunit="m"/>
  </solids>
  <structure>
    <volume name="volLArActive">
      <materialref ref="G4_lAr"/>
      <solidref ref="LArActive_box"/>
      <auxiliary auxtype="SegmentDetector" auxvalue="{SD_NAME}"/>
    </volume>
    <volume name="volWorld">
      <materialref ref="G4_AIR"/>
      <solidref ref="World_box"/>
      <physvol name="LArActive_pv">
        <volumeref ref="volLArActive"/>
        <position name="LArActive_pos" x="0" y="0" z="0" unit="m"/>
      </physvol>
    </volume>
  </structure>
  <setup name="Default" version="1.0">
    <world ref="volWorld"/>
  </setup>
</gdml>
"""
