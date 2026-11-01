"""ND-LAr's placement in BdNMC's beam frame, the card's cuboid keys, and the detector frame for `write_hepevt`.

Frames (see data/nd_lar.toml):
- beam frame: origin at the target, +z along the beam axis, x horizontal and transverse, y = z x x (up);
- hall frame: z horizontal along the beam's horizontal projection, y vertical up, x the same as the beam frame's.

The beam points down by the slope s in the hall frame: beam direction (0, -sin s, cos s). A vector's hall components
are R v_beam with R = rot_x(s). ND-LAr's axes are the hall axes ("5 m deep, the direction nearly parallel to the
beam", ND CDR p. 2-69), so in the beam frame the cuboid is rotated by R^T = rot_x(-s): BdNMC's X-Y-X Euler angles
(det-phi, det-theta, det-psi) = (-s, 0, 0).

On axis, the beam passes through the active volume's centre (ND CDR p. 8-250), at the distance L along the beam.
DUNE-PRISM moves the detector horizontally, transverse to the beam: a hall displacement (x_off, 0, 0), which is the
same vector in the beam frame.

The detector frame written to HEPEVT is the cuboid's own frame (its centre and local axes), shifted by the
simulation geometry's offset g (where the active-volume centre sits in the GDML world): r_det = A (r - c) + g, with A
the matrix whose rows are the cuboid's local axes in the beam frame. In write_hepevt's form, rotation @ (r - origin):
rotation = A, origin = c - A^T g. `frame_from_card` reads c and A from a card's detector keys, so the frame always
matches the detector BdNMC used.
"""
import math

import numpy as np


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def beam_to_hall(slope_rad):
    """R with v_hall = R v_beam."""
    return rot_x(slope_rad)


def xyx_rotate(t1, t2, t3, v):
    """BdNMC's detector.cpp XYX_rotate (the same expression as bdnmc_jax.detector.xyx_rotate), numpy only."""
    x, y, z = v
    c1, s1, c2, s2, c3, s3 = math.cos(t1), math.sin(t1), math.cos(t2), math.sin(t2), math.cos(t3), math.sin(t3)
    return np.array([
        x * c2 + z * c3 * s2 + y * s2 * s3,
        x * s1 * s2 + z * (-c2 * c3 * s1 - c1 * s3) + y * (c1 * c3 - c2 * s1 * s3),
        -x * c1 * s2 + y * (c3 * s1 + c1 * c2 * s3) + z * (c1 * c2 * c3 - s1 * s3),
    ])


def cuboid_axes(phi, theta, psi):
    """Rows: the cuboid's local x, y, z unit axes in the beam frame (detector_cuboid's face directions)."""
    return np.array([xyx_rotate(phi, theta, psi, e) for e in np.eye(3)])


def euler_angles(slope_rad):
    """(det-phi, det-theta, det-psi) for a cuboid aligned with the hall frame."""
    return (-slope_rad, 0.0, 0.0)


def center_beam_frame(distance, off_axis_x=0.0, vertical_offset=0.0, slope_rad=0.0):
    """ND-LAr active-volume centre in the beam frame (m): on-axis point at `distance`, plus the hall displacement
    (off_axis_x, vertical_offset, 0) rotated into the beam frame."""
    d_hall = np.array([off_axis_x, vertical_offset, 0.0])
    return np.array([0.0, 0.0, distance]) + beam_to_hall(slope_rad).T @ d_hall


def placement(cfg, off_axis_x=0.0):
    """Card detector keys for ND-LAr at horizontal off-axis displacement `off_axis_x` (m): a dict of
    x/y/z-position, width, height, length, det-phi, det-theta, det-psi."""
    s = cfg.slope_rad
    c = center_beam_frame(cfg.value("nd_lar.distance_from_target"), off_axis_x, cfg.value("nd_lar.vertical_offset"),
                          s)
    w, h, l = cfg.active_size
    phi, theta, psi = euler_angles(s)
    return {"x-position": c[0], "y-position": c[1], "z-position": c[2], "width": w, "height": h, "length": l,
            "det-phi": phi, "det-theta": theta, "det-psi": psi}


def frame(center, axes, gdml_offset=(0.0, 0.0, 0.0)):
    """(origin m, rotation 3x3) for bdnmc_jax.output.write_hepevt: r_det = axes (r - center) + gdml_offset."""
    axes = np.asarray(axes, dtype=float)
    return np.asarray(center, dtype=float) - axes.T @ np.asarray(gdml_offset, dtype=float), axes


def frame_from_card(card, gdml_offset=(0.0, 0.0, 0.0)):
    """The detector frame of a card's cuboid (a bdnmc_jax.card.Card or a path)."""
    from bdnmc_jax import card as cardmod
    if not hasattr(card, "keymap"):
        card = cardmod.parse_card(card)
    k = card.keymap
    if k.get("detector") != "cuboid":
        raise ValueError(f"detector {k.get('detector')!r}: the ND-LAr frame needs a cuboid")
    num = lambda key: cardmod.stod(k[key])  # noqa: E731
    center = [num(a) for a in ("x-position", "y-position", "z-position")]
    return frame(center, cuboid_axes(num("det-phi"), num("det-theta"), num("det-psi")), gdml_offset)


def frame_from_config(cfg, off_axis_x=0.0):
    """The same frame from the configuration (for checks): the hall axes at ND-LAr's centre."""
    p = placement(cfg, off_axis_x)
    center = [p["x-position"], p["y-position"], p["z-position"]]
    return frame(center, beam_to_hall(cfg.slope_rad), cfg.value("frame.gdml_offset"))
