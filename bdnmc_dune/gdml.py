"""Placements in a GDML geometry, read with the standard library's XML parser and numpy: the world-frame position and
orientation of every placement of a logical volume, the union box of box-shaped volumes, and the sensitive-detector
auxiliaries. For `frame.gdml_offset`: where ND-LAr's active volume sits in the world of the geometry edep-sim runs.

Transforms follow Geant4's GDML reader (G4GDMLReadDefine::GetRotationMatrix, G4GDMLReadStructure::PhysvolRead):
a `<rotation x y z>` is the matrix M = Rz(z) Ry(y) Rx(x) (CLHEP's rotateX, then rotateY, then rotateZ, each
multiplying from the left), and a physvol places its daughter with G4Transform3D(M^-1, position), so a point p of the
daughter is at M^T p + position in the mother. Lengths are returned in mm unless a function says otherwise.

Only what geometries made with GeGeDe (DUNE's dunendggd) use is supported: numeric attributes (no expressions),
`<position>`/`<rotation>` defines or inline in a physvol, no scale (reflections), no assemblies, replicas or
parameterised volumes; anything else raises.
"""
from dataclasses import dataclass, field
import math
import xml.etree.ElementTree as ET

import numpy as np

LENGTH_MM = {"nm": 1e-6, "um": 1e-3, "mm": 1.0, "cm": 10.0, "m": 1000.0, "km": 1e6}
ANGLE_RAD = {"rad": 1.0, "radian": 1.0, "mrad": 1e-3, "deg": math.pi / 180.0, "degree": math.pi / 180.0}
UNSUPPORTED = {"scale", "scaleref", "assembly", "replicavol", "paramvol", "divisionvol", "bordersurface",
               "skinsurface", "loop"}


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotation_matrix(x, y, z):
    """Geant4's matrix for a GDML <rotation> (angles in rad): Rz(z) Ry(y) Rx(x)."""
    return _rz(z) @ _ry(y) @ _rx(x)


def _num(el, key, default=0.0):
    v = el.get(key)
    if v is None:
        return default
    try:
        return float(v)
    except ValueError:
        raise ValueError(f"<{el.tag} name={el.get('name')!r}> {key}={v!r}: expressions are not supported") from None


def _position(el):
    """A <position> element -> mm (GDML's default length unit is mm)."""
    u = LENGTH_MM[el.get("unit", "mm")]
    return np.array([_num(el, k) for k in "xyz"]) * u


def _rotation(el):
    """A <rotation> element -> (x, y, z) in rad (GDML's default angle unit is rad)."""
    u = ANGLE_RAD[el.get("unit", "rad")]
    return tuple(_num(el, k) * u for k in "xyz")


@dataclass
class Placement:
    mother: str
    daughter: str
    name: str
    rot: np.ndarray  # daughter -> mother: p_mother = rot @ p_daughter + pos
    pos: np.ndarray  # mm
    copy: int = 0


@dataclass
class Geometry:
    world: str
    volumes: dict = field(default_factory=dict)  # name -> {"solid", "material", "aux": [(type, value)]}
    solids: dict = field(default_factory=dict)  # name -> (tag, attributes)
    placements: list = field(default_factory=list)
    world_aux: list = field(default_factory=list)

    def __post_init__(self):
        self.index()

    def index(self):
        self._up, self._cache = {}, {}
        for p in self.placements:
            self._up.setdefault(p.daughter, []).append(p)
        return self

    def instances(self, volume):
        """Every placement of `volume` in the world: a tuple of (path, rot, pos), with rot (3x3) and pos (mm) the
        daughter -> world transform and path the physvol names from the world down."""
        if volume in self._cache:
            return self._cache[volume]
        if volume == self.world:
            out = [((), np.eye(3), np.zeros(3))]
        elif volume not in self.volumes:
            raise KeyError(f"no volume {volume!r}")
        else:
            out = [(path + (p.name or f"{p.daughter}_{p.copy}",), r @ p.rot, r @ p.pos + t)
                   for p in self._up.get(volume, []) for path, r, t in self.instances(p.mother)]
        self._cache[volume] = tuple(out)
        return self._cache[volume]

    def box_half(self, volume):
        """Half-lengths (mm) of a box-shaped volume."""
        tag, att = self.solids[self.volumes[volume]["solid"]]
        if tag != "box":
            raise ValueError(f"{volume}: solid is a {tag}, not a box")
        u = LENGTH_MM[att.get("lunit", "mm")]
        return np.array([float(att[k]) for k in "xyz"]) * u / 2.0

    def sensitive(self, auxtypes=("SegmentDetector", "SensDet")):
        """{detector name: [volumes]} for edep-sim's segment detectors: the auxiliary SegmentDetector, or SensDet
        (deprecated in edep-sim, the same thing; EDepSimUserDetectorConstruction.cc)."""
        out = {}
        for name, v in self.volumes.items():
            for t, val in v["aux"]:
                if t in auxtypes:
                    out.setdefault(val, []).append(name)
        return out


def load(path):
    """Read a GDML file into a Geometry (iteratively, so large files stay cheap)."""
    pos, rot = {}, {}
    geom = Geometry(world="")
    for event, el in ET.iterparse(path, events=("start", "end")):
        tag = el.tag
        if event == "start":
            if tag in UNSUPPORTED:
                raise ValueError(f"GDML element <{tag}> is not supported")
            continue
        if tag == "define":
            for d in el:
                if d.tag == "position":
                    pos[d.get("name")] = _position(d)
                elif d.tag == "rotation":
                    rot[d.get("name")] = _rotation(d)
            el.clear()
        elif tag == "solids":
            for s in el:
                geom.solids[s.get("name")] = (s.tag, dict(s.attrib))
            el.clear()
        elif tag == "materials":
            el.clear()
        elif tag == "volume":
            name = el.get("name")
            aux = [(a.get("auxtype"), a.get("auxvalue")) for a in el.findall("auxiliary")]
            mat, sol = el.find("materialref"), el.find("solidref")
            geom.volumes[name] = {"material": mat.get("ref") if mat is not None else None,
                                  "solid": sol.get("ref") if sol is not None else None, "aux": aux}
            for pv in el.findall("physvol"):
                ref = pv.find("volumeref").get("ref")
                if pv.find("file") is not None:
                    raise ValueError("physvol <file> is not supported")
                p = pv.find("position")
                p = _position(p) if p is not None else pos[pv.find("positionref").get("ref")] \
                    if pv.find("positionref") is not None else np.zeros(3)
                r = pv.find("rotation")
                r = _rotation(r) if r is not None else rot[pv.find("rotationref").get("ref")] \
                    if pv.find("rotationref") is not None else (0.0, 0.0, 0.0)
                geom.placements.append(Placement(name, ref, pv.get("name", ""), rotation_matrix(*r).T, p,
                                                 int(pv.get("copynumber", 0))))
            el.clear()
        elif tag == "world":
            geom.world = el.get("ref")
    if not geom.world:
        raise ValueError(f"{path}: no <setup><world>")
    geom.world_aux = geom.volumes[geom.world]["aux"]
    return geom.index()


def union_box(geom, volume):
    """World-frame axis-aligned union box (mm) of every placement of a box-shaped volume: (low, high)."""
    h = geom.box_half(volume)
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * h
    pts = np.concatenate([corners @ r.T + t for _, r, t in geom.instances(volume)])
    return pts.min(axis=0), pts.max(axis=0)


def frame_report(geom, active="volTPCActive", landmarks=()):
    """What `frame.gdml_offset` needs from a geometry, in m: the active volume's placements (count, largest deviation
    of their rotation from the identity, determinant), their union box and its centre (the offset), the active
    volume summed over placements, the sensitive detectors, and the world-frame origins of `landmarks` (volumes
    whose order shows which way the axes point). The union box is axis-aligned in the world frame; the placements'
    distinct rotations show whether the boxes are turned (a 180-degree turn of a box about its own centre, as for
    TPCs that drift in opposite directions, leaves the union unchanged)."""
    inst = geom.instances(active)
    low, high = union_box(geom, active)
    h = geom.box_half(active)
    out = {
        "world": geom.world,
        "world_aux": dict(geom.world_aux),
        "world_box_size_m": (2 * geom.box_half(geom.world) / 1e3).tolist(),
        "sensitive_detectors": {k: {"volumes": len(v), "first": v[:3]} for k, v in geom.sensitive().items()},
        "active": {
            "volume": active,
            "material": geom.volumes[active]["material"],
            "placements": len(inst),
            "max_abs_rotation_minus_identity": float(max(np.abs(r - np.eye(3)).max() for _, r, _ in inst)),
            "determinants": sorted({round(float(np.linalg.det(r)), 12) for _, r, _ in inst}),
            "distinct_rotations": [list(map(list, m)) for m in sorted({tuple(map(tuple, np.round(r, 9) + 0.0))
                                                                         for _, r, _ in inst})],
            "box_size_m": (2 * h / 1e3).tolist(),
            "union_low_m": (low / 1e3).tolist(),
            "union_high_m": (high / 1e3).tolist(),
            "union_size_m": ((high - low) / 1e3).tolist(),
            "union_centre_m": ((low + high) / 2e3).tolist(),
            "summed_volume_m3": float(len(inst) * np.prod(2 * h / 1e3)),
        },
        "active_centres_m": [(t / 1e3).tolist() for _, _, t in inst],
        "landmarks_m": {},
    }
    for name in landmarks:
        out["landmarks_m"][name] = [{"origin": (t / 1e3).tolist(),
                                     "max_abs_rotation_minus_identity": float(np.abs(r - np.eye(3)).max())}
                                    for _, r, t in geom.instances(name)]
    return out
