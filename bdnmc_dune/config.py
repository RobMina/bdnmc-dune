"""The ND-LAr configuration: published numbers with their sources (`data/nd_lar.toml`), and user overrides.

Every quantity in the TOML file is a table `{value, unit, source}`. `load()` reads the packaged file, merges any
override files over it (an override may give a bare value, `key = 6.0`, or a full table) and returns a `Config`
whose `value(path)` returns the number and `source(path)` the citation.
"""
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
import math
import tomllib

DEFAULT = "nd_lar.toml"


def _is_quantity(x):
    return isinstance(x, dict) and "value" in x


def _merge(base, over, where=""):
    out = dict(base)
    for k, v in over.items():
        here = f"{where}.{k}" if where else k
        if k not in base:
            raise KeyError(f"unknown configuration key {here!r}")
        if _is_quantity(base[k]):
            if _is_quantity(v):
                out[k] = base[k] | v
            else:
                out[k] = base[k] | {"value": v, "source": "user override"}
        elif isinstance(base[k], dict) and isinstance(v, dict):
            out[k] = _merge(base[k], v, here)
        else:
            out[k] = v
    return out


@dataclass(frozen=True)
class Config:
    data: dict

    def _get(self, path):
        node = self.data
        for part in path.split("."):
            node = node[part]
        return node

    def value(self, path):
        q = self._get(path)
        if not _is_quantity(q):
            raise KeyError(f"{path!r} is not a quantity")
        return q["value"]

    def source(self, path):
        return self._get(path)["source"]

    def unit(self, path):
        return self._get(path)["unit"]

    def quantities(self):
        """(path, quantity table) for every quantity, in file order."""
        def walk(node, where):
            for k, v in node.items():
                here = f"{where}.{k}" if where else k
                if _is_quantity(v):
                    yield here, v
                elif isinstance(v, dict) and k != "sources":
                    yield from walk(v, here)
        return list(walk(self.data, ""))

    # Derived numbers -------------------------------------------------------------------------------------------
    @property
    def slope_rad(self):
        return math.radians(self.value("beam.slope_at_nd"))

    @property
    def active_size(self):
        """(width x, height y, length z) of ND-LAr's active volume, m."""
        return tuple(float(self.value(f"nd_lar.active_{k}")) for k in ("width", "height", "length"))

    @property
    def lar_number_density(self):
        """Argon atoms per cm^3: density / A x N_A."""
        return self.value("lar.density") / self.value("lar.atomic_mass") * self.value("lar.avogadro")

    @property
    def lar_nucleus_mass(self):
        """Argon mass in GeV (atomic mass x u); BdNMC's material `mass`, used only for coherent scattering."""
        return self.value("lar.atomic_mass") * self.value("lar.atomic_mass_unit")

    @property
    def active_mass_t(self):
        w, h, l = self.active_size
        return w * h * l * self.value("lar.density")  # m^3 x g/cm^3 = t

    @property
    def pot(self):
        return self.value("beam.pot_per_year") * self.value("generator.years")


def load(*overrides):
    """The packaged configuration with the override files (TOML paths) merged over it, in order."""
    data = tomllib.loads(resources.files("bdnmc_dune").joinpath("data", DEFAULT).read_text())
    for path in overrides:
        if path:
            data = _merge(data, tomllib.loads(Path(path).read_text()))
    return Config(data)
