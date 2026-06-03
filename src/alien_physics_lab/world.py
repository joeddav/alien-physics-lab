from __future__ import annotations

from dataclasses import dataclass
import math

EARTH_GRAVITY_M_S2 = 9.80665
EARTH_DIAMETER_M = 12_742_000.0
EARTH_MASS_KG = 5.9722e24
EARTH_SPIN_RAD_S = 7.2921159e-5
GRAVITATIONAL_CONSTANT = 6.67430e-11


@dataclass(frozen=True)
class WorldParams:
    """Hidden alien world parameters.

    `gravity_m_s2` is the simplest first-pass knob. If it is omitted, effective
    lab gravity is computed from mass, diameter, spin, and lab latitude.
    """

    gravity_m_s2: float | None = None
    world_diameter_m: float = EARTH_DIAMETER_M
    world_spin_rad_s: float = EARTH_SPIN_RAD_S
    lab_latitude_deg: float = 35.0
    world_mass_kg: float = EARTH_MASS_KG
    atmosphere_drag: float = 0.0
    measurement_noise: float = 0.005
    seed: int | None = None

    @property
    def effective_gravity_m_s2(self) -> float:
        if self.gravity_m_s2 is not None:
            return self.gravity_m_s2

        radius_m = self.world_diameter_m / 2.0
        base_gravity = GRAVITATIONAL_CONSTANT * self.world_mass_kg / radius_m**2
        latitude_rad = math.radians(self.lab_latitude_deg)
        centrifugal = self.world_spin_rad_s**2 * radius_m * math.cos(latitude_rad) ** 2
        return max(0.001, base_gravity - centrifugal)

    def public_summary(self) -> dict[str, float]:
        """Return non-hidden configuration details safe to show an agent."""

        return {
            "world_diameter_m": self.world_diameter_m,
            "world_spin_rad_s": self.world_spin_rad_s,
            "lab_latitude_deg": self.lab_latitude_deg,
            "world_mass_kg": self.world_mass_kg,
        }
