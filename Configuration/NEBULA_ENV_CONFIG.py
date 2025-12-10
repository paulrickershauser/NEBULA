"""
NEBULA_ENV_CONFIG.py

This module defines the core environmental constants used by the NEBULA
simulation framework.  These values capture the large-scale properties
of the Earth and its immediate orbital environment that are required
by orbital propagation, visibility, and line-of-sight geometry models.

All constants are defined in explicit units (typically kilometers
and seconds) and are intended to be imported by other modules without
introducing any side effects such as file I/O or logging setup.
"""

# Import the math module so we can compute derived angular/temporal quantities
# such as the length of a sidereal day from the Earth's rotation rate.
import math


# === Earth / geometry constants ===

# Earth radius (WGS-84 mean equatorial) in kilometers.
# This value is used as the reference radius for orbital geometry,
# horizon checks, and line-of-sight blocking computations.
R_EARTH = 6378.137  # km

# Atmospheric clearance buffer above the nominal Earth radius in kilometers.
# Adding this buffer ensures that line-of-sight tests clear the bulk of the
# atmosphere, providing a more conservative blocking radius for visibility.
ATM_HEIGHT = 100.0  # km

# Effective blocking radius for visibility tests in kilometers.
# This is the sum of the Earth radius and the atmospheric buffer and
# represents the radius of the "solid" Earth used in line-of-sight checks.
R_BLOCK = R_EARTH + ATM_HEIGHT  # km

# Gravitational parameter of the Earth (standard gravitational parameter)
# in units of km^3 / s^2.  This constant is used in two-body orbital
# dynamics, including conversions between semi-major axis and mean motion.
MU_EARTH = 398600.4418  # km^3 / s^2


# === Earth rotation / sidereal constants ===

# Earth's sidereal rotation rate in radians per second.
# This is the inertial rotation rate of the Earth relative to the
# fixed stars and is used when modeling sidereal tracking and
# rate-tracking camera modes in NEBULA.
OMEGA_EARTH_SIDEREAL = 7.2921159e-5  # rad / s

# Length of one sidereal day in seconds.
# This is derived from the Earth's sidereal rotation rate as the time
# required for a full 2π radian rotation relative to the inertial frame.
EARTH_SIDEREAL_DAY_S = (2.0 * math.pi) / OMEGA_EARTH_SIDEREAL  # s


# === Reference orbital radii ===

# Nominal geostationary orbit radius measured from the Earth's center in
# kilometers.  This value is used when constructing synthetic GEO reference
# points (for example, along a fixed boresight direction) for line-of-sight
# and visibility analyses.
GEO_RADIUS_KM = 42164.0  # km

# Convenience alias for the Earth's gravitational parameter.
# Some modules may refer to the gravitational parameter generically as MU;
# importing MU from this module keeps those uses consistent with MU_EARTH.
MU = MU_EARTH  # km^3 / s^2
