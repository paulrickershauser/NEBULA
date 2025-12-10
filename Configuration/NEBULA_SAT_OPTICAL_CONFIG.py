"""
NEBULA_SAT_OPTICAL_CONFIG.py

Optical / radiometric configuration for the NEBULA simulation framework.

This file defines:
    1) A simple identifier for the Lambertian-sphere optical model.
    2) Orbit-regime labels (LEO, MEO, GEO, HEO).
    3) A dataclass that stores all scalar parameters needed to describe
       a Lambertian-equivalent sphere in a given photometric band.
    4) Default Lambertian-sphere configurations for each orbit regime.
    5) A helper function to print a human-readable description of the
       chosen configuration (useful for logs).

No actual radiometry or BRDF math is performed here. This module is
pure configuration: other NEBULA utilities import this file and use
the constants / dataclasses when computing fluxes and photon rates.
"""

from dataclasses import dataclass  # Used to create a simple config container class
from typing import Optional        # Used to allow some fields to be None (e.g., mag_zero_point)


# ---------------------------------------------------------------------------
# Optical model identifier
# ---------------------------------------------------------------------------

# Name/label for the Lambertian-sphere optical model.
# - "Lambertian sphere" means:
#     * The satellite is modeled as an equivalent sphere.
#     * The surface reflects light diffusely (Lambertian).
#     * Brightness vs phase angle follows the Lambertian sphere phase law.
OPTICAL_MODEL_LAM_SPHERE = "LAM_SPHERE"


# ---------------------------------------------------------------------------
# Orbit-regime identifiers
# ---------------------------------------------------------------------------

# These strings are used throughout NEBULA to tag satellites by orbit regime.
# They allow utilities to select different default optical parameters
# for LEO, MEO, GEO, and HEO regimes.
ORBIT_REGIME_LEO = "LEO"  # Low Earth Orbit
ORBIT_REGIME_MEO = "MEO"  # Medium Earth Orbit
ORBIT_REGIME_GEO = "GEO"  # Geosynchronous / Geostationary region
ORBIT_REGIME_HEO = "HEO"  # Highly Elliptical Orbit

# ---------------------------------------------------------------------------
# Gaia DR3 G-band photometric / solar reference values
# ---------------------------------------------------------------------------

# Broad-band name used in NEBULA for this passband
GAIA_G_BAND_NAME = "Gaia_G"

# Effective (pivot) wavelength and approximate effective width of the Gaia G band.
# For a solar-type spectrum the pivot wavelength is ~622 nm, and the band is very
# broad (~330–1050 nm) with an effective/FWHM width of a few hundred nm.
# (See Weiler 2018 and Gaia DR2/EDR3 passband papers.):contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}
GAIA_G_LAMBDA_EFF_NM = 622.0     # nm, pivot/effective wavelength
GAIA_G_BANDWIDTH_EFF_NM = 400.0  # nm, rough effective width; refine later if needed

# Solar absolute and apparent magnitudes in Gaia G.
# Casagrande & VandenBerg (2018) give M_G,⊙ ≈ 4.67, implying m_G,⊙(1 AU) ≈ -26.9.:contentReference[oaicite:2]{index=2}:contentReference[oaicite:3]{index=3}
GAIA_G_M_SUN_ABS = 4.67      # Absolute magnitude of Sun in G
GAIA_G_M_SUN_APP_1AU = -26.90  # Apparent G magnitude of Sun at 1 AU

# Standard bolometric solar constant at 1 AU [W m^-2].
# This is *not* band-limited; radiometry code can use ratios so the exact
# band integral cancels when working relative to the Sun.
SOLAR_CONSTANT_BOL_W_M2 = 1361.0

# ---------------------------------------------------------------------------
# Physical constants (SI units) used by radiometry utilities
# ---------------------------------------------------------------------------

# Planck's constant [J s]; exact by 2019 SI definition.
PLANCK_H_J_S = 6.62607015e-34

# Speed of light in vacuum [m s^-1]; exact by definition.
SPEED_OF_LIGHT_M_S = 299792458.0

# ---------------------------------------------------------------------------
# Dataclass describing a Lambertian-sphere optical configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LambertSphereModelConfig:
    """
    Configuration object describing a Lambertian-equivalent sphere model
    for a given orbit regime and photometric band.

    This is a *pure data* container: it holds the scalar values that a
    radiometry utility will need to compute reflected-light flux from a
    satellite modeled as a Lambertian sphere.

    Fields
    ------
    radius_m_default : float
        Default Lambertian-equivalent radius [meters].
        For GEO, NEBULA uses ~2.5 m as a canonical effective radius.
        This follows common GEO photometry practice where a large
        satellite is approximated by a diffuse sphere of radius 2–3 m.

    radius_m_min : float
        Lower bound on plausible radius [meters] for this regime.
        This is not a hard constraint, but a reasonable lower limit for
        sensitivity studies and parameter sweeps.

    radius_m_max : float
        Upper bound on plausible radius [meters] for this regime.

    albedo_default : float
        Default broadband reflectance (dimensionless, 0–1).
        For GEO, values around 0.2 are common in debris / RSO size-
        estimation models.

    albedo_min : float
        Lower bound on plausible albedo for this regime.

    albedo_max : float
        Upper bound on plausible albedo for this regime.

    solar_irradiance_W_m2 : float
        Solar irradiance at the satellite [W m^-2].
        For Earth-bound orbits (LEO/MEO/GEO/HEO), NEBULA assumes the
        standard solar constant at 1 AU (~1361 W/m^2) and treats it as
        constant.
        Solar irradiance used by the model [W m^-2].
        In the current NEBULA defaults this is the *bolometric* solar
        constant at 1 AU (~1361 W m^-2). Radiometry utilities may combine
        this with GAIA_G_* band constants to form an effective G-band
        irradiance, but the configuration itself stays agnostic.

    band_name : str
        Short name for the photometric band in which radiometry is
        modeled (e.g. "Gaia_G"). This ensures consistency between the
        satellite radiometry and star/background models.

    lambda_eff_nm : float
        Effective (pivot) wavelength of the band [nanometers].
        For a Gaia-like G band, this is typically in the red (~640–680 nm).

    bandwidth_nm : float
        Effective width of the photometric band [nanometers].
        For a broad Gaia-like G band, this can be several hundred nm.
        NEBULA uses a simplified effective width here for photon-count
        estimates.

    mag_zero_point : Optional[float]
        Optional magnitude zero point for this band (e.g. Vega-system
        or AB-system). If set, radiometry utilities can convert from
        flux to apparent magnitude directly. If None, the conversion is
        deferred to a higher-level photometry module.

    notes : str
        Free-form string with provenance, citations, or comments about
        how these defaults were chosen. Purely informational; not used
        in computations.
    """

    # Default effective radius of the Lambert sphere [m]
    radius_m_default: float

    # Plausible lower/upper bounds on effective radius [m]
    radius_m_min: float
    radius_m_max: float

    # Default broadband albedo (0–1)
    albedo_default: float

    # Plausible lower/upper bounds on albedo (0–1)
    albedo_min: float
    albedo_max: float

    # Solar irradiance used for the model [W m^-2]
    solar_irradiance_W_m2: float

    # Photometric band name (e.g. "Gaia_G")
    band_name: str

    # Effective wavelength of the band [nm]
    lambda_eff_nm: float

    # Effective bandwidth of the band [nm]
    bandwidth_nm: float

    # Optional magnitude zero point for this band
    mag_zero_point: Optional[float]

    # Human-readable description / provenance
    notes: str


# ---------------------------------------------------------------------------
# Default Lambertian-sphere configurations by orbit regime
# ---------------------------------------------------------------------------

# -----------------------
# GEO Lambert-sphere config
# -----------------------
# This configuration is intended for large GEO communication satellites
# and similar objects.  The chosen defaults:
#   - radius_m_default ≈ 2.5 m
#   - albedo_default   ≈ 0.20
# are consistent with common GEO brightness models that treat such
# satellites as diffuse spheres with a few-meter effective radius and
# moderate reflectance.

LAM_SPHERE_GEO_DEFAULTS = LambertSphereModelConfig(
    radius_m_default=2.5,   # Default Lambertian radius for GEO [m]
    radius_m_min=1.0,       # Lower plausible radius bound [m]
    radius_m_max=4.0,       # Upper plausible radius bound [m]

    albedo_default=0.20,    # Default broadband albedo for GEO objects
    albedo_min=0.10,        # Lower plausible albedo bound
    albedo_max=0.30,        # Upper plausible albedo bound

    # Use the global bolometric solar constant; the G-band behavior is
    # handled by GAIA_G_* constants in this module and by the radiometry code.
    solar_irradiance_W_m2=SOLAR_CONSTANT_BOL_W_M2,

    # Bind this regime to the Gaia-like G band defined above
    band_name=GAIA_G_BAND_NAME,
    lambda_eff_nm=GAIA_G_LAMBDA_EFF_NM,
    bandwidth_nm=GAIA_G_BANDWIDTH_EFF_NM,


    mag_zero_point=None,    # Gaia-G zero point not hard-coded here

    notes=(
        "Canonical GEO Lambertian sphere: R_eff ≈ 2.5 m, albedo ≈ 0.2, "
        "in a Gaia-like broad optical band. Parameters chosen to align "
        "with typical GEO photometry assumptions (large GEO satellites "
        "modeled as diffuse spheres at ~36,000 km)."
    ),
)


# -----------------------
# LEO Lambert-sphere config
# -----------------------
# This configuration is a provisional choice for LEO objects such as
# small satellites and debris.  These values are not yet tuned to any
# specific dataset and can be refined later if NEBULA is extended to
# LEO-focused optical studies.

LAM_SPHERE_LEO_DEFAULTS = LambertSphereModelConfig(
    radius_m_default=1.0,   # Representative LEO effective radius [m]
    radius_m_min=0.3,       # Lower plausible radius [m]
    radius_m_max=2.0,       # Upper plausible radius [m]

    albedo_default=0.18,    # Representative LEO broadband albedo
    albedo_min=0.10,
    albedo_max=0.30,

    # Use the global bolometric solar constant; the G-band behavior is
    # handled by GAIA_G_* constants in this module and by the radiometry code.
    solar_irradiance_W_m2=SOLAR_CONSTANT_BOL_W_M2,

    # Bind this regime to the Gaia-like G band defined above
    band_name=GAIA_G_BAND_NAME,
    lambda_eff_nm=GAIA_G_LAMBDA_EFF_NM,
    bandwidth_nm=GAIA_G_BANDWIDTH_EFF_NM,


    mag_zero_point=None,

    notes=(
        "Representative Lambertian sphere for LEO satellites/debris. "
        "Values are approximate and serve as a starting point for "
        "LEO optical modeling; they should be revisited if LEO "
        "scenarios become a primary focus."
    ),
)


# -----------------------
# MEO Lambert-sphere config
# -----------------------
# This configuration is a placeholder for MEO objects (e.g. GNSS-like
# satellites).  As with the LEO configuration, these values are
# approximate and can be refined based on specific mission classes.

LAM_SPHERE_MEO_DEFAULTS = LambertSphereModelConfig(
    radius_m_default=1.5,   # Representative MEO effective radius [m]
    radius_m_min=0.5,
    radius_m_max=3.0,

    albedo_default=0.20,    # Representative MEO albedo
    albedo_min=0.10,
    albedo_max=0.35,

    # Use the global bolometric solar constant; the G-band behavior is
    # handled by GAIA_G_* constants in this module and by the radiometry code.
    solar_irradiance_W_m2=SOLAR_CONSTANT_BOL_W_M2,

    # Bind this regime to the Gaia-like G band defined above
    band_name=GAIA_G_BAND_NAME,
    lambda_eff_nm=GAIA_G_LAMBDA_EFF_NM,
    bandwidth_nm=GAIA_G_BANDWIDTH_EFF_NM,


    mag_zero_point=None,

    notes=(
        "Representative Lambertian sphere for MEO (e.g. GNSS) spacecraft. "
        "Parameters are approximate and intended as provisional defaults."
    ),
)


# -----------------------
# HEO Lambert-sphere config
# -----------------------
# For highly elliptical orbits, the solar irradiance can vary across
# the orbit as the spacecraft's distance to the Sun changes slightly.
# NEBULA's default configuration still uses the 1 AU solar constant
# for simplicity.  If more accuracy is needed, a HEO-specific model
# can override solar_irradiance_W_m2 as a function of time.

LAM_SPHERE_HEO_DEFAULTS = LambertSphereModelConfig(
    radius_m_default=2.0,   # Representative HEO effective radius [m]
    radius_m_min=0.5,
    radius_m_max=4.0,

    albedo_default=0.20,
    albedo_min=0.10,
    albedo_max=0.35,

    # Use the global bolometric solar constant; the G-band behavior is
    # handled by GAIA_G_* constants in this module and by the radiometry code.
    solar_irradiance_W_m2=SOLAR_CONSTANT_BOL_W_M2,

    # Bind this regime to the Gaia-like G band defined above
    band_name=GAIA_G_BAND_NAME,
    lambda_eff_nm=GAIA_G_LAMBDA_EFF_NM,
    bandwidth_nm=GAIA_G_BANDWIDTH_EFF_NM,


    mag_zero_point=None,

    notes=(
        "Representative Lambertian sphere for highly elliptical orbits. "
        "Solar irradiance is treated as constant at 1 AU; refine this "
        "if high-fidelity HEO radiometry is required."
    ),
)


# ---------------------------------------------------------------------------
# Convenience mapping: orbit regime -> Lambert-sphere config
# ---------------------------------------------------------------------------

# This dictionary allows utilities to look up the appropriate Lambertian
# sphere defaults given an orbit regime string.
#
# Example usage in another module:
#   from Configuration.NEBULA_SAT_OPTICAL_CONFIG import (
#       ORBIT_REGIME_GEO, LAM_SPHERE_DEFAULTS_BY_REGIME
#   )
#   cfg = LAM_SPHERE_DEFAULTS_BY_REGIME[ORBIT_REGIME_GEO]
#   radius = cfg.radius_m_default
LAM_SPHERE_DEFAULTS_BY_REGIME = {
    ORBIT_REGIME_LEO: LAM_SPHERE_LEO_DEFAULTS,
    ORBIT_REGIME_MEO: LAM_SPHERE_MEO_DEFAULTS,
    ORBIT_REGIME_GEO: LAM_SPHERE_GEO_DEFAULTS,
    ORBIT_REGIME_HEO: LAM_SPHERE_HEO_DEFAULTS,
}


# ---------------------------------------------------------------------------
# Human-readable description helper
# ---------------------------------------------------------------------------

def describe_lambert_sphere_config(regime: str = ORBIT_REGIME_GEO) -> str:
    """
    Build a human-readable multi-line description of the Lambertian
    sphere configuration for a given orbit regime.

    This is purely for logging / debugging.  A typical use is to call
    this once at the start of a simulation to record what optical
    assumptions NEBULA is using for a particular scenario.

    Parameters
    ----------
    regime : str
        One of the ORBIT_REGIME_* constants, e.g. ORBIT_REGIME_GEO.

    Returns
    -------
    desc : str
        Multi-line string describing the Lambertian-sphere configuration
        for the requested regime.  If an unknown regime is requested,
        a brief error message is returned instead of raising an error.
    """
    # Look up the config mapping for all regimes
    cfg_map = LAM_SPHERE_DEFAULTS_BY_REGIME

    # If the requested regime is not known, return a short error string
    if regime not in cfg_map:
        return f"[NEBULA_SAT_OPTICAL_CONFIG] Unknown regime: {regime!r}\n"

    # Extract the configuration for the requested regime
    cfg = cfg_map[regime]

    # Build a multi-line description string showing all key fields
    desc = (
        f"Lambert-sphere optical configuration for regime {regime}:\n"
        f"  model name:          {OPTICAL_MODEL_LAM_SPHERE}\n"
        f"  radius_m_default:    {cfg.radius_m_default:.3f} m\n"
        f"  radius_m_range:      "
        f"[{cfg.radius_m_min:.3f}, {cfg.radius_m_max:.3f}] m\n"
        f"  albedo_default:      {cfg.albedo_default:.3f}\n"
        f"  albedo_range:        "
        f"[{cfg.albedo_min:.3f}, {cfg.albedo_max:.3f}]\n"
        f"  solar_irradiance:    {cfg.solar_irradiance_W_m2:.1f} W/m^2\n"
        f"  band_name:           {cfg.band_name}\n"
        f"  lambda_eff_nm:       {cfg.lambda_eff_nm:.1f} nm\n"
        f"  bandwidth_nm:        {cfg.bandwidth_nm:.1f} nm\n"
        f"  mag_zero_point:      {cfg.mag_zero_point}\n"
        f"  notes:               {cfg.notes}\n"
    )

    # Return the assembled description string
    return desc
