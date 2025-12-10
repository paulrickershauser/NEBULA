"""
NEBULA_FLUX
===========

Lambertian-sphere radiometry in Gaia G band for a single observer–target pair.

This module is designed to operate *after* NEBULA_SKYFIELD_ILLUMINATION has
been run. That earlier step:

    • Propagates satellites and observer
    • Uses Skyfield to determine:
        - whether the target is sunlit (illum_is_sunlit)
        - phase angle α(t) (illum_phase_angle_rad)
        - fraction illuminated (illum_fraction_illuminated)

Those illumination fields are stored in per-target track dictionaries and
typically pickled out as, e.g.:

    target_tracks_with_vis_and_illum.pkl

NEBULA_FLUX takes:

    • One "observer_track" dictionary (e.g., SBSS in GEO)
    • One "target_track" dictionary (e.g., a specific GEO comsat)

that already contain the illumination fields, and computes:

    • G-band reflected flux at the observer (W m⁻²)
    • G-band photon flux at the observer (photons m⁻² s⁻¹)
    • Apparent Gaia G magnitude of the satellite

using a simple Lambertian-sphere model in the Gaia G band.

New arrays written back into the target_track are:

    • "rad_range_obs_sat_m"          – observer–target range Δ(t) [m]
    • "rad_lambert_phase_function"   – Φ_L(α) dimensionless phase function
    • "rad_flux_g_w_m2"              – band-integrated G flux at observer [W m⁻²]
    • "rad_photon_flux_g_m2_s"       – G-band photon flux at observer
                                       [photons m⁻² s⁻¹]
    • "rad_app_mag_g"                – apparent Gaia G magnitude of target

Core assumptions and equations
------------------------------

1) Lambertian phase function for a sphere
   --------------------------------------
We adopt the standard disk-integrated Lambertian phase function Φ_L(α):

    Φ_L(α) = [ sin(α) + (π − α) cos(α) ] / π

where α is the Sun–target–observer phase angle in radians, 0 ≤ α ≤ π.

This expression goes back to Russell (1916) and is now standard in planetary
and exoplanet reflection modeling (e.g., Seager 2010, "Exoplanet
Atmospheres"). It is normalized so that Φ_L(0) = 1 (full phase) and
Φ_L(π) = 0 (new phase).

2) Basic reflected-flux scaling for a small body
   ---------------------------------------------
For an unresolved Lambertian sphere with radius R and geometric albedo A_G
illuminated by the Sun and viewed at distance Δ, a first-order flux model is:

    F_sat,G ≈ F_⊙,G(1 AU) · A_G · Φ_L(α) · (R² / Δ²),

where:
    • F_sat,G is the G-band flux at the observer [W m⁻²]
    • F_⊙,G(1 AU) is the Sun's G-band flux at 1 AU [W m⁻²]
    • A_G is "geometric albedo" in the Gaia G band (0–1, effective parameter)
    • Φ_L(α) is the Lambertian phase function
    • R is satellite effective radius [m]
    • Δ is observer–satellite range [m]

The more general small-body photometry formula includes an explicit Sun–target
distance r (∝ 1/r²). For near-Earth or GEO satellites, r ≈ 1 AU and r-variation
modulates the flux only at the few-percent level, well below other modeling
uncertainties in a first-cut GEO reflections model.

3) Magnitude–flux relation and Sun as internal G-band calibrator
   --------------------------------------------------------------
The astronomical magnitude system satisfies:

    m2 − m1 = −2.5 log10(F2 / F1),

where F is band-integrated flux and m is magnitude.

If we take the Sun as our internal Gaia G-band calibrator:

    p ≡ F_sat,G / F_⊙,G(1 AU),

then:

    m_sat,G = m_⊙,G(1 AU) − 2.5 log10(p),

where:
    • m_⊙,G(1 AU) is the Sun's apparent G magnitude at 1 AU
    • p is the satellite–Sun flux ratio in G.

Using the scaling from (2):

    p = A_G · Φ_L(α) · (R² / Δ²),

so:

    m_sat,G = m_⊙,G(1 AU)
              − 2.5 log10 [ A_G · Φ_L(α) · (R² / Δ²) ].

For Gaia, Casagrande & VandenBerg (2018) give:
    • M_G,⊙ ≈ 4.67 (absolute)
    • m_G,⊙(1 AU) ≈ −26.9 (apparent at 1 AU),

which we encode in NEBULA_SAT_OPTICAL_CONFIG as GAIA_G_M_SUN_ABS and
GAIA_G_M_SUN_APP_1AU.

4) Photon flux approximation from band-integrated flux
   ---------------------------------------------------
For band-integrated energy flux F_G [W m⁻²], a simple "effective" photon rate is:

    Ṅ_G ≈ F_G / E_photon,G,

where E_photon,G = h c / λ_eff,G, with λ_eff,G ≈ 620–670 nm for the Gaia
G band for Sun-like spectra. This collapses Gaia's full band integration and
passband calibration into an effective wavelength and photon energy,
appropriate for a first-cut instrument model.

5) Solar flux normalization
   -------------------------
For absolute fluxes we need F_⊙,G(1 AU). As a first approximation we use:

    F_⊙,G(1 AU) ≈ SOLAR_CONSTANT_BOL_W_M2,

i.e., the total bolometric solar constant at 1 AU (~1361 W m⁻²). In reality,
only a fraction of this power lies in the Gaia G band; we therefore interpret
A_G as an "effective G-band albedo" that implicitly absorbs this factor.

Later, you can refine this by introducing GAIA_G_IRRADIANCE_1AU_W_M2 in the
optical config (true G-band solar irradiance from a solar SED + Gaia G
passband) and using that here.

Inputs and outputs (NEBULA context)
-----------------------------------
The main user-facing function is:

    attach_lambertian_radiometry_to_target(observer_track, target_track, ...)

Inputs:
    observer_track : dict
        NEBULA track dictionary for the observer, with at least:
            • "times"   – 1D array of times
            • "r_eci_km"     – (N, 3) ECI position in km

    target_track : dict
        NEBULA track dictionary for the target, with at least:
            • "times"                – same length as observer
            • "r_eci_km"                  – (N, 3) ECI position in km
            • "illum_is_sunlit"           – boolean array from
                                             NEBULA_SKYFIELD_ILLUMINATION
            • "illum_phase_angle_rad"     – phase angle α(t) in radians
            • "illum_fraction_illuminated"– fraction illuminated (unused here,
                                             but checked for consistency)

        Optionally, per-target optical properties (scalars):
            • "optical_radius_m"          – effective Lambertian radius [m]
            • "optical_geometric_albedo_g"– effective G-band albedo A_G

    eta_eff : float, optional
        Overall throughput/quantum-efficiency factor η, combining optics
        transmission and detector quantum efficiency (0–1). Use η = 1.0
        if you want photon flux at the entrance pupil.

    logger : logging.Logger, optional
        Logger for info/debug messages. If None, a module-level logger is used.

Outputs:
    LambertianRadiometryResult dataclass instance, *and* side-effects on
    target_track (arrays added as new keys).

    Result fields:
        • times
        • is_sunlit
        • phase_angle_rad
        • lambert_phase_function
        • range_obs_sat_m
        • flux_g_w_m2
        • photon_flux_g_m2_s
        • app_mag_g

    target_track keys added:
        • "rad_range_obs_sat_m"
        • "rad_lambert_phase_function"
        • "rad_flux_g_w_m2"
        • "rad_photon_flux_g_m2_s"
        • "rad_app_mag_g"

This module is intentionally **one observer–one target**. A higher-level
wrapper elsewhere in NEBULA can iterate over many targets for a single
observer and then re-pickle the augmented target_tracks dictionary.

"""

# ---------------------------------------------------------------------------
# Standard library imports
# ---------------------------------------------------------------------------

# Import logging to provide informative messages and debugging hooks.
import logging

# Import dataclass to define simple containers for radiometry outputs.
from dataclasses import dataclass

# Import typing helpers for type hints.
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Third-party imports
# ---------------------------------------------------------------------------

# Import numpy for numerical array handling.
import numpy as np

# ---------------------------------------------------------------------------
# NEBULA configuration imports
# ---------------------------------------------------------------------------

# Import environment constants (e.g., AU) if needed for later scaling.
# from Configuration.NEBULA_ENV_CONFIG import ASTRONOMICAL_UNIT_M

# Import optical configuration, including Gaia G-band constants and defaults.
from Configuration.NEBULA_SAT_OPTICAL_CONFIG import (
    GAIA_G_LAMBDA_EFF_NM,       # Effective G-band wavelength [nm]
    GAIA_G_M_SUN_APP_1AU,       # Sun's apparent G magnitude at 1 AU
    SOLAR_CONSTANT_BOL_W_M2,    # Bolometric solar constant at 1 AU [W m⁻²]
    PLANCK_H_J_S,               # Planck's constant [J s]
    SPEED_OF_LIGHT_M_S,         # Speed of light [m s⁻¹]
    LAM_SPHERE_GEO_DEFAULTS,    # Lambertian-sphere defaults for GEO regime
)

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------

# Create a logger specific to this module; external code can adjust its level.
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Derived Gaia G-band constants for photon-energy calculations
# ---------------------------------------------------------------------------

# Convert Gaia G effective wavelength from nanometers to meters.
GAIA_G_LAMBDA_EFF_M = GAIA_G_LAMBDA_EFF_NM * 1e-9

# Compute effective photon energy in the G band:
#   E_photon = h * c / λ_eff
GAIA_G_PHOTON_ENERGY_J = (
    PLANCK_H_J_S * SPEED_OF_LIGHT_M_S / GAIA_G_LAMBDA_EFF_M
)

# For now, use the bolometric solar constant as an approximate G-band solar
# irradiance normalization at 1 AU. See detailed discussion in the module
# docstring; later you can replace this with a proper GAIA_G_IRRADIANCE_1AU_W_M2.
F_SUN_G_1AU_W_M2 = SOLAR_CONSTANT_BOL_W_M2

# ---------------------------------------------------------------------------
# Small numerical epsilon to avoid log10(0) and division by zero
# ---------------------------------------------------------------------------

NUM_EPS = 1e-30  # Dimensionless small number for safe log10 / division

# ---------------------------------------------------------------------------
# Data container for radiometry results
# ---------------------------------------------------------------------------

@dataclass
class LambertianRadiometryResult:
    """
    Container for Lambertian-sphere radiometry results in Gaia G band.

    Attributes
    ----------
    times : np.ndarray
        1D array of time stamps (same as input tracks).

    is_sunlit : np.ndarray
        Boolean array indicating whether the satellite is sunlit (from
        NEBULA_SKYFIELD_ILLUMINATION / Skyfield).

    phase_angle_rad : np.ndarray
        Phase angle α(t) in radians (Sun–target–observer).

    lambert_phase_function : np.ndarray
        Φ_L(α) for each time step (dimensionless).

    range_obs_sat_m : np.ndarray
        Observer–satellite range Δ(t) in meters.

    flux_g_w_m2 : np.ndarray
        Reflected Gaia G-band flux at the observer [W m⁻²].

    photon_flux_g_m2_s : np.ndarray
        Photon flux in G band at the observer [photons m⁻² s⁻¹], after
        applying any effective throughput/quantum-efficiency factor η.

    app_mag_g : np.ndarray
        Apparent Gaia G-band magnitude of the satellite at each timestep.
        Non-sunlit samples are set to +inf (effectively "invisible").
    """
    times: np.ndarray
    is_sunlit: np.ndarray
    phase_angle_rad: np.ndarray
    lambert_phase_function: np.ndarray
    range_obs_sat_m: np.ndarray
    flux_g_w_m2: np.ndarray
    photon_flux_g_m2_s: np.ndarray
    app_mag_g: np.ndarray


# ---------------------------------------------------------------------------
# Internal helpers to access / set track fields
# ---------------------------------------------------------------------------

def _get_track_field(
    track: Dict[str, Any],
    key: str,
    required: bool = True,
    default: Optional[Any] = None,
) -> Any:
    """
    Internal helper to read a field from a track dictionary.

    Parameters
    ----------
    track : dict
        NEBULA track dictionary (observer or target).

    key : str
        Name of the field to retrieve.

    required : bool, optional
        If True and the key is missing, raise a KeyError. If False and the
        key is missing, return default instead. Default is True.

    default : Any, optional
        Fallback value if key is missing and required is False.

    Returns
    -------
    Any
        Value associated with the key, or the provided default.

    Raises
    ------
    KeyError
        If required is True and the field is missing.
    """
    # If the key is present in the dictionary, return it directly.
    if key in track:
        return track[key]

    # If the key is missing and it is required, raise a clear error.
    if required:
        raise KeyError(f"Track is missing required field '{key}'")

    # Otherwise, return the provided default value.
    return default


def _set_track_field(
    track: Dict[str, Any],
    key: str,
    value: Any,
) -> None:
    """
    Internal helper to write a field into a track dictionary.

    Parameters
    ----------
    track : dict
        NEBULA track dictionary to modify.

    key : str
        Name of the field to set.

    value : Any
        Value to assign; numpy arrays are stored as-is, scalars are stored
        directly. This function does not copy arrays; the caller should pass
        copies if isolation is needed.
    """
    # Directly assign the value to the dictionary under the given key.
    track[key] = value


# ---------------------------------------------------------------------------
# Lambertian phase function
# ---------------------------------------------------------------------------

def lambertian_phase_function(alpha_rad: np.ndarray) -> np.ndarray:
    """
    Compute the Lambertian disk phase function Φ_L(α) for a sphere.

    This implements the classic Lambertian phase function:

        Φ_L(α) = [ sin(α) + (π − α) cos(α) ] / π,

    where α is the Sun–target–observer phase angle in radians.

    Parameters
    ----------
    alpha_rad : np.ndarray
        Phase angle α in radians. Values outside [0, π] will be clipped.

    Returns
    -------
    np.ndarray
        Φ_L(α) evaluated at each α (dimensionless, 0–1).
    """
    # Convert input to a numpy array for vectorized operations.
    alpha = np.asarray(alpha_rad, dtype=float)

    # Clip α to [0, π] for numerical safety and physical consistency.
    alpha_clipped = np.clip(alpha, 0.0, np.pi)

    # Compute sin(α) for each phase angle.
    sin_alpha = np.sin(alpha_clipped)

    # Compute cos(α) for each phase angle.
    cos_alpha = np.cos(alpha_clipped)

    # Compute (π − α) for each phase angle.
    pi_minus_alpha = np.pi - alpha_clipped

    # Implement Φ_L(α) = [ sin(α) + (π − α) cos(α) ] / π.
    phi_l = (sin_alpha + pi_minus_alpha * cos_alpha) / np.pi

    # Return the resulting phase function array.
    return phi_l


# ---------------------------------------------------------------------------
# Core radiometry computation for a single observer–target pair
# ---------------------------------------------------------------------------

def compute_lambertian_radiometry_for_pair(
    observer_track: Dict[str, Any],
    target_track: Dict[str, Any],
    eta_eff: float = 1.0,
    logger: Optional[logging.Logger] = None,
) -> LambertianRadiometryResult:
    """
    Compute Gaia G-band radiometry for one observer–target pair.

    This function:
        • Extracts common times and ECI positions for observer and target.
        • Computes observer–satellite range Δ(t) from ECI positions.
        • Uses existing illumination information on the target:
            - illum_is_sunlit (boolean)
            - illum_phase_angle_rad (phase angle α)
        • Applies a Lambertian-sphere model with:
            - Effective radius R (optical_radius_m or config default)
            - Effective G-band geometric albedo A_G (optical_geometric_albedo_g
              or config default)
        • Computes:
            - Φ_L(α)               (Lambertian phase function)
            - G-band flux at observer (W m⁻²)
            - G-band photon flux (photons m⁻² s⁻¹)
            - Apparent Gaia G magnitude

    Parameters
    ----------
    observer_track : dict
        NEBULA track dictionary for the observer, containing:
            • "times" : 1D array of times
            • "r_eci_km"   : (N, 3) ECI positions [km]

    target_track : dict
        NEBULA track dictionary for the target, containing:
            • "times"                : 1D array of times
            • "r_eci_km"                  : (N, 3) ECI positions [km]
            • "illum_is_sunlit"           : boolean array
            • "illum_phase_angle_rad"     : phase angle α(t) [rad]
            • "illum_fraction_illuminated": fraction illuminated (unused here)

        Optionally, scalar optical properties:
            • "optical_radius_m"          : effective radius R [m]
            • "optical_geometric_albedo_g": effective G-band A_G

    eta_eff : float, optional
        Overall throughput/quantum-efficiency factor η (0–1). This multiplies
        the photon flux to reflect detector + optics efficiency. Default is 1.0.

    logger : logging.Logger, optional
        Logger for diagnostic messages. If None, uses the module-level logger.

    Returns
    -------
    LambertianRadiometryResult
        Dataclass containing the computed radiometric quantities.
    """
    # If no logger is provided, fall back to the module-level logger.
    log = logger or globals().get("logger", None) or logging.getLogger(__name__)

    # ----------------------------------------------------------------------
    # 1) Extract and sanity-check time and geometry
    # ----------------------------------------------------------------------

    # Load target times (assumed to be the master time grid).
    t = np.asarray(_get_track_field(target_track, "times", required=True))

    # Load observer and target ECI positions in kilometers.
    r_obs_km = np.asarray(_get_track_field(observer_track, "r_eci_km", required=True))
    r_tar_km = np.asarray(_get_track_field(target_track, "r_eci_km", required=True))

    # Check that observer and target position arrays have the same shape.
    if r_obs_km.shape != r_tar_km.shape:
        raise ValueError(
            f"Observer and target r_eci_km shapes differ: "
            f"{r_obs_km.shape} vs {r_tar_km.shape}"
        )

    # Check that the number of time samples matches the position vector length.
    if t.shape[0] != r_obs_km.shape[0]:
        raise ValueError(
            f"Timestamp length {t.shape[0]} does not match position length "
            f"{r_obs_km.shape[0]}"
        )

    # Compute observer–satellite vector in km for each timestep.
    delta_r_km = r_tar_km - r_obs_km

    # Convert observer–satellite range to meters:
    #   Δ [m] = ||delta_r|| [km] * 1e3.
    range_obs_sat_m = np.linalg.norm(delta_r_km, axis=1) * 1e3

    # ----------------------------------------------------------------------
    # 2) Extract illumination arrays from target track
    # ----------------------------------------------------------------------

    # Load boolean array indicating whether the satellite is sunlit.
    illum_is_sunlit = np.asarray(
        _get_track_field(target_track, "illum_is_sunlit", required=True),
        dtype=bool,
    )

    # Load phase angle α(t) in radians.
    phase_angle_rad = np.asarray(
        _get_track_field(target_track, "illum_phase_angle_rad", required=True),
        dtype=float,
    )

    # Optionally load fraction_illuminated for consistency checks (not used here).
    frac_illum = _get_track_field(
        target_track,
        "illum_fraction_illuminated",
        required=False,
        default=None,
    )

    # Clip α to [0, π] to ensure physical values and numerical stability.
    phase_angle_rad_clipped = np.clip(phase_angle_rad, 0.0, np.pi)

    # ----------------------------------------------------------------------
    # 3) Determine optical properties (radius R and G-band albedo A_G)
    # ----------------------------------------------------------------------

    # Try to get an effective radius R [m] from the target track; if not present,
    # fall back to the Lambertian-sphere default in the optical config.
    if "optical_radius_m" in target_track:
        radius_m = float(target_track["optical_radius_m"])
    else:
        # Use the GEO Lambert-sphere default radius from the optical config.
        radius_m = float(LAM_SPHERE_GEO_DEFAULTS.radius_m_default)

    # Similarly, get the effective G-band geometric albedo A_G; if missing, use
    # the default broadband albedo from the optical config.  Here we interpret
    # `albedo_default` in NEBULA_SAT_OPTICAL_CONFIG as the effective G-band
    # geometric albedo for the Lambertian sphere.
    if "optical_geometric_albedo_g" in target_track:
        albedo_g = float(target_track["optical_geometric_albedo_g"])
    else:
        albedo_g = float(LAM_SPHERE_GEO_DEFAULTS.albedo_default)


    # Log the optical parameters for debugging.
    log.debug(
        "Lambertian radiometry: radius_m=%.3f m, albedo_g=%.3f",
        radius_m,
        albedo_g,
    )

    # ----------------------------------------------------------------------
    # 4) Compute Lambertian phase function Φ_L(α)
    # ----------------------------------------------------------------------

    # Evaluate Φ_L(α) for each phase angle using the helper function.
    lambert_phase = lambertian_phase_function(phase_angle_rad_clipped)

    # ----------------------------------------------------------------------
    # 5) Compute flux ratio p = F_sat,G / F_⊙,G(1 AU)
    # ----------------------------------------------------------------------

    # Compute radius-to-range ratio for each time sample:
    #   (R / Δ)² is the geometric dilution factor for a small unresolved body.
    radius_over_range = radius_m / np.maximum(range_obs_sat_m, NUM_EPS)

    # Compute the squared ratio (R / Δ)².
    radius_over_range_sq = radius_over_range ** 2

    # Combine albedo, phase function, and geometric dilution:
    #   p = A_G · Φ_L(α) · (R² / Δ²).
    p_flux_ratio = albedo_g * lambert_phase * radius_over_range_sq

    # For non-sunlit samples, set the flux ratio to zero (no reflected light).
    p_flux_ratio = np.where(illum_is_sunlit, p_flux_ratio, 0.0)

    # ----------------------------------------------------------------------
    # 6) Convert flux ratio to absolute G-band flux and photon flux
    # ----------------------------------------------------------------------

    # Convert dimensionless flux ratio p into an absolute G-band flux at
    # the observer by scaling with F_⊙,G(1 AU). For now, F_⊙,G(1 AU) is
    # approximated by SOLAR_CONSTANT_BOL_W_M2; see module docstring.
    flux_g_w_m2 = F_SUN_G_1AU_W_M2 * p_flux_ratio

    # Convert energy flux to photon flux using:
    #   Ṅ_G = F_G / E_photon,G.
    photon_flux_g_m2_s = flux_g_w_m2 / max(GAIA_G_PHOTON_ENERGY_J, NUM_EPS)

    # Apply overall instrument efficiency η (optics throughput × QE).
    photon_flux_g_m2_s *= float(eta_eff)

    # ----------------------------------------------------------------------
    # 7) Convert flux ratio to Gaia G magnitude
    # ----------------------------------------------------------------------

    # Avoid log10(0) by enforcing a small floor on the flux ratio.
    p_safe = np.maximum(p_flux_ratio, NUM_EPS)

    # Compute satellite apparent magnitude in Gaia G using the Sun as the
    # reference source:
    #
    #   m_sat,G = m_⊙,G(1 AU) − 2.5 log10(p),
    #
    # where p = F_sat,G / F_⊙,G(1 AU).
    app_mag_g = GAIA_G_M_SUN_APP_1AU - 2.5 * np.log10(p_safe)

    # For non-sunlit samples, set magnitude to +inf (effectively invisible).
    app_mag_g = np.where(illum_is_sunlit & (p_flux_ratio > 0.0), app_mag_g, np.inf)

    # ----------------------------------------------------------------------
    # 8) Package results into a dataclass and return
    # ----------------------------------------------------------------------

    # Create the dataclass instance with all radiometric results.
    result = LambertianRadiometryResult(
        times=t,
        is_sunlit=illum_is_sunlit,
        phase_angle_rad=phase_angle_rad_clipped,
        lambert_phase_function=lambert_phase,
        range_obs_sat_m=range_obs_sat_m,
        flux_g_w_m2=flux_g_w_m2,
        photon_flux_g_m2_s=photon_flux_g_m2_s,
        app_mag_g=app_mag_g,
    )

    # Return the radiometry result to the caller.
    return result


# ---------------------------------------------------------------------------
# User-facing helper: attach results back to target_track
# ---------------------------------------------------------------------------

def attach_lambertian_radiometry_to_target(
    observer_track: Dict[str, Any],
    target_track: Dict[str, Any],
    eta_eff: float = 1.0,
    logger: Optional[logging.Logger] = None,
) -> LambertianRadiometryResult:
    """
    Compute Lambertian radiometry and attach results to the target track.

    This is the main entry point you will call from higher-level NEBULA code.
    It wraps compute_lambertian_radiometry_for_pair and writes the resulting
    arrays into the target_track dictionary.

    Parameters
    ----------
    observer_track : dict
        NEBULA observer track dictionary.

    target_track : dict
        NEBULA target track dictionary; will be modified in-place to include
        new radiometric fields.

    eta_eff : float, optional
        Overall throughput/quantum-efficiency factor η (0–1). Default is 1.0.

    logger : logging.Logger, optional
        Logger for diagnostic messages. If None, uses the module-level logger.

    Returns
    -------
    LambertianRadiometryResult
        Dataclass containing the computed radiometric quantities.
    """
    # Compute radiometry for this observer–target pair.
    result = compute_lambertian_radiometry_for_pair(
        observer_track=observer_track,
        target_track=target_track,
        eta_eff=eta_eff,
        logger=logger,
    )

    # Write observer–satellite range Δ(t) into the target track.
    _set_track_field(target_track, "rad_range_obs_sat_m", result.range_obs_sat_m)

    # Write the Lambertian phase function Φ_L(α) into the target track.
    _set_track_field(
        target_track,
        "rad_lambert_phase_function",
        result.lambert_phase_function,
    )

    # Write G-band flux (W m⁻²) into the target track.
    _set_track_field(target_track, "rad_flux_g_w_m2", result.flux_g_w_m2)

    # Write G-band photon flux (photons m⁻² s⁻¹) into the target track.
    _set_track_field(
        target_track,
        "rad_photon_flux_g_m2_s",
        result.photon_flux_g_m2_s,
    )

    # Write apparent Gaia G magnitude into the target track.
    _set_track_field(target_track, "rad_app_mag_g", result.app_mag_g)

    # Return the dataclass result to the caller for any further analysis.
    return result
