"""
NEBULA_VISIBILITY_DISPATCHER.py

Snapshot-level dispatcher for line-of-sight (LOS) visibility regimes.

This module provides a single high-level function that, given the
instantaneous state of two satellites (position vectors and basic
orbital elements), selects the most appropriate analytic Cinelli
regime implemented in Utility.NEBULA_VISIBILITY and returns a
LOSResult describing:

    • Whether the current geometry is visible (strict h > R_BLOCK),
    • The minimum distance h from Earth's center to the LOS line,
    • A regime-dependent symmetric phasing limit |Δν|_lim [rad],
    • A string label identifying which regime was applied,
    • Optional diagnostics.

Design philosophy
=================
- This is a *snapshot* dispatcher: it evaluates one pair of states
  (observer, target) at a single epoch. A higher-level wrapper
  should loop over timesteps and accumulate visibility arrays.

- The dispatcher only decides **which analytic case** to use based
  on approximate “regime classification” tolerances (coplanar vs.
  non-coplanar, same vs. different radius, circular vs. weakly
  eccentric, etc.).

- All heavy analytic work (Cinelli §3, §4, §5) is delegated to the
  low-level functions in Utility.NEBULA_VISIBILITY.

Inputs
======
The main entry point is:

    dispatch_los_snapshot(
        r1, r2,
        a1, e1, i1, raan1, nu1,
        a2, e2, i2, raan2, nu2,
        Rb=R_BLOCK,
        custom_tolerances=None,
    )

where:

    r1, r2 : np.ndarray, shape (3,)
        Inertial position vectors of satellite 1 and 2 [km].

    a1, a2 : float
        Semi-major axes of the two orbits [km].

    e1, e2 : float
        Eccentricities of the two orbits (dimensionless).

    i1, i2 : float
        Inclinations of the two orbits [rad].

    raan1, raan2 : float
        Right ascension of ascending node (RAAN) for each orbit [rad].

    nu1, nu2 : float
        True anomalies ν₁, ν₂ for each orbit at the snapshot epoch [rad].
        These are only needed by the Cinelli §4.1 inclination case.

    Rb : float, optional
        Blocking radius [km], defaulting to NEBULA_ENV_CONFIG.R_BLOCK
        (Earth radius + atmospheric buffer). This is used in the strict
        visibility test h > Rb.

    custom_tolerances : dict or None, optional
        Optional dictionary overriding any of the default tolerance
        constants defined below (e.g. "ecc_circular_max", "ecc_weak_max").
        Values should be floats in the same units as the defaults.

Outputs
=======
LOSResult (from Utility.NEBULA_VISIBILITY):

    visible      : bool
        True iff the current geometry satisfies h > Rb.

    h            : float
        Minimum distance from Earth's center to the LOS line [km].

    delta_nu_lim : np.ndarray, shape (1,)
        Non-negative symmetric phasing limit |Δν|_lim [rad] for the
        selected regime. For the generic fallback, this is NaN.

    regime       : str
        Short label naming the applied regime, e.g.
        "coplanar_same_radius", "coplanar_different_radii",
        "noncoplanar_inclination", "noncoplanar_raan_case42",
        "weakly_eccentric/active_51", "weakly_eccentric/active_52",
        or "generic_geometry".

    fallback     : bool
        True only if a degenerate or generic fallback path is taken.

    diagnostics  : Optional[dict]
        Optional dictionary with extra information (norms, angles,
        selected regime, etc.) to help debugging and validation.
"""

# --- Standard library imports -------------------------------------------------

# Import logging so we can emit debug / warning messages about regime selection.
import logging
# Import typing helpers for type annotations on optional dict parameters.
from typing import Optional, Dict, Any

# --- Third-party imports ------------------------------------------------------

# Import numpy for vector math, norms, and transcendental functions.
import numpy as np

# --- NEBULA configuration imports --------------------------------------------

# Import the effective blocking radius R_BLOCK and the physical Earth
# radius R_EARTH from NEBULA_ENV_CONFIG.  R_BLOCK is used for the strict
# visibility test, while R_EARTH is used in Cinelli §5 analytic limits.
from Configuration.NEBULA_ENV_CONFIG import R_BLOCK, R_EARTH

# --- NEBULA visibility regime imports ----------------------------------------

# Import the LOSResult container and the geometric helper from NEBULA_VISIBILITY.
from Utility.LOS.NEBULA_VISIBILITY import (
    LOSResult,
    los_distance,
    _coplanar_same_radius,
    _coplanar_different_radii,
    _noncoplanar_inclination,
    _noncoplanar_raan_case42,
    _weakly_eccentric_case51,
    _weakly_eccentric_case52,
)


# --- Tolerance constants for regime classification ----------------------------

# Define the default relative tolerance for treating two radii as "equal".
# If |r1 - r2| / max(r1, r2) ≤ RADIUS_EQUAL_REL_MAX, we regard the orbits
# as living on the same spherical shell (for Cinelli §3.1, §4.1, §4.2, §5).
RADIUS_EQUAL_REL_MAX: float = 0.01  # 1% relative difference

# Define the relative tolerance for treating two semi-major axes as "equal".
# This is used for weakly eccentric case §5 where a₁ ≈ a₂ is required.
SMA_EQUAL_REL_MAX: float = 0.01  # 1% relative difference

# Define the eccentricity threshold below which an orbit is treated as
# effectively circular for regime classification (Cinelli §3 and §4).
ECC_CIRCULAR_MAX: float = 0.01  # |e| ≤ 0.01 ⇒ "circular"

# Define the upper bound on "weakly eccentric" orbits for Cinelli §5.
# Cinelli Eq. (27) notes that truncating at first order in e yields a
# maximum true-anomaly error of ~1° for e < 0.12.  Here we adopt a
# slightly more conservative default of 0.10, which can be changed by
# the user via custom_tolerances if desired.
ECC_WEAK_MAX: float = 0.10  # |e| ≤ 0.10 ⇒ "weakly eccentric" (Section 5)

# Define the maximum angle (in radians) between orbital planes for the
# orbits to be regarded as coplanar in the strict sense (same plane).
# This is measured via the angle between the orbital normal vectors.
PLANE_COPLANAR_MAX: float = np.deg2rad(0.5)  # 0.5 degrees

# Define the maximum RAAN difference (in radians) for which we regard
# the RAANs as "equal" (for distinguishing §3.x / §4.1 from §4.2).
RAAN_EQUAL_MAX: float = np.deg2rad(0.5)  # 0.5 degrees

# Define the maximum inclination difference (in radians) for which we
# regard two inclinations as "equal".  This is used in §4.2, which
# assumes the same inclination but different RAAN.
INCL_EQUAL_MAX: float = np.deg2rad(0.5)  # 0.5 degrees


# --- Small internal helpers ---------------------------------------------------

def _wrap_to_pi(angle: float) -> float:
    """
    Wrap a scalar angle (in radians) into the principal interval (-π, π].

    Parameters
    ----------
    angle : float
        Input angle in radians.

    Returns
    -------
    float
        Angle wrapped into (-π, π].
    """
    # Shift the angle by +π, take modulo 2π, then shift back by -π to
    # obtain a value in the interval (-π, π].
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _orbit_plane_normal(i: float, raan: float) -> np.ndarray:
    """
    Construct the unit normal vector to an orbit plane from inclination
    and RAAN (assuming a standard prograde orbit).

    Parameters
    ----------
    i : float
        Inclination [rad].
    raan : float
        Right ascension of ascending node Ω [rad].

    Returns
    -------
    np.ndarray
        3-element unit vector normal to the orbital plane in the same
        inertial frame as the input angles.
    """
    # Compute sin(i) once for reuse in the normal components.
    sin_i = np.sin(i)
    # Build the non-normalized normal vector components in ECI/TEME.
    n_vec = np.array(
        [
            sin_i * np.sin(raan),    # x-component of the orbital normal
            -sin_i * np.cos(raan),   # y-component of the orbital normal
            np.cos(i),               # z-component of the orbital normal
        ],
        dtype=float,
    )
    # Compute the norm of the normal vector to normalize it.
    n_norm = float(np.linalg.norm(n_vec))
    # Guard against degenerate inclinations (n_norm ≈ 0) by returning a
    # zero vector in that case (caller will handle coplanarity logic).
    if n_norm == 0.0:
        return np.zeros(3, dtype=float)
    # Return the unit normal vector.
    return n_vec / n_norm


def _merge_tolerances(custom: Optional[Dict[str, float]]) -> Dict[str, float]:
    """
    Merge user-provided tolerance overrides into the default tolerance
    dictionary and return the resulting mapping.

    Parameters
    ----------
    custom : dict or None
        Dictionary of tolerance name → float overrides.  Any key not
        present in this dictionary retains its default value.

    Returns
    -------
    dict
        Dictionary of effective tolerances to be used by the dispatcher.
    """
    # Start from the default tolerance values defined at the module level.
    tol: Dict[str, float] = {
        "radius_equal_rel_max": RADIUS_EQUAL_REL_MAX,
        "sma_equal_rel_max": SMA_EQUAL_REL_MAX,
        "ecc_circular_max": ECC_CIRCULAR_MAX,
        "ecc_weak_max": ECC_WEAK_MAX,
        "plane_coplanar_max": PLANE_COPLANAR_MAX,
        "raan_equal_max": RAAN_EQUAL_MAX,
        "incl_equal_max": INCL_EQUAL_MAX,
    }
    # If the caller provided a custom dict, iterate over its items and
    # replace matching keys in the tolerance dictionary.
    if custom is not None:
        for key, value in custom.items():
            tol[key] = float(value)
    # Return the effective tolerance mapping.
    return tol


# --- Main snapshot dispatcher -------------------------------------------------

def dispatch_los_snapshot(
    r1: np.ndarray,
    r2: np.ndarray,
    a1: float,
    e1: float,
    i1: float,
    raan1: float,
    nu1: float,
    a2: float,
    e2: float,
    i2: float,
    raan2: float,
    nu2: float,
    Rb: float = R_BLOCK,
    custom_tolerances: Optional[Dict[str, float]] = None,
) -> LOSResult:
    """
    Dispatch a single snapshot of two satellites into the appropriate
    Cinelli visibility regime and return a LOSResult.

    This function performs *regime classification* based on approximate
    tolerances on radius, semi-major axis, eccentricity, inclination,
    RAAN, and orbital plane alignment, then calls the corresponding
    low-level function from Utility.NEBULA_VISIBILITY:

        - §3.1 coplanar, same radius, circular        → _coplanar_same_radius
        - §3.2 coplanar, different radii, circular    → _coplanar_different_radii
        - §4.1 non-coplanar, same radius, diff. inc.  → _noncoplanar_inclination
        - §4.2 non-coplanar, same radius & inc, ΔΩ≠0  → _noncoplanar_raan_case42
        - §5 weakly eccentric (one circular, one weak)
          → combination of _weakly_eccentric_case51 and _weakly_eccentric_case52

    If none of the analytic regimes apply (e.g. strongly eccentric,
    large semi-major axis mismatch, or strongly misaligned planes),
    the dispatcher falls back to a generic geometric LOS check using
    los_distance and returns a LOSResult with regime="generic_geometry"
    and delta_nu_lim = [NaN].

    Parameters
    ----------
    r1, r2 : np.ndarray
        Inertial position vectors of the two satellites [km], shape (3,).
    a1, a2 : float
        Semi-major axes of the two orbits [km].
    e1, e2 : float
        Eccentricities of the two orbits (dimensionless).
    i1, i2 : float
        Inclinations of the two orbits [rad].
    raan1, raan2 : float
        Right ascension of the ascending node Ω for each orbit [rad].
    nu1, nu2 : float
        True anomalies ν₁, ν₂ for each orbit at the snapshot epoch [rad].
    Rb : float, optional
        Blocking radius [km] for the strict visibility test, defaulting
        to NEBULA_ENV_CONFIG.R_BLOCK.
    custom_tolerances : dict or None, optional
        Optional mapping of tolerance-name → float overriding any of
        the defaults defined in this module.

    Returns
    -------
    LOSResult
        Structured result of the LOS evaluation for the selected regime.
    """
    # Merge default tolerance values with any user-provided overrides so
    # we have a single effective tolerance dictionary for classification.
    tol = _merge_tolerances(custom_tolerances)

    # Compute the Euclidean norm of r1 [km] for radius comparison and diagnostics.
    r1_norm = float(np.linalg.norm(r1))
    # Compute the Euclidean norm of r2 [km] similarly.
    r2_norm = float(np.linalg.norm(r2))

    # If either radius is non-positive, the geometry is unphysical; in
    # that case, immediately return a degenerate LOSResult with no
    # visibility and zero phase span.
    if r1_norm <= 0.0 or r2_norm <= 0.0:
        logging.error(
            "dispatch_los_snapshot: non-positive radius (r1=%.3f, r2=%.3f); "
            "returning degenerate LOSResult.",
            r1_norm,
            r2_norm,
        )
        return LOSResult(
            visible=False,
            h=0.0,
            delta_nu_lim=np.array([0.0], dtype=float),
            regime="degenerate_input",
            fallback=True,
            diagnostics={"r1_norm": r1_norm, "r2_norm": r2_norm},
        )

    # Compute the relative difference in instantaneous radii to decide
    # whether the orbits are effectively on the same spherical shell.
    radius_rel_diff = abs(r1_norm - r2_norm) / max(r1_norm, r2_norm)
    # Decide if the radii are "equal enough" for same-radius regimes.
    same_radius = bool(radius_rel_diff <= tol["radius_equal_rel_max"])

    # Compute the relative difference in semi-major axis to decide on
    # same-altitude / same-a assumptions (Cinelli §5).
    sma_rel_diff = abs(a1 - a2) / max(a1, a2)
    # Decide if the semi-major axes are equal enough.
    same_sma = bool(sma_rel_diff <= tol["sma_equal_rel_max"])

    # Classify each orbit as effectively circular based on the magnitude
    # of its eccentricity compared to ECC_CIRCULAR_MAX.
    e1_circ = bool(abs(e1) <= tol["ecc_circular_max"])
    e2_circ = bool(abs(e2) <= tol["ecc_circular_max"])
    # Both circular if each individual orbit is circular.
    both_circular = e1_circ and e2_circ

    # Classify each orbit as "weakly eccentric" based on ECC_WEAK_MAX.
    e1_weak = bool(abs(e1) <= tol["ecc_weak_max"])
    e2_weak = bool(abs(e2) <= tol["ecc_weak_max"])
    # Both weak in the sense required for Cinelli §5.
    both_weak = e1_weak and e2_weak

    # Compute the angular separation of the orbital planes using their
    # normal vectors and treat planes as coplanar if the *unsigned*
    # angle between normals is below PLANE_COPLANAR_MAX.
    n1 = _orbit_plane_normal(i1, raan1)
    n2 = _orbit_plane_normal(i2, raan2)
    # Compute the cosine of the angle between plane normals, using the
    # absolute value to treat ±n as the same plane orientation.
    cos_plane_angle = float(
        np.clip(
            abs(float(np.dot(n1, n2))),
            0.0,
            1.0,
        )
    )
    # Recover the plane angle from its cosine.
    plane_angle = float(np.arccos(cos_plane_angle))
    # Decide if the planes are coplanar within the tolerance.
    planes_coplanar = bool(plane_angle <= tol["plane_coplanar_max"])

    # Compute the wrapped RAAN difference ΔΩ ∈ (-π, π].
    delta_omega = _wrap_to_pi(float(raan2 - raan1))
    # Take the absolute value |ΔΩ| for comparisons.
    delta_omega_abs = abs(delta_omega)
    # Decide whether the RAANs are effectively equal.
    same_raan = bool(delta_omega_abs <= tol["raan_equal_max"])

    # Compute the inclination difference Δi for diagnostic purposes and
    # for same-inclination checks in §4.2.
    delta_i = abs(float(i2 - i1))
    # Decide if the inclinations are equal within the specified tolerance.
    same_inclination = bool(delta_i <= tol["incl_equal_max"])

    # Pre-compute the generic geometric LOS altitude so that the generic
    # fallback and some diagnostics can reuse it without recomputing.
    h_line = float(los_distance(r1, r2))
    # Apply the strict NEBULA visibility rule: visible iff h_line > Rb.
    visible_generic = bool(h_line > Rb)

    # --- Regime 1: Weakly eccentric configuration (Cinelli §5) ---------------

    # In Section 5, S1 is circular (e ≈ 0) and S2 is weakly eccentric
    # with the same semi-major axis a; the orbits are coplanar.  We
    # generalize slightly by:
    #   - requiring same_sma and planes_coplanar,
    #   - requiring both orbits to be "weak" in |e|,
    #   - treating the smaller-e orbit as the circular reference.
    if same_sma and planes_coplanar and both_weak:
        # Identify which orbit is "more circular" and which is "more eccentric".
        if abs(e1) <= abs(e2):
            # Orbit 1 is more circular; orbit 2 is the weakly eccentric S2.
            e_circ = e1
            e_weak = e2
        else:
            # Orbit 2 is more circular; orbit 1 is the weakly eccentric S2.
            e_circ = e2
            e_weak = e1

        # Only apply §5 if at least one orbit is "circular enough" and
        # the other is not purely circular (otherwise §3.x is better).
        if abs(e_circ) <= tol["ecc_circular_max"] and abs(e_weak) > tol["ecc_circular_max"]:
            # Choose a reference semi-major axis based on the average of a1 and a2.
            r_ref = 0.5 * (float(a1) + float(a2))
            # Use the more eccentric magnitude as the Section 5 eccentricity.
            e_ref = float(e_weak)

            # Compute case 5.1 LOSResult using the weakly eccentric helper.
            res51 = _weakly_eccentric_case51(
                r1=r1,
                r2=r2,
                Rb=Rb,
                r=r_ref,
                e=e_ref,
            )
            # Compute case 5.2 LOSResult using the weakly eccentric helper.
            res52 = _weakly_eccentric_case52(
                r1=r1,
                r2=r2,
                Rb=Rb,
                r=r_ref,
                e=e_ref,
                R_E=R_EARTH,  # use physical Earth radius in Eq. (35)
            )

            # Extract the scalar phase limits from each case, guarding
            # against empty arrays or NaNs by treating them as +∞.
            try:
                lim51 = float(res51.delta_nu_lim[0])
            except Exception:
                lim51 = float("inf")
            try:
                lim52 = float(res52.delta_nu_lim[0])
            except Exception:
                lim52 = float("inf")

            # Decide which branch yields the tighter (smaller) admissible
            # symmetric phase bound |Δν|_lim.
            if lim51 <= lim52:
                active_lim = lim51
                active_label = "weakly_eccentric/active_51"
            else:
                active_lim = lim52
                active_label = "weakly_eccentric/active_52"

            # Overwrite visibility with a fresh strict check using the
            # generic LOS altitude to avoid minor inconsistencies.
            visible = bool(h_line > Rb)
            # Package the active phase limit into a 1D array.
            delta_nu_lim_vec = np.array([active_lim], dtype=float)

            # Return a combined LOSResult that respects the strict
            # visibility criterion and the more restrictive of the two
            # analytic weak-eccentric phasing bounds.
            return LOSResult(
                visible=visible,
                h=h_line,
                delta_nu_lim=delta_nu_lim_vec,
                regime=active_label,
                fallback=False,
                diagnostics={
                    "r1_norm": r1_norm,
                    "r2_norm": r2_norm,
                    "a1": float(a1),
                    "a2": float(a2),
                    "e1": float(e1),
                    "e2": float(e2),
                    "r_ref": r_ref,
                    "e_ref": e_ref,
                    "delta_omega_abs": delta_omega_abs,
                    "plane_angle": plane_angle,
                    "delta_i": delta_i,
                    "Rb": Rb,
                    "h_line": h_line,
                    "delta_nu_lim_51": lim51,
                    "delta_nu_lim_52": lim52,
                },
            )

    # --- Regime 2: Circular orbits (Cinelli §3 and §4) -----------------------

    # For circular regimes, require both orbits to be effectively circular
    # according to ECC_CIRCULAR_MAX.
    if both_circular:
        # Subregime 2a: Coplanar circular orbits (Cinelli §3).
        if planes_coplanar:
            # If radii are equal within tolerance, use §3.1 same-radius case.
            if same_radius:
                return _coplanar_same_radius(
                    r1=r1,
                    r2=r2,
                    Rb=Rb,
                )
            # Otherwise use §3.2 different-radius case.
            else:
                return _coplanar_different_radii(
                    r1=r1,
                    r2=r2,
                    Rb=Rb,
                )

        # Subregime 2b: Non-coplanar circular orbits (Cinelli §4).
        # Here the orbital planes differ appreciably; distinguish between
        # different-inclination (§4.1) and different-RAAN (§4.2) cases.
        else:
            # If radii are equal and RAANs are effectively equal, the
            # configuration matches Cinelli §4.1 (same RAAN, different i).
            if same_radius and same_raan:
                return _noncoplanar_inclination(
                    r1=r1,
                    r2=r2,
                    Rb=Rb,
                    nu1=float(nu1),
                    nu2=float(nu2),
                    i1=float(i1),
                    i2=float(i2),
                )

            # If radii are equal and inclinations are equal but RAANs differ,
            # the configuration matches Cinelli §4.2 (same i, different RAAN).
            if same_radius and same_inclination and (not same_raan):
                return _noncoplanar_raan_case42(
                    r1=r1,
                    r2=r2,
                    Rb=Rb,
                    incl1=float(i1),
                    incl2=float(i2),
                    raan1=float(raan1),
                    raan2=float(raan2),
                )

    # --- Generic fallback: arbitrary geometry (no analytic regime) -----------

    # If none of the analytic regimes apply (e.g., strongly eccentric or
    # misaligned without a clean Cinelli analogue), fall back to a purely
    # geometric LOS test: visible ⇔ h_line > Rb, with no analytic phase
    # limit (delta_nu_lim = NaN).
    visible = visible_generic
    # Use NaN for the phase limit to indicate "no analytic bound".
    delta_nu_lim_vec = np.array([float("nan")], dtype=float)

    # Emit a debug message so it is easy to trace when the generic
    # fallback is used during simulation runs.
    logging.debug(
        "dispatch_los_snapshot: falling back to generic_geometry regime "
        "(planes_coplanar=%s, both_circular=%s, same_radius=%s, "
        "same_sma=%s, both_weak=%s, delta_omega_abs=%.6f, delta_i=%.6f)",
        planes_coplanar,
        both_circular,
        same_radius,
        same_sma,
        both_weak,
        delta_omega_abs,
        delta_i,
    )

    # Return the generic LOSResult with basic diagnostics.
    return LOSResult(
        visible=visible,
        h=h_line,
        delta_nu_lim=delta_nu_lim_vec,
        regime="generic_geometry",
        fallback=True,
        diagnostics={
            "r1_norm": r1_norm,
            "r2_norm": r2_norm,
            "a1": float(a1),
            "a2": float(a2),
            "e1": float(e1),
            "e2": float(e2),
            "i1": float(i1),
            "i2": float(i2),
            "raan1": float(raan1),
            "raan2": float(raan2),
            "nu1": float(nu1),
            "nu2": float(nu2),
            "radius_rel_diff": radius_rel_diff,
            "sma_rel_diff": sma_rel_diff,
            "planes_coplanar": planes_coplanar,
            "delta_omega_abs": delta_omega_abs,
            "delta_i": delta_i,
            "Rb": Rb,
            "h_line": h_line,
        },
    )
