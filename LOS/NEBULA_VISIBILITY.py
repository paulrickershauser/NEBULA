# --- Standard library & third-party imports ----------------------------------
# Import dataclass so we can define LOSResult as a typed container.
from dataclasses import dataclass
# Import logging so we can emit warnings for degenerate or suspicious inputs.
import logging
# Import typing helpers for optional fields and flexible diagnostic payloads.
from typing import Optional, Dict, Any
# Import numpy for vector math, norms, dot products, and transcendental functions.
import numpy as np


# --- Small numeric helpers used by multiple cases -----------------------------
# Define a helper for arccos with clamped domain to avoid NaN from tiny
# floating-point overshoots (keeps the argument in [-1, 1]).
def acos_clamped(x: float) -> float:
    # First clamp the input x into the closed interval [-1, 1] using numpy.
    clamped = float(np.clip(x, -1.0, 1.0))
    # Then apply arccos to the clamped value and convert the result to a plain float.
    return float(np.arccos(clamped))


# --- Result container for LOS checks -----------------------------------------
@dataclass
class LOSResult:
    """
    Line-Of-Sight (LOS) result for a single geometric regime.

    visible      : True iff the regime’s visibility condition is satisfied (strict h > Rb).
    h            : Minimum Earth-center distance for the regime [km].
    delta_nu_lim : Non-negative phasing limit |Δν|_lim [rad] for this regime, stored as a 1D array.
    regime       : Short label naming the regime/branch used (e.g., "coplanar_same_radius").
    fallback     : True iff this result comes from a degenerate or numerically-defensive code path.
    diagnostics  : Optional extra info (e.g., r1, r2, eps, Δν, Δi) for debugging/validation.
    """

    # Store whether this particular geometric configuration is visible (h > Rb).
    visible: bool
    # Store the minimum distance from Earth's center to the line-of-sight [km].
    h: float
    # Store the non-negative phasing limit |Δν|_lim [rad] as a numpy array for consistency
    # with other regimes (usually shape (1,), but kept as an array for flexibility).
    delta_nu_lim: np.ndarray
    # Store a short label describing which analytic regime produced this result.
    regime: str
    # Flag whether this result is coming from a degenerate / fallback handling path.
    fallback: bool = False
    # Optional dictionary to hold extra diagnostic information (e.g., input norms, angles).
    diagnostics: Optional[Dict[str, Any]] = None


# --- Basic geometric helper used by multiple cases ----------------------------
# Compute the shortest distance from Earth's center to the *segment* r1→r2 (km).
def los_distance(r1: np.ndarray, r2: np.ndarray) -> float:
    """
    Minimum distance from Earth's center to the finite line-of-sight segment
    joining r1 and r2.

    This is the correct geometry for Earth occultation: we only care whether
    the finite line-of-sight between the two satellites intersects the
    blocking sphere of radius Rb, not the infinite line that extends beyond
    both spacecraft.
    """
    # Vector from S1 to S2.
    v = r2 - r1

    # Squared length of the segment.
    v2 = float(np.dot(v, v))

    # Degenerate case: r1 == r2 → segment collapses to a single point.
    if v2 == 0.0:
        # Just return the distance of that point to the origin.
        return float(np.linalg.norm(r1))

    # Parameter t* at which the infinite line r(t) = r1 + t v is closest to the origin.
    t_star = -float(np.dot(r1, v)) / v2

    # Clamp to the segment: t in [0, 1].
    if t_star < 0.0:
        t = 0.0
    elif t_star > 1.0:
        t = 1.0
    else:
        t = t_star

    # Closest point on the *segment* to the origin.
    r_closest = r1 + t * v

    # Distance from Earth's center to that closest point.
    return float(np.linalg.norm(r_closest))




# --- Coplanar, same-radius, circular (§3.1; Eqs. 2–4) ------------------------
def _coplanar_same_radius(r1: np.ndarray, r2: np.ndarray, Rb: float) -> LOSResult:
    """
    Coplanar, same-radius, circular case (Cinelli §3.1).

    Implements exactly:
      (Eq. 2)  h(r, Δν) = r * cos(Δν / 2)
      (Eq. 3)  visibility iff  h ≥ R_b     (R_b: Earth+margin blocking radius)
      (Eq. 4)  |Δν|_lim = 2 * arccos(R_b / r)   (clipped to the domain [-1, 1])

    In NEBULA we adopt a strict visibility rule h > R_b (never allow grazing),
    but keep the analytic expressions otherwise identical to Cinelli.

    Inputs
    ------
    r1, r2 : np.ndarray
        Position vectors of observer and target in a common inertial frame,
        in kilometers (km). This routine assumes |r1| ≈ |r2| by construction.
    Rb : float
        Blocking radius in kilometers (km): typically Rb = R_Earth + atmosphere buffer.

    Returns
    -------
    LOSResult
        - visible: True iff h > Rb
        - h: the computed minimum distance h in km
        - delta_nu_lim: np.array([|Δν|_lim]) in radians, non-negative
        - regime: "coplanar_same_radius"
        - fallback: True only for degenerate inputs (e.g., zero-radius vectors)
    """

    # Compute the norm of r1 [km] for use in dot products and basic guarding.
    r1_norm = float(np.linalg.norm(r1))
    # Compute the norm of r2 [km] similarly.
    r2_norm = float(np.linalg.norm(r2))

    # If either radius is exactly zero, the geometry is unphysical (satellite at Earth's center).
    if r1_norm == 0.0 or r2_norm == 0.0:
        # Emit a warning to the log so the caller can trace problematic inputs.
        logging.warning("Zero radius encountered in _coplanar_same_radius; returning fallback LOSResult.")
        # Construct a fallback LOSResult with no visibility, zero distance, and zero phase limit.
        return LOSResult(
            visible=False,                           # Force invisible for unphysical inputs.
            h=0.0,                                   # No meaningful h can be defined.
            delta_nu_lim=np.array([0.0], dtype=float),  # No allowed phasing in this regime.
            regime="coplanar_same_radius/degenerate",   # Mark this result as degenerate.
            fallback=True,                           # Explicitly indicate fallback handling.
            diagnostics={"r1_norm": r1_norm, "r2_norm": r2_norm}  # Provide norms for debugging.
        )

    # Compute the average radius r = (|r1| + |r2|)/2 [km] to smooth tiny mismatches between |r1|
    # and |r2|. In the exact Cinelli regime, the radii are identical, so this is numerically safe.
    r = 0.5 * (r1_norm + r2_norm)

    # Compute the cosine of the relative true anomaly Δν via the dot product:
    # cos(Δν) = (r1 · r2) / (|r1| * |r2|).
    cos_delta_nu = float(np.dot(r1, r2) / (r1_norm * r2_norm))
    # Clamp the cosine into [-1, 1] to guard against floating-point overshoot before arccos.
    cos_delta_nu = float(np.clip(cos_delta_nu, -1.0, 1.0))
    # Recover the principal value of Δν in radians using the clamped cosine (range [0, π]).
    delta_nu = acos_clamped(cos_delta_nu)

    # Compute h per Cinelli Eq. (2): h = r * cos(Δν / 2).
    # This is the minimum distance from Earth's center to the line joining the two satellites.
    h = r * np.cos(0.5 * delta_nu)

    # Compute the argument for the phase limit per Eq. (4): Rb / r.
    ratio = float(Rb / r)
    # If Rb > r, even Δν = 0 cannot produce h ≥ Rb, so there is no real admissible phase span.
    if ratio > 1.0:
        # In this pathological case, return a zero phasing limit to signal "no admissible Δν".
        delta_nu_lim = np.array([0.0], dtype=float)
    else:
        # Clamp the ratio into [-1, 1] and apply the analytic formula
        # |Δν|_lim = 2 * arccos(Rb / r) using the clamped argument.
        arg = float(np.clip(ratio, -1.0, 1.0))
        # Use acos_clamped for consistency, even though we already clipped the argument.
        delta_nu_lim = np.array([2.0 * acos_clamped(arg)], dtype=float)

    # Apply the visibility condition with NEBULA's strict policy: h must be strictly greater
    # than Rb. This enforces a non-grazing line-of-sight even if the analytic boundary is met.
    visible = bool(h > Rb)

    # Return the structured LOSResult for this regime with fallback=False (non-degenerate path).
    return LOSResult(
        visible=visible,                     # Whether the current configuration is visible.
        h=float(h),                          # Minimum Earth-center distance along the LOS.
        delta_nu_lim=delta_nu_lim,           # Non-negative phasing limit |Δν|_lim as a 1D array.
        regime="coplanar_same_radius",       # Label identifying this analytic regime.
        fallback=False,                      # This result came from the main (non-fallback) path.
        diagnostics={                        # Optional diagnostics for debugging/validation.
            "r1_norm": r1_norm,
            "r2_norm": r2_norm,
            "r_avg": r,
            "delta_nu": delta_nu,
            "Rb_over_r": ratio,
        },
    )



# --- Coplanar, different-radius, circular (§3.2; Eqs. 6–12) -------------------
def _coplanar_different_radii(r1: np.ndarray,
                              r2: np.ndarray,
                              Rb: float) -> LOSResult:
    """
    Coplanar, circular, *different radii* (Cinelli §3.2).

    This is a low-level regime implementation that assumes the following
    preconditions have already been checked by a higher-level dispatcher:

      • The two orbits are effectively coplanar.
      • Both orbits are nearly circular.
      • The radii are sufficiently different that the same-radius (§3.1)
        approximation is not appropriate.

    Under these assumptions, this function applies Cinelli's §3.2 formulas
    exactly, using the convention that r1 is the inner orbit (smaller radius)
    and r2 is the outer orbit (larger radius):

      • §3.2.1 (obtuse case): if |Δν| <= Δν_α with
            Δν_α = arccos(1 / (1 + ε))              [Eq. (7)]
        then the minimum distance is h = r1, and visibility is h > Rb.

      • §3.2.2 (acute case): otherwise use
            h(r, ε, Δν)                              [Eq. (10)]
            |Δν|_lim from the arccos expression      [Eq. (12)]
        and visibility is again h > Rb (strict, no grazing).

    Inputs
    ------
    r1, r2 : np.ndarray
        Position vectors of the inner and outer satellites in a common
        inertial frame, in kilometers (km). If r1 is not the inner orbit,
        this function will swap r1 and r2 internally to enforce r1 <= r2.
    Rb : float
        Blocking radius in kilometers (km): R_Earth plus any safety margin.

    Returns
    -------
    LOSResult
        visible      : True iff h > Rb under the current Δν.
        h            : Minimum Earth-center distance for the configuration [km].
        delta_nu_lim : 1D array with the non-negative symmetric bound |Δν|_lim [rad].
        regime       : String label identifying which §3.2 branch was used.
        fallback     : True only for degenerate inputs (e.g., zero-radius).
        diagnostics  : Optional dict with intermediate quantities for debugging.
    """

    # Compute the Euclidean norm of r1 [km] for use in ε and dot products.
    r1_norm = float(np.linalg.norm(r1))
    # Compute the Euclidean norm of r2 [km] similarly.
    r2_norm = float(np.linalg.norm(r2))

    # If either radius is exactly zero, the geometry is unphysical (satellite at Earth's center).
    if r1_norm == 0.0 or r2_norm == 0.0:
        # Log an error to indicate that the inputs violate basic physical constraints.
        logging.error("_coplanar_different_radii: zero-radius input; returning degenerate LOSResult.")
        # Return a degenerate LOSResult with no visibility and zero phase span.
        return LOSResult(
            visible=False,                                   # Force invisible in this regime.
            h=0.0,                                           # No meaningful minimum distance.
            delta_nu_lim=np.array([0.0], dtype=float),       # No admissible phase span.
            regime="coplanar_different_radii/degenerate",    # Mark this result as degenerate.
            fallback=True,                                   # Indicate this is a fallback path.
            diagnostics={"r1_norm": r1_norm, "r2_norm": r2_norm},  # Provide norms for debugging.
        )

    # If r1 is currently the outer orbit (larger radius), swap labels so that
    # r1 refers to the inner orbit and r2 to the outer orbit. This enforces the
    # convention r1 <= r2 used in Cinelli §3.2 and guarantees ε >= 0.
    if r1_norm > r2_norm:
        # Swap the position vectors so that r1 is always the inner orbit.
        r1, r2 = r2, r1
        # Swap the precomputed norms to remain consistent with the swapped vectors.
        r1_norm, r2_norm = r2_norm, r1_norm

    # Set the reference radius r := r1 (inner orbit) as in Cinelli Eq. (5).
    r = r1_norm
    # Compute ε := (r2 - r1)/r1, the fractional difference between the radii. [Eq. (5)]
    eps = (r2_norm - r1_norm) / r1_norm

    # Compute the cosine of the true-anomaly separation Δν via the dot product:
    # cos(Δν) = (r1 · r2) / (|r1| |r2|).
    cos_delta_nu = float(np.dot(r1, r2) / (r1_norm * r2_norm))
    # Clamp cos(Δν) into [-1, 1] to guard against floating-point drift before arccos.
    cos_delta_nu = float(np.clip(cos_delta_nu, -1.0, 1.0))
    # Recover the principal value of Δν ∈ [0, π] using the clamped cosine.
    delta_nu = acos_clamped(cos_delta_nu)
    # Store the absolute value |Δν| for comparisons with analytic limits.
    abs_delta_nu = abs(delta_nu)

    # Compute the “obtuse threshold” Δν_α = arccos(1 / (1 + ε)) from Eq. (7).
    # This is the maximum |Δν| for which the minimum distance is attained at r1.
    arg_alpha = 1.0 / (1.0 + eps)
    # Clamp the argument into [-1, 1] for numerical safety before arccos.
    arg_alpha = float(np.clip(arg_alpha, -1.0, 1.0))
    # Compute Δν_α as the non-negative principal value.
    delta_nu_alpha = acos_clamped(arg_alpha)

    # -------- Branch 1: §3.2.1 obtuse geometry (α > 90°) if |Δν| ≤ Δν_α --------

    # If the current phase separation is within the obtuse threshold, the minimum
    # Earth-center distance is simply the inner radius r1.
    if abs_delta_nu <= delta_nu_alpha:
        # Set h to the inner radius r1 (since we enforced r1 <= r2 above).
        h = r1_norm

        # The admissible symmetric phase limit for this branch is exactly Δν_α. [Eq. (7)]
        delta_nu_lim = np.array([delta_nu_alpha], dtype=float)

        # Apply NEBULA's strict visibility policy: visible ⇔ h > Rb (no grazing allowed).
        visible = bool(h > Rb)

        # Return the LOSResult for the obtuse-geometry branch with diagnostics.
        return LOSResult(
            visible=visible,                             # Whether the configuration is visible.
            h=h,                                         # Minimum distance to Earth's center [km].
            delta_nu_lim=delta_nu_lim,                   # Non-negative |Δν|_lim in radians.
            regime="coplanar_different_radii/obtuse_alpha_gt_90",  # Regime label for logging.
            fallback=False,                              # This is the nominal (non-fallback) path.
            diagnostics={                                # Optional diagnostics for debugging.
                "r1_norm": r1_norm,
                "r2_norm": r2_norm,
                "eps": eps,
                "delta_nu": delta_nu,
                "delta_nu_alpha": delta_nu_alpha,
                "Rb_over_r": Rb / r if r > 0.0 else np.inf,
            },
        )

    # -------- Branch 2: §3.2.2 acute geometry (α < 90°, β < 90°) ---------------

    # For the acute case, use h(r, ε, Δν) from Eq. (10) and |Δν|_lim from Eq. (12).

    # Compute the numerator of Eq. (10): r(1+ε) sin(Δν).
    # Since Δν ∈ [0, π], sin(Δν) ≥ 0, so no absolute value is required.
    num = r * (1.0 + eps) * np.sin(delta_nu)

    # Compute the argument of the square root in the denominator of Eq. (10):
    # 2 + 2ε + ε^2 − 2(1+ε) cos(Δν).
    den_arg = 2.0 + 2.0 * eps + eps * eps - 2.0 * (1.0 + eps) * cos_delta_nu
    # Guard against tiny negative values due to floating-point error by flooring at zero.
    den_arg = float(max(den_arg, 0.0))
    # Take the square root to obtain the denominator magnitude.
    den = float(np.sqrt(den_arg))

    # If the denominator is positive, compute h; otherwise set h to +∞ (no effective Earth block).
    h = num / den if den > 0.0 else float("inf")

    # Compute the discriminant under the square root in Eq. (12):
    # (1+ε)^2 (r^2 − Rb^2) (r^2(1+ε)^2 − Rb^2).
    term1 = (r * r) - (Rb * Rb)
    # Second factor in the discriminant: r^2(1+ε)^2 − Rb^2.
    term2 = (r * r) * (1.0 + eps) * (1.0 + eps) - (Rb * Rb)
    # Full discriminant for Eq. (12): (1+ε)^2 * term1 * term2.
    disc = (1.0 + eps) * (1.0 + eps) * term1 * term2
    # Floor the discriminant at zero to prevent negative values from roundoff.
    disc = float(max(disc, 0.0))
    # Take the square root of the discriminant.
    root = float(np.sqrt(disc))

    # Compute the denominator r^2(1+ε) appearing in Eq. (12).
    denom = (r * r) * (1.0 + eps)

    # If the denominator is non-positive (which should not happen for valid inputs),
    # fall back to a conservative bound with zero admissible phase span.
    if denom <= 0.0:
        delta_nu_lim = np.array([0.0], dtype=float)
    else:
        # Compute the argument of arccos in Eq. (12):
        # ((1+ε) Rb^2 − sqrt( ... )) / (r^2(1+ε)).
        val = ((1.0 + eps) * (Rb * Rb) - root) / denom
        # Clamp the argument into [-1, 1] to keep it in the domain of arccos.
        val = float(np.clip(val, -1.0, 1.0))
        # Compute the non-negative principal value of |Δν|_lim from Eq. (12).
        delta_nu_lim = np.array([acos_clamped(val)], dtype=float)

    # Apply NEBULA's strict visibility policy at the current Δν: visible ⇔ h > Rb.
    visible = bool(h > Rb)

    # Return the LOSResult for the acute-geometry branch with diagnostics.
    return LOSResult(
        visible=visible,                                   # Whether the configuration is visible.
        h=h,                                               # Minimum distance to Earth's center [km].
        delta_nu_lim=delta_nu_lim,                         # Non-negative |Δν|_lim in radians.
        regime="coplanar_different_radii/acute_alpha_lt_90_beta_lt_90",  # Regime label.
        fallback=False,                                    # This is the nominal (non-fallback) path.
        diagnostics={                                      # Optional diagnostics for debugging.
            "r1_norm": r1_norm,
            "r2_norm": r2_norm,
            "eps": eps,
            "delta_nu": delta_nu,
            "delta_nu_alpha": delta_nu_alpha,
            "Rb_over_r": Rb / r if r > 0.0 else np.inf,
            "den_arg": den_arg,
            "disc": disc,
        },
    )




# --- Non-coplanar, circular, different-inclination (§4.1; Eqs. 17–19) --------
def _noncoplanar_inclination(r1: np.ndarray,
                             r2: np.ndarray,
                             Rb: float,
                             nu1: float,
                             nu2: float,
                             i1: float,
                             i2: float) -> LOSResult:
    """
    Non-coplanar, circular, different-inclination case (Cinelli §4.1).

    This low-level regime assumes the caller has already selected appropriate
    orbits, i.e.:
      • Nearly circular,
      • Same radius r (to within whatever tolerance the dispatcher chooses),
      • Same RAAN (so Δi is just |i2 - i1|), but different inclinations.

    Under those assumptions, this function implements:

      Eq. (17):  h_min(r, Δν, Δi) =
                     (r / 2) * sqrt( 3 − cosΔi + cosΔν * (1 + cosΔi) )

      Eq. (18/19): h_min ≥ Rb ⇒
                     cosΔν_lim =
                        ( (2Rb/r)^2 − 3 + cosΔi ) / (1 + cosΔi ),
                     |Δν|_lim = arccos( clip(cosΔν_lim, −1, 1) )

    with NEBULA policies:
      • Strict visibility: visible ⇔ h_min > Rb (no grazing).
      • Uses Rb (Earth + margin) instead of bare R_Earth.
      • Special handling for Δi ≈ π (denominator → 0).

    Inputs
    ------
    r1, r2 : np.ndarray
        Position vectors of the two satellites in a common inertial frame [km].
        Their norms should be (almost) equal for this regime to be valid.
    Rb : float
        Blocking radius [km] (R_Earth plus any safety margin).
    nu1, nu2 : float
        True anomalies ν₁, ν₂ of satellite 1 and 2 [rad].
    i1, i2 : float
        Inclinations i₁, i₂ of the two orbits [rad].

    Returns
    -------
    LOSResult
        visible      : True iff h_min > Rb at the current (ν₁, ν₂, i₁, i₂).
        h            : h_min(r, Δν, Δi) in km.
        delta_nu_lim : np.array([ |Δν|_lim ]) in radians, non-negative.
        regime       : "noncoplanar_inclination" (or "/degenerate" on fallback).
        fallback     : True only for degenerate inputs (e.g., zero-radius).
        diagnostics  : Optional intermediate values for debugging.
    """

    # Compute the Euclidean norm of r1 [km] for use in the radius parameter r.
    r1n = float(np.linalg.norm(r1))
    # Compute the Euclidean norm of r2 [km] similarly.
    r2n = float(np.linalg.norm(r2))

    # Guard against degenerate inputs where either satellite sits at the origin.
    if r1n == 0.0 or r2n == 0.0:
        # Log an error so the caller can trace the bad geometry.
        logging.error("_noncoplanar_inclination: zero-radius input; returning degenerate LOSResult.")
        # Return a fallback LOSResult with no visibility and zero phase span.
        return LOSResult(
            visible=False,                                       # Force invisible.
            h=0.0,                                               # No meaningful minimum distance.
            delta_nu_lim=np.array([0.0], dtype=float),           # No admissible phase span.
            regime="noncoplanar_inclination/degenerate",         # Mark as degenerate.
            fallback=True,                                       # Indicate fallback handling.
            diagnostics={"r1_norm": r1n, "r2_norm": r2n},        # Provide norms for debugging.
        )

    # Choose the reference radius r [km]. In Cinelli §4.1, r1 = r2 = r exactly.
    # Here we take r := |r1| and rely on the higher-level dispatcher to ensure
    # that |r1| ≈ |r2| (equal-radius assumption) before calling this regime.
    r = r1n

    # Compute the true-anomaly separation Δν = ν₂ − ν₁ [rad].
    # Only cosΔν enters the formulas, so the sign is irrelevant (cos is even),
    # but keeping Δν is useful for diagnostics.
    delta_nu = float(nu2 - nu1)

    # Compute the inclination difference Δi = i₂ − i₁ [rad].
    # Again, only cosΔi enters, so the sign does not matter.
    delta_i = float(i2 - i1)

    # Compute cosΔν and cosΔi once for reuse.
    cos_delta_nu = float(np.cos(delta_nu))
    cos_delta_i = float(np.cos(delta_i))

    # Compute the term inside the square root in Eq. (17):
    # term = 3 − cosΔi + cosΔν * (1 + cosΔi).
    term = 3.0 - cos_delta_i + cos_delta_nu * (1.0 + cos_delta_i)
    # Clamp term from below at 0 to avoid sqrt of tiny negative values.
    term_clamped = float(np.clip(term, 0.0, None))

    # Apply Eq. (17): h_min = (r / 2) * sqrt( term ).
    h_min = (r / 2.0) * np.sqrt(term_clamped)

    # Compute the denominator appearing in Eq. (18/19): denom = 1 + cosΔi.
    denom = 1.0 + cos_delta_i

    # SPECIAL BRANCH: Δi ≈ π ⇒ denom ≈ 0, so Eq. (18/19) is singular.
    # In that limit, Eq. (17) collapses to h_min = r, independent of Δν.
    if np.isclose(denom, 0.0, atol=1e-12):
        # Apply strict visibility policy at this inclination: visible ⇔ r > Rb.
        visible = bool(r > Rb)
        # If visible, every phase is allowed ⇒ |Δν|_lim = π; else none ⇒ 0.
        delta_nu_lim_mag = np.pi if visible else 0.0

        # Return the LOSResult for this special inclination configuration.
        return LOSResult(
            visible=visible,                                     # Visible at all Δν or not at all.
            h=float(h_min),                                     # Equals r here (from Eq. 17).
            delta_nu_lim=np.array([delta_nu_lim_mag], dtype=float),  # Symmetric |Δν|_lim.
            regime="noncoplanar_inclination",                   # Regime label.
            fallback=False,                                     # Non-degenerate special case.
            diagnostics={                                       # Optional diagnostics.
                "r1_norm": r1n,
                "r2_norm": r2n,
                "r": r,
                "delta_nu": delta_nu,
                "delta_i": delta_i,
                "cos_delta_i": cos_delta_i,
            },
        )

    # GENERAL BRANCH: use Eq. (18/19) to compute the symmetric phase bound |Δν|_lim.

    # Compute (2Rb / r)^2; this is the scaled blocking term in Eq. (18/19).
    twoRb_over_r_sq = (2.0 * Rb / r) ** 2

    # Compute the numerator: (2Rb/r)^2 − 3 + cosΔi.
    numerator = twoRb_over_r_sq - 3.0 + cos_delta_i

    # Apply Eq. (18/19): cosΔν_lim = numerator / (1 + cosΔi).
    cos_delta_nu_lim = numerator / denom

    # Clamp cosΔν_lim into [-1, 1] to stay within the acos domain.
    cos_delta_nu_lim = float(np.clip(cos_delta_nu_lim, -1.0, 1.0))

    # Compute the non-negative magnitude of the symmetric phase limit:
    # |Δν|_lim = arccos( cosΔν_lim ).
    delta_nu_lim_mag = float(acos_clamped(cos_delta_nu_lim))

    # Apply strict visibility at the CURRENT phase: visible ⇔ h_min > Rb.
    visible = bool(h_min > Rb)

    # Return the LOSResult for the general different-inclination configuration.
    return LOSResult(
        visible=visible,                                       # Whether current Δν is visible.
        h=float(h_min),                                       # h_min(r, Δν, Δi) [km].
        delta_nu_lim=np.array([delta_nu_lim_mag], dtype=float),  # Symmetric |Δν|_lim [rad].
        regime="noncoplanar_inclination",                     # Regime label.
        fallback=False,                                       # Non-fallback path.
        diagnostics={                                         # Optional diagnostics.
            "r1_norm": r1n,
            "r2_norm": r2n,
            "r": r,
            "delta_nu": delta_nu,
            "delta_i": delta_i,
            "cos_delta_nu": cos_delta_nu,
            "cos_delta_i": cos_delta_i,
            "cos_delta_nu_lim": cos_delta_nu_lim,
        },
    )


# case 4.2

# case 4.2

def _arg_of_latitude(r_vec: np.ndarray, i: float, raan: float) -> float:
    """
    Argument of latitude u for a circular orbit with inclination i and RAAN Ω,
    assuming r_vec is expressed in the same inertial frame as i and Ω.
    u is measured from the ascending node.
    """
    r = float(np.linalg.norm(r_vec))
    if r == 0.0:
        raise ValueError("Zero radius in _arg_of_latitude")

    sin_i = np.sin(i)
    if abs(sin_i) < 1e-8:
        # Nearly equatorial orbit; argument of latitude is ill-conditioned.
        # Fallback: use projection into equatorial plane only.
        x, y, z = r_vec
        u = float(np.arctan2(y, x) - raan)
        # Wrap into (-π, π]
        u = (u + np.pi) % (2.0 * np.pi) - np.pi
        return u

    x, y, z = r_vec

    # cos u = (x cosΩ + y sinΩ) / r
    cos_u = (x * np.cos(raan) + y * np.sin(raan)) / r
    # sin u = z / (r sin i)
    sin_u = z / (r * sin_i)

    # Clamp to avoid tiny numerical excursions.
    cos_u = float(np.clip(cos_u, -1.0, 1.0))
    sin_u = float(np.clip(sin_u, -1.0, 1.0))

    u = float(np.arctan2(sin_u, cos_u))
    # Wrap into (-π, π]
    u = (u + np.pi) % (2.0 * np.pi) - np.pi
    return u


def _h_case42_eq21_from_vectors(r1: np.ndarray, r2: np.ndarray) -> float:
    """
    Compute the LOS altitude h for equal-radius points using the Eq. (21)
    geometry:

        h = sqrt( r^2 - ||r2 - r1||^2 / 4 )

    where r is the (approximately common) radius of r1 and r2.

    This is algebraically equivalent to Cinelli's Eq. (21) for the
    case |r1| ≈ |r2| = r, but expressed in a compact geometric form.
    """
    # Compute the norm of each position vector [km].
    r1n = float(np.linalg.norm(r1))
    r2n = float(np.linalg.norm(r2))

    # Take the average radius r to smooth tiny mismatches.
    r = 0.5 * (r1n + r2n)

    # Compute the chord vector between the two points.
    diff = r2 - r1
    # Compute the chord length d = ||r2 - r1|| [km].
    d = float(np.linalg.norm(diff))

    # Apply the altitude formula: h = sqrt( r^2 - (d^2)/4 ).
    inside = max(r * r - 0.25 * d * d, 0.0)
    h = float(np.sqrt(inside))

    return h


def _delta_nu_lim_case42_closed_form(
    r: float,
    i: float,
    delta_omega_abs: float,
    Rb: float,
    inc_polar_tol: float = 1e-6,
) -> float:
    """
    Analytic |Δν|_lim for Cinelli case 4.2 via Eq. (23) and Eq. (24).

    If |i - π/2| < inc_polar_tol, use the polar simplification Eq. (24).
    Otherwise use the full Eq. (23) with the corrected inner-bracket term:

        r^2(-3 + cos(2i)*(cos ΔΩ - 1)) - cos ΔΩ + 4 Rb^2

    Inputs
    ------
    r              : common radius [km]
    i              : common inclination [rad]
    delta_omega_abs: |ΔΩ| [rad] in [0, π]
    Rb             : blocking radius [km]
    inc_polar_tol  : tolerance around 90° for using Eq. (24)

    Returns
    -------
    delta_nu_lim_abs : principal non-negative |Δν|_lim in [0, π].
    """

    # --- Polar simplification (Eq. 24) ---------------------------------------
    if abs(i - 0.5 * np.pi) < inc_polar_tol:
        # Eq. (24): |Δν_lim| = ±2 cos^-1( ± sqrt(Rb^2 sec^2(ΔΩ/2) / r^2) ).
        half_dO = 0.5 * delta_omega_abs
        sec_half_dO_sq = 1.0 / max(np.cos(half_dO) ** 2, 1e-16)
        arg = (Rb ** 2 * sec_half_dO_sq) / (r ** 2)

        # Clip argument into [0, 1]; if >=1, no solution => 0.
        if arg >= 1.0:
            return 0.0

        root = np.sqrt(arg)
        root = float(np.clip(root, -1.0, 1.0))
        # Take the principal positive solution 2 arccos(root).
        return 2.0 * float(np.arccos(root))

    # --- General case: full Eq. (23) -----------------------------------------
    cos_2i = np.cos(2.0 * i)
    cos_4i = np.cos(4.0 * i)
    cos_dO = np.cos(delta_omega_abs)
    cos_2dO = np.cos(2.0 * delta_omega_abs)
    sin_i = np.sin(i)

    # Denominator D = r^4 [3 + cos(2i) + 2 cos ΔΩ sin^2 i]^2
    D = (r ** 4) * (3.0 + cos_2i + 2.0 * cos_dO * (sin_i ** 2)) ** 2
    if D <= 0.0:
        return 0.0

    # Base numerator part N_base (all the cos terms plus the 8 r^2 Rb^2[...] piece)
    N_base = (
        13.0 * r ** 4
        + 16.0 * r ** 4 * cos_2i
        + 3.0 * r ** 4 * cos_4i
        - 12.0 * r ** 4 * cos_dO
        - 16.0 * r ** 4 * cos_2i * cos_dO
        - 4.0 * r ** 4 * cos_4i * cos_dO
        - r ** 4 * cos_2dO
        + r ** 4 * cos_4i * cos_2dO
        + 8.0 * r ** 2 * Rb ** 2
        * (
            2.0 - 2.0 * cos_2i
            + np.cos(2.0 * i - delta_omega_abs)
            + 6.0 * cos_dO
            + np.cos(2.0 * i + delta_omega_abs)
        )
    )

    # Inner radical piece:
    # - r^4 Rb^2 cos^2 i sin^2 ΔΩ [ r^2(-3 + cos(2i)*(cos ΔΩ - 1)) - cos ΔΩ + 4 Rb^2 ]
    inner_bracket = (
        r ** 2 * (-3.0 + np.cos(2.0 * i) * (cos_dO - 1.0))  # corrected grouping
        - cos_dO
        + 4.0 * Rb ** 2
    )
    radical_inner = - (r ** 4) * (Rb ** 2) * (np.cos(i) ** 2) * (np.sin(delta_omega_abs) ** 2) * inner_bracket

    if radical_inner < 0.0:
        radical_inner = 0.0
    sqrt_term = np.sqrt(radical_inner)

    # Candidate numerators with ±32 sqrt(...).
    N_plus = N_base + 32.0 * sqrt_term
    N_minus = N_base - 32.0 * sqrt_term

    # Build candidate cos^-1 arguments from ±1/2 * sqrt( N / D ).
    candidates = []
    for N_inner in (N_plus, N_minus):
        frac = N_inner / D
        if frac < 0.0:
            continue
        base_val = 0.5 * np.sqrt(frac)  # ± 1/2 outside sqrt
        candidates.append(base_val)
        candidates.append(-base_val)

    delta_nu_candidates = []
    for cand in candidates:
        # Clamp for safety.
        arg = float(np.clip(cand, -1.0, 1.0))
        try:
            val = 2.0 * float(np.arccos(arg))
        except ValueError:
            continue
        # We care about |Δν| in [0, π].
        val_abs = abs(val)
        if 0.0 <= val_abs <= np.pi + 1e-6:
            delta_nu_candidates.append(val_abs)

    if not delta_nu_candidates:
        return 0.0

    # For now, use the largest admissible |Δν|_lim in [0, π].
    # (This matches the previous behaviour you plotted.)
    delta_nu_lim_abs = max(delta_nu_candidates)
    return float(delta_nu_lim_abs)


def _noncoplanar_raan_case42(
    r1: np.ndarray,
    r2: np.ndarray,
    Rb: float,
    incl1: float,
    incl2: float,
    raan1: float,
    raan2: float,
    root_tol: float = 1e-6,   # kept for signature compatibility; unused
    max_iter: int = 64,       # kept for signature compatibility; unused
) -> LOSResult:
    """
    Cinelli case 4.2: non-coplanar circular orbits with same radius & inclination
    but different RAAN (ΔΩ ≠ 0).

    This implementation:
      1. Treats r1, r2 as the instantaneous position vectors of S1, S2
         in a common inertial frame (TEME or ECI), in km.
      2. Uses incl1, incl2 and raan1, raan2 (rad) to recover the arguments
         of latitude u1, u2 and the current phase Δν ≈ u2 - u1.
      3. Computes:
           - h_line  = geometric LOS distance via los_distance(r1, r2),
           - |Δν|_lim = closed-form limit from Cinelli Eq. (23),
             with the polar Eq. (24) when |i - 90°| < inc_polar_tol,
             using Rb in place of R_E.
      4. Returns a LOSResult with:
           visible      = (h_line > Rb)   [strict, no grazing],
           h            = h_line,
           delta_nu_lim = np.array([|Δν|_lim]),

    Notes
    -----
    - This regime assumes:
        * r1 and r2 are on (approximately) the same circular shell: |r1| ≈ |r2|.
        * incl1 ≈ incl2 (same inclination).
        * Eccentricities are small (handled at a higher level in the dispatcher).
    - Rb is a blocking radius (Earth + atmosphere margin), so we are effectively
      applying Cinelli Eq. (23) with R_E -> Rb.
    """

    # --- Radii and sanity checks ---------------------------------------------
    r1_norm = float(np.linalg.norm(r1))
    r2_norm = float(np.linalg.norm(r2))

    if r1_norm <= 0.0 or r2_norm <= 0.0:
        logging.warning(
            "_noncoplanar_raan_case42: non-positive radius: r1=%.3f r2=%.3f",
            r1_norm,
            r2_norm,
        )
        return LOSResult(
            visible=False,
            h=0.0,
            delta_nu_lim=np.array([0.0], dtype=float),
            regime="noncoplanar_raan_case42/degenerate",
            fallback=True,
        )

    # Average radius and inclination for the analytic geometry.
    r = 0.5 * (r1_norm + r2_norm)
    i_avg = 0.5 * (float(incl1) + float(incl2))

    # Warn if inclinations differ more than expected for this regime.
    if abs(incl1 - incl2) > 1e-3:
        logging.debug(
            "_noncoplanar_raan_case42: incl1=%.6f, incl2=%.6f differ by >1e-3 rad",
            incl1,
            incl2,
        )

    # --- RAAN difference ΔΩ, wrapped and absoluted ---------------------------
    delta_omega = (float(raan2) - float(raan1) + np.pi) % (2.0 * np.pi) - np.pi
    delta_omega_abs = abs(delta_omega)

    # --- Arguments of latitude and current phase Δν --------------------------
    try:
        u1 = _arg_of_latitude(r1, float(incl1), float(raan1))
        u2 = _arg_of_latitude(r2, float(incl2), float(raan2))
    except ValueError as exc:
        logging.warning("_noncoplanar_raan_case42: %s", exc)
        return LOSResult(
            visible=False,
            h=0.0,
            delta_nu_lim=np.array([0.0], dtype=float),
            regime="noncoplanar_raan_case42/degenerate",
            fallback=True,
        )

    delta_nu = (u2 - u1 + np.pi) % (2.0 * np.pi) - np.pi
    delta_nu_abs = abs(delta_nu)

    # --- Exact LOS distance from the line r1->r2 -----------------------------
    h_line = float(los_distance(r1, r2))

    # --- Analytic |Δν|_lim via Eq. (23)/(24) ---------------------------------
    delta_nu_lim_analytic = _delta_nu_lim_case42_closed_form(
        r=r,
        i=i_avg,
        delta_omega_abs=delta_omega_abs,
        Rb=Rb,
        inc_polar_tol=1e-6,
    )

    visible = bool(h_line > Rb)

    # Pack the closed-form limit into delta_nu_lim (single-entry array).
    delta_nu_lim_vec = np.array([delta_nu_lim_analytic], dtype=float)

    return LOSResult(
        visible=visible,
        h=h_line,
        delta_nu_lim=delta_nu_lim_vec,
        regime="noncoplanar_raan_case42",
        fallback=False,
        diagnostics={
            "r1_norm": r1_norm,
            "r2_norm": r2_norm,
            "r_avg": r,
            "i_avg": i_avg,
            "delta_omega_abs": delta_omega_abs,
            "delta_nu_current_abs": delta_nu_abs,
            "h_eq21": _h_case42_eq21_from_vectors(r1, r2),
        },
    )


# # Optional standalone test for case 4.2
# if __name__ == "__main__":
#     import numpy as np

#     # Example: LEO-like shell with small RAAN offset.
#     r_mag = 7000.0          # orbit radius [km]
#     i_deg = 53.0            # inclination [deg]
#     dO_deg = 30.0           # RAAN difference ΔΩ [deg]
#     Rb = 6378.137 + 50.0    # blocking radius [km] (Earth + 50 km margin)

#     i_rad = np.deg2rad(i_deg)
#     dO_rad = np.deg2rad(dO_deg)

#     # Choose an along-track phase offset Δν₀ [deg] for the test geometry.
#     dnu0_deg = 10.0
#     dnu0_rad = np.deg2rad(dnu0_deg)

#     # S1 on orbit 1 with RAAN Ω1 = 0 and argument of latitude u1 = 0.
#     u1 = 0.0
#     x1 = r_mag * np.cos(u1)
#     y1 = r_mag * np.cos(i_rad) * np.sin(u1)
#     z1 = r_mag * np.sin(i_rad) * np.sin(u1)
#     r1 = np.array([x1, y1, z1], dtype=float)

#     # S2 on orbit 2 with RAAN Ω2 = ΔΩ and argument of latitude u2 = Δν₀.
#     u2 = dnu0_rad
#     cos_u2 = np.cos(u2)
#     sin_u2 = np.sin(u2)
#     cos_dO = np.cos(dO_rad)
#     sin_dO = np.sin(dO_rad)

#     x2 = r_mag * (cos_u2 * cos_dO - sin_u2 * np.cos(i_rad) * sin_dO)
#     y2 = r_mag * (cos_u2 * sin_dO + sin_u2 * np.cos(i_rad) * cos_dO)
#     z2 = r_mag * (sin_u2 * np.sin(i_rad))
#     r2 = np.array([x2, y2, z2], dtype=float)

#     res = _noncoplanar_raan_case42(
#         r1=r1,
#         r2=r2,
#         Rb=Rb,
#         incl1=i_rad,
#         incl2=i_rad,
#         raan1=0.0,
#         raan2=dO_rad,
#     )

#     h_vec = los_distance(r1, r2)
#     h_eq21 = _h_case42_eq21_from_vectors(r1, r2)

#     print("=== Case 4.2 test (closed-form only) ===")
#     print(f"r = {r_mag:.3f} km, i = {i_deg:.3f} deg, ΔΩ = {dO_deg:.3f} deg")
#     print(f"Rb = {Rb:.3f} km")
#     print(f"h_line (from _noncoplanar_raan_case42) = {res.h:.6f} km")
#     print(f"h_line (direct los_distance)          = {h_vec:.6f} km")
#     print(f"h_line (Eq. 21 altitude form)         = {h_eq21:.6f} km")
#     print(f"|Δν|_lim analytic   = {np.rad2deg(res.delta_nu_lim[0]):.6f} deg")
#     print(f"visible (h_line>Rb) = {res.visible}")


#Case 5 weakly eccentric

def delta_M_lim_case_51(e: float) -> float:
    """
    Case 5.1 (α > 90°), Cinelli Eq. (31).
    
    Parameters
    ----------
    e : float
        Eccentricity of S2 (dimensionless, small e).
    
    Returns
    -------
    float
        ΔM_lim in radians, from Eq. (31):
        ΔM_lim = arccos( (1 - e sin(2e)) / (1 - e^2) ) - 2e.
    """
    # Eq. (31): argument of arccos
    num = 1.0 - e * np.sin(2.0 * e)
    den = 1.0 - e**2
    arg = num / den if den != 0.0 else np.nan

    # Numerical safety: clip to valid domain of arccos
    arg = np.clip(arg, -1.0, 1.0)

    # Direct transcription of Eq. (31)
    delta_M_lim = np.arccos(arg) - 2.0 * e
    return delta_M_lim

def delta_r_case_52(r: float, e: float, delta_f: float) -> float:
    """
    Case 5.2, distance between satellites, Cinelli Eq. (32).

    Δr = r * sqrt(
            1
            + (e^2 - 1)^2 / [1 - e sin(2e)]^2
            - 2 (e^2 - 1) cos(Δf) / [e sin(2e) - 1]
         )

    Parameters
    ----------
    r : float
        Reference radius (same as a2 = r in the paper).
    e : float
        Eccentricity of S2 (dimensionless, small e).
    delta_f : float
        True-anomaly phasing Δf (rad).

    Returns
    -------
    float
        Δr from Eq. (32), same units as r.
    """
    denom_1 = 1.0 - e * np.sin(2.0 * e)   # [1 - e sin(2e)]
    denom_2 = e * np.sin(2.0 * e) - 1.0   # [e sin(2e) - 1]

    # Inner bracket of the square root in Eq. (32)
    inner = (
        1.0
        + (e**2 - 1.0)**2 / (denom_1**2)
        - 2.0 * (e**2 - 1.0) * np.cos(delta_f) / denom_2
    )

    # Numerical safety: avoid small negative due to roundoff
    inner = np.clip(inner, 0.0, None)

    delta_r = r * np.sqrt(inner)
    return delta_r

def area_OS1S2_case_52(r: float, e: float, delta_f: float) -> float:
    """
    Case 5.2, triangle area A_OS1S2, Cinelli Eq. (33).

    A_OS1S2 = 1/2 * r^2 * sqrt(
                    (e^2 - 1)^2 * sin^2(Δf)
                    / [1 - e sin(2e)]^2
                )

    Parameters
    ----------
    r : float
        Reference radius.
    e : float
        Eccentricity of S2 (dimensionless).
    delta_f : float
        True-anomaly phasing Δf (rad).

    Returns
    -------
    float
        A_OS1S2 from Eq. (33), in units of r^2.
    """
    denom_1 = 1.0 - e * np.sin(2.0 * e)   # [1 - e sin(2e)]

    inner = (
        (e**2 - 1.0)**2
        * (np.sin(delta_f)**2)
        / (denom_1**2)
    )
    inner = np.clip(inner, 0.0, None)

    A_OS1S2 = 0.5 * r**2 * np.sqrt(inner)
    return A_OS1S2

def h_case_52(r: float, e: float, delta_f: float) -> float:
    """
    Case 5.2, height h(r, e, Δf), Cinelli Eq. (34).

    We implement the definition:
        h = 2 * A_OS1S2 / Δr
    where A_OS1S2 is Eq. (33) and Δr is Eq. (32).

    Parameters
    ----------
    r : float
        Reference radius.
    e : float
        Eccentricity of S2 (dimensionless).
    delta_f : float
        True-anomaly phasing Δf (rad).

    Returns
    -------
    float
        h(r, e, Δf), same units as r.
    """
    delta_r = delta_r_case_52(r, e, delta_f)
    if delta_r == 0.0:
        # Degenerate configuration; in practice shouldn't happen
        return 0.0

    A = area_OS1S2_case_52(r, e, delta_f)

    # Eq. (34): h = 2 A / Δr
    h = 2.0 * A / delta_r
    return h

def delta_f_M_lim_case_52(r: float, e: float, R_E: float) -> tuple[float, float]:
    """
    Case 5.2, limit phasing in true and mean anomaly, Cinelli Eqs. (35)–(36).

    Implements:
        Δf_lim = arccos( ARG )       [taking the positive magnitude branch]
        ΔM_lim = Δf_lim - 2e

    where ARG is the bracketed expression in Eq. (35).

    Parameters
    ----------
    r : float
        Reference radius (same units as R_E).
    e : float
        Eccentricity of S2 (dimensionless, small e).
    R_E : float
        Earth radius RE (same units as r).

    Returns
    -------
    (Δf_lim, ΔM_lim) : tuple of floats
        Δf_lim  : magnitude of the limit true-anomaly phasing (rad), Eq. (35).
        ΔM_lim  : limit mean-anomaly phasing (rad), Eq. (36).
    """
    one_minus_e2 = 1.0 - e**2
    one_minus_e2_sq = one_minus_e2**2

    # termA = R_E^2 (e^2 - 1) [e sin(2e) - 1]
    termA = (R_E**2) * (e**2 - 1.0) * (e * np.sin(2.0 * e) - 1.0)

    # Inner pieces inside the big square root in Eq. (35)
    inner1 = 2.0 * one_minus_e2_sq * (r**4)
    inner2 = (r**2) * (R_E**2) * (
        -2.0 * e**4 + 3.0 * e**2 + e**2 * np.cos(4.0 * e) + 4.0 * e * np.sin(2.0 * e) - 4.0
    )
    inner3 = 2.0 * (R_E**4) * (e * np.sin(2.0 * e) - 1.0)**2

    # big_inner = (1 - e^2)^2 * [ 2(1 - e^2)^2 r^4 + r^2 R_E^2 (...) + 2 R_E^4 (...)^2 ]
    big_inner = one_minus_e2_sq * (inner1 + inner2 + inner3)
    big_inner = np.clip(big_inner, 0.0, None)

    sqrt_term = np.sqrt(big_inner)

    # Numerator inside the {...} in Eq. (35)
    numerator = termA - (1.0 / np.sqrt(2.0)) * sqrt_term

    # Denominator: r^2 (1 - e^2)^2
    denom = (r**2) * (one_minus_e2_sq)

    if denom == 0.0:
        # Degenerate case; in practice, should not happen for e < 1 and r > 0
        return np.nan, np.nan

    arg = numerator / denom

    # Clip into domain of arccos
    arg = np.clip(arg, -1.0, 1.0)

    # Eq. (35): we take the principal magnitude branch (0 ≤ Δf_lim ≤ π)
    delta_f_lim = np.arccos(arg)

    # Eq. (36): ΔM_lim = Δf_lim - 2e
    delta_M_lim = delta_f_lim - 2.0 * e

    return delta_f_lim, delta_M_lim

def _weakly_eccentric_case51(
    r1: np.ndarray,
    r2: np.ndarray,
    Rb: float,
    r: float,
    e: float,
) -> LOSResult:
    """
    Cinelli case 5.1: weakly eccentric coplanar orbits with α > 90° (Eq. 31).

    Geometry / assumptions
    ----------------------
    - S1 on a circular orbit with semi-major axis a₁ = r, e₁ = 0.
    - S2 on a weakly eccentric orbit with a₂ = r, eccentricity e (small).
    - Same orbital plane (coplanar).
    - Case 5.1 assumes that, in the *worst condition* used to derive the
      permanent-visibility bound, the triangle angle α at S1 satisfies α > 90°.

    What this regime does
    ---------------------
    1. Uses the instantaneous inertial position vectors r1, r2 [km] to compute
       the current line-of-sight altitude:
           h_line = los_distance(r1, r2)
       i.e. the distance from Earth’s center to the line joining S1 and S2.

    2. Applies NEBULA’s strict visibility rule:
           visible ⇔ h_line > Rb
       where Rb is a blocking radius (Earth + atmosphere margin).

    3. Uses Cinelli Eq. (31) for case 5.1 to compute the *orbit-level*
       mean-anomaly limit:
           ΔM_lim,5.1 = arccos( (1 - e sin(2e)) / (1 - e²) ) - 2e

       and Eq. (29) in the “worst condition” (sin(f₀ + ΔM) = 1) to relate
       true anomaly and mean anomaly:
           Δf = ΔM + 2e  ⇒  Δf_lim,5.1 = ΔM_lim,5.1 + 2e.

       We then expose this worst-condition limit as the symmetric phasing
       bound in true anomaly:
           |Δν|_lim = |Δf_lim,5.1|.

    Inputs
    ------
    r1, r2 : np.ndarray
        Current position vectors of satellites S1 and S2 in a common inertial
        frame [km].
    Rb : float
        Blocking radius [km] (Earth + margin) used in the strict visibility test.
    r : float
        Common semi-major axis [km] appearing in Cinelli’s weakly eccentric
        derivation (a₁ = a₂ = r). This is stored for diagnostics but the case
        5.1 ΔM_lim formula itself depends only on e.
    e : float
        Eccentricity of S2 (dimensionless, small).

    Returns
    -------
    LOSResult
        visible      : True iff h_line > Rb (strict, no grazing).
        h            : h_line [km], distance from Earth center to current LOS.
        delta_nu_lim : np.array([ |Δν|_lim ]) [rad], with
                           |Δν|_lim = |Δf_lim,5.1|.
        regime       : "weakly_eccentric_case51".
        fallback     : True only for degenerate inputs (e.g., zero radius).
        diagnostics  : Dict with geometry and Cinelli §5.1 values.
    """
    # Compute norms of the current position vectors [km].
    r1_norm = float(np.linalg.norm(r1))
    r2_norm = float(np.linalg.norm(r2))

    # Guard against degenerate or unphysical inputs.
    if r1_norm <= 0.0 or r2_norm <= 0.0 or r <= 0.0:
        logging.error(
            "_weakly_eccentric_case51: degenerate input "
            "(non-positive radius or semi-major axis)."
        )
        return LOSResult(
            visible=False,
            h=0.0,
            delta_nu_lim=np.array([0.0], dtype=float),
            regime="weakly_eccentric_case51/degenerate",
            fallback=True,
            diagnostics={
                "r1_norm": r1_norm,
                "r2_norm": r2_norm,
                "r_ref": r,
                "e": e,
            },
        )

    # Current LOS altitude from full geometry (no small-e approximation).
    h_line = los_distance(r1, r2)

    # Instantaneous true-anomaly separation Δν from r1, r2.
    cos_delta_nu = float(np.dot(r1, r2) / (r1_norm * r2_norm))
    cos_delta_nu = float(np.clip(cos_delta_nu, -1.0, 1.0))
    delta_nu_current = acos_clamped(cos_delta_nu)

    # Strict NEBULA visibility: no grazing allowed.
    visible = bool(h_line > Rb)

    # Cinelli Eq. (31): ΔM_lim for case 5.1.
    delta_M_lim_51 = float(delta_M_lim_case_51(e))

    # Eq. (29) in worst condition (sin(f₀ + ΔM) = 1): Δf = ΔM + 2e.
    delta_f_lim_51 = delta_M_lim_51 + 2.0 * e
    # We expose the *magnitude* of the symmetric bound |Δν|_lim.
    delta_f_lim_51 = abs(delta_f_lim_51)

    # Pack into a 1D array for consistency with other regimes.
    delta_nu_lim_vec = np.array([delta_f_lim_51], dtype=float)

    return LOSResult(
        visible=visible,
        h=float(h_line),
        delta_nu_lim=delta_nu_lim_vec,
        regime="weakly_eccentric_case51",
        fallback=False,
        diagnostics={
            "r1_norm": r1_norm,
            "r2_norm": r2_norm,
            "r_ref": r,
            "e": e,
            "Rb": Rb,
            "h_line": h_line,
            "delta_nu_current": delta_nu_current,
            "delta_M_lim_51": delta_M_lim_51,
            "delta_f_lim_51": delta_f_lim_51,
        },
    )


def _weakly_eccentric_case52(
    r1: np.ndarray,
    r2: np.ndarray,
    Rb: float,
    r: float,
    e: float,
    R_E: float,
) -> LOSResult:
    """
    Cinelli case 5.2: weakly eccentric coplanar orbits with α < 90°, β < 90°
    (Eqs. 32–36).

    Geometry / assumptions
    ----------------------
    - Same basic setup as case 5.1:
        * S1 on circular orbit: a₁ = r, e₁ = 0.
        * S2 on weakly eccentric orbit: a₂ = r, eccentricity e (small).
        * Coplanar orbits.
    - Case 5.2 further assumes that in the *limiting configuration*:
        α < 90°, β < 90°.

    What this regime does
    ---------------------
    1. Uses the instantaneous inertial position vectors r1, r2 [km] to compute
       the current line-of-sight altitude:
           h_line = los_distance(r1, r2).

    2. Applies NEBULA’s strict visibility rule:
           visible ⇔ h_line > Rb.

    3. Uses Cinelli Eqs. (35)–(36) (via delta_f_M_lim_case_52) to compute the
       *orbit-level* phasing limits:
           Δf_lim,5.2  (Eq. 35)
           ΔM_lim,5.2 = Δf_lim,5.2 - 2e  (Eq. 36)

       and exposes:
           |Δν|_lim = |Δf_lim,5.2|.

       Here R_E is the radius appearing explicitly in Eq. (35). In NEBULA you
       can choose to pass either the physical Earth radius or Rb if you want
       the closed form to include the margin (similar to what we did for case 4.2).

    Inputs
    ------
    r1, r2 : np.ndarray
        Current position vectors of satellites S1 and S2 in a common inertial
        frame [km].
    Rb : float
        Blocking radius [km] (Earth + margin) used in the strict visibility test.
    r : float
        Common semi-major axis [km] of the weakly eccentric configuration
        (a₁ = a₂ = r) as used in Eqs. (32)–(36).
    e : float
        Eccentricity of S2 (dimensionless, small).
    R_E : float
        Radius used in Eq. (35) [km]. In the original paper this is R_Earth;
        in NEBULA you may pass Rb instead if you want the margin baked into
        the analytic limit.

    Returns
    -------
    LOSResult
        visible      : True iff h_line > Rb.
        h            : h_line [km], distance from Earth center to current LOS.
        delta_nu_lim : np.array([ |Δν|_lim ]) [rad], with
                           |Δν|_lim = |Δf_lim,5.2|.
        regime       : "weakly_eccentric_case52".
        fallback     : True only for degenerate inputs.
        diagnostics  : Dict with geometry and Cinelli §5.2 outputs.
    """
    # Norms of the current position vectors [km].
    r1_norm = float(np.linalg.norm(r1))
    r2_norm = float(np.linalg.norm(r2))

    # Basic sanity checks on radii / semi-major axis.
    if r1_norm <= 0.0 or r2_norm <= 0.0 or r <= 0.0 or R_E <= 0.0:
        logging.error(
            "_weakly_eccentric_case52: degenerate input "
            "(non-positive radius, semi-major axis, or R_E)."
        )
        return LOSResult(
            visible=False,
            h=0.0,
            delta_nu_lim=np.array([0.0], dtype=float),
            regime="weakly_eccentric_case52/degenerate",
            fallback=True,
            diagnostics={
                "r1_norm": r1_norm,
                "r2_norm": r2_norm,
                "r_ref": r,
                "e": e,
                "R_E": R_E,
            },
        )

    # Current LOS altitude from full geometry.
    h_line = los_distance(r1, r2)

    # Instantaneous true-anomaly separation Δν from current r1, r2.
    cos_delta_nu = float(np.dot(r1, r2) / (r1_norm * r2_norm))
    cos_delta_nu = float(np.clip(cos_delta_nu, -1.0, 1.0))
    delta_nu_current = acos_clamped(cos_delta_nu)

    # inside _weakly_eccentric_case52, after you compute delta_nu_current etc.
    h_analytic = h_case_52(r, e, delta_nu_current)

    # Strict visibility at the current geometry.
    visible = bool(h_line > Rb)

    # Cinelli Eqs. (35)–(36) via your helper:
    #   delta_f_lim_52  : Δf_lim,5.2 (true-anomaly limit) [rad]
    #   delta_M_lim_52  : ΔM_lim,5.2 (mean-anomaly limit) [rad]
    delta_f_lim_52, delta_M_lim_52 = delta_f_M_lim_case_52(r=r, e=e, R_E=R_E)

    # Enforce non-negative magnitude for the symmetric |Δν|_lim.
    delta_f_lim_52 = abs(float(delta_f_lim_52))

    # Pack into array for LOSResult.
    delta_nu_lim_vec = np.array([delta_f_lim_52], dtype=float)

    return LOSResult(
        visible=visible,
        h=float(h_line),
        delta_nu_lim=delta_nu_lim_vec,
        regime="weakly_eccentric_case52",
        fallback=False,
        diagnostics={
            "r1_norm": r1_norm,
            "r2_norm": r2_norm,
            "r_ref": r,
            "e": e,
            "R_E": R_E,
            "Rb": Rb,
            "h_line": h_line,
            "h_analytic_case52": float(h_analytic),
            "delta_nu_current": delta_nu_current,
            "delta_f_lim_52": delta_f_lim_52,
            "delta_M_lim_52": float(delta_M_lim_52),
        },
    )

#Geometric Fallback

def _general_3d_los(
    r1: np.ndarray,
    r2: np.ndarray,
    Rb: float,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> LOSResult:
    """
    General 3D line-of-sight (LOS) evaluation for arbitrary satellite geometry.

    This helper is used when a satellite pair does not cleanly match any of the
    analytic Cinelli cases (3.1, 3.2, 4.1, 4.2, 5.1, 5.2), for example:
        - Mixed inclination and RAAN differences that are not purely "4.1" or "4.2".
        - Orbits with eccentricities outside the "weakly eccentric" range.
        - Different semi-major axes and non-coplanar geometry.

    In these situations we:
        * Compute the exact instantaneous line-of-sight distance h between r1 and r2.
        * Decide visibility based purely on the condition h > Rb.
        * Do NOT provide any analytic phasing limit |Δν|_lim; this is returned as NaN.

    Parameters
    ----------
    r1 : np.ndarray
        Position vector of the first satellite in an inertial frame [km], shape (3,).
    r2 : np.ndarray
        Position vector of the second satellite in the same inertial frame [km], shape (3,).
    Rb : float
        Blocking radius [km]; if h <= Rb, the line-of-sight intersects the Earth (or margin).
    diagnostics : dict, optional
        Optional dictionary for passing or collecting extra diagnostic information
        (e.g., norms of r1/r2, mutual inclination, Δa_rel, Δi, ΔΩ). If None, an
        empty dictionary is created.

    Returns
    -------
    LOSResult
        A LOSResult object with:
            visible      : True iff h > Rb for this instantaneous geometry.
            h            : Minimum Earth-center distance from the S1–S2 line [km].
            delta_nu_lim : np.array([np.nan]) to indicate no analytic |Δν|_lim is available.
            regime       : "general_3d_geometry".
            fallback     : False (this is a deliberate regime, not a degenerate path).
            diagnostics  : The provided or newly created diagnostics dictionary.
    """
    # If no diagnostics dictionary was provided by the caller, create an empty one
    # so that downstream code can safely attach additional debug information.
    if diagnostics is None:
        diagnostics = {}

    # Compute the minimum distance from Earth's center to the line connecting r1 and r2.
    # This uses the exact 3D geometry via the existing los_distance helper.
    h_line = los_distance(r1, r2)

    # Determine instantaneous visibility by comparing the line-of-sight distance h_line
    # against the blocking radius Rb. Visibility is strict: h_line must be greater than Rb.
    visible = bool(h_line > Rb)

    # Since this general 3D geometry does not correspond to any specific Cinelli case,
    # we cannot provide a meaningful global phasing limit |Δν|_lim. We encode this as NaN.
    delta_nu_lim = np.array([np.nan], dtype=float)

    # Construct the LOSResult instance summarizing this regime's outcome:
    #   - visible: current LOS result,
    #   - h: the computed minimum Earth-center distance,
    #   - delta_nu_lim: NaN because no analytic phasing limit applies,
    #   - regime: label indicating this is the general 3D geometry path,
    #   - fallback: False, since this is not a numerical fallback but an intentional regime,
    #   - diagnostics: optional extra info for debugging or analysis.
    result = LOSResult(
        visible=visible,
        h=float(h_line),
        delta_nu_lim=delta_nu_lim,
        regime="general_3d_geometry",
        fallback=False,
        diagnostics=diagnostics,
    )

    # Return the LOSResult object to the caller so the dispatcher or higher-level
    # logic can use this instantaneous visibility information.
    return result
