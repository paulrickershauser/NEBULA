# NEBULA_COE.py
# -*- coding: utf-8 -*-

"""
NEBULA_COE
==========
Purpose
-------
Convert Cartesian state (r, v) or an SGP4 Satrec at a given time into
Classical Orbital Elements (COEs): a, e, i, RAAN (Ω), argument of perigee (ω),
and true anomaly (ν). Designed to feed Cinelli-style visibility “case” functions
without making the caller pre-compute angles.

Key points
----------
- Inputs:
  * Either (r [km], v [km/s]) in TEME or ECI-like frame
  * Or (sat: sgp4.api.Satrec, t: datetime) to propagate via SGP4
- Outputs:
  * Dataclass `OrbitalElements` with fields (a, e, i, raan, argp, nu, p, h_norm)
  * Angles in radians; distances in km
- Robust angle handling:
  * Uses atan2 with the orbit-normal to get signed in-plane angles
  * Proper edge-case handling for:
      - near-circular orbits (e < e_tol): ω undefined → set ω=0 and use
        argument of latitude u as ν (so downstream “Δν” logic still works)
      - near-equatorial orbits (i < i_tol): RAAN undefined → set Ω=0 and
        use true longitude λ when needed
- Dependencies:
  * numpy, dataclasses, typing
  * sgp4.api (Satrec, jday) for optional Satrec→(r,v)
  * NEBULA_ENV_CONFIG.MU_EARTH for the gravitational parameter (km^3/s^2)

Units
-----
- Position r in km
- Velocity v in km/s
- μ in km^3/s^2
- Angles in radians

This module is intentionally small and pure-Python so it is easy to audit and
portable with the NEBULA repository.
"""

# -----------------------------
# Standard library imports
# -----------------------------
# Dataclass for a clean result container
from dataclasses import dataclass
# Typing helpers for signatures
from typing import Tuple, Optional

# -----------------------------
# Third-party imports
# -----------------------------
# Numerical arrays and linear algebra
import numpy as np
# SGP4 types and Julian date helper (only needed for Satrec → (r,v))
from sgp4.api import Satrec, jday

# -----------------------------
# NEBULA configuration imports
# -----------------------------
# Gravitational parameter μ (km^3/s^2)
from Configuration.NEBULA_ENV_CONFIG import MU_EARTH


# -----------------------------
# Small numeric helpers
# -----------------------------
# Wrap any angle to (-π, π]
def _wrap_to_pi(x: float) -> float:
    # Use numpy’s angle wrapping via arctan2 on sin/cos for numerical stability
    return float(np.arctan2(np.sin(x), np.cos(x)))


# Return unit vector and its norm, guarding zero
def _unit(vec: np.ndarray) -> Tuple[np.ndarray, float]:
    # Compute Euclidean norm
    n = float(np.linalg.norm(vec))
    # If zero vector, return itself and zero norm
    if n == 0.0:
        return vec, 0.0
    # Else return normalized vector and its norm
    return vec / n, n


# Angle from vector a to b measured in the plane whose normal is h_hat
def _angle_in_plane(a: np.ndarray, b: np.ndarray, h_hat: np.ndarray) -> float:
    # Compute dot product for the cosine term
    c = float(np.dot(a, b))
    # Compute signed sine term via scalar triple product ĥ·(a×b)
    s = float(np.dot(h_hat, np.cross(a, b)))
    # atan2(s, c) gives signed angle in (-π, π]
    return float(np.arctan2(s, c))


# -----------------------------
# Result dataclass
# -----------------------------
@dataclass(frozen=True)
class OrbitalElements:
    """Classical Orbital Elements (angles in radians; distances in km)."""
    # Semi-major axis (km); for near-parabolic cases this may be very large in magnitude
    a: float
    # Eccentricity (unitless, >=0)
    e: float
    # Inclination (rad, ∈ [0, π])
    i: float
    # Right ascension of ascending node (Ω, rad, ∈ [−π, π])
    raan: float
    # Argument of perigee (ω, rad, ∈ [−π, π]); set to 0 if e < e_tol
    argp: float
    # True anomaly (ν, rad, ∈ [−π, π]); for e < e_tol this stores argument of latitude u
    nu: float
    # Semi-latus rectum p = a(1 − e^2) (km); equals h^2/μ
    p: float
    # Specific angular momentum magnitude |h| (km^2/s)
    h_norm: float

    # Convenience: difference in true anomaly wrapped to (−π, π]
    def delta_nu(self, other: "OrbitalElements") -> float:
        # Subtract true anomalies and wrap
        return _wrap_to_pi(self.nu - other.nu)


# -----------------------------
# Core RV → COE conversion
# -----------------------------
def coe_from_rv(r_km: np.ndarray,
                v_kms: np.ndarray,
                mu: float = MU_EARTH,
                e_tol: float = 1e-6,
                i_tol: float = 1e-8) -> OrbitalElements:
    """
    Convert Cartesian state (r, v) to Classical Orbital Elements.

    Parameters
    ----------
    r_km : np.ndarray
        Position vector [km], shape (3,)
    v_kms : np.ndarray
        Velocity vector [km/s], shape (3,)
    mu : float
        Gravitational parameter [km^3/s^2]
    e_tol : float
        Eccentricity threshold below which the orbit is treated as circular
    i_tol : float
        Inclination threshold below which the orbit is treated as equatorial

    Returns
    -------
    OrbitalElements
        Dataclass with a, e, i, Ω, ω, ν, p, |h|

    Notes
    -----
    - For circular orbits (e < e_tol), ω is undefined. We set ω = 0 and define ν
      to be the **argument of latitude** u (angle from node vector to r).
    - For equatorial orbits (i < i_tol), Ω is undefined. We set Ω = 0 and measure
      in-plane angles relative to the inertial x-axis.
    """
    # Ensure numpy arrays as float64
    r = np.asarray(r_km, dtype=float).reshape(3)
    v = np.asarray(v_kms, dtype=float).reshape(3)

    # Specific angular momentum vector h = r × v
    h_vec = np.cross(r, v)
    # Unit orbit normal and its magnitude |h|
    h_hat, h_norm = _unit(h_vec)

    # Node vector n = k × h (where k = [0,0,1])
    k_hat = np.array([0.0, 0.0, 1.0], dtype=float)
    n_vec = np.cross(k_hat, h_vec)
    # Unit node and its magnitude |n|
    n_hat, n_norm = _unit(n_vec)

    # Radius and speed
    r_norm = float(np.linalg.norm(r))
    v_norm = float(np.linalg.norm(v))

    # Eccentricity vector e_vec = (v×h)/μ − r/|r|
    e_vec = (np.cross(v, h_vec) / mu) - (r / max(r_norm, 1e-16))
    # Eccentricity magnitude e
    e = float(np.linalg.norm(e_vec))
    # Unit eccentricity vector (undefined if e ≈ 0)
    e_hat = e_vec / max(e, 1e-16)

    # Specific mechanical energy ε = v^2/2 − μ/r
    energy = 0.5 * v_norm * v_norm - mu / max(r_norm, 1e-16)

    # Semi-major axis a = −μ / (2ε)  (for elliptic; also valid for hyperbolic/degenerate)
    # Guard extremely small |ε| to avoid overflow.
    if abs(energy) < 1e-16:
        # Near-parabolic; set a to a very large magnitude with the correct sign
        a = float(np.sign(energy) * 1e16)
    else:
        a = float(-mu / (2.0 * energy))

    # Semi-latus rectum p = |h|^2 / μ
    p = float(h_norm * h_norm / mu)

    # Inclination i = arccos(h_z / |h|)
    # Guard h_norm = 0 (degenerate); fall back to 0
    i = float(np.arccos(np.clip(h_vec[2] / max(h_norm, 1e-16), -1.0, 1.0))) if h_norm > 0.0 else 0.0
    # Wrap inclination into [0, π]
    i = float(np.clip(i, 0.0, np.pi))

    # Right ascension of ascending node Ω
    # If near-equatorial (i < i_tol), Ω is undefined; set Ω = 0
    if i < i_tol or n_norm == 0.0:
        raan = 0.0
    else:
        # Ω = atan2(n_y, n_x) wrapped to (−π, π]
        raan = float(np.arctan2(n_vec[1], n_vec[0]))
        raan = _wrap_to_pi(raan)

    # Argument of perigee ω and true anomaly ν
    if e < e_tol:
        # Circular case: ω undefined; choose ω = 0 and set ν := u (argument of latitude)
        argp = 0.0
        if i < i_tol or n_norm == 0.0:
            # Circular + equatorial: use true longitude λ = atan2(r_y, r_x)
            nu = float(np.arctan2(r[1], r[0]))
            nu = _wrap_to_pi(nu)
        else:
            # Circular but inclined: u = angle from node vector to r in the orbital plane
            nu = _angle_in_plane(n_hat, r / max(r_norm, 1e-16), h_hat)
            nu = _wrap_to_pi(nu)
    else:
        # Non-circular: compute ω as angle from node vector to eccentricity vector
        if i < i_tol or n_norm == 0.0:
            # Equatorial but eccentric: set Ω = 0 already; measure ω from x-axis to ê
            argp = float(np.arctan2(e_hat[1], e_hat[0]))
            argp = _wrap_to_pi(argp)
        else:
            # ω = angle_in_plane(n̂ → ê) using the orbit normal for sign
            argp = _angle_in_plane(n_hat, e_hat, h_hat)
            argp = _wrap_to_pi(argp)

        # True anomaly ν = angle_in_plane(ê → r̂)
        nu = _angle_in_plane(e_hat, r / max(r_norm, 1e-16), h_hat)
        nu = _wrap_to_pi(nu)

    # Return the dataclass with all fields populated
    return OrbitalElements(
        a=a,
        e=e,
        i=i,
        raan=raan,
        argp=argp,
        nu=nu,
        p=p,
        h_norm=h_norm,
    )


# -----------------------------
# Satrec + datetime → COE helper
# -----------------------------
def coe_from_satrec_at(sat: Satrec,
                       t) -> OrbitalElements:
    """
    Convenience: propagate a Satrec to time t with SGP4, then convert to COE.

    Parameters
    ----------
    sat : sgp4.api.Satrec
        Parsed TLE as a Satrec
    t : datetime.datetime
        UTC datetime to evaluate

    Returns
    -------
    OrbitalElements
        COE elements at epoch t (angles in radians; distances in km)

    Raises
    ------
    RuntimeError
        If SGP4 propagation fails (non-zero error code)
    """
    # Extract Julian Date split (jd integer day, fr fractional day) from datetime
    jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second + t.microsecond * 1e-6)
    # Run SGP4 propagation: e==0 means success
    e, r_km, v_kms = sat.sgp4(jd, fr)
    # If error code non-zero, raise with details
    if e != 0:
        raise RuntimeError(f"SGP4 propagation error code {e} for satellite {getattr(sat, 'satnum', 'unknown')}")
    # Convert (r, v) to COE using the core function
    return coe_from_rv(np.array(r_km, dtype=float),
                       np.array(v_kms, dtype=float),
                       mu=MU_EARTH)
