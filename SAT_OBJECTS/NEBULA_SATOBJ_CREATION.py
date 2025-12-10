# NEBULA_SATOBJ_CREATION.py
# ---------------------------------------------------------------------------
# NEBULA Satellite Object Creation
# ---------------------------------------------------------------------------
"""
Utility to build NEBULA satellite objects from:

  * TLE-based SGP4 objects (sgp4.api.Satrec)
  * A specified time grid (list of datetime objects)
  * The existing propagation + COE utilities

Purpose
-------
This module centralizes the creation of a *single, rich satellite object*
that NEBULA can pass around instead of juggling separate arrays:

  - NEBULA_TLE_READER gives us: Dict[str, Satrec]
  - NEBULA_TIME_CONFIG + NEBULA_TIME_PARSER give us: start, end, dt
  - NEBULA_PROPAGATOR propagates Satrec → (r(t), v(t))
  - NEBULA_COE converts (r(t), v(t)) → COEs (including true anomaly ν(t))

Here we wrap that into a dataclass `SatelliteTrack` with:

  * Name / identifier
  * Original Satrec (for debugging / reuse)
  * Static, TLE-derived orbital elements:
      - semi-major axis a [km]
      - eccentricity e
      - inclination i [rad]
      - RAAN Ω [rad]
      - argument of perigee ω [rad]
      - mean anomaly at epoch M0 [rad]
      - mean motion n [rad/min]
      - semi-latus rectum p = a(1 - e^2) [km]
  * Time-series arrays over the selected window:
      - times[k] : datetime UTC
      - r_eci_km[k, :] : position in ECI/TEME [km]
      - v_eci_km_s[k, :] : velocity in ECI/TEME [km/s]
      - nu_rad[k] : true anomaly ν(t_k) [rad]
  * A place to attach visibility / tracking results later:
      - visibility_flags: Dict[str, np.ndarray], e.g. per-observer boolean flags

Typical flow in a main script
-----------------------------
1) Read TLEs:
   >>> from NEBULA_TLE_READER import read_tle_file
   >>> sats = read_tle_file(OBS_TLE_FILE)  # or TAR_TLE_FILE

2) Build time grid using NEBULA_TIME_CONFIG + NEBULA_TIME_PARSER:
   >>> from Configuration.NEBULA_TIME_CONFIG import DEFAULT_TIME_WINDOW, DEFAULT_PROPAGATION_STEPS
   >>> from Utility.NEBULA_TIME_PARSER import parse_time_window_config
   >>> start_dt, end_dt = parse_time_window_config(DEFAULT_TIME_WINDOW)
   >>> dt_s = DEFAULT_PROPAGATION_STEPS.base_dt_s
   >>> times = ...
       # build a list of datetimes from start_dt to end_dt using dt_s

3) Build satellite objects:
   >>> from Utility.NEBULA_SATOBJ_CREATION import build_satellite_tracks
   >>> tracks = build_satellite_tracks(sats, times)

4) Use those tracks in NEBULA_VISIBILITY and elsewhere:
   >>> obs = tracks["SBSS (USA 216)"]
   >>> tar = tracks["GEO_TARGET"]
   >>> k = 10  # time index
   >>> r_obs = obs.r_eci_km[k]
   >>> r_tar = tar.r_eci_km[k]
   (pass into your geometry / visibility cases)

This keeps the satellites as first-class objects in your simulation,
which is exactly the OOP direction you described.
"""

# ---------------------------------------------------------------------------
# Standard library imports
# ---------------------------------------------------------------------------
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Optional
from datetime import datetime

# Optional progress bar for long runs (targets)
try:
    from tqdm.auto import tqdm  # type: ignore
except ImportError:
    tqdm = None  # graceful fallback if tqdm is not installed

# ---------------------------------------------------------------------------
# Third-party imports
# ---------------------------------------------------------------------------
import numpy as np
from sgp4.api import Satrec

# ---------------------------------------------------------------------------
# NEBULA imports
# ---------------------------------------------------------------------------
# Gravitational parameter μ (km^3/s^2) for Earth
from Configuration.NEBULA_ENV_CONFIG import MU_EARTH  # type: ignore

# Propagation helper: Satrec + times → r(t), v(t)
from Utility.PROPAGATION.NEBULA_PROPAGATOR import propagate_teme_state  # type: ignore

# COE helper: (r, v, μ) → OrbitalElements (includes ν)
from Utility.PROPAGATION.NEBULA_COE import coe_from_rv  # type: ignore


# ---------------------------------------------------------------------------
# Satellite track dataclass
# ---------------------------------------------------------------------------
@dataclass
class SatelliteTrack:
    """
    NEBULA representation of a single satellite over a time grid.

    Attributes
    ----------
    name : str
        Human-readable name, usually taken directly from the TLE line 0
        (e.g. "SBSS (USA 216)").
    satrec : Satrec
        Original SGP4 `Satrec` object produced from the TLE.  We keep this for
        debugging and for any additional on-the-fly propagation you may want.

    # Static, TLE-derived elements (nominal)
    a_km : float
        Semi-major axis [km], derived from mean motion `no_kozai` and μ.
    e : float
        Eccentricity (dimensionless).
    inc_rad : float
        Inclination i [rad].
    raan_rad : float
        Right ascension of ascending node Ω [rad].
    argp_rad : float
        Argument of perigee ω [rad].
    mean_anomaly_epoch_rad : float
        Mean anomaly M0 [rad] at the TLE epoch.
    mean_motion_rad_per_min : float
        Mean motion n [rad/min] from the TLE (no_kozai).
    p_km : float
        Semi-latus rectum p = a(1 - e^2) [km], useful for some analytic work.

    # Time series
    times : List[datetime]
        List of UTC datetimes defining the propagation grid.
    r_eci_km : np.ndarray
        Positions in ECI/TEME frame, shape (N, 3), units km.
    v_eci_km_s : np.ndarray
        Velocities in ECI/TEME frame, shape (N, 3), units km/s.
    nu_rad : np.ndarray
        True anomaly ν(t_k) [rad] for each time step, shape (N,).

    # Visibility / tracking annotations (to be filled later)
    visibility_flags : Dict[str, np.ndarray]
        Optional per-observer or per-regime visibility flags.
        Keys might be things like "GEO_OBS_1" or "coplanar_case_3p1".
        Values should be boolean arrays of length N, aligned with `times`.
    """

    # --- Identity & raw SGP4 reference ---------------------------------------
    name: str
    #satrec: Satrec

    # --- Static orbital elements (from TLE at epoch) ------------------------
    a_km: float
    e: float
    inc_rad: float
    raan_rad: float
    argp_rad: float
    mean_anomaly_epoch_rad: float
    mean_motion_rad_per_min: float
    p_km: float

    # --- Time-series state ---------------------------------------------------
    times: List[datetime]
    r_eci_km: np.ndarray
    v_eci_km_s: np.ndarray
    nu_rad: np.ndarray

    # --- Visibility / tracking annotations ----------------------------------
    visibility_flags: Dict[str, np.ndarray] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helper: derive static orbital elements from a Satrec
# ---------------------------------------------------------------------------
def _static_elements_from_satrec(sat: Satrec) -> Dict[str, float]:
    """
    Compute "nominal" orbital elements from a Satrec using its TLE fields.

    This is intentionally lightweight and does *not* propagate anything;
    it just converts mean motion to semi-major axis and pulls the rest of the
    classical elements directly from the Satrec.

    Parameters
    ----------
    sat : Satrec
        SGP4 satellite record created from a TLE.

    Returns
    -------
    elems : Dict[str, float]
        Dictionary with keys:
          - a_km
          - e
          - inc_rad
          - raan_rad
          - argp_rad
          - mean_anomaly_epoch_rad
          - mean_motion_rad_per_min
          - p_km
    """
    # Mean motion n in rad/min (SGP4's `no_kozai` field).
    n_rad_per_min = float(sat.no_kozai)

    # Convert mean motion to rad/s for consistency with μ [km^3/s^2].
    n_rad_per_s = n_rad_per_min / 60.0

    # Use the two-body relation n^2 a^3 = μ  →  a = (μ / n^2)^(1/3).
    a_km = float((MU_EARTH / (n_rad_per_s ** 2)) ** (1.0 / 3.0))

    # Eccentricity (dimensionless).
    e = float(sat.ecco)

    # Inclination, RAAN, argument of perigee, mean anomaly (all in radians).
    inc_rad = float(sat.inclo)
    raan_rad = float(sat.nodeo)
    argp_rad = float(sat.argpo)
    mean_anomaly_epoch_rad = float(sat.mo)

    # Semi-latus rectum p = a(1 - e^2) [km].
    p_km = a_km * (1.0 - e * e)

    return dict(
        a_km=a_km,
        e=e,
        inc_rad=inc_rad,
        raan_rad=raan_rad,
        argp_rad=argp_rad,
        mean_anomaly_epoch_rad=mean_anomaly_epoch_rad,
        mean_motion_rad_per_min=n_rad_per_min,
        p_km=p_km,
    )


# ---------------------------------------------------------------------------
# Public API: build a single SatelliteTrack
# ---------------------------------------------------------------------------
def build_satellite_track(
    name: str,
    sat: Satrec,
    times: Sequence[datetime],
) -> SatelliteTrack:
    """
    Build a `SatelliteTrack` for one satellite over a given time grid.

    Parameters
    ----------
    name : str
        Identifier for this satellite (usually the TLE name).
    sat : Satrec
        SGP4 satellite record (from NEBULA_TLE_READER).
    times : Sequence[datetime]
        Monotonic list/sequence of UTC datetimes defining the propagation grid.

    Returns
    -------
    track : SatelliteTrack
        A NEBULA satellite object containing:
          - static TLE-derived elements (a, e, i, Ω, ω, M0, n, p)
          - time-series r_eci_km, v_eci_km_s, nu_rad
          - empty visibility_flags dict ready to be filled later.
    """
    # Ensure we have a concrete list of datetimes (not just a view).
    time_list: List[datetime] = [t for t in times]

    # --- 1) Static elements (no propagation yet) -----------------------------
    static_elems = _static_elements_from_satrec(sat)

    # --- 2) Propagate to get r(t), v(t) on the grid --------------------------
    # Uses NEBULA_PROPAGATOR: Satrec + times → arrays (N, 3).
    r_eci_km, v_eci_km_s = propagate_teme_state(sat, time_list)

    # --- 3) Compute ν(t) via NEBULA_COE for each time step -------------------
    # We only need true anomaly here, but we reuse the full COE machinery for
    # consistency and edge-case handling.
    N = r_eci_km.shape[0]
    nu_rad = np.zeros(N, dtype=float)

    for k in range(N):
        # Extract position and velocity at this step.
        r_k = r_eci_km[k, :]
        v_k = v_eci_km_s[k, :]

        # Compute orbital elements from (r, v, μ).
        coe_k = coe_from_rv(r_k, v_k, mu=MU_EARTH)

        # In degenerate cases (e ~ 0 or i ~ 0) coe_from_rv may set nu in a
        # particular way; we just store whatever ν it returns.
        nu_val = getattr(coe_k, "nu", None)
        nu_rad[k] = float(nu_val) if nu_val is not None else 0.0

    # --- 4) Package everything into a SatelliteTrack -------------------------
    track = SatelliteTrack(
        name=name,
        #satrec=sat,
        a_km=static_elems["a_km"],
        e=static_elems["e"],
        inc_rad=static_elems["inc_rad"],
        raan_rad=static_elems["raan_rad"],
        argp_rad=static_elems["argp_rad"],
        mean_anomaly_epoch_rad=static_elems["mean_anomaly_epoch_rad"],
        mean_motion_rad_per_min=static_elems["mean_motion_rad_per_min"],
        p_km=static_elems["p_km"],
        times=time_list,
        r_eci_km=r_eci_km,
        v_eci_km_s=v_eci_km_s,
        nu_rad=nu_rad,
        visibility_flags={},  # start empty; NEBULA_VISIBILITY can fill this
    )

    return track


def build_satellite_tracks(
    sat_dict: Dict[str, Satrec],
    times: Sequence[datetime],
) -> Dict[str, SatelliteTrack]:
    """
    Convenience wrapper: build SatelliteTrack objects for many satellites.

    This version uses a simple serial loop (no multiprocessing) and will
    display a progress bar via tqdm if it is available.

    Parameters
    ----------
    sat_dict : Dict[str, Satrec]
        Dictionary mapping satellite names → Satrec objects, as returned by
        NEBULA_TLE_READER.read_tle_file().
    times : Sequence[datetime]
        Time grid (list/sequence of UTC datetimes) shared by all satellites.

    Returns
    -------
    tracks : Dict[str, SatelliteTrack]
        Dictionary mapping the same keys to fully-populated SatelliteTrack
        objects.
    """
    items = list(sat_dict.items())
    n_sats = len(items)
    tracks: Dict[str, SatelliteTrack] = {}

    if n_sats == 0:
        return tracks

    # Optional progress bar for long runs
    if tqdm is not None:
        iterable = tqdm(
            items,
            total=n_sats,
            desc="Building SatelliteTrack objects",
        )
    else:
        iterable = items

    for name, sat in iterable:
        track = build_satellite_track(name, sat, times)
        tracks[name] = track

    return tracks
