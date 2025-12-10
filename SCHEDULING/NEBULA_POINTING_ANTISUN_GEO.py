"""
NEBULA_POINTING_ANTISUN_GEO

Sidereal anti-Sun GEO-belt pointing schedule for a dawn–dusk
sun-synchronous LEO observer.

This module implements a variant of the anti-Sun pointing law that is
tailored specifically to observing GEO from a terminator LEO in a
sidereal stare configuration, as described in the pointing strategy
paper. :contentReference[oaicite:7]{index=7}

Given:
    - an observer "track" (dict-like or object-like) containing NEBULA
      state fields for a LEO platform on some time grid, and
    - a PointingConfig whose mode is PointingMode.ANTI_SUN_GEO_BELT,

this module:

  1. Loads (or accepts) a Skyfield DE440s ephemeris using the shared
     helper in NEBULA_SKYFIELD_ILLUMINATION (no hard-coded paths). :contentReference[oaicite:8]{index=8}
  2. Builds a Skyfield Time array from the observer times.
  3. Selects a single reference time (the midpoint of the time grid)
     to represent the "midnight sector" / anti-Sun geometry for the
     observation window.
  4. Computes the Sun RA at that reference time and defines a fixed
     boresight:
         RA_ref  = RA_sun(t_ref) + 180 deg  (mod 360)
         Dec_ref = 0 deg  (aligned with the GEO belt)
  5. Converts (RA_ref, Dec_ref) into a constant boresight unit vector
     in ICRS, which is held fixed for all timesteps (sidereal stare).
  6. For each timestep, computes:
        - whether that boresight passes too close to the Earth limb
          using R_BLOCK + earth_avoid_margin_deg (Earth-block mask),
        - whether the observer is sunlit (sensor_is_sunlit) using
          Skyfield's is_sunlit() machinery,
        - the angular separation between boresight and Sun direction
          (sun_angle_deg),
        - whether that separation violates sun_exclusion_angle_deg
          (sun_excluded mask).
  7. Combines those masks plus the PointingConfig flags into a final
     valid_for_projection mask indicating which timesteps should be
     used by downstream WCS / sensor-projection code.
  8. Returns a PointingSchedule dataclass with these arrays and (if
     requested) attaches a subset of them back onto the observer track
     for convenient inspection in tools like Spyder.

This is analogous to NEBULA_POINTING_ANTISUN, but with a fixed
GEO-optimized boresight instead of a time-varying anti-Sun vector.
"""

from dataclasses import dataclass          # For the PointingSchedule container.
from typing import Any, Optional           # For type annotations.
import logging                             # For optional logging.
import numpy as np                         # For array math.

from skyfield.api import load              # For TimeScale construction.
from skyfield.positionlib import ICRF      # For ICRF states.
from skyfield.sgp4lib import TEME         # For TEME -> ICRF conversion.
from skyfield.api import Distance, Velocity

# Earth blocking radius (includes atmosphere, etc.) from ENV config.
from Configuration.NEBULA_ENV_CONFIG import R_BLOCK

# Pointing configuration types.
from Configuration.NEBULA_POINTING_CONFIG import (
    PointingConfig,
    PointingMode,
)

# Shared Skyfield helpers (ephemeris loading and track field accessors).
from Utility.RADIOMETRY.NEBULA_SKYFIELD_ILLUMINATION import (
    load_de440s_ephemeris,
    _get_track_field,
    _set_track_field,
)


@dataclass
class PointingSchedule:
    """
    Container for per-timestep GEO-optimized anti-Sun pointing outputs.

    Attributes
    ----------
    observer_name : str
        Name or identifier of the observer platform.
    mode : PointingMode
        Pointing mode used to generate this schedule.  Should be
        PointingMode.ANTI_SUN_GEO_BELT for this module.
    times : np.ndarray
        Array of datetime objects (shape (N,)).
    boresight_ra_deg : np.ndarray
        Fixed right ascension of the boresight in ICRS, degrees,
        shape (N,) but with a single constant value.
    boresight_dec_deg : np.ndarray
        Fixed declination of the boresight in ICRS, degrees, shape
        (N,) but with a single constant value (nominally 0 deg).
    roll_deg : float
        Focal-plane roll angle about the boresight, in degrees.
    earth_blocked : np.ndarray
        Boolean array (N,) indicating Earth intersection of the
        boresight (R_BLOCK + limb margin).
    sensor_is_sunlit : np.ndarray
        Boolean array (N,) indicating whether the observer is in
        direct sunlight.
    sun_angle_deg : np.ndarray
        Angular separation between boresight and Sun direction,
        degrees, shape (N,).
    sun_excluded : np.ndarray
        Boolean array (N,) indicating violation of the configured
        Sun-exclusion angle.
    valid_for_projection : np.ndarray
        Boolean array (N,) combining all constraints.
    """

    observer_name: str
    mode: PointingMode
    times: np.ndarray
    boresight_ra_deg: np.ndarray
    boresight_dec_deg: np.ndarray
    roll_deg: float
    earth_blocked: np.ndarray
    sensor_is_sunlit: np.ndarray
    sun_angle_deg: np.ndarray
    sun_excluded: np.ndarray
    valid_for_projection: np.ndarray


def build_pointing_schedule_antisun_geo(
    observer_track: Any,
    config: PointingConfig,
    eph=None,
    ephemeris_path: Optional[str] = None,
    store_on_observer: bool = True,
    logger: Optional[logging.Logger] = None,
) -> PointingSchedule:
    """
    Build a GEO-optimized anti-Sun pointing schedule for an observer.

    This function implements the ANTI_SUN_GEO_BELT mode described in
    the module docstring.

    Parameters
    ----------
    observer_track : dict or SatelliteTrack-like
        NEBULA observer track with at least:
            - 'times'      : sequence of datetime objects (length N)
            - 'r_eci_km'   : array (N,3) positions [km]
            - 'v_eci_km_s' : array (N,3) velocities [km/s]
    config : PointingConfig
        Pointing configuration.  Should have mode ==
        PointingMode.ANTI_SUN_GEO_BELT.  Uses:
            - roll_deg
            - earth_avoid_margin_deg
            - require_sensor_sunlit
            - sun_exclusion_angle_deg
    eph : skyfield.jpllib.SpiceKernel or None, optional
        Optional pre-loaded DE440s ephemeris.  If None, this function
        will call load_de440s_ephemeris(ephemeris_path).
    ephemeris_path : str or None, optional
        Optional explicit path to "de440s.bsp".  Passed through to the
        ephemeris loader if eph is None.
    store_on_observer : bool, optional
        If True, key arrays are written back onto observer_track with
        'pointing_*' field names (boresight RA/Dec, masks, etc.).
    logger : logging.Logger or None, optional
        Optional logger for status messages.

    Returns
    -------
    PointingSchedule
        Dataclass containing the GEO-optimized anti-Sun boresight,
        environment flags, and valid_for_projection mask.
    """
    # Build or reuse a logger.
    log = logger or logging.getLogger(__name__)

    # Extract times and observer states.
    times = np.asarray(_get_track_field(observer_track, "times"), dtype=object)
    n_steps = times.shape[0]

    r_obs = np.asarray(_get_track_field(observer_track, "r_eci_km"), dtype=float)[
        :n_steps, :
    ]
    v_obs = np.asarray(_get_track_field(observer_track, "v_eci_km_s"), dtype=float)[
        :n_steps, :
    ]

    # Build Skyfield Time array.
    ts = load.timescale()
    t = ts.from_datetimes(times.tolist())

    # Load ephemeris if needed.
    if eph is None:
        eph = load_de440s_ephemeris(ephemeris_path=ephemeris_path, logger=log)

    earth = eph["earth"]
    sun = eph["sun"]

    # ------------------------------------------------------------------
    # Step 1: choose a reference time near the middle of the window.
    # ------------------------------------------------------------------
    mid_index = n_steps // 2
    t_ref = t[mid_index]

    # Compute Sun RA at the reference time as seen from Earth.
    astrometric_sun_ref = earth.at(t_ref).observe(sun)
    ra_sun_ref, dec_sun_ref, _ = astrometric_sun_ref.radec()
    ra_sun_ref_deg = ra_sun_ref.degrees

    # Define the fixed boresight RA/Dec:
    #   RA_ref  = RA_sun_ref + 180 deg (mod 360)
    #   Dec_ref = 0 deg  (GEO belt)
    ra_ref_deg = (ra_sun_ref_deg + 180.0) % 360.0
    dec_ref_deg = 0.0

    # Convert (RA_ref, Dec_ref) into a unit vector.
    ra_ref_rad = np.deg2rad(ra_ref_deg)
    dec_ref_rad = np.deg2rad(dec_ref_deg)
    bx = np.cos(dec_ref_rad) * np.cos(ra_ref_rad)
    by = np.cos(dec_ref_rad) * np.sin(ra_ref_rad)
    bz = np.sin(dec_ref_rad)
    boresight_vec = np.array([bx, by, bz], dtype=float)
    boresight_vec /= np.linalg.norm(boresight_vec)

    # Tile the scalar RA/Dec into arrays of length N for convenience.
    boresight_ra_deg = np.full(n_steps, ra_ref_deg, dtype=float)
    boresight_dec_deg = np.full(n_steps, dec_ref_deg, dtype=float)

    # ------------------------------------------------------------------
    # Step 2: environment geometry per timestep.
    # ------------------------------------------------------------------

    # Unit vector from observer to Earth's center.
    dir_to_earth = -r_obs / np.linalg.norm(r_obs, axis=1, keepdims=True)

    # Angle between boresight and Earth's center.
    cos_theta_earth = np.einsum("i,ij->j", boresight_vec, dir_to_earth.T)
    cos_theta_earth = np.clip(cos_theta_earth, -1.0, 1.0)
    theta_earth_rad = np.arccos(cos_theta_earth)

    # Earth apparent half-angle as seen from observer.
    r_obs_norm = np.linalg.norm(r_obs, axis=1)
    alpha_rad = np.arcsin(np.clip(R_BLOCK / r_obs_norm, -1.0, 1.0))

    # Apply limb margin.
    margin_rad = np.deg2rad(config.earth_avoid_margin_deg)
    alpha_eff_rad = alpha_rad + margin_rad

    # Earth blocking flag.
    earth_blocked = theta_earth_rad <= alpha_eff_rad

    # Build Distance/Velocity for TEME -> ICRF conversion.
    dist_obs = Distance(km=r_obs.T)
    vel_obs = Velocity(km_per_s=v_obs.T)
    pos_obs_icrf = ICRF.from_time_and_frame_vectors(t, TEME, dist_obs, vel_obs)
    pos_obs_icrf.center = 399  # Earth-centered.

    # Sensor sunlit flag via Skyfield.
    sensor_is_sunlit = np.asarray(pos_obs_icrf.is_sunlit(eph), dtype=bool)

    # Sun direction as unit vectors at all times.
    astrometric_sun = earth.at(t).observe(sun)
    r_sun_eci_km = astrometric_sun.position.km.T
    sun_vec = r_sun_eci_km / np.linalg.norm(r_sun_eci_km, axis=1, keepdims=True)

    # Angle between fixed boresight and Sun direction at each timestep.
    cos_theta_sun = np.einsum("i,ij->j", boresight_vec, sun_vec.T)
    cos_theta_sun = np.clip(cos_theta_sun, -1.0, 1.0)
    theta_sun_rad = np.arccos(cos_theta_sun)
    sun_angle_deg = np.degrees(theta_sun_rad)

    # Sun exclusion mask.
    sun_excluded = np.zeros_like(sun_angle_deg, dtype=bool)
    if config.sun_exclusion_angle_deg > 0.0:
        sun_excluded = sun_angle_deg < float(config.sun_exclusion_angle_deg)

    # Valid-for-projection mask.
    valid = ~earth_blocked
    if config.require_sensor_sunlit:
        valid = valid & sensor_is_sunlit
    if config.sun_exclusion_angle_deg > 0.0:
        valid = valid & (~sun_excluded)

    # Observer name for bookkeeping.
    observer_name = _get_track_field(observer_track, "name", required=False) or "UNKNOWN"

    schedule = PointingSchedule(
        observer_name=observer_name,
        mode=config.mode,
        times=times,
        boresight_ra_deg=boresight_ra_deg,
        boresight_dec_deg=boresight_dec_deg,
        roll_deg=float(config.roll_deg),
        earth_blocked=earth_blocked,
        sensor_is_sunlit=sensor_is_sunlit,
        sun_angle_deg=sun_angle_deg,
        sun_excluded=sun_excluded,
        valid_for_projection=valid,
    )

    # Optionally attach fields back onto the observer track.
    if store_on_observer:
        _set_track_field(observer_track, "pointing_boresight_ra_deg", boresight_ra_deg)
        _set_track_field(observer_track, "pointing_boresight_dec_deg", boresight_dec_deg)

        # NEW: attach the focal-plane roll used by the WCS builder.
        # This can be a scalar; WCS code will broadcast as needed.
        _set_track_field(observer_track, "roll_deg", float(config.roll_deg))

        _set_track_field(observer_track, "pointing_earth_blocked", earth_blocked)
        _set_track_field(observer_track, "pointing_sensor_is_sunlit", sensor_is_sunlit)
        _set_track_field(observer_track, "pointing_sun_angle_deg", sun_angle_deg)
        _set_track_field(observer_track, "pointing_sun_excluded", sun_excluded)
        _set_track_field(observer_track, "pointing_valid_for_projection", valid)


    return schedule
