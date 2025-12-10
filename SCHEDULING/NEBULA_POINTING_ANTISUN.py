"""
NEBULA_POINTING_ANTISUN.py

Anti-Sun pointing schedule for a dawn–dusk sun-synchronous LEO observer.

This module implements a specific high-level pointing mode for NEBULA:
an "anti-Sun stare" geometry intended for dawn–dusk sun-synchronous
LEO platforms (e.g., SBSS / NEOSSat-style missions) that observe GEO
targets near the Earth's midnight sector.

Given:
    - an observer "track" (dict-like or object-like) containing NEBULA
      state fields for a LEO platform on some time grid, and
    - a PointingConfig whose mode is PointingMode.ANTI_SUN_STARE,

this module:

  1. Loads (or accepts) a Skyfield DE440s ephemeris using the shared
     helper in NEBULA_SKYFIELD_ILLUMINATION (no hard-coded paths).
  2. Builds a Skyfield Time array from the observer times.
  3. Computes the Sun direction in an Earth-centered inertial frame
     at each timestep.
  4. Defines the boresight as the *anti-Sun* unit vector at each
     timestep and converts that to ICRS right ascension and declination.
  5. Computes:
        - whether that boresight passes too close to the Earth limb
          using R_BLOCK + earth_avoid_margin_deg (Earth-block mask),
        - whether the observer is sunlit (sensor_is_sunlit) using
          Skyfield's is_sunlit() machinery,
        - the angular separation between boresight and Sun direction
          (sun_angle_deg),
        - whether that separation violates sun_exclusion_angle_deg
          (sun_excluded mask).
  6. Combines those masks plus the PointingConfig flags into a final
     valid_for_projection mask indicating which timesteps should be
     used by downstream WCS / sensor-projection code.
  7. Returns a PointingSchedule dataclass with these arrays and (if
     requested) attaches a subset of them back onto the observer track
     for convenient inspection in tools like Spyder.

Functions
---------

PointingSchedule (dataclass)
    Container for per-timestep anti-Sun pointing outputs.

build_pointing_schedule_antisun(observer_track, config, eph=None,
                                ephemeris_path=None,
                                store_on_observer=True, logger=None)
    Core routine that computes the anti-Sun pointing schedule for a
    given observer track and PointingConfig.

attach_pointing_to_observer(observer_track, schedule)
    Helper that writes selected PointingSchedule arrays back onto the
    observer track (dict or object), mirroring the pattern used in
    NEBULA_SKYFIELD_ILLUMINATION for illumination results.

Notes
-----
- This module performs *only* the pointing logic for the anti-Sun mode.
  It does not perform target-specific visibility checks or any sensor
  projection; those tasks are handled by NEBULA_VISIBILITY and
  downstream WCS / radiometry modules.
- All geometry is expressed in Earth-centered inertial coordinates
  consistent with NEBULA's existing use of Skyfield and TEME → ICRF
  transforms.
"""

# Import typing helpers for type annotations.
from dataclasses import dataclass
from typing import Any, Optional
import logging

# Import numpy for array math.
import numpy as np

# Import Skyfield tools for time, geometry, and frame transforms.
from skyfield.api import load, Distance, Velocity
from skyfield.positionlib import ICRF
from skyfield.sgp4lib import TEME

# Import NEBULA environment constants, in particular the Earth blocking radius.
from Configuration.NEBULA_ENV_CONFIG import R_BLOCK

# Import pointing configuration types (mode enum and config dataclass).
from Configuration.NEBULA_POINTING_CONFIG import PointingConfig, PointingMode

# Import ephemeris loader and track helpers from the shared Skyfield module.
from Utility.RADIOMETRY.NEBULA_SKYFIELD_ILLUMINATION import (
    load_de440s_ephemeris,
    _get_track_field,
    _set_track_field,
)

# --------------------------------------------------------------------------- #
# Data container for anti-Sun pointing results
# --------------------------------------------------------------------------- #


@dataclass
class PointingSchedule:
    """
    Container for per-timestep anti-Sun pointing outputs.

    This dataclass represents the time series of boresight geometry
    and environment flags produced by the anti-Sun pointing logic.

    Attributes
    ----------
    observer_name : str
        Name or identifier of the observer platform.  Taken from the
        'name' field of the observer track if present; otherwise a
        generic placeholder.
    mode : PointingMode
        Pointing mode used to generate this schedule.  Should be
        PointingMode.ANTI_SUN_STARE for this module.
    times : np.ndarray
        Array of datetime objects (shape (N,)) representing the time
        grid used for all subsequent arrays.
    boresight_ra_deg : np.ndarray
        Right ascension of the anti-Sun boresight in the ICRS frame,
        in degrees, shape (N,).
    boresight_dec_deg : np.ndarray
        Declination of the anti-Sun boresight in the ICRS frame, in
        degrees, shape (N,).
    roll_deg : float
        Focal-plane roll angle about the boresight, in degrees.  This
        is taken directly from PointingConfig.roll_deg and is assumed
        constant over the schedule.
    earth_blocked : np.ndarray
        Boolean array (shape (N,)) indicating whether the boresight
        intersects the Earth disk, using R_BLOCK + the limb margin
        from config.earth_avoid_margin_deg.
    sensor_is_sunlit : np.ndarray
        Boolean array (shape (N,)) indicating whether the observer
        platform is in direct sunlight (True) or in Earth's shadow
        (False), as determined by Skyfield's is_sunlit().
    sun_angle_deg : np.ndarray
        Angular separation between the anti-Sun boresight and the Sun
        direction, in degrees, shape (N,).  This value will be close
        to 180 degrees by construction, but is retained for debugging
        and for any future offset pointing strategies.
    sun_excluded : np.ndarray
        Boolean array (shape (N,)) indicating whether the boresight
        violates config.sun_exclusion_angle_deg (True if the Sun is
        closer than the requested exclusion angle).
    valid_for_projection : np.ndarray
        Boolean array (shape (N,)) indicating which timesteps are
        considered valid for downstream sensor projection, after
        combining Earth-blocking, sensor-sunlit, and Sun-exclusion
        constraints with the flags in PointingConfig.
    """

    # Store the name/identifier of the observer platform.
    observer_name: str
    # Store the pointing mode used to generate this schedule.
    mode: PointingMode
    # Store the time grid as a numpy array of datetime objects.
    times: np.ndarray
    # Store the boresight right ascension in degrees (ICRS).
    boresight_ra_deg: np.ndarray
    # Store the boresight declination in degrees (ICRS).
    boresight_dec_deg: np.ndarray
    # Store the focal-plane roll angle in degrees (assumed constant).
    roll_deg: float
    # Store the Earth-blocking flag for each timestep.
    earth_blocked: np.ndarray
    # Store the sensor-sunlit flag for each timestep.
    sensor_is_sunlit: np.ndarray
    # Store the Sun–boresight separation angle in degrees.
    sun_angle_deg: np.ndarray
    # Store the Sun-exclusion flag for each timestep.
    sun_excluded: np.ndarray
    # Store the final "valid for projection" mask for each timestep.
    valid_for_projection: np.ndarray


# --------------------------------------------------------------------------- #
# Core anti-Sun pointing routine
# --------------------------------------------------------------------------- #


def build_pointing_schedule_antisun(
    observer_track: Any,
    config: PointingConfig,
    eph=None,
    ephemeris_path: Optional[str] = None,
    store_on_observer: bool = True,
    logger: Optional[logging.Logger] = None,
) -> PointingSchedule:
    """
    Build an anti-Sun pointing schedule for a given observer track.

    This function computes, for each timestep in the observer track:

      - The anti-Sun boresight direction in ICRS (RA/Dec).
      - Whether that boresight intersects the Earth disk using R_BLOCK
        and config.earth_avoid_margin_deg.
      - Whether the observer platform is in direct sunlight using
        Skyfield's is_sunlit() method.
      - The angular separation between boresight and Sun direction, and
        whether that violates config.sun_exclusion_angle_deg.
      - A combined "valid_for_projection" mask that uses the flags in
        PointingConfig to decide which timesteps should be considered
        usable for sensor projection.

    Parameters
    ----------
    observer_track : dict or SatelliteTrack-like
        NEBULA observer track.  Must provide at least:
            - 'times'     : sequence of datetime objects (length N)
            - 'r_eci_km'  : array of shape (N, 3) with positions [km]
            - 'v_eci_km_s': array of shape (N, 3) with velocities [km/s]
          The track may be a dict (as produced by NEBULA_SAT_PICKLER)
          or an object with attributes of the same names.
    config : PointingConfig
        Pointing configuration object.  The mode field should be
        PointingMode.ANTI_SUN_STARE for this function.  The following
        fields are used:
            - roll_deg
            - earth_avoid_margin_deg
            - require_sensor_sunlit
            - sun_exclusion_angle_deg
    eph : skyfield.jpllib.SpiceKernel or None, optional
        Pre-loaded DE440s ephemeris.  If None, this function will load
        the ephemeris using load_de440s_ephemeris(ephemeris_path).
    ephemeris_path : str or None, optional
        Explicit path to "de440s.bsp".  Only used if eph is None.  If
        omitted, NEBULA_SKYFIELD_ILLUMINATION's project-relative default
        path is used, making this function portable across systems.
    store_on_observer : bool, optional
        If True, a subset of the resulting arrays is attached back onto
        observer_track using _set_track_field, with keys:
            - 'pointing_boresight_ra_deg'
            - 'pointing_boresight_dec_deg'
            - 'pointing_earth_blocked'
            - 'pointing_sensor_is_sunlit'
            - 'pointing_sun_angle_deg'
            - 'pointing_sun_excluded'
            - 'pointing_valid_for_projection'
        This mirrors the attachment pattern used by the illumination
        utilities and facilitates inspection in interactive tools.
    logger : logging.Logger or None, optional
        Optional logger for status messages and warnings.

    Returns
    -------
    PointingSchedule
        Dataclass containing the boresight RA/Dec, environment flags,
        and valid_for_projection mask on the observer's time grid.

    Raises
    ------
    KeyError or AttributeError
        If required fields are missing from observer_track.
    FileNotFoundError
        If the DE440s ephemeris cannot be found at the requested path.
    """
    # Create a logger if one was not supplied, using this module's name.
    log = logger or logging.getLogger(__name__)

    # Extract the time grid from the observer track using the shared helper.
    times = np.asarray(_get_track_field(observer_track, "times"), dtype=object)
    # Determine the number of timesteps from the length of the time array.
    n_steps = times.shape[0]

    # Extract the observer positions in ECI/TEME coordinates [km].
    r_obs = np.asarray(_get_track_field(observer_track, "r_eci_km"), dtype=float)[
        :n_steps, :
    ]
    # Extract the observer velocities in ECI/TEME coordinates [km/s].
    v_obs = np.asarray(_get_track_field(observer_track, "v_eci_km_s"), dtype=float)[
        :n_steps, :
    ]

    # Build a Skyfield Time array from the datetime objects.
    ts = load.timescale()
    # Convert the numpy datetime array into a Skyfield Time object.
    t = ts.from_datetimes(times.tolist())

    # If an ephemeris object was not passed in, load DE440s using the shared helper.
    if eph is None:
        eph = load_de440s_ephemeris(ephemeris_path=ephemeris_path, logger=log)

    # Extract the Earth body from the ephemeris.
    earth = eph["earth"]
    # Extract the Sun body from the ephemeris.
    sun = eph["sun"]

    # Compute the astrometric position of the Sun as seen from Earth at each timestep.
    astrometric_sun = earth.at(t).observe(sun)
    # Extract the Sun position vectors in km and transpose to shape (N, 3).
    r_sun_eci_km = astrometric_sun.position.km.T

    # Compute the unit vector pointing from Earth to the Sun at each timestep.
    sun_vec = r_sun_eci_km / np.linalg.norm(r_sun_eci_km, axis=1, keepdims=True)
    # Define the anti-Sun unit vector as the negative of the Sun direction.
    boresight_vec = -sun_vec

    # Extract components of the boresight unit vector for convenience.
    bx = boresight_vec[:, 0]
    # Extract the y-component of the boresight unit vector.
    by = boresight_vec[:, 1]
    # Extract the z-component of the boresight unit vector.
    bz = boresight_vec[:, 2]

    # Compute boresight right ascension in radians using arctan2(y, x).
    ra_rad = np.arctan2(by, bx)
    # Wrap RA into the [0, 2π) interval for consistency.
    ra_rad = np.mod(ra_rad, 2.0 * np.pi)
    # Convert RA from radians to degrees.
    boresight_ra_deg = np.degrees(ra_rad)

    # Compute boresight declination in radians using arcsin(z).
    dec_rad = np.arcsin(np.clip(bz, -1.0, 1.0))
    # Convert Declination from radians to degrees.
    boresight_dec_deg = np.degrees(dec_rad)

    # Compute the direction from observer to Earth center as a unit vector.
    dir_to_earth = -r_obs / np.linalg.norm(r_obs, axis=1, keepdims=True)

    # Compute the angular separation between boresight and Earth's center in radians.
    cos_theta_earth = np.einsum("ij,ij->i", boresight_vec, dir_to_earth)
    # Clip the cosine values to the valid domain of arccos to avoid numerical issues.
    cos_theta_earth = np.clip(cos_theta_earth, -1.0, 1.0)
    # Compute the angular separation angle in radians.
    theta_earth_rad = np.arccos(cos_theta_earth)

    # Compute the observer's distance from Earth's center at each timestep.
    r_obs_norm = np.linalg.norm(r_obs, axis=1)
    # Compute the apparent half-angle of the Earth disk as seen from the observer.
    alpha_rad = np.arcsin(np.clip(R_BLOCK / r_obs_norm, -1.0, 1.0))

    # Convert the Earth-avoidance margin from degrees to radians.
    margin_rad = np.deg2rad(config.earth_avoid_margin_deg)
    # Compute the effective half-angle including the avoidance margin.
    alpha_eff_rad = alpha_rad + margin_rad

    # Determine Earth-blocking: boresight intersects the Earth disk if
    # the separation is smaller than the effective half-angle.
    earth_blocked = theta_earth_rad <= alpha_eff_rad

    # Build Distance and Velocity objects for the observer states.
    dist_obs = Distance(km=r_obs.T)
    # Build the Velocity object from the observer velocity array.
    vel_obs = Velocity(km_per_s=v_obs.T)

    # Convert observer TEME states to Skyfield's internal ICRF frame.
    pos_obs_icrf = ICRF.from_time_and_frame_vectors(t, TEME, dist_obs, vel_obs)
    # Tell Skyfield that this is an Earth-centered orbit (NAIF id 399).
    pos_obs_icrf.center = 399

    # Use Skyfield to determine whether the observer is in direct sunlight.
    sensor_is_sunlit = np.asarray(pos_obs_icrf.is_sunlit(eph), dtype=bool)

    # Compute the cosine of the angle between boresight and Sun direction.
    cos_theta_sun = np.einsum("ij,ij->i", boresight_vec, sun_vec)
    # Clip the cosine values for numerical safety.
    cos_theta_sun = np.clip(cos_theta_sun, -1.0, 1.0)
    # Compute the separation angle in radians.
    theta_sun_rad = np.arccos(cos_theta_sun)
    # Convert the Sun–boresight separation angle to degrees.
    sun_angle_deg = np.degrees(theta_sun_rad)

    # Initialize the Sun-exclusion mask to all False by default.
    sun_excluded = np.zeros_like(sun_angle_deg, dtype=bool)
    # If a positive Sun-exclusion angle is requested in the config, apply it.
    if config.sun_exclusion_angle_deg > 0.0:
        # Mark timesteps where the Sun–boresight angle is smaller than the exclusion angle.
        sun_excluded = sun_angle_deg < float(config.sun_exclusion_angle_deg)

    # Build the base "valid" mask by requiring that the boresight is not Earth-blocked.
    valid = ~earth_blocked

    # If the config requires the sensor to be sunlit, enforce that constraint.
    if config.require_sensor_sunlit:
        # Combine the existing mask with the sensor_is_sunlit flag.
        valid = valid & sensor_is_sunlit

    # If a positive Sun-exclusion angle is configured, exclude those timesteps.
    if config.sun_exclusion_angle_deg > 0.0:
        # Combine the existing mask with the inverse of the Sun-excluded mask.
        valid = valid & (~sun_excluded)

    # Extract the observer name from the track if available, else use a placeholder.
    observer_name = _get_track_field(observer_track, "name", required=False) or "UNKNOWN"

    # Construct the PointingSchedule dataclass with all computed arrays.
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

    # If requested, attach selected arrays back onto the observer track for convenience.
    if store_on_observer:
        # Store the boresight RA series on the observer track.
        _set_track_field(observer_track, "pointing_boresight_ra_deg", boresight_ra_deg)
        # Store the boresight Dec series on the observer track.
        _set_track_field(observer_track, "pointing_boresight_dec_deg", boresight_dec_deg)
        # Store the Earth-blocking mask on the observer track.
        _set_track_field(observer_track, "pointing_earth_blocked", earth_blocked)
        # Store the sensor-sunlit mask on the observer track.
        _set_track_field(observer_track, "pointing_sensor_is_sunlit", sensor_is_sunlit)
        # Store the Sun–boresight separation angle on the observer track.
        _set_track_field(observer_track, "pointing_sun_angle_deg", sun_angle_deg)
        # Store the Sun-exclusion mask on the observer track.
        _set_track_field(observer_track, "pointing_sun_excluded", sun_excluded)
        # Store the final valid-for-projection mask on the observer track.
        _set_track_field(observer_track, "pointing_valid_for_projection", valid)

    # Return the constructed pointing schedule to the caller.
    return schedule


# --------------------------------------------------------------------------- #
# Optional helper for explicit attachment (mirrors pattern in illumination)
# --------------------------------------------------------------------------- #


def attach_pointing_to_observer(
    observer_track: Any,
    schedule: PointingSchedule,
) -> None:
    """
    Attach a PointingSchedule's arrays back onto an observer track.

    This helper simply writes the boresight and mask arrays from a
    PointingSchedule into fields on observer_track.  It is useful if
    build_pointing_schedule_antisun() was called with
    store_on_observer=False but you later decide that attaching the
    arrays would aid debugging or visualization.

    Parameters
    ----------
    observer_track : dict or SatelliteTrack-like
        NEBULA observer track to which the arrays will be attached.
    schedule : PointingSchedule
        Anti-Sun pointing schedule whose arrays should be written onto
        observer_track.

    Returns
    -------
    None
        The function modifies observer_track in place and returns None.
    """
    # Attach the boresight right ascension series to the observer track.
    _set_track_field(
        observer_track, "pointing_boresight_ra_deg", schedule.boresight_ra_deg
    )
    # Attach the boresight declination series to the observer track.
    _set_track_field(
        observer_track, "pointing_boresight_dec_deg", schedule.boresight_dec_deg
    )
    # Attach the Earth-blocking mask to the observer track.
    _set_track_field(
        observer_track, "pointing_earth_blocked", schedule.earth_blocked
    )
    # Attach the sensor-sunlit mask to the observer track.
    _set_track_field(
        observer_track, "pointing_sensor_is_sunlit", schedule.sensor_is_sunlit
    )
    # Attach the Sun–boresight separation angle series to the observer track.
    _set_track_field(
        observer_track, "pointing_sun_angle_deg", schedule.sun_angle_deg
    )
    # Attach the Sun-exclusion mask to the observer track.
    _set_track_field(
        observer_track, "pointing_sun_excluded", schedule.sun_excluded
    )
    # Attach the final valid-for-projection mask to the observer track.
    _set_track_field(
        observer_track, "pointing_valid_for_projection", schedule.valid_for_projection
    )
