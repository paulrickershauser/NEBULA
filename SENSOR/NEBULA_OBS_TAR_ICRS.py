"""
NEBULA_OBS_TAR_ICRS.py

Observer–target ICRS geometry helpers for the NEBULA simulation framework.

This module provides functions that operate on pickled NEBULA satellite
"track" dictionaries (or objects) and compute:

    1) Absolute positions in the ICRS frame for a single track, given
       its TEME position history and time stamps.

    2) The apparent line-of-sight (LOS) from an observer to a target,
       expressed as:
           - ICRS right ascension (deg)
           - ICRS declination (deg)
           - LOS range (km)
           - LOS unit vector in ICRS (dimensionless, 3-component)

These outputs are designed to be stored back onto the existing NEBULA
track objects and later consumed by NEBULA_WCS for pixel projection.

Conventions
-----------
- Input positions are assumed to be Cartesian TEME Earth-centered
  inertial coordinates in kilometers, e.g. from SGP4, stored in the
  "r_eci_km" field of each track.

- Input time stamps are taken from the "times" field of each track and
  are converted to astropy.time.Time with scale="utc".

- All intermediate transformations use Astropy's built-in TEME frame
  and its transformation graph (TEME -> GCRS -> ICRS) for consistency
  with Vallado's definition of TEME and modern ICRS practice.

- All returned positions, LOS vectors, and RA/Dec values are in ICRS,
  which is the same celestial frame used by NEBULA_WCS and by external
  catalogs such as Gaia.

Functions
---------
convert_track_teme_to_icrs(track) -> dict
    Given a NEBULA track with "times" and "r_eci_km" (TEME), compute
    the absolute ICRS Cartesian coordinates (x, y, z) and return them
    in a small dictionary suitable for attaching back to the track.

compute_observer_target_icrs_geometry(observer_track, target_track) -> dict
    Given an observer track and a target track, both with TEME
    positions and times, compute:
        - absolute observer ICRS position (x, y, z) [km],
        - absolute target ICRS position (x, y, z) [km],
        - observer->target ICRS line-of-sight RA/Dec [deg],
        - observer->target range [km],
        - observer->target LOS unit vector in ICRS (3-component).

    The returned dictionary is intended to be used by higher-level
    pickler modules (e.g. NEBULA_ICRS_PAIR_PICKLER) which will attach
    these arrays to the appropriate track objects.
"""

# Import future annotations to allow forward references in type hints.
from __future__ import annotations

# Import typing helpers for describing function arguments and return types.
from typing import Any, Dict, Sequence, Tuple

# Import NumPy for numerical array handling and vectorized math.
import numpy as np

# Import Astropy's SkyCoord, TEME frame, and CartesianRepresentation
# for robust coordinate frame handling and 3D vector representations.
from astropy.coordinates import SkyCoord, TEME, CartesianRepresentation

# Import Astropy Time for handling time stamps in a uniform way.
from astropy.time import Time

# Import Astropy units to attach physical units (km, deg, etc.) to quantities.
import astropy.units as u


# --------------------------------------------------------------------------- #
# Internal helper for accessing fields on NEBULA "track" objects
# --------------------------------------------------------------------------- #

def _get_track_field(track: Any, field_name: str, required: bool = True) -> Any:
    """
    Safely extract a field from a NEBULA track object.

    This helper mirrors the behavior used in other NEBULA modules and
    supports both dict-like and attribute-style access for the track
    fields.

    Parameters
    ----------
    track : Any
        NEBULA track object, typically a dictionary created by one of
        the SAT_OBJECTS pickler modules, but may also be a custom
        object with attributes.

    field_name : str
        Name of the field to extract from the track (e.g. "times",
        "r_eci_km", "name", etc.).

    required : bool, optional
        If True (default), raise an error if the field is missing.
        If False, return None when the field is not found.

    Returns
    -------
    Any
        The value stored under the requested field name.

    Raises
    ------
    KeyError
        If `track` is dict-like and the field is missing and
        required=True.

    AttributeError
        If `track` is object-like and the field is missing and
        required=True.
    """
    # Check whether the track behaves like a dictionary.
    if isinstance(track, dict):
        # If the field is present in the dictionary, return it.
        if field_name in track:
            return track[field_name]
        # If the field is required but missing, raise a KeyError.
        if required:
            raise KeyError(f"Track dict missing required field '{field_name}'")
        # If the field is not required and missing, return None.
        return None

    # If the track is not a dict, check for an attribute with the given name.
    if hasattr(track, field_name):
        # If the attribute exists, return its value.
        return getattr(track, field_name)

    # If the attribute is missing and required, raise an AttributeError.
    if required:
        raise AttributeError(f"Track object missing required field '{field_name}'")

    # If the attribute is not required and missing, return None.
    return None


# --------------------------------------------------------------------------- #
# Low-level TEME -> ICRS conversion for a single track
# --------------------------------------------------------------------------- #

def _teme_xyz_to_icrs_xyz(
    times: Sequence[Any],
    r_eci_km: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert TEME Cartesian positions to ICRS Cartesian positions.

    This internal helper performs the actual Astropy coordinate
    transformation from TEME (True Equator, Mean Equinox) to ICRS for
    a set of time-stamped positions.

    Parameters
    ----------
    times : sequence of datetime-like or Time-like
        Sequence of time stamps corresponding to the position samples.
        These are converted to astropy.time.Time with scale="utc".

    r_eci_km : numpy.ndarray, shape (N, 3)
        TEME Earth-centered inertial position vectors in kilometers,
        where N is the number of time steps.

    Returns
    -------
    x_icrs_km : numpy.ndarray, shape (N,)
        ICRS x-coordinates in kilometers.

    y_icrs_km : numpy.ndarray, shape (N,)
        ICRS y-coordinates in kilometers.

    z_icrs_km : numpy.ndarray, shape (N,)
        ICRS z-coordinates in kilometers.

    Notes
    -----
    - The transformation uses Astropy's TEME frame and its
      transformation graph (TEME -> GCRS -> ICRS), which follows the
      conventions described by Vallado (2006) and the SGP4 standard.
    """
    # Convert the input times to an Astropy Time object with UTC scale.
    t_astropy = Time(times, scale="utc")

    # Convert the input position array to a NumPy array of type float.
    r_arr = np.asarray(r_eci_km, dtype=float)

    # Check that the final dimension of the position array has length 3.
    if r_arr.shape[-1] != 3:
        # If not, raise a ValueError to signal incorrect input shape.
        raise ValueError(
            f"r_eci_km must have shape (N, 3); got {r_arr.shape}"
        )

    # Extract the x-component of the TEME positions and attach km units.
    x_teme = r_arr[:, 0] * u.km

    # Extract the y-component of the TEME positions and attach km units.
    y_teme = r_arr[:, 1] * u.km

    # Extract the z-component of the TEME positions and attach km units.
    z_teme = r_arr[:, 2] * u.km

    # Build a CartesianRepresentation from the TEME position components.
    teme_rep = CartesianRepresentation(x=x_teme, y=y_teme, z=z_teme)

    # Construct a SkyCoord in the TEME frame with the given positions and times.
    coord_teme = SkyCoord(teme_rep, frame=TEME(obstime=t_astropy))

    # Transform the TEME coordinates to the ICRS frame.
    coord_icrs = coord_teme.icrs

    # Extract the CartesianRepresentation of the ICRS coordinates.
    icrs_rep = coord_icrs.cartesian

    # Convert the ICRS x-component to a plain NumPy array in kilometers.
    x_icrs_km = icrs_rep.x.to_value(u.km)

    # Convert the ICRS y-component to a plain NumPy array in kilometers.
    y_icrs_km = icrs_rep.y.to_value(u.km)

    # Convert the ICRS z-component to a plain NumPy array in kilometers.
    z_icrs_km = icrs_rep.z.to_value(u.km)

    # Return the three ICRS coordinate component arrays.
    return x_icrs_km, y_icrs_km, z_icrs_km


def convert_track_teme_to_icrs(track: Any) -> Dict[str, np.ndarray]:
    """
    Compute absolute ICRS Cartesian coordinates for a single NEBULA track.

    This function reads the "times" and "r_eci_km" fields from a NEBULA
    track (typically produced by NEBULA_SAT_PICKLER / NEBULA_SCHEDULE_PICKLER),
    interprets "r_eci_km" as TEME positions, and converts them to ICRS
    Cartesian coordinates using Astropy.

    Parameters
    ----------
    track : Any
        NEBULA track dictionary or object.  It must contain:
            - "times" : sequence of datetime-like objects, and
            - "r_eci_km" : array-like of shape (N, 3) with TEME positions [km].

    Returns
    -------
    icrs_state : dict
        Dictionary containing:
            - "icrs_x_km" : numpy.ndarray, shape (N,)
            - "icrs_y_km" : numpy.ndarray, shape (N,)
            - "icrs_z_km" : numpy.ndarray, shape (N,)

        These arrays can be attached back onto the track object by
        higher-level code or picklers.

    Raises
    ------
    KeyError or AttributeError
        If the required fields "times" or "r_eci_km" are missing.

    ValueError
        If the "r_eci_km" array does not have shape (N, 3).
    """
    # Extract the time stamps from the track using the helper function.
    times = _get_track_field(track, "times")

    # Extract the TEME position array from the track.
    r_eci_km = _get_track_field(track, "r_eci_km")

    # Call the internal helper to convert TEME positions to ICRS positions.
    x_icrs_km, y_icrs_km, z_icrs_km = _teme_xyz_to_icrs_xyz(times, r_eci_km)

    # Build a small dictionary bundling the ICRS component arrays.
    icrs_state: Dict[str, np.ndarray] = {
        "icrs_x_km": x_icrs_km,
        "icrs_y_km": y_icrs_km,
        "icrs_z_km": z_icrs_km,
    }

    # Return the dictionary of ICRS state arrays.
    return icrs_state


# --------------------------------------------------------------------------- #
# Observer–target ICRS geometry (line-of-sight)
# --------------------------------------------------------------------------- #

def compute_observer_target_icrs_geometry(
    observer_track: Any,
    target_track: Any,
) -> Dict[str, np.ndarray]:
    """
    Compute ICRS observer–target line-of-sight geometry for a track pair.

    This function takes an observer track (e.g., SBSS) and a target track
    (e.g., a GEO satellite), each with TEME positions and times, and
    computes:

        - Absolute observer ICRS Cartesian coordinates (x, y, z) [km].
        - Absolute target ICRS Cartesian coordinates (x, y, z) [km].
        - Observer->target ICRS line-of-sight right ascension [deg].
        - Observer->target ICRS line-of-sight declination [deg].
        - Observer->target range [km] (Euclidean distance in ICRS).
        - Observer->target LOS unit vector in ICRS (dimensionless, shape (N, 3)).

    The returned arrays are meant to be attached to the corresponding
    tracks by a higher-level pickler, for example under per-observer
    keys on the target track.

    Parameters
    ----------
    observer_track : Any
        NEBULA track dictionary or object representing the observer
        (sensor-carrying satellite).  Must contain:
            - "times" : sequence of datetime-like objects,
            - "r_eci_km" : array-like of shape (N, 3) with TEME positions [km].

    target_track : Any
        NEBULA track dictionary or object representing the target
        satellite.  Must contain the same fields as the observer:
            - "times" : sequence of datetime-like objects,
            - "r_eci_km" : array-like of shape (N, 3) with TEME positions [km].

        The "times" fields for observer and target are expected to be
        aligned (same length and same epochs) as produced by the NEBULA
        propagation and scheduling pipeline.

    Returns
    -------
    geom : dict
        Dictionary containing:
            - "obs_icrs_x_km"      : numpy.ndarray, shape (N,)
            - "obs_icrs_y_km"      : numpy.ndarray, shape (N,)
            - "obs_icrs_z_km"      : numpy.ndarray, shape (N,)
            - "tar_icrs_x_km"      : numpy.ndarray, shape (N,)
            - "tar_icrs_y_km"      : numpy.ndarray, shape (N,)
            - "tar_icrs_z_km"      : numpy.ndarray, shape (N,)
            - "los_icrs_ra_deg"    : numpy.ndarray, shape (N,)
            - "los_icrs_dec_deg"   : numpy.ndarray, shape (N,)
            - "los_icrs_range_km"  : numpy.ndarray, shape (N,)
            - "los_icrs_unit_vec"  : numpy.ndarray, shape (N, 3)

    Raises
    ------
    ValueError
        If the observer and target have mismatched time arrays or
        incompatible position array shapes.
    """
    # Extract the observer time stamps using the helper function.
    obs_times = _get_track_field(observer_track, "times")

    # Extract the target time stamps using the helper function.
    tar_times = _get_track_field(target_track, "times")

    # Convert the observer times to an Astropy Time object for comparison.
    obs_t_astropy = Time(obs_times, scale="utc")

    # Convert the target times to an Astropy Time object for comparison.
    tar_t_astropy = Time(tar_times, scale="utc")

    # Check that the observer and target have the same number of time samples.
    if obs_t_astropy.shape != tar_t_astropy.shape:
        # If the shapes differ, raise a ValueError indicating the mismatch.
        raise ValueError(
            f"Observer and target times have different shapes: "
            f"{obs_t_astropy.shape} vs {tar_t_astropy.shape}"
        )

    # Check that the actual time values are identical (within Astropy's compare).
    if not np.all(obs_t_astropy == tar_t_astropy):
        # If any time stamp differs, raise a ValueError describing the problem.
        raise ValueError(
            "Observer and target times are not aligned; NEBULA scheduling "
            "is expected to provide matching time grids."
        )

    # Extract the observer TEME position array from the track.
    obs_r_eci_km = _get_track_field(observer_track, "r_eci_km")

    # Extract the target TEME position array from the track.
    tar_r_eci_km = _get_track_field(target_track, "r_eci_km")

    # Convert the observer TEME positions to ICRS Cartesian coordinates.
    obs_x_icrs_km, obs_y_icrs_km, obs_z_icrs_km = _teme_xyz_to_icrs_xyz(
        obs_times,
        obs_r_eci_km,
    )

    # Convert the target TEME positions to ICRS Cartesian coordinates.
    tar_x_icrs_km, tar_y_icrs_km, tar_z_icrs_km = _teme_xyz_to_icrs_xyz(
        tar_times,
        tar_r_eci_km,
    )

    # Stack the observer ICRS components into a single (N, 3) array for convenience.
    obs_icrs_xyz = np.stack(
        [obs_x_icrs_km, obs_y_icrs_km, obs_z_icrs_km],
        axis=1,
    )

    # Stack the target ICRS components into a single (N, 3) array for convenience.
    tar_icrs_xyz = np.stack(
        [tar_x_icrs_km, tar_y_icrs_km, tar_z_icrs_km],
        axis=1,
    )

    # Compute the line-of-sight vector from observer to target in ICRS (km).
    los_icrs_xyz = tar_icrs_xyz - obs_icrs_xyz  # shape (N, 3), units: km

    # For RA/Dec, we only care about direction, so we can reuse the km scale.
    # Attach km units to each component to build a SkyCoord for angular coords.
    los_x_icrs = los_icrs_xyz[:, 0] * u.km
    los_y_icrs = los_icrs_xyz[:, 1] * u.km
    los_z_icrs = los_icrs_xyz[:, 2] * u.km

    # Build a CartesianRepresentation from the LOS vector components.
    los_rep = CartesianRepresentation(
        x=los_x_icrs,
        y=los_y_icrs,
        z=los_z_icrs,
    )

    # Construct a SkyCoord in the ICRS frame from the LOS representation.
    los_coord_icrs = SkyCoord(los_rep, frame="icrs")

    # Extract the LOS right ascension in degrees from the ICRS SkyCoord.
    los_icrs_ra_deg = los_coord_icrs.ra.to_value(u.deg)

    # Extract the LOS declination in degrees from the ICRS SkyCoord.
    los_icrs_dec_deg = los_coord_icrs.dec.to_value(u.deg)

    # ------------------------------------------------------------------
    # Range and unit vector: do this explicitly with NumPy.
    # ------------------------------------------------------------------

    # Compute the Euclidean norm of the LOS vectors to get the range [km].
    los_icrs_range_km = np.linalg.norm(los_icrs_xyz, axis=1)

    # Avoid divide-by-zero by clipping extremely small ranges.
    # (In practice, observer and target should never coincide exactly.)
    eps = 1e-12
    safe_range = np.clip(los_icrs_range_km, eps, None)

    # Compute the unit line-of-sight vector in ICRS (dimensionless).
    los_icrs_unit_vec = los_icrs_xyz / safe_range[:, None]


    # Build the geometry dictionary bundling all outputs for this observer–target pair.
    geom: Dict[str, np.ndarray] = {
        "obs_icrs_x_km": obs_x_icrs_km,
        "obs_icrs_y_km": obs_y_icrs_km,
        "obs_icrs_z_km": obs_z_icrs_km,
        "tar_icrs_x_km": tar_x_icrs_km,
        "tar_icrs_y_km": tar_y_icrs_km,
        "tar_icrs_z_km": tar_z_icrs_km,
        "los_icrs_ra_deg": los_icrs_ra_deg,
        "los_icrs_dec_deg": los_icrs_dec_deg,
        "los_icrs_range_km": los_icrs_range_km,
        "los_icrs_unit_vec": los_icrs_unit_vec,
    }

    # Return the dictionary containing the observer–target ICRS geometry.
    return geom
