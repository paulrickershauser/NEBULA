"""
NEBULA_VIS_PICKLE.py

High-level wrappers for computing line-of-sight (LOS) visibility over time
between an observer and a target SatelliteTrack loaded from NEBULA pickles.

This module is designed to sit on top of:
    - Utility.NEBULA_VISIBILITY_DISPATCHER.dispatch_los_snapshot
    - The "track" objects produced by NEBULA_SAT_PICKLER (or similar),
      which are assumed to expose:
          times : array-like of shape (N,)
          r_eci : array-like of shape (N, 3)
          a, e, i, raan, nu : scalar or array-like of shape (N,)

Two main functions are provided:

    1) evaluate_los_timeseries_for_pair(observer_track, target_track, ...)
       → returns a dict of arrays: times, visible, h, delta_nu_lim, regime, fallback.

    2) attach_los_visibility_to_target(observer_track, target_track, ...)
       → runs the evaluation AND attaches a 0/1 visibility array and related
         LOS metadata directly onto the *target* track object, so that the
         target "remembers" its visibility relative to the chosen observer.

All distances are in kilometers, all angles in radians.
"""

# --------------------------------------------------------------------------- #
# Standard library imports
# --------------------------------------------------------------------------- #

import logging  # For optional warnings and debug info about mismatches.
from typing import Any, Dict, Optional  # For type annotations.

# --------------------------------------------------------------------------- #
# Third-party imports
# --------------------------------------------------------------------------- #

import numpy as np  # For array handling and numeric operations.

# --------------------------------------------------------------------------- #
# NEBULA configuration imports
# --------------------------------------------------------------------------- #

# Import the effective blocking radius R_BLOCK (Earth radius + buffer) used
# for the strict h > Rb visibility test.
from Configuration.NEBULA_ENV_CONFIG import R_BLOCK

# --------------------------------------------------------------------------- #
# NEBULA visibility dispatcher import
# --------------------------------------------------------------------------- #

# Import the snapshot-level LOS dispatcher that selects the Cinelli regime
# and computes LOS geometry for a single time instant.
from Utility.LOS.NEBULA_VISIBILITY_DISPATCHER import dispatch_los_snapshot


# --------------------------------------------------------------------------- #
# Helper: generic getter for track fields
# --------------------------------------------------------------------------- #

def _get_track_field(track: Any, key: str) -> Any:
    """
    Retrieve a field from a "track-like" object.

    This helper tries two access patterns in order:
        1. Attribute access:  getattr(track, key)
        2. Dictionary access: track[key]   (if track is a dict)

    Parameters
    ----------
    track : Any
        The observer or target track object. Typically this is either a
        dataclass / simple object with attributes, or a dict-like object.
    key : str
        Name of the field to retrieve (e.g., "times", "r_eci", "a", "e", "i").

    Returns
    -------
    Any
        The value stored under the requested field.

    Raises
    ------
    AttributeError
        If the field cannot be found as either an attribute or a dict key.
    """
    # First, check if the track exposes the field as an attribute.
    if hasattr(track, key):
        return getattr(track, key)

    # Next, if the track behaves like a dictionary, try key-based access.
    if isinstance(track, dict) and key in track:
        return track[key]

    # If neither access pattern worked, raise a clear error to the caller.
    raise AttributeError(
        f"Track object of type {type(track)!r} has no field '{key}' "
        "as either an attribute or dictionary key."
    )


# --------------------------------------------------------------------------- #
# Helper: scalar-or-series orbital element extractor
# --------------------------------------------------------------------------- #

def _get_scalar_or_series_value(track: Any, key: str, index: int) -> float:
    """
    Extract a scalar orbital element at a given index, allowing either
    a scalar or a 1D time-series array.

    This helper is designed for fields such as 'a', 'e', 'i', 'raan', 'nu'
    that may be stored as:
        - a single scalar value (constant over the track), or
        - a 1D array whose length matches the time grid.

    Parameters
    ----------
    track : Any
        The observer or target track.
    key : str
        Name of the orbital element field (e.g., "a", "e", "i", "raan", "nu").
    index : int
        Time index at which the scalar value should be extracted.

    Returns
    -------
    float
        The scalar orbital element value at the requested index.
    """
    # Retrieve the raw field (could be scalar or array-like).
    value = _get_track_field(track, key)

    # Convert the raw field to a numpy array of floats for uniform handling.
    arr = np.asarray(value, dtype=float)

    # If the converted value is 0D (scalar), return it directly for any index.
    if arr.ndim == 0:
        return float(arr)

    # Otherwise, treat it as a 1D time-series and return the value at this index.
    return float(arr[index])


# --------------------------------------------------------------------------- #
# Helper: generic setter for track fields
# --------------------------------------------------------------------------- #

def _set_track_field(track: Any, key: str, value: Any) -> None:
    """
    Set a field on a "track-like" object, handling both attribute-style
    and dictionary-style storage.

    For objects like SatelliteTrack (dataclasses with a __dict__), we allow
    adding *new* attributes via setattr. For plain dicts, we assign into the
    mapping. For anything else, we raise a clear error.
    """
    # If the track behaves like a dictionary, store the value under the key.
    if isinstance(track, dict):
        track[key] = value
        return

    # If the object has a __dict__, we can safely add or overwrite attributes.
    if hasattr(track, "__dict__"):
        setattr(track, key, value)
        return

    # Otherwise, we don't know how to set a field on this object.
    raise AttributeError(
        f"Cannot set field '{key}' on track of type {type(track)!r}: "
        "object is neither dict-like nor supports attribute assignment."
    )



# --------------------------------------------------------------------------- #
# Core function: evaluate LOS timeseries for an observer/target pair
# --------------------------------------------------------------------------- #

def evaluate_los_timeseries_for_pair(
    observer_track: Any,
    target_track: Any,
    Rb: float = R_BLOCK,
    custom_tolerances: Optional[Dict[str, float]] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Evaluate line-of-sight (LOS) visibility over time between an observer
    and a target track using the NEBULA_VISIBILITY_DISPATCHER snapshot logic.

    This function assumes that the observer and target tracks share a common
    time grid (same length and, ideally, identical timestamps) and that each
    track provides:
        - times : array-like of shape (N,)
        - r_eci : array-like of shape (N, 3)
        - a, e, i, raan, nu : scalar or array-like of shape (N,)

    At each timestep k, the function:
        1. Extracts the instantaneous states (r1, r2, a1, e1, i1, raan1, nu1,
           a2, e2, i2, raan2, nu2).
        2. Calls dispatch_los_snapshot(...) to select the appropriate Cinelli
           regime and compute LOS for that snapshot.
        3. Stores:
              visible[k]      : bool, h[k] : float,
              delta_nu_lim[k] : float (or NaN if not available),
              regime[k]       : str,
              fallback[k]     : bool.

    Parameters
    ----------
    observer_track : Any
        Track-like object representing the observer satellite. Must expose
        the fields described above (either as attributes or dict keys).
    target_track : Any
        Track-like object representing the target satellite, with the same
        conventions as observer_track.
    Rb : float, optional
        Blocking radius [km] used in the strict LOS test h > Rb. By default,
        this is NEBULA_ENV_CONFIG.R_BLOCK.
    custom_tolerances : dict or None, optional
        Optional mapping of tolerance-name → float that overrides any of the
        default regime classification tolerances defined in
        NEBULA_VISIBILITY_DISPATCHER (e.g. "ecc_circular_max", "ecc_weak_max").
    logger : logging.Logger or None, optional
        Optional logger instance. If provided, this function will emit a
        warning if the observer and target time grids do not match exactly.

    Returns
    -------
    Dict[str, Any]
        Dictionary with the following keys:

        - "times"        : np.ndarray, shape (N,)
            The time grid used for the LOS evaluation (observer times
            truncated to the common length with the target, if necessary).
        - "visible"      : np.ndarray, shape (N,), dtype=bool
            True/False LOS visibility at each timestep based on h > Rb.
        - "h"            : np.ndarray, shape (N,), dtype=float
            Minimum Earth-center distance to the LOS line [km] at each step.
        - "delta_nu_lim" : np.ndarray, shape (N,), dtype=float
            Symmetric phase limits |Δν|_lim [rad]; NaN where no analytic
            limit is available (generic fallback).
        - "regime"       : np.ndarray, shape (N,), dtype=object
            String label of the active Cinelli regime at each timestep
            (e.g., "coplanar_same_radius", "weakly_eccentric/active_52").
        - "fallback"     : np.ndarray, shape (N,), dtype=bool
            True where the dispatcher resorted to the generic geometry
            fallback, False otherwise.
    """
    # ------------------------------------------------------------------ #
    # Extract and align the time grids for observer and target.
    # ------------------------------------------------------------------ #

    # Retrieve the observer time array and coerce it to a numpy array.
    times_obs = np.asarray(_get_track_field(observer_track, "times"))
    # Retrieve the target time array and coerce it to a numpy array.
    times_tar = np.asarray(_get_track_field(target_track, "times"))

    # Compute the number of timesteps for observer and target separately.
    n_obs = int(times_obs.shape[0])
    n_tar = int(times_tar.shape[0])

    # Determine the number of steps to evaluate as the minimum of the two.
    n_steps = min(n_obs, n_tar)

    # If the lengths differ, optionally warn and truncate to the common length.
    if n_obs != n_tar and logger is not None:
        logger.warning(
            "evaluate_los_timeseries_for_pair: observer and target time grids "
            "have different lengths (obs=%d, tar=%d); truncating to %d steps.",
            n_obs,
            n_tar,
            n_steps,
        )

    # Truncate both time arrays to the common number of steps.
    times = times_obs[:n_steps]

    # If the actual timestamp values differ (but lengths match), warn if desired.
    if n_obs == n_tar:
        if not np.array_equal(times_obs, times_tar) and logger is not None:
            logger.warning(
                "evaluate_los_timeseries_for_pair: observer and target times "
                "are not identical; using observer times for output."
            )

    # ------------------------------------------------------------------ #
    # Extract and align the ECI position arrays.
    # ------------------------------------------------------------------ #

    # Retrieve the observer ECI position array and coerce it to float.
    # SatelliteTrack stores this as r_eci_km with shape (N, 3).
    r1_all = np.asarray(_get_track_field(observer_track, "r_eci_km"), dtype=float)
    # Retrieve the target ECI position array and coerce it to float.
    r2_all = np.asarray(_get_track_field(target_track, "r_eci_km"), dtype=float)


    # Ensure we do not exceed the available number of position entries.
    n_steps = min(n_steps, int(r1_all.shape[0]), int(r2_all.shape[0]))
    # Truncate the time array once more if position arrays are shorter.
    times = times[:n_steps]

    # ------------------------------------------------------------------ #
    # Pre-allocate output arrays for LOS evaluation results.
    # ------------------------------------------------------------------ #

    # Allocate a boolean array for the visibility flag at each timestep.
    visible = np.zeros(n_steps, dtype=bool)
    # Allocate a float array for the minimum LOS distance h at each timestep.
    h = np.full(n_steps, np.nan, dtype=float)
    # Allocate a float array for the symmetric phase limit |Δν|_lim at each step.
    delta_nu_lim = np.full(n_steps, np.nan, dtype=float)
    # Allocate an object array for the regime label at each timestep.
    regime = np.empty(n_steps, dtype=object)
    # Allocate a boolean array indicating whether the generic fallback was used.
    fallback = np.zeros(n_steps, dtype=bool)

    # ------------------------------------------------------------------ #
    # Main loop: iterate over timesteps and call the snapshot dispatcher.
    # ------------------------------------------------------------------ #

    for k in range(n_steps):
        # Extract the observer position vector r1 at timestep k (shape (3,)).
        r1 = np.asarray(r1_all[k, :], dtype=float)
        # Extract the target position vector r2 at timestep k (shape (3,)).
        r2 = np.asarray(r2_all[k, :], dtype=float)

        # Extract scalar orbital elements for the observer at timestep k.
        # SatelliteTrack stores:
        #   a_km      : semi-major axis [km]
        #   e         : eccentricity
        #   inc_rad   : inclination [rad]
        #   raan_rad  : RAAN [rad]
        #   nu_rad    : true anomaly [rad]
        a1 = _get_scalar_or_series_value(observer_track, "a_km", k)
        e1 = _get_scalar_or_series_value(observer_track, "e", k)
        i1 = _get_scalar_or_series_value(observer_track, "inc_rad", k)
        raan1 = _get_scalar_or_series_value(observer_track, "raan_rad", k)
        nu1 = _get_scalar_or_series_value(observer_track, "nu_rad", k)

        # Extract scalar orbital elements for the target at timestep k.
        a2 = _get_scalar_or_series_value(target_track, "a_km", k)
        e2 = _get_scalar_or_series_value(target_track, "e", k)
        i2 = _get_scalar_or_series_value(target_track, "inc_rad", k)
        raan2 = _get_scalar_or_series_value(target_track, "raan_rad", k)
        nu2 = _get_scalar_or_series_value(target_track, "nu_rad", k)


        # Call the snapshot dispatcher to classify the regime and compute LOS.
        res = dispatch_los_snapshot(
            r1=r1,
            r2=r2,
            a1=a1,
            e1=e1,
            i1=i1,
            raan1=raan1,
            nu1=nu1,
            a2=a2,
            e2=e2,
            i2=i2,
            raan2=raan2,
            nu2=nu2,
            Rb=Rb,
            custom_tolerances=custom_tolerances,
        )

        # Store the visibility flag (True/False) for this timestep.
        visible[k] = bool(res.visible)
        # Store the minimum LOS distance h [km] for this timestep.
        h[k] = float(res.h)

        # Attempt to extract the scalar phase limit from the 1D array.
        try:
            delta_nu_lim[k] = float(res.delta_nu_lim[0])
        except Exception:
            delta_nu_lim[k] = float("nan")

        # Store the regime label string for this timestep.
        regime[k] = res.regime
        # Store whether the generic fallback path was used.
        fallback[k] = bool(res.fallback)

    # ------------------------------------------------------------------ #
    # Package the results into a dictionary and return.
    # ------------------------------------------------------------------ #

    result: Dict[str, Any] = {
        "times": times,
        "visible": visible,
        "h": h,
        "delta_nu_lim": delta_nu_lim,
        "regime": regime,
        "fallback": fallback,
    }

    return result


# --------------------------------------------------------------------------- #
# Convenience function: attach LOS visibility directly to target track
# --------------------------------------------------------------------------- #

def attach_los_visibility_to_target(
    observer_track: Any,
    target_track: Any,
    visibility_field: str = "los_visible",
    h_field: str = "los_h",
    regime_field: str = "los_regime",
    fallback_field: str = "los_fallback",
    Rb: float = R_BLOCK,
    custom_tolerances: Optional[Dict[str, float]] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Compute LOS visibility between an observer and a target over time and
    ATTACH the results directly to the target track.

    Conceptually, this function:
        1. Calls evaluate_los_timeseries_for_pair(...) to obtain:
               times[k], visible[k], h[k], delta_nu_lim[k],
               regime[k], fallback[k].
        2. Converts the boolean visibility array to a 0/1 integer array.
        3. Writes the following new fields onto the target track:
               - visibility_field : np.ndarray[int]  (0 or 1)
               - h_field          : np.ndarray[float]
               - regime_field     : np.ndarray[object]
               - fallback_field   : np.ndarray[bool]
        4. Returns the full result dictionary from the underlying
           timeseries evaluation for optional further analysis.

    This matches the pattern:
        "append a visibility line-of-sight boolean (0 or 1) to the
         target object's attributes in an array that matches the
         length of the time array."

    Parameters
    ----------
    observer_track : Any
        Track-like object representing the observer satellite. Must expose:
            times, r_eci, a, e, i, raan, nu
        either as attributes or dictionary keys.
    target_track : Any
        Track-like object representing the target satellite, with the same
        field conventions as observer_track. This object will be modified
        in-place to include LOS visibility information relative to the
        given observer.
    visibility_field : str, optional
        Name of the field on target_track under which the 0/1 visibility
        array will be stored. Defaults to "los_visible".
    h_field : str, optional
        Name of the field on target_track under which the LOS distance h[k]
        array will be stored. Defaults to "los_h".
    regime_field : str, optional
        Name of the field on target_track under which the string regime
        labels will be stored. Defaults to "los_regime".
    fallback_field : str, optional
        Name of the field on target_track under which the boolean fallback
        flags will be stored. Defaults to "los_fallback".
    Rb : float, optional
        Blocking radius [km] used for the strict visibility criterion
        h > Rb. Defaults to NEBULA_ENV_CONFIG.R_BLOCK.
    custom_tolerances : dict or None, optional
        Optional mapping of tolerance-name → float that overrides the
        default regime classification tolerances used by the dispatcher.
    logger : logging.Logger or None, optional
        Optional logger instance used for any time-grid mismatch warnings.

    Returns
    -------
    Dict[str, Any]
        The full result dictionary returned by
        evaluate_los_timeseries_for_pair(...), containing:
            - "times"        : np.ndarray (N,)
            - "visible"      : np.ndarray (N,) dtype=bool
            - "h"            : np.ndarray (N,) dtype=float
            - "delta_nu_lim" : np.ndarray (N,) dtype=float
            - "regime"       : np.ndarray (N,) dtype=object
            - "fallback"     : np.ndarray (N,) dtype=bool

        Note: target_track is also modified in-place to carry a subset
        of these arrays as new fields.
    """
    # Step 1: run the underlying timeseries LOS evaluation for this pair.
    result = evaluate_los_timeseries_for_pair(
        observer_track=observer_track,
        target_track=target_track,
        Rb=Rb,
        custom_tolerances=custom_tolerances,
        logger=logger,
    )

    # Step 2: convert boolean visibility into a 0/1 integer array.
    visible_bool = np.asarray(result["visible"], dtype=bool)
    visible_int = visible_bool.astype(np.int8)

    # Step 3: attach LOS-related arrays directly to the target track.
    _set_track_field(target_track, visibility_field, visible_int)
    _set_track_field(target_track, h_field, np.asarray(result["h"], dtype=float))
    _set_track_field(target_track, regime_field, np.asarray(result["regime"], dtype=object))
    _set_track_field(target_track, fallback_field, np.asarray(result["fallback"], dtype=bool))

    # Step 4: return the full result dictionary for optional inspection.
    return result
