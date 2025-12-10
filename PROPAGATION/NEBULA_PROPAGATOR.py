"""
NEBULA_PROPAGATOR.py

Satellite propagation utilities for the NEBULA
(Neuromorphic Event-Based Luminance Asset-tracking) simulation framework.

This module provides two core utilities:

  * propagate_teme_state:
        Propagate a single `Satrec` object (from SGP4) to a list of
        UTC times, returning position and velocity vectors in the TEME
        frame (km and km/s).

  * compute_adaptive_dt:
        Given two semi-major axes and an analytic limit on true-anomaly
        separation (delta_f_lim), compute a time step that resolves the
        predicted maximum visibility window sufficiently, otherwise
        fall back to a default base step.

These functions are the NEBULA equivalents of the ones in the original
AMOS_propagator module, adapted to use NEBULA configuration where
appropriate (e.g. MU_EARTH from NEBULA_ENV_CONFIG). :contentReference[oaicite:2]{index=2}

Design notes
------------
* This file lives in NEBULA/Utility/ because it provides "gears"
  (numeric propagation and dt selection), not configuration constants.

* The implementation is deliberately functional: each function is
  stateless given its arguments.  In the future, you may choose to wrap
  these in a higher-level `NebulaPropagator` class, but that is not
  required to use them effectively.
"""

# Import Satrec (SGP4 satellite record) and jday (Julian date helper)
# from the sgp4 library.  Satrec holds the TLE-derived orbit parameters,
# and jday converts a UTC datetime into Julian date + fraction for SGP4.
from sgp4.api import jday, Satrec

# Import datetime and timezone so we can ensure that input times are
# timezone-aware UTC, as required by SGP4 propagation.
from datetime import datetime, timezone

# Import typing helpers: List and Tuple for annotating argument and
# return types.  You can also switch to Sequence in the future.
from typing import List, Tuple

# Import numpy so we can return arrays of positions and velocities in
# a convenient form for later numeric operations.
import numpy as np

# Import logging so we can emit useful messages or error reports when
# propagation fails for a given epoch.
import logging

# Import math for square roots and absolute-value operations needed in
# the adaptive time-step calculation.
import math

# Import Earth's gravitational parameter from the NEBULA environment
# configuration instead of hard-coding it in this module.  This keeps
# physical constants centralized in NEBULA_ENV_CONFIG.
from Configuration.NEBULA_ENV_CONFIG import MU_EARTH  # type: ignore


def propagate_teme_state(sat: Satrec, times: List[datetime]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Propagate a Satrec object to each specified epoch and return state vectors.

    This function is a NEBULA version of the original AMOS_propagator
    propagate_teme_state. :contentReference[oaicite:3]{index=3}

    It expects a list of timezone-aware UTC datetime objects and uses
    the SGP4 library to compute the satellite state in the TEME frame
    at each epoch.

    Parameters
    ----------
    sat : Satrec
        The SGP4 satellite record to propagate (typically built from TLE).

    times : List[datetime]
        A list of datetime objects, each of which must be timezone-aware
        and convertible to UTC (i.e., have tzinfo not None).

    Returns
    -------
    (np.ndarray, np.ndarray)
        A pair (positions, velocities), where:

          * positions has shape (N, 3) with units of km,
          * velocities has shape (N, 3) with units of km/s,

        and N is the number of input times.

    Raises
    ------
    ValueError
        If any datetime in `times` is naive (no tzinfo).

    RuntimeError
        If SGP4 returns a non-zero error code for any epoch.
    """

    # Initialize Python lists to accumulate the position and velocity
    # vectors as we propagate to each time.  We will convert these
    # lists to numpy arrays at the end.
    positions = []
    velocities = []

    # Loop through each requested epoch in the input list.
    for t in times:
        # If the datetime has no timezone information (tzinfo is None),
        # we reject it to avoid ambiguous propagation results.
        if t.tzinfo is None:
            raise ValueError(f"Datetime {t!r} must be timezone-aware (with tzinfo) and in UTC")

        # Normalize the time to UTC explicitly, in case it has another
        # timezone attached (e.g., local time).  SGP4 expects UTC input.
        t_utc = t.astimezone(timezone.utc)

        # Convert the UTC datetime into the Julian date (jd) and
        # fractional part (fr) that SGP4 uses internally.
        jd, fr = jday(
            t_utc.year,
            t_utc.month,
            t_utc.day,
            t_utc.hour,
            t_utc.minute,
            # Combine seconds and microseconds into a floating-point second.
            t_utc.second + t_utc.microsecond * 1e-6,
        )

        # Call the SGP4 propagator for this Julian date, which returns
        # an error code plus position (r) and velocity (v) in TEME.
        error_code, r, v = sat.sgp4(jd, fr)

        # If the error code is non-zero, SGP4 encountered a problem
        # (e.g., numerical issues or an out-of-bounds epoch).
        if error_code != 0:
            # Log a detailed error message with the Julian date and time.
            logging.error(
                "SGP4 propagation error code %d for time %s (jd=%.6f, fr=%.6f)",
                error_code,
                t_utc.isoformat(),
                jd,
                fr,
            )
            # Raise a RuntimeError so the caller can decide how to handle
            # this failure (e.g., abort, skip, etc.).
            raise RuntimeError(
                f"SGP4 propagation error code {error_code} at time {t_utc.isoformat()}"
            )

        # If propagation succeeded, append the position and velocity
        # vectors to our accumulator lists.
        positions.append(r)
        velocities.append(v)

    # Convert the accumulated Python lists into numpy arrays with shape
    # (N, 3), where N is the number of epochs.  This format is convenient
    # for downstream vectorized computations.
    return np.array(positions), np.array(velocities)


def compute_adaptive_dt(a1: float, a2: float, delta_f_lim: float, default_dt: float) -> float:
    """
    Compute an adaptive time step based on the predicted maximum visibility interval.

    This function uses an analytic estimate for the maximum duration of
    a continuous visibility window between two satellites with semi-major
    axes a1 and a2, given a limit on their true-anomaly separation.

    The underlying model assumes:
        n = sqrt(mu / a^3)  (mean motion in rad/s)

    and the paper-derived expression:
        T_max = 2 * |delta_f_lim| / |n2 - n1|

    where:
        - n1, n2 are the mean motions of the two orbits,
        - delta_f_lim is the allowed true-anomaly separation (radians),
        - T_max is the predicted maximum continuous visibility duration.

    The returned dt is chosen as follows:

      * If there is effectively no drift in mean motion (n1 ≈ n2),
        the function returns `default_dt`.

      * Otherwise, T_max is computed.  If T_max is shorter than 10 times
        default_dt, the function refines the time step so that there are
        roughly 20 samples across T_max, but never smaller than 1 second.

      * If T_max is longer than or equal to 10 * default_dt, then
        `default_dt` is used as-is.

    Parameters
    ----------
    a1 : float
        Semi-major axis of satellite 1 [km].

    a2 : float
        Semi-major axis of satellite 2 [km].

    delta_f_lim : float
        Limit on true-anomaly separation [radians] derived from the
        analytic visibility regime (e.g., from your line-of-sight paper).

    default_dt : float
        Base time step in seconds (for example, the `base_dt_s` value
        from NEBULA_TIME_CONFIG).

    Returns
    -------
    float
        Suggested time step in seconds, subject to the logic described
        above.

    Notes
    -----
    This is a NEBULA adaptation of the original AMOS_propagator
    compute_adaptive_dt, but uses MU_EARTH imported from
    NEBULA_ENV_CONFIG instead of re-defining the gravitational
    parameter locally. :contentReference[oaicite:4]{index=4}
    """

    # Compute mean motion n1 of the first orbit in radians per second
    # using the two-body relation n = sqrt(mu / a^3).
    n1 = math.sqrt(MU_EARTH / a1**3)

    # Compute mean motion n2 of the second orbit.
    n2 = math.sqrt(MU_EARTH / a2**3)

    # Compute the magnitude of the relative mean motion (rad/s).
    # This quantity describes how quickly the true anomaly difference
    # between the two orbits changes over time.
    delta_n = abs(n2 - n1)

    # If there is effectively no drift in mean motion (delta_n == 0),
    # then the relative true-anomaly difference does not evolve in this
    # simplified model, so there is no strong need to refine dt.
    if delta_n == 0.0:
        return default_dt

    # Compute the predicted maximum continuous visibility duration T_max
    # using the analytic expression from your line-of-sight paper:
    #   T_max = 2 * |delta_f_lim| / |delta_n|.
    # Source is paper by Cinelli: Geometrical approach for an optimal inter-satellite visibility
    T_max = 2.0 * abs(delta_f_lim) / delta_n

    # If this predicted visibility window is shorter than 10 coarse
    # time steps, then using default_dt would give fewer than 10 samples
    # across T_max.  In that case, refine dt to aim for roughly 20
    # samples across the window.
    if T_max < 10.0 * default_dt:
        # Proposed refined dt is T_max / 20, but we enforce a floor of
        # 1.0 second to avoid impractically small time steps.
        dt = max(1.0, T_max / 20.0)
    else:
        # If T_max is not particularly short, the base time step is
        # deemed sufficient to resolve the visibility behavior.
        dt = default_dt

    # Return the selected time step to the caller.
    return dt
