"""
NEBULA_TIMESTEP_PLANNER.py

Time–step and time–grid planning utilities for the NEBULA
(Neuromorphic Event-Based Luminance Asset-tracking) simulation framework.

This module sits logically between:

  * Configuration.NEBULA_TIME_CONFIG
        (which only *describes* the nominal time window and step sizes)
  * Utility.NEBULA_TIME_PARSER
        (which converts those strings into UTC datetime objects)
  * Utility.NEBULA_PROPAGATOR
        (which does the actual SGP4 propagation and contains the
         analytic `compute_adaptive_dt` helper based on Cinelli et al.)

Responsibilities of this module
--------------------------------
1. Build a uniform list of UTC datetimes spanning a configured time
   window using a configured base time step:
       build_time_grid(...)

2. Wrap the analytic `compute_adaptive_dt` function so that any
   chosen adaptive dt is clamped to the min/max bounds specified in
   PropagationStepConfig:
       choose_dt_for_pair(...)

Nothing in this file knows about specific satellites or visibility
cases; it only deals with *when* the simulation should sample the
orbits.

Typical usage
-------------
    from Configuration.NEBULA_TIME_CONFIG import DEFAULT_TIME_WINDOW, DEFAULT_PROPAGATION_STEPS
    from Utility.NEBULA_TIMESTEP_PLANNER import build_time_grid, choose_dt_for_pair

    # 1) Build a uniform time grid for a run:
    times = build_time_grid(DEFAULT_TIME_WINDOW, DEFAULT_PROPAGATION_STEPS)

    # 2) Later, after a visibility case gives you |Δν|_lim and you know
    #    the semi-major axes a_obs, a_tar, pick an adaptive dt:
    dt_pair = choose_dt_for_pair(a_obs, a_tar, delta_nu_lim, DEFAULT_PROPAGATION_STEPS)

This keeps time-window configuration, parsing, and step-planning cleanly
separated, which will make NEBULA_MAIN much easier to read.
"""

# ---------------------------------------------------------------------------
# Standard library imports
# ---------------------------------------------------------------------------

# Datetime is used for representing the actual UTC time grid.
from datetime import datetime, timedelta, timezone

# Logging lets us emit small diagnostic messages without configuring
# handlers in this module.
import logging

# Typing helpers for function signatures.
from typing import List, Sequence

# ---------------------------------------------------------------------------
# NEBULA configuration / utilities
# ---------------------------------------------------------------------------

# Import the time-window and propagation-step *configuration* objects.
from Configuration.NEBULA_TIME_CONFIG import (
    TimeWindowConfig,
    PropagationStepConfig,
    DEFAULT_TIME_WINDOW,
    DEFAULT_PROPAGATION_STEPS,
)

# Import the parser that converts TimeWindowConfig into real datetimes.
from Utility.PROPAGATION.NEBULA_TIME_PARSER import parse_time_window_config

# Import the analytic adaptive-dt helper based on Cinelli Eq. (13).
from Utility.PROPAGATION.NEBULA_PROPAGATOR import compute_adaptive_dt


# ---------------------------------------------------------------------------
# Time-grid construction
# ---------------------------------------------------------------------------

def build_time_grid(
    window: TimeWindowConfig = DEFAULT_TIME_WINDOW,
    steps: PropagationStepConfig = DEFAULT_PROPAGATION_STEPS,
) -> List[datetime]:
    """
    Build a uniform list of UTC datetimes spanning the configured window.

    Parameters
    ----------
    window : TimeWindowConfig, optional
        The time-window description to use.  By default this is
        `DEFAULT_TIME_WINDOW` from NEBULA_TIME_CONFIG.

    steps : PropagationStepConfig, optional
        The propagation-step description to use.  Only `base_dt_s` is
        used here.  By default this is `DEFAULT_PROPAGATION_STEPS`.

    Returns
    -------
    List[datetime]
        A list of timezone-aware UTC datetime objects starting at the
        parsed window start time, stepping forward by `steps.base_dt_s`,
        and *including* the end time if it does not fall exactly on a
        step.

    Notes
    -----
    - This function is intentionally small and side-effect free: it
      parses the window, walks forward in fixed increments, and returns
      the resulting list.
    - Any adaptive refinement (e.g., using Cinelli-based dt) is handled
      separately by `choose_dt_for_pair`.
    """

    # Convert the string-based configuration into actual UTC datetimes.
    start_dt, end_dt = parse_time_window_config(window)

    # Extract the base time step in seconds from the configuration.
    dt_seconds = float(steps.base_dt_s)

    # Sanity check: require a strictly positive base dt.
    if dt_seconds <= 0.0:
        raise ValueError(f"base_dt_s must be > 0, got {dt_seconds}")

    # Create a timedelta object representing one propagation step.
    step = timedelta(seconds=dt_seconds)

    # Initialize the list of times with the starting epoch.
    times: List[datetime] = [start_dt]

    # Walk forward in fixed steps until we pass the end time.
    t = start_dt
    while t + step <= end_dt:
        t = t + step
        times.append(t)

    # If the last sample is strictly before the requested end time,
    # append the exact end time as a final sample.  This ensures that
    # downstream code can always assume the time grid covers the full
    # window [start, end].
    if times[-1] < end_dt:
        times.append(end_dt)

    # Optionally log a small summary for debugging / sanity checks.
    logging.info(
        "build_time_grid: %d epochs from %s to %s (base_dt_s=%.1f)",
        len(times),
        start_dt.isoformat(),
        end_dt.isoformat(),
        dt_seconds,
    )

    # Return the constructed list of UTC datetimes.
    return times


# ---------------------------------------------------------------------------
# Adaptive dt selection wrapper
# ---------------------------------------------------------------------------

def choose_dt_for_pair(
    a1: float,
    a2: float,
    delta_nu_lim: float,
    steps: PropagationStepConfig = DEFAULT_PROPAGATION_STEPS,
) -> float:
    """
    Choose a time step for a *pair* of orbits using Cinelli's T_max idea.

    This is a thin wrapper around `compute_adaptive_dt` that also
    enforces the min/max dt bounds from PropagationStepConfig.

    Conceptually, the steps are:

      1. Use Cinelli Eq. (13) via compute_adaptive_dt to get a dt that
         should give ~`steps.adaptive_samples_per_window` points across
         a visibility window of length T_max.

      2. Clamp that dt into [adaptive_min_dt_s, adaptive_max_dt_s] from
         the configuration, so that dt is never absurdly small or large.

    Parameters
    ----------
    a1 : float
        Semi-major axis of satellite 1 [km].

    a2 : float
        Semi-major axis of satellite 2 [km].

    delta_nu_lim : float
        Non-negative limit on true-anomaly separation |Δν|_lim [radians]
        obtained from the appropriate NEBULA_VISIBILITY case.

    steps : PropagationStepConfig, optional
        Propagation-step configuration describing base_dt_s and the
        adaptive min/max bounds.  Defaults to DEFAULT_PROPAGATION_STEPS.

    Returns
    -------
    float
        A recommended dt [s] for propagating this *pair* of orbits,
        respecting both the analytic Cinelli scaling and the configured
        min/max dt bounds.

    Raises
    ------
    ValueError
        If `delta_nu_lim` is not positive.
    """

    # Require a positive phasing limit; a zero or negative value would
    # make Eq. (13) meaningless.
    if delta_nu_lim <= 0.0:
        raise ValueError(f"delta_nu_lim must be > 0 (radians), got {delta_nu_lim}")

    # Use the analytic helper from NEBULA_PROPAGATOR to get a first
    # guess at a suitable dt [s], given the two semi-major axes and the
    # phasing limit from visibility.
    dt_raw = compute_adaptive_dt(
        a1=a1,
        a2=a2,
        delta_f_lim=delta_nu_lim,
        default_dt=steps.base_dt_s,
    )

    # Clamp the raw dt into the configured [min, max] interval to avoid
    # pathological values (e.g., extremely small dt that would explode
    # runtime, or extremely large dt that would skip over short windows).
    dt_clamped = max(steps.adaptive_min_dt_s, min(dt_raw, steps.adaptive_max_dt_s))

    # Optionally log the decision for later inspection.
    logging.info(
        "choose_dt_for_pair: a1=%.1f km, a2=%.1f km, |Δν|_lim=%.4f rad -> "
        "dt_raw=%.3f s, dt_clamped=%.3f s",
        a1,
        a2,
        delta_nu_lim,
        dt_raw,
        dt_clamped,
    )

    # Return the final dt to the caller.
    return dt_clamped


# ---------------------------------------------------------------------------
# Optional helper: summarize a time grid (useful while debugging)
# ---------------------------------------------------------------------------

def summarize_time_grid(times: Sequence[datetime]) -> str:
    """
    Build a small human-readable summary string for a time grid.

    This is purely for logging / debugging convenience and has no
    side effects.

    Parameters
    ----------
    times : Sequence[datetime]
        The list (or other sequence) of UTC datetimes making up the
        time grid.

    Returns
    -------
    str
        A multi-line string with the number of samples and the first /
        last timestamps.
    """

    # If the grid is empty, return a short message.
    if not times:
        return "Time grid: (empty)"

    # Otherwise, report length and endpoints.
    first = times[0]
    last = times[-1]
    return (
        "Time grid:\n"
        f"  samples: {len(times)}\n"
        f"  first:   {first.isoformat()}\n"
        f"  last:    {last.isoformat()}\n"
    )
