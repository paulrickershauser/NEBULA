"""
NEBULA_TIME_CONFIG.py

Time-window and propagation-step configuration for the NEBULA
(Neuromorphic Event-Based Luminance Asset-tracking) simulation framework.

This module is responsible ONLY for describing:

  - The default simulation time window (start and end timestamps).
  - The default "base" propagation time step.
  - Tuning parameters for adaptive time stepping (e.g. minimum and
    maximum allowed dt, and how finely to sample short visibility
    windows).

It deliberately does NOT implement any actual time-parsing or SGP4
propagation.  Those behaviors will live in other NEBULA modules
(e.g. a future NEBULA_TIME_UTIL for parsing, and NEBULA_PROPAGATOR
for orbit propagation).  This file just provides configuration
objects that those modules can read.
"""

# Import the dataclass decorator so we can define small configuration
# classes that behave like simple objects with named fields.
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Time window configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TimeWindowConfig:
    """
    Configuration object describing a single simulation time window.

    This object stores the start and end of a time interval as strings,
    rather than as datetime objects, because the *parsing* of those
    strings into timezone-aware UTC datetimes is the job of a separate
    utility (for example, a NEBULA_TIME_UTIL module).

    Fields
    ------
    start_utc :
        Start of the simulation window, expressed as an ISO8601-like
        string (e.g. "2025-11-10T00:00:00").

    end_utc :
        End of the simulation window, expressed in the same format.

    label :
        Optional human-readable description of what this window
        represents (e.g. "Nominal 24-hour GEO survey").
    """

    # Simulation window start time as a UTC string.
    # This should be in a format that a future NEBULA time parser
    # can easily interpret (ISO8601 style is recommended).
    start_utc: str

    # Simulation window end time as a UTC string.
    # As with start_utc, this is intentionally kept as text here;
    # parsing to a datetime will occur elsewhere.
    end_utc: str

    # Optional label that describes this time window in a human-friendly
    # way (for logs, plots, or configuration summaries).
    label: str = ""


# Define the default time window NEBULA will use if a specific window
# is not provided elsewhere.  These values come directly from your
# existing AMOS_config defaults:
#
#   START_TIME = "2025-11-10T00:00:00"
#   END_TIME   = "2025-11-11T00:00:00"
#
# You can change them here as needed without touching any other code.
DEFAULT_TIME_WINDOW = TimeWindowConfig(
    # Start time of the default simulation window (UTC, ISO-like string).
    start_utc="2025-11-25T00:00:00",

    # End time of the default simulation window (UTC, ISO-like string).
    end_utc="2025-11-26T00:00:00",

    # Human-readable label describing this particular window.
    label="Default 24-hour NEBULA simulation window",
)


# ---------------------------------------------------------------------------
# Propagation / time-step configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PropagationStepConfig:
    """
    Configuration object describing NEBULA's propagation time steps.

    This object groups together both the base (fixed) time step that
    you use for coarse propagation, and the tuning parameters that any
    *adaptive* time-stepping logic should respect.

    It does not perform any computations itself; instead, a future
    NEBULA_PROPAGATOR module can import this configuration and use the
    values when implementing functions like `compute_adaptive_dt`.

    Fields
    ------
    base_dt_s :
        Default propagation time step in seconds.  This corresponds to
        the DEFAULT_DT value you previously used in AMOS_config.

    adaptive_min_dt_s :
        Minimum allowable time step when refining adaptively.  This
        protects you from choosing absurdly small dt values.

    adaptive_max_dt_s :
        Maximum allowable time step when refining adaptively.  This
        protects you from dt becoming too large and skipping over short
        visibility windows.

    adaptive_samples_per_window :
        Target number of time samples across a "short" visibility
        window when adaptive stepping is in effect.  For example, a
        value of 20 means: if the analytic geometry predicts a short
        window of duration T_max, the adaptive logic should aim for
        roughly 20 samples across that interval.
    """

    # Base propagation time step [seconds].
    # This is the coarse time step used when no strong constraints on
    # dt are present.  In your original AMOS_config, this was set to 60.
    base_dt_s: float

    # Minimum time step [seconds] that any adaptive scheme is allowed
    # to choose.  This prevents runaway refinement to extremely small
    # dt values that could make the simulation impractically slow.
    adaptive_min_dt_s: float

    # Maximum time step [seconds] that any adaptive scheme is allowed
    # to choose.  Even when relative motion is small, this ensures
    # that dt cannot grow so large that short visibility windows are
    # completely skipped.
    adaptive_max_dt_s: float

    # Target number of samples across a short visibility window when
    # adaptive stepping is active.  In your previous AMOS_propagator,
    # you effectively targeted about 20 samples by using dt ≈ T_max / 20
    # when T_max was small compared to the base step.
    adaptive_samples_per_window: int


# Define the default propagation step configuration for NEBULA.
# These values are chosen to match the spirit of your existing AMOS
# setup:
#
#   - base_dt_s = 60 seconds (DEFAULT_DT)
#   - adaptive_min_dt_s = 1 second   (same lower bound you used when
#                                     clipping dt in your sidereal code)
#   - adaptive_max_dt_s = 20 seconds (same upper bound you used there)
#   - adaptive_samples_per_window = 20 (matches the "~20 samples" logic
#                                       from compute_adaptive_dt)
DEFAULT_PROPAGATION_STEPS = PropagationStepConfig(
    # Coarse / default propagation time step in seconds.
    base_dt_s=15.0,

    # Minimum allowed time step when adaptively refining dt [s].
    adaptive_min_dt_s=1.0,

    # Maximum allowed time step when adaptively refining dt [s].
    adaptive_max_dt_s=20.0,

    # Target number of samples across a short visibility window.
    adaptive_samples_per_window=20,
)

# ---------------------------------------------------------------------------
# High-resolution window (re-propagation) time-step configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HighResWindowStepConfig:
    """
    Configuration object describing the time sampling used when
    re-propagating inside short, observer-centric visibility / pointing
    windows to generate high-resolution tracks for frame / event
    simulation.

    Fields
    ------
    dt_fine_s :
        Fixed time step in seconds to use inside each high-resolution
        window. This is typically smaller than the coarse
        DEFAULT_PROPAGATION_STEPS.base_dt_s (for example, 1.0 s).
    """

    # Fixed fine time step [seconds] used inside each high-resolution
    # window when regenerating observer and target tracks.
    dt_fine_s: float


# Default high-resolution window sampling configuration for NEBULA.
DEFAULT_HIGHRES_WINDOW_STEPS = HighResWindowStepConfig(
    # Fine time step to use when re-propagating inside pointing-valid
    # windows. Adjust here (and only here) if you want denser or
    # sparser sampling, e.g. 0.5 s, 1.0 s, 2.0 s, etc.
    dt_fine_s=1.0,
)

# ---------------------------------------------------------------------------
# Convenience helper for logging / debugging
# ---------------------------------------------------------------------------

def describe_time_configuration() -> str:
    """
    Build a human-readable multi-line string that summarizes the current
    NEBULA time-window and propagation-step configuration.

    This function does not parse or validate the times; it simply
    formats the configuration objects defined above into a string that
    can be written to logs or printed for debugging.
    """

    # Take local references to the configuration objects to make the
    # formatting code below more readable.
    tw = DEFAULT_TIME_WINDOW
    steps = DEFAULT_PROPAGATION_STEPS
    highres = DEFAULT_HIGHRES_WINDOW_STEPS

    # Construct the time-window portion of the description string.
    window_info = (
        "Time window:\n"
        f"  start_utc: {tw.start_utc}\n"
        f"  end_utc:   {tw.end_utc}\n"
        f"  label:     {tw.label}\n"
        "\n"
        "High-resolution windows:\n"
        f"  dt_fine_s:                 {highres.dt_fine_s:.1f} s\n"
    )

    # Construct the propagation-step portion of the description string.
    step_info = (
        "Propagation steps:\n"
        f"  base_dt_s:                 {steps.base_dt_s:.1f} s\n"
        f"  adaptive_min_dt_s:         {steps.adaptive_min_dt_s:.1f} s\n"
        f"  adaptive_max_dt_s:         {steps.adaptive_max_dt_s:.1f} s\n"
        f"  adaptive_samples_per_window: {steps.adaptive_samples_per_window}\n"
    )

    # Combine the two pieces into a single multi-line string and return it.
    return window_info + step_info


