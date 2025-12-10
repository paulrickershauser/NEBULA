"""
NEBULA_TIME_PARSER.py

Time parsing utilities for the NEBULA
(Neuromorphic Event-Based Luminance Asset-tracking) simulation framework.

This module is responsible for converting the string-based time window
configuration defined in NEBULA_TIME_CONFIG into timezone-aware UTC
`datetime` objects that the rest of NEBULA can use.

It is the NEBULA replacement for the original AMOS_time_parser module:
  - It can parse both space-separated and ISO8601-like time strings.
  - It always returns timezone-aware UTC datetimes.
  - It validates that the end time is strictly after the start time.

Design notes
------------
* This file lives in the NEBULA/Utility/ folder because it contains
  "gears" (functions that do work), not configuration constants.

* The implementation is deliberately functional (plain functions) rather
  than object-oriented.  For this task, a simple set of small, focused
  functions is clearer and easier to test than introducing a class.

* NEBULA modules should typically:
    - import a TimeWindowConfig from NEBULA_TIME_CONFIG, then
    - call parse_time_window_config(...) to obtain (start_dt, end_dt).
"""

# Import the datetime class to represent points in time, and the timezone
# object so we can explicitly mark datetimes as UTC.
from datetime import datetime, timezone

# Import Tuple so we can annotate functions that return (start_dt, end_dt).
from typing import Tuple

# Import the TimeWindowConfig dataclass that stores start/end strings.
# This keeps the configuration (strings) in Configuration/, and the parser
# logic here in Utility/.
from Configuration.NEBULA_TIME_CONFIG import TimeWindowConfig  # type: ignore


def _parse_time_string(time_str: str) -> datetime:
    """
    Parse a single time string into a timezone-aware UTC datetime.

    This helper function accepts several formats:

      - "YYYY-MM-DD HH:MM:SS"      (space-separated, assumed UTC)
      - "YYYY-MM-DDTHH:MM:SS"      (ISO8601 basic)
      - "YYYY-MM-DDTHH:MM:SSZ"     (ISO8601 with 'Z' suffix for UTC)
      - "YYYY-MM-DDTHH:MM:SS+HH:MM" (ISO8601 with explicit offset)

    It mirrors the behavior of the _parse(...) inner function from the
    original AMOS_time_parser.parse_window implementation. :contentReference[oaicite:1]{index=1}

    Parameters
    ----------
    time_str : str
        Text representation of the time to parse.

    Returns
    -------
    datetime
        A timezone-aware `datetime` object in UTC.
    """

    # First, try to interpret the input as a space-separated UTC time:
    #   "YYYY-MM-DD HH:MM:SS"
    # This matches the most common format used in your original config.
    try:
        # Attempt to parse using strptime with the explicit format.
        dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")

        # Attach a UTC timezone to the naive datetime, since this format
        # is assumed to represent UTC.
        return dt.replace(tzinfo=timezone.utc)

    except ValueError:
        # If parsing in the space-separated format fails, we simply
        # fall through and try a more general ISO8601-style parse below.
        pass

    # If the string ends with 'Z', that usually means "UTC" in ISO8601.
    # Python's datetime.fromisoformat does not understand 'Z' directly,
    # but it does understand "+00:00", so we convert it here.
    if time_str.endswith("Z"):
        # Replace the trailing 'Z' with an explicit UTC offset.
        iso_str = time_str.replace("Z", "+00:00")
    else:
        # Otherwise, use the string as-is for ISO8601 parsing.
        iso_str = time_str

    # Use datetime.fromisoformat to parse the ISO8601-style string.
    # This can handle:
    #   - "YYYY-MM-DDTHH:MM:SS"
    #   - "YYYY-MM-DDTHH:MM:SS+HH:MM"
    dt = datetime.fromisoformat(iso_str)

    # If the parsed datetime has no timezone information, we assume
    # that the input was already in UTC and attach a UTC tzinfo.
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        # If the datetime *does* have timezone information (for example,
        # +02:00), convert it explicitly to UTC for consistency.
        dt = dt.astimezone(timezone.utc)

    # Return the final timezone-aware UTC datetime object.
    return dt


def parse_window_strings(start: str, end: str) -> Tuple[datetime, datetime]:
    """
    Parse two time strings into a validated (start, end) UTC datetime pair.

    This is the NEBULA analogue of AMOS_time_parser.parse_window: it
    calls _parse_time_string on both strings and then checks that the
    end time occurs strictly after the start time. :contentReference[oaicite:2]{index=2}

    Parameters
    ----------
    start : str
        Start time as a string (space-separated or ISO8601-like).

    end : str
        End time as a string (same accepted formats as `start`).

    Returns
    -------
    (datetime, datetime)
        A pair (start_dt, end_dt), both timezone-aware in UTC.

    Raises
    ------
    ValueError
        If the end time is not strictly after the start time.
    """

    # Parse the start time string into a UTC datetime.
    start_dt = _parse_time_string(start)

    # Parse the end time string into a UTC datetime.
    end_dt = _parse_time_string(end)

    # Validate that the end time is strictly greater than the start time.
    # If not, raise a ValueError with a helpful error message.
    if end_dt <= start_dt:
        raise ValueError(
            f"End time ({end_dt.isoformat()}) must be after start time ({start_dt.isoformat()})"
        )

    # Return the validated (start_dt, end_dt) pair to the caller.
    return start_dt, end_dt


def parse_time_window_config(window: TimeWindowConfig) -> Tuple[datetime, datetime]:
    """
    Parse a TimeWindowConfig object into a (start, end) UTC datetime pair.

    This function is the main entry point NEBULA code should use when it
    wants to go from configuration (string times in NEBULA_TIME_CONFIG)
    to actual `datetime` objects that can be used for propagation.

    Internally it simply forwards to parse_window_strings using the
    `start_utc` and `end_utc` fields of the configuration object. :contentReference[oaicite:3]{index=3}

    Parameters
    ----------
    window : TimeWindowConfig
        The configuration object that stores the start and end times as
        strings (typically DEFAULT_TIME_WINDOW from NEBULA_TIME_CONFIG).

    Returns
    -------
    (datetime, datetime)
        A pair (start_dt, end_dt), both timezone-aware in UTC.
    """

    # Delegate to the string-based parser using the fields from the
    # TimeWindowConfig, which keeps all parsing logic in one place.
    return parse_window_strings(window.start_utc, window.end_utc)


def describe_parsed_window(window: TimeWindowConfig) -> str:
    """
    Build a human-readable description of a TimeWindowConfig, including
    both the original strings and the parsed UTC datetime values.

    This is purely a convenience function for logging or debugging;
    it does not introduce any new behavior beyond parse_time_window_config.

    Parameters
    ----------
    window : TimeWindowConfig
        The configuration object describing the time window.

    Returns
    -------
    str
        A multi-line string describing the window in both string and
        parsed UTC datetime form.
    """

    # Use the parser to convert the configuration into actual datetimes.
    start_dt, end_dt = parse_time_window_config(window)

    # Build a formatted multi-line string with both the raw and parsed
    # representations for easy inspection in logs.
    return (
        "NEBULA time window:\n"
        f"  start (config): {window.start_utc}\n"
        f"  end   (config): {window.end_utc}\n"
        f"  start (parsed): {start_dt.isoformat()}\n"
        f"  end   (parsed): {end_dt.isoformat()}\n"
    )
