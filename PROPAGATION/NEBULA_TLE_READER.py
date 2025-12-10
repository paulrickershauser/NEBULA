"""
NEBULA_TLE_READER.py

TLE (Two-Line Element) file reader utilities for the NEBULA
(Neuromorphic Event-Based Luminance Asset-tracking) simulation framework.

This module is responsible for reading plain-text TLE files and
converting them into `Satrec` objects from the `sgp4` library, which
can then be used by NEBULA's propagator and visibility modules.

It is the NEBULA replacement for the original AMOS_tle_reader module.
The behavior is intentionally very similar:

  - Input files are expected to contain repeated 3-line blocks:
        NAME
        TLE line 1
        TLE line 2

  - TLE line 1 and line 2 are validated to start with "1 " and "2 "
    respectively.

  - The epoch is parsed from TLE line 1 using the standard TLE rule:
        yy ddd.dddddd
    where:
        yy   = last two digits of the year
        ddd  = day of year (1..365/366) with fractional part
        year < 57 -> 2000 + yy
        year >=57 -> 1900 + yy

  - If multiple TLE blocks share the same satellite NAME, the entry
    with the *newest* epoch is kept and older ones are discarded.

In addition to a general-purpose `read_tle_file(...)` function, this
module provides small convenience wrappers that use the default TLE
paths defined in NEBULA_PATH_CONFIG (observer and target TLE files).

Design notes
------------
* This file lives in NEBULA/Utility/ because it contains "gears"
  (functions that do work), not configuration constants.

* The implementation is functional rather than object-oriented.  For
  this task, a small set of clear, focused functions is easier to work
  with than introducing a dedicated class.  In the future, a higher-
  level "NEBULA Orbit Manager" object could call these utilities.

* NEBULA modules should typically:
    - obtain a path (e.g. from NEBULA_PATH_CONFIG or user input),
    - call read_tle_file(path) to get a {name -> Satrec} dictionary.
"""

# Import logging so we can emit useful messages while reading TLE files.
import logging

# Import datetime and timedelta so we can construct epoch datetimes from
# the TLE year/day-of-year fields.
from datetime import datetime, timedelta

# Import typing helpers: Dict for mapping names to Satrec objects,
# and Union so we can accept both str and Path-like inputs.
from typing import Dict, Union

# Import Path so we can treat file paths in an OS-independent way.
from pathlib import Path

# Import Satrec, the SGP4 satellite record class used for orbit propagation.
from sgp4.api import Satrec

# Import default TLE file locations from the path configuration module.
# These are the observer/target TLE files under the NEBULA input directory.
from Configuration.NEBULA_PATH_CONFIG import OBS_TLE_FILE, TAR_TLE_FILE  # type: ignore


# ---------------------------------------------------------------------------
# Internal helper: parse epoch from TLE line 1
# ---------------------------------------------------------------------------

def _parse_epoch_from_tle_line1(line1: str) -> datetime:
    """
    Extract the epoch from a TLE "line 1" string.

    TLE line 1 encodes the epoch using:
      - columns 19-20: last two digits of year (yy)
      - columns 21-32: day of year with fractional portion (ddd.dddddd)

    The standard TLE year interpretation rule is:
      - if yy < 57  -> year = 2000 + yy
      - if yy >= 57 -> year = 1900 + yy

    Example (columns 19-32):
      " 25 123.456789" -> year 2025, day 123.456789

    The returned datetime is naive (no tzinfo) but is understood to be
    in UTC, consistent with how TLE epochs are defined.

    """

    # TLE specs use 1-indexed column positions; Python slicing is 0-indexed.
    # Columns 19-20 -> indices 18:20 for the 2-digit year.
    year_str = line1[18:20]

    # Columns 21-32 -> indices 20:32 for the day-of-year with fraction.
    day_of_year_str = line1[20:32]

    try:
        # Convert the year substring to an integer (0..99).
        yy = int(year_str)

        # Apply standard TLE rule: years from 00 to 56 map to 2000+, others to 1900+.
        if yy < 57:
            year = 2000 + yy
        else:
            year = 1900 + yy

        # Convert the day-of-year substring to a float, including fractional part.
        day_of_year = float(day_of_year_str)

    except Exception as exc:
        # If parsing fails, raise a ValueError with a helpful message.
        raise ValueError(f"Failed to parse epoch from TLE line1 '{line1}': {exc}") from exc

    # Separate the integer day and fractional part:
    #   - day_integer: whole number of days (1..365/366)
    #   - fraction: fractional part of the day (0..1)
    day_integer = int(day_of_year)
    fraction = day_of_year - day_integer

    # Construct the epoch as:
    #   Jan 1 of the given year
    #   + (day_integer - 1) days
    #   + fraction days (for the fractional part of the day).
    epoch = (
        datetime(year, 1, 1)
        + timedelta(days=day_integer - 1)
        + timedelta(days=fraction)
    )

    # The returned datetime is naive; by convention TLE epochs are UTC.
    return epoch


# ---------------------------------------------------------------------------
# Core TLE reader
# ---------------------------------------------------------------------------

def read_tle_file(path: Union[str, Path]) -> Dict[str, Satrec]:
    """
    Read a text file of TLEs and parse them into Satrec objects.

    Expected file format:
        NAME
        TLE line 1
        TLE line 2
        (repeated for each satellite)

    Blank lines are ignored.  Non-blank lines are grouped into blocks of
    three; any trailing incomplete block is ignored with a warning.

    Duplicate satellite names:
        If multiple TLE blocks share the same NAME, this function keeps
        only the TLE with the *newest epoch* as determined from line 1.
        Older TLEs for the same name are discarded with an informational
        log message.

    Parameters
    ----------
    path : str or Path
        Path to the TLE file on disk.

    Returns
    -------
    Dict[str, Satrec]
        A dictionary mapping satellite name -> Satrec instance.

    Raises
    ------
    RuntimeError
        If no valid satellites are parsed from the file.

    Notes
    -----
    The core logic is adapted from your original AMOS_tle_reader.read_tles
    implementation.  The main differences are:
      - the function name is now read_tle_file,
      - the path argument accepts both str and Path,
      - more detailed error and logging messages are provided. :contentReference[oaicite:0]{index=0}
    """

    # Convert the input path to a Path object for OS-independent handling.
    path = Path(path)

    # Dictionary to hold the final mapping from satellite name -> Satrec.
    sats: Dict[str, Satrec] = {}

    # Dictionary to track the epoch associated with each satellite name.
    # This is used to decide which TLE to keep when duplicates occur.
    epochs: Dict[str, datetime] = {}

    try:
        # Open the TLE file in text mode and read all lines.
        with path.open("r") as f:
            raw_lines = f.readlines()
    except Exception as exc:
        # Log and re-raise any file I/O errors.
        logging.error("Failed to open TLE file '%s': %s", path, exc)
        raise

    # Strip leading/trailing whitespace from each line and discard blank lines.
    lines = [line.strip() for line in raw_lines if line.strip()]

    # If the number of non-blank lines is not a multiple of 3, there is at
    # least one incomplete block at the end; warn but proceed.
    if len(lines) % 3 != 0:
        logging.warning(
            "TLE file '%s' has %d nonblank lines; expected a multiple of 3. "
            "Last incomplete block (if any) will be ignored.",
            path,
            len(lines),
        )

    # Process the lines in groups of three:
    #   index  i   -> NAME
    #   index  i+1 -> TLE line 1
    #   index  i+2 -> TLE line 2
    for i in range(0, len(lines) - 2, 3):
        # Extract the three-line block for this satellite.
        name = lines[i]
        line1 = lines[i + 1]
        line2 = lines[i + 2]

        # Basic sanity check: TLE lines should start with "1 " and "2 ".
        if not (line1.startswith("1 ") and line2.startswith("2 ")):
            logging.warning(
                "Skipping block starting at logical line %d in '%s' because TLE "
                "lines do not have expected prefixes: '%s', '%s'",
                i + 1,
                path,
                line1[:2],
                line2[:2],
            )
            continue

        try:
            # Parse the epoch encoded in TLE line 1.
            epoch = _parse_epoch_from_tle_line1(line1)
        except Exception as exc:
            # If epoch parsing fails, log a warning and skip this block.
            logging.warning(
                "Could not parse epoch for satellite '%s' in '%s': %s. Skipping.",
                name,
                path,
                exc,
            )
            continue

        try:
            # Use SGP4 to construct a Satrec object from the two TLE lines.
            satrec = Satrec.twoline2rv(line1, line2)
        except Exception as exc:
            # If Satrec creation fails, log an error and skip this block.
            logging.error(
                "Failed to parse TLE for satellite '%s' from '%s': %s",
                name,
                path,
                exc,
            )
            continue

        # Handle duplicates: if we've already seen this satellite name,
        # compare epochs and keep the newer one.
        if name in sats:
            prev_epoch = epochs[name]
            if epoch <= prev_epoch:
                # This TLE is older or equal; keep the existing entry.
                logging.info(
                    "Encountered older or equal TLE for '%s' (epoch %s); keeping existing newer epoch %s.",
                    name,
                    epoch,
                    prev_epoch,
                )
                continue
            else:
                # This TLE is newer; replace the existing entry.
                logging.info(
                    "Found newer TLE for '%s' (epoch %s) replacing previous epoch %s.",
                    name,
                    epoch,
                    prev_epoch,
                )
        else:
            # First time we've seen this satellite name; record it.
            logging.info(
                "Parsed TLE for satellite '%s' from '%s' with epoch %s",
                name,
                path,
                epoch,
            )

        # Store/overwrite the Satrec and epoch for this satellite name.
        sats[name] = satrec
        epochs[name] = epoch

    # After processing all blocks, verify that we parsed at least one satellite.
    if not sats:
        logging.error("No valid satellites parsed from '%s'", path)
        raise RuntimeError(f"No valid satellites parsed from '{path}'")

    # Return the final mapping from satellite name -> Satrec object.
    return sats


# ---------------------------------------------------------------------------
# Convenience wrappers using NEBULA default TLE paths
# ---------------------------------------------------------------------------

def read_default_observer_tles() -> Dict[str, Satrec]:
    """
    Read the default observer TLE file as defined in NEBULA_PATH_CONFIG.

    This helper simply calls read_tle_file(...) on the OBS_TLE_FILE path,
    which is typically:

        NEBULA/input/tle/GEO_observers.txt

    but is fully controlled by NEBULA_PATH_CONFIG. :contentReference[oaicite:2]{index=2}

    Returns
    -------
    Dict[str, Satrec]
        Dictionary mapping observer satellite name -> Satrec instance.
    """
    return read_tle_file(OBS_TLE_FILE)


def read_default_target_tles() -> Dict[str, Satrec]:
    """
    Read the default target TLE file as defined in NEBULA_PATH_CONFIG.

    This helper simply calls read_tle_file(...) on the TAR_TLE_FILE path,
    which is typically:

        NEBULA/input/tle/geo_target.txt

    but is fully controlled by NEBULA_PATH_CONFIG. :contentReference[oaicite:3]{index=3}

    Returns
    -------
    Dict[str, Satrec]
        Dictionary mapping target satellite name -> Satrec instance.
    """
    return read_tle_file(TAR_TLE_FILE)
