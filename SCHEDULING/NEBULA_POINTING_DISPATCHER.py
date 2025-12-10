"""
NEBULA_POINTING_DISPATCHER.py

High-level dispatcher for NEBULA pointing modes.

This module provides a single public function,
`build_pointing_schedule()`, that inspects a PointingConfig and routes
to the appropriate pointing-mode implementation.  This keeps the rest
of NEBULA from needing to know about individual pointing-mode modules
(e.g. anti-Sun stare vs fixed ICRS stare).

Currently supported mode:

    - PointingMode.ANTI_SUN_STARE
        Implemented by Utility.SCHEDULING.NEBULA_POINTING_ANTISUN

Future modes (e.g. fixed ICRS, rate tracking, explicit slews) can be
added here without changing any callers: they will simply select a
different mode in NEBULA_POINTING_CONFIG.
"""

# Import typing helpers for function annotations.
from typing import Any, Optional, Union
import logging

# Import the pointing configuration dataclass and mode enumeration.
from Configuration.NEBULA_POINTING_CONFIG import (
    PointingConfig,
    PointingMode,
    DEFAULT_POINTING_CONFIG,
)

# Import the anti-Sun pointing schedule dataclass and builder.
from Utility.SCHEDULING.NEBULA_POINTING_ANTISUN import (
    PointingSchedule as AntiSunSchedule,
    build_pointing_schedule_antisun,
)
from Utility.SCHEDULING.NEBULA_POINTING_ANTISUN_GEO import (
    PointingSchedule as AntiSunGeoSchedule,
    build_pointing_schedule_antisun_geo,
)
# Unified return type for all pointing modes in this dispatcher.
PointingSchedule = Union[AntiSunSchedule, AntiSunGeoSchedule]



def build_pointing_schedule(
    observer_track: Any,
    config: PointingConfig = DEFAULT_POINTING_CONFIG,
    eph=None,
    ephemeris_path: Optional[str] = None,
    store_on_observer: bool = True,
    logger: Optional[logging.Logger] = None,
) -> PointingSchedule:
    """
    Dispatch to the appropriate pointing-mode implementation.

    This function examines `config.mode` and calls the corresponding
    pointing-mode builder, returning a PointingSchedule on the
    observer's time grid.

    Parameters
    ----------
    observer_track : dict or SatelliteTrack-like
        NEBULA observer track.  Must contain at least the fields
        required by the selected pointing-mode implementation (for the
        anti-Sun mode, 'times', 'r_eci_km', and 'v_eci_km_s').
    config : PointingConfig, optional
        High-level pointing configuration.  If omitted,
        DEFAULT_POINTING_CONFIG is used.  The `mode` field determines
        which pointing-mode module is invoked.
    eph : skyfield.jpllib.SpiceKernel or None, optional
        Optional pre-loaded DE440s ephemeris, passed through to the
        underlying pointing implementation.
    ephemeris_path : str or None, optional
        Optional explicit path to "de440s.bsp".  Only used if `eph` is
        None; otherwise ignored.  This path is passed through to the
        underlying pointing implementation, which in turn uses the
        shared loader in NEBULA_SKYFIELD_ILLUMINATION.
    store_on_observer : bool, optional
        If True, the selected pointing-mode implementation will attach
        key arrays back onto `observer_track` (e.g. boresight RA/Dec,
        Earth-block mask, valid_for_projection mask).  If False, the
        track is left unchanged.
    logger : logging.Logger or None, optional
        Logger for status messages.  If None, the underlying pointing
        implementation will create or reuse its own logger as needed.

    Returns
    -------
    PointingSchedule
        Dataclass with boresight geometry, environment flags, and a
        valid_for_projection mask for the chosen pointing mode.

    Raises
    ------
    ValueError
        If `config.mode` selects a pointing mode that has not yet been
        implemented in this dispatcher.
    """
    # If no logger was provided, build a module-local logger.
    log = logger or logging.getLogger(__name__)

    # Log which pointing mode is about to be used (helpful for debugging).
    log.info("Building pointing schedule with mode: %s", config.mode.value)

    if config.mode == PointingMode.ANTI_SUN_STARE:
        return build_pointing_schedule_antisun(
            observer_track=observer_track,
            config=config,
            eph=eph,
            ephemeris_path=ephemeris_path,
            store_on_observer=store_on_observer,
            logger=log,
        )

    elif config.mode == PointingMode.ANTI_SUN_GEO_BELT:
        return build_pointing_schedule_antisun_geo(
            observer_track=observer_track,
            config=config,
            eph=eph,
            ephemeris_path=ephemeris_path,
            store_on_observer=store_on_observer,
            logger=log,
        )


    # Placeholder for future modes such as FIXED_ICRS_EARTH_AVOID, RATE_TRACK, etc.
    # When those modules are created, they should be imported above and added
    # to this dispatch table.
    msg = f"Pointing mode '{config.mode.value}' is not implemented in the dispatcher."
    # Log a clear error message before raising, to aid debugging.
    log.error(msg)
    # Raise an exception so the caller knows this mode is not yet supported.
    raise ValueError(msg)
