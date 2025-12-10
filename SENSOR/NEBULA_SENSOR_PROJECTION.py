# NEBULA_SENSOR_PROJECTION.py
# ---------------------------------------------------------------------------
# Project observer–target ICRS LOS geometry onto the sensor pixel array
# ---------------------------------------------------------------------------
"""
NEBULA_SENSOR_PROJECTION
========================

Purpose
-------

This module is the **bridge** between:

- The observer–target ICRS geometry produced by
  NEBULA_OBS_TAR_ICRS / NEBULA_ICRS_PAIR_PICKLER, and
- The sensor pixel grid described by NEBULA_SENSOR_CONFIG and NEBULA_WCS.

Given:

    * Observer tracks with pointing attached:
        - pointing_boresight_ra_deg
        - pointing_boresight_dec_deg
        - roll_deg

    * Target tracks with ICRS line-of-sight (LOS) per observer:
        - by_observer[obs_name]["los_icrs_ra_deg"]
        - by_observer[obs_name]["los_icrs_dec_deg"]

    * A SensorConfig instance (e.g. EVK4_SENSOR)

this module:

    1. Builds a NebulaWCS (or list of NebulaWCS objects) for each observer
       using the pointing information and sensor geometry.

    2. Uses that WCS to project the ICRS LOS RA/Dec for each
       observer–target pair into detector pixel coordinates (pix_x, pix_y).

    3. Computes a pure geometric "on_detector" mask indicating whether
       the LOS falls on the active pixel array (0 <= x < cols,
       0 <= y < rows).

    4. Optionally combines "on_detector" with existing LOS/illumination
       flags:

           - los_visible          (Earth-occlusion result)
           - illum_is_sunlit      (eclipse / sunlight result)

       to create:

           - on_detector_and_visible
           - on_detector_visible_sunlit

Layering
--------

This module is intended to be used **after**:

    1) NEBULA_FLUX_PICKLER
    2) NEBULA_SCHEDULE_PICKLER.attach_pointing_to_all_observers()
    3) NEBULA_ICRS_PAIR_PICKLER.attach_icrs_to_all_pairs()

and **before** any radiometry or event-generation modules that need to
know which pixels are hit by which targets over time.

Public API
----------

attach_pixel_geometry_to_all_pairs(obs_tracks, tar_tracks,
                                   sensor_config=ACTIVE_SENSOR,
                                   logger=None)
    Main entry point. Builds WCS objects for all observers and projects
    every observer–target ICRS LOS onto the sensor pixel grid, augmenting
    the target tracks with:

        by_observer[obs_name]["pix_x"]
        by_observer[obs_name]["pix_y"]
        by_observer[obs_name]["on_detector"]

    and, if LOS/illumination flags are present:

        by_observer[obs_name]["on_detector_and_visible"]
        by_observer[obs_name]["on_detector_visible_sunlit"]

Notes
-----

- This module does **not** recompute LOS or illumination; it only
  consumes existing "los_visible" / "illum_is_sunlit" arrays when they
  exist and combines them logically with the "on_detector" mask.

- No pickling or file I/O is performed here. Higher-level "pickler"
  modules are expected to call this on in-memory track dictionaries,
  then handle persistence themselves.
"""

from typing import Any, Dict, Optional, Union, List, Tuple

import logging
import numpy as np

# Import the sensor configuration (pixel grid, FOV, etc.).
from Configuration.NEBULA_SENSOR_CONFIG import SensorConfig, ACTIVE_SENSOR

# Import NebulaWCS helpers for building WCS and projecting RA/Dec to pixels.
from Utility.SENSOR.NEBULA_WCS import (
    NebulaWCS,
    build_wcs_for_observer,
    project_radec_to_pixels,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_logger(logger: Optional[logging.Logger]) -> logging.Logger:
    """
    Return a usable logger.

    If the caller did not supply a logger, construct a simple module-level
    logger that logs to the root handlers.
    """
    if logger is not None:
        return logger

    # Create a logger using this module's name.
    logger = logging.getLogger(__name__)
    # Do not change handlers or levels; assume caller configured logging.
    return logger


def _build_wcs_for_all_observers(
    obs_tracks: Dict[str, Any],
    sensor_config: SensorConfig,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Union[NebulaWCS, List[NebulaWCS]]]:
    """
    Build NebulaWCS objects for every observer track.

    Parameters
    ----------
    obs_tracks : dict
        Dictionary mapping observer names to observer track dictionaries
        (or light-weight objects) that carry pointing fields:
            - "pointing_boresight_ra_deg"
            - "pointing_boresight_dec_deg"
            - "roll_deg"

    sensor_config : SensorConfig
        Sensor configuration describing the detector geometry and FOV.

    logger : logging.Logger, optional
        Logger for status messages. If None, a default logger is used.

    Returns
    -------
    wcs_map : dict
        Dictionary mapping each observer name to either:
            - a single NebulaWCS (static pointing), or
            - a list of NebulaWCS objects (time-varying pointing),
        exactly mirroring the behavior of build_wcs_for_observer().
    """
    log = _get_logger(logger)

    wcs_map: Dict[str, Union[NebulaWCS, List[NebulaWCS]]] = {}

    # Loop over all observers and build their WCS representation.
    for obs_name, obs_track in obs_tracks.items():
        log.info(
            "NEBULA_SENSOR_PROJECTION: Building WCS for observer '%s'.",
            obs_name,
        )

        # Delegate to build_wcs_for_observer(), which inspects the pointing
        # arrays/scalars and returns either a single NebulaWCS or a list.
        wcs_obj_or_list = build_wcs_for_observer(
            observer_track=obs_track,
            sensor_config=sensor_config,
        )

        # Store the resulting WCS object(s) under this observer's name.
        wcs_map[obs_name] = wcs_obj_or_list

    return wcs_map


def _project_single_pair_to_pixels_and_masks(
    observer_track: Any,
    target_track: Any,
    obs_name: str,
    nebula_wcs_entry: Union[NebulaWCS, List[NebulaWCS]],
    sensor_config: SensorConfig,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Project a single observer–target ICRS LOS into pixel space and
    attach masks.

    This helper operates on **one** observer–target pair. It assumes:

        * target_track["by_observer"][obs_name]["los_icrs_ra_deg"]
        * target_track["by_observer"][obs_name]["los_icrs_dec_deg"]

    already exist (created by NEBULA_ICRS_PAIR_PICKLER). It will:

        1) Project these RA/Dec arrays through the supplied NebulaWCS
           (or list of NebulaWCS objects) to produce pix_x, pix_y arrays.

        2) Build an "on_detector" mask using the sensor pixel dimensions.

        3) Optionally combine "on_detector" with existing LOS/illumination
           booleans:

                - los_visible
                - illum_is_sunlit

           to produce:

                - on_detector_and_visible
                - on_detector_visible_sunlit

    All outputs are attached in-place under:

        target_track["by_observer"][obs_name][<field_name>]
    """
    log = _get_logger(logger)

    # Ensure the target has a by_observer dictionary; if not, there is
    # nothing to do for this target.
    by_observer = target_track.get("by_observer", None)
    if by_observer is None or obs_name not in by_observer:
        log.debug(
            "NEBULA_SENSOR_PROJECTION: target '%s' has no by_observer['%s']; "
            "skipping pixel projection.",
            getattr(target_track, "name", "<unnamed>"),
            obs_name,
        )
        return

    # Dictionary holding observer-relative fields for this target.
    by_obs_dict = by_observer[obs_name]

    # Extract the ICRS LOS RA/Dec arrays; if they are missing, we cannot
    # project anything for this pair.
    if "los_icrs_ra_deg" not in by_obs_dict or "los_icrs_dec_deg" not in by_obs_dict:
        log.warning(
            "NEBULA_SENSOR_PROJECTION: target '%s' missing los_icrs RA/Dec "
            "for observer '%s'; skipping.",
            getattr(target_track, "name", "<unnamed>"),
            obs_name,
        )
        return

    ra_icrs = np.asarray(by_obs_dict["los_icrs_ra_deg"], dtype=float)
    dec_icrs = np.asarray(by_obs_dict["los_icrs_dec_deg"], dtype=float)

    # The number of timesteps for this observer–target pair.
    n_times = ra_icrs.shape[0]

    # Determine whether the WCS is static or time-varying.
    if isinstance(nebula_wcs_entry, NebulaWCS):
        # Static pointing: a single NebulaWCS for all timesteps.
        nebula_wcs_static = nebula_wcs_entry

        # Use the functional wrapper to project all RA/Dec values at once.
        pix_x, pix_y = project_radec_to_pixels(
            nebula_wcs=nebula_wcs_static,
            ra_deg=ra_icrs,
            dec_deg=dec_icrs,
        )

        # Sensor dimensions from the WCS / sensor_config.
        rows = sensor_config.rows
        cols = sensor_config.cols

    else:
        # Time-varying pointing: nebula_wcs_entry is a list of NebulaWCS.
        wcs_list: List[NebulaWCS] = nebula_wcs_entry

        # Sanity check: WCS list length must match number of timesteps.
        if len(wcs_list) != n_times:
            raise ValueError(
                f"NEBULA_SENSOR_PROJECTION: length of WCS list "
                f"({len(wcs_list)}) does not match number of timesteps "
                f"({n_times}) for observer '{obs_name}'."
            )

        # Allocate arrays for pixel coordinates.
        pix_x = np.empty(n_times, dtype=float)
        pix_y = np.empty(n_times, dtype=float)

        # Loop over each timestep and project using the corresponding WCS.
        for i in range(n_times):
            wcs_i = wcs_list[i]
            ra_i = ra_icrs[i]
            dec_i = dec_icrs[i]

            # Project a single RA/Dec pair through the time-varying WCS.
            x_i, y_i = project_radec_to_pixels(
                nebula_wcs=wcs_i,
                ra_deg=ra_i,
                dec_deg=dec_i,
            )

            pix_x[i] = x_i
            pix_y[i] = y_i

        # Sensor dimensions are constant across all WCS entries.
        rows = sensor_config.rows
        cols = sensor_config.cols

    # ------------------------------------------------------------------
    # Build "on_detector" mask: pure geometric FOV clipping
    # ------------------------------------------------------------------

    # Start by assuming all timesteps are off-detector and then mark those
    # that fall within [0, cols) x [0, rows) as True.
    on_detector = (
        (pix_x >= 0.0)
        & (pix_x < float(cols))
        & (pix_y >= 0.0)
        & (pix_y < float(rows))
    )

    # If any pixel coordinates are NaN, explicitly mark them as off-detector.
    nan_mask = np.isnan(pix_x) | np.isnan(pix_y)
    if np.any(nan_mask):
        on_detector[nan_mask] = False

    # ------------------------------------------------------------------
    # Optionally combine with LOS / illumination flags
    # ------------------------------------------------------------------

    # Try to fetch los_visible and illum_is_sunlit if they exist.
    los_visible = by_obs_dict.get("los_visible", None)
    illum_is_sunlit = by_obs_dict.get("illum_is_sunlit", None)

    on_detector_and_visible = None
    on_detector_visible_sunlit = None

    # If LOS visibility exists, check its shape and build combined mask.
    if los_visible is not None:
        los_visible = np.asarray(los_visible, dtype=bool)

        if los_visible.shape[0] != n_times:
            raise ValueError(
                f"NEBULA_SENSOR_PROJECTION: los_visible length "
                f"({los_visible.shape[0]}) does not match number of "
                f"timesteps ({n_times}) for observer '{obs_name}'."
            )

        # True only when the target is on the detector AND line-of-sight
        # is clear (no Earth occultation).
        on_detector_and_visible = on_detector & los_visible

    # If both LOS and illumination flags exist, build the fully combined mask.
    if (los_visible is not None) and (illum_is_sunlit is not None):
        illum_is_sunlit = np.asarray(illum_is_sunlit, dtype=bool)

        if illum_is_sunlit.shape[0] != n_times:
            raise ValueError(
                f"NEBULA_SENSOR_PROJECTION: illum_is_sunlit length "
                f"({illum_is_sunlit.shape[0]}) does not match number of "
                f"timesteps ({n_times}) for observer '{obs_name}'."
            )

        # True only when:
        #   * the target falls on the sensor,
        #   * there is geometric LOS, and
        #   * the target is illuminated by the Sun.
        on_detector_visible_sunlit = on_detector & los_visible & illum_is_sunlit

    # ------------------------------------------------------------------
    # Attach results to target_track.by_observer[obs_name]
    # ------------------------------------------------------------------

    by_obs_dict["pix_x"] = pix_x
    by_obs_dict["pix_y"] = pix_y
    by_obs_dict["on_detector"] = on_detector

    if on_detector_and_visible is not None:
        by_obs_dict["on_detector_and_visible"] = on_detector_and_visible

    if on_detector_visible_sunlit is not None:
        by_obs_dict["on_detector_visible_sunlit"] = on_detector_visible_sunlit

    # Optional: tiny diagnostic log (fraction of time visible, etc.).
    # This is cheap and can be helpful when debugging pointing / FOV choices.
    # We now emit it at DEBUG level so normal runs (INFO) stay clean.
    if log.isEnabledFor(logging.DEBUG):
        frac_on = float(on_detector.mean())
        msg = (
            f"NEBULA_SENSOR_PROJECTION: target='{getattr(target_track, 'name', '<unnamed>')}', "
            f"obs='{obs_name}': on={frac_on:.3f}"
        )

        if los_visible is not None:
            frac_los = float(los_visible.mean())
            msg += f", los={frac_los:.3f}"

        if on_detector_and_visible is not None:
            frac_on_los = float(on_detector_and_visible.mean())
            msg += f", on&los={frac_on_los:.3f}"

        if on_detector_visible_sunlit is not None:
            frac_full = float(on_detector_visible_sunlit.mean())
            msg += f", on&los&sun={frac_full:.3f}"

        log.debug(msg)



# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def attach_pixel_geometry_to_all_pairs(
    obs_tracks: Dict[str, Any],
    tar_tracks: Dict[str, Any],
    sensor_config: SensorConfig = ACTIVE_SENSOR,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Attach pixel coordinates and on-detector masks to all observer–target pairs.

    Parameters
    ----------
    obs_tracks : dict
        Dictionary mapping observer names to observer track dictionaries
        (or light-weight objects). Each observer track is expected to carry
        pointing fields compatible with NEBULA_WCS.build_wcs_for_observer():

            - "pointing_boresight_ra_deg"
            - "pointing_boresight_dec_deg"
            - "roll_deg"

    tar_tracks : dict
        Dictionary mapping target names to target track dictionaries.
        Each target track that has a "by_observer" entry for a given
        observer is expected to carry ICRS LOS geometry:

            by_observer[obs_name]["los_icrs_ra_deg"]
            by_observer[obs_name]["los_icrs_dec_deg"]

        Optionally, it may also carry:

            by_observer[obs_name]["los_visible"]
            by_observer[obs_name]["illum_is_sunlit"]

        If present, these will be combined with the geometric "on_detector"
        mask to yield "on_detector_and_visible" and
        "on_detector_visible_sunlit".

    sensor_config : SensorConfig, optional
        Sensor configuration describing the detector geometry and FOV.
        Defaults to ACTIVE_SENSOR.

    logger : logging.Logger, optional
        Logger for status and diagnostic messages. If None, a default
        module-level logger is used.

    Returns
    -------
    obs_tracks_out : dict
        The same observer tracks dictionary that was passed in, returned
        for convenience and symmetry.

    tar_tracks_out : dict
        The same target tracks dictionary, modified in-place to include
        per-observer pixel coordinates and masks under:

            target["by_observer"][obs_name]["pix_x"]
            target["by_observer"][obs_name]["pix_y"]
            target["by_observer"][obs_name]["on_detector"]

        and, when applicable:

            target["by_observer"][obs_name]["on_detector_and_visible"]
            target["by_observer"][obs_name]["on_detector_visible_sunlit"]
    """
    log = _get_logger(logger)

    log.info("NEBULA_SENSOR_PROJECTION: Building WCS for all observers.")
    wcs_map = _build_wcs_for_all_observers(
        obs_tracks=obs_tracks,
        sensor_config=sensor_config,
        logger=log,
    )

    log.info("NEBULA_SENSOR_PROJECTION: Projecting all observer–target pairs.")
    # Loop over all targets and their per-observer entries.
    for tar_name, tar_track in tar_tracks.items():
        by_observer = tar_track.get("by_observer", None)
        if not by_observer:
            # No per-observer data for this target; nothing to do.
            log.debug(
                "NEBULA_SENSOR_PROJECTION: target '%s' has no by_observer; skipping.",
                tar_name,
            )
            continue

        for obs_name, by_obs_dict in by_observer.items():
            # Skip if this observer has no WCS (should not normally happen).
            if obs_name not in wcs_map:
                log.warning(
                    "NEBULA_SENSOR_PROJECTION: observer '%s' not found in WCS map; "
                    "skipping target '%s' for this observer.",
                    obs_name,
                    tar_name,
                )
                continue

            # Retrieve the WCS object or list for this observer.
            nebula_wcs_entry = wcs_map[obs_name]

            # Project LOS for this observer–target pair and attach masks.
            _project_single_pair_to_pixels_and_masks(
                observer_track=obs_tracks[obs_name],
                target_track=tar_track,
                obs_name=obs_name,
                nebula_wcs_entry=nebula_wcs_entry,
                sensor_config=sensor_config,
                logger=log,
            )

    return obs_tracks, tar_tracks
