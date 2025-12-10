# NEBULA_PIXEL_PICKLER.py
# ---------------------------------------------------------------------------
# Attach pixel-level observer–target geometry to all NEBULA tracks and pickle
# ---------------------------------------------------------------------------
"""
NEBULA_PIXEL_PICKLER
====================

High-level post-processing pickler that sits *on top of*
NEBULA_ICRS_PAIR_PICKLER and NEBULA_SENSOR_PROJECTION.

Pipeline layering
-----------------

1. NEBULA_FLUX_PICKLER
   - Computes intrinsic flux / magnitude vs time for all targets.

2. NEBULA_LOS_FLUX_PICKLER
   - Combines LOS visibility with flux and writes LOS-gated flux products.

3. NEBULA_SCHEDULE_PICKLER
   - Ensures LOS+illum+flux pickles exist (via NEBULA_LOS_FLUX_PICKLER).
   - Attaches pointing information to all observer tracks.

4. NEBULA_ICRS_PAIR_PICKLER
   - Uses NEBULA_OBS_TAR_ICRS to convert TEME state vectors to ICRS.
   - Computes observer–target ICRS line-of-sight (LOS) geometry and
     retrofits existing per-observer fields under target["by_observer"].

5. NEBULA_PIXEL_PICKLER  (this module)
   - Uses NEBULA_SENSOR_PROJECTION to project ICRS LOS (RA, Dec) for each
     observer–target pair onto the sensor pixel grid.
   - Attaches, per observer, the following pixel-level arrays to each
     target track:

         by_observer[obs_name]["pix_x"]  (float, x pixel index)
         by_observer[obs_name]["pix_y"]  (float, y pixel index)
         by_observer[obs_name]["on_detector"]  (bool mask)

     and, when upstream LOS/illumination flags are present:

         by_observer[obs_name]["on_detector_and_visible"]
         by_observer[obs_name]["on_detector_visible_sunlit"]

   - Persists the augmented observer/target tracks to dedicated pickles
     for later reuse by radiometry, event-generation, and frame-building
     code.

Public API
----------

attach_pixels_to_all_pairs(force_recompute=False,
                           sensor_config=ACTIVE_SENSOR,
                           logger=None)
    Main entry point. Ensures upstream ICRS pair pickles exist by
    calling NEBULA_ICRS_PAIR_PICKLER.attach_icrs_to_all_pairs(), then
    uses NEBULA_SENSOR_PROJECTION to attach pixel coordinates and
    on-detector masks for every observer–target pair.

    Returns (observer_tracks, target_tracks), where both dicts contain
    the augmented tracks.

Notes
-----

- This module is intended to be the *last* geometry-layer pickler before
  any sensor radiometry or event-based simulation.  After this, every
  target knows, for each observer, which pixel it falls on (if any) at
  each timestep.

- Like the other picklers, this module implements simple caching: if
  PIXEL_SatPickles already exists and force_recompute is False, it will
  reload the existing pixel-augmented pickles instead of recomputing
  them.
"""

# Import typing helpers for type hints.
from typing import Any, Dict, Tuple, Optional

# Import standard-library modules for file paths, directories, and pickling.
import os
import pickle
import logging

# Import tqdm for progress bars, with a graceful fallback if not installed.
try:
    # Import tqdm.auto for automatic notebook / console behavior.
    from tqdm.auto import tqdm  # type: ignore
except ImportError:
    # If tqdm is not available, set tqdm to None and fall back to plain loops.
    tqdm = None  # type: ignore

# Import NEBULA configuration for the base output directory.
from Configuration.NEBULA_PATH_CONFIG import NEBULA_OUTPUT_DIR

# Import the sensor configuration (pixel grid, FOV, etc.).
from Configuration.NEBULA_SENSOR_CONFIG import SensorConfig, ACTIVE_SENSOR

# Import the upstream ICRS pair pickler to ensure ICRS geometry exists.
from Utility.SAT_OBJECTS import NEBULA_ICRS_PAIR_PICKLER

# Import the sensor projection helpers for ICRS -> pixel mapping.
from Utility.SENSOR import NEBULA_SENSOR_PROJECTION


# ---------------------------------------------------------------------------
# Output paths for pixel-augmented pickles
# ---------------------------------------------------------------------------

# Directory where pixel-augmented tracks will be written,
# under the main NEBULA_OUTPUT directory.
PIXEL_TRACKS_DIR = os.path.join(NEBULA_OUTPUT_DIR, "PIXEL_SatPickles")

# Full path to the observer tracks pickle with pixel data.
OBS_PIXEL_PICKLE_PATH = os.path.join(
    PIXEL_TRACKS_DIR,
    "observer_tracks_with_pixels.pkl",
)

# Full path to the target tracks pickle with pixel data.
TAR_PIXEL_PICKLE_PATH = os.path.join(
    PIXEL_TRACKS_DIR,
    "target_tracks_with_pixels.pkl",
)


# ---------------------------------------------------------------------------
# Internal helpers: logging, output dirs, and pickle I/O
# ---------------------------------------------------------------------------

def _build_default_logger() -> logging.Logger:
    """
    Build a simple console logger if the caller does not supply one.

    Returns
    -------
    logging.Logger
        Logger instance with INFO level and a basic stream handler.
    """
    # Get a logger instance for this module, using its __name__.
    logger = logging.getLogger(__name__)
    # If the logger has no handlers yet, configure a basic stream handler.
    if not logger.handlers:
        # Create a stream handler that writes to stderr.
        handler = logging.StreamHandler()
        # Define a simple log message format with timestamp, name, and level.
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        # Attach the formatter to the handler.
        handler.setFormatter(formatter)
        # Add the handler to the logger so it becomes active.
        logger.addHandler(handler)
        # Set the default logging level to INFO for this logger.
        logger.setLevel(logging.INFO)
    # Return the configured logger instance.
    return logger


def _ensure_output_dirs(logger: logging.Logger) -> None:
    """
    Ensure that NEBULA_OUTPUT_DIR and PIXEL_TRACKS_DIR exist on disk.

    Parameters
    ----------
    logger : logging.Logger
        Logger used to report directory creation and status messages.
    """
    # If the base NEBULA_OUTPUT_DIR does not exist yet, create it.
    if not os.path.isdir(NEBULA_OUTPUT_DIR):
        logger.info(
            "Creating NEBULA_OUTPUT_DIR at '%s' for pixel pickles.",
            NEBULA_OUTPUT_DIR,
        )
        os.makedirs(NEBULA_OUTPUT_DIR, exist_ok=True)

    # If the PIXEL_TRACKS_DIR does not exist yet, create it.
    if not os.path.isdir(PIXEL_TRACKS_DIR):
        logger.info(
            "Creating PIXEL_TRACKS_DIR at '%s' for pixel-augmented tracks.",
            PIXEL_TRACKS_DIR,
        )
        os.makedirs(PIXEL_TRACKS_DIR, exist_ok=True)


def _load_pickle(path: str, logger: logging.Logger) -> Any:
    """
    Load a pickle file from disk.

    Parameters
    ----------
    path : str
        Full path to the pickle file.
    logger : logging.Logger
        Logger used to report what this helper is doing.

    Returns
    -------
    Any
        Deserialized Python object stored in the pickle file.
    """
    # Log which file we are loading.
    logger.info("Loading pickle from '%s'.", path)
    # Open the pickle file in binary read mode and deserialize.
    with open(path, "rb") as f:
        obj = pickle.load(f)
    # Return the loaded object to the caller.
    return obj


def _save_pickle(obj: Any, path: str, logger: logging.Logger) -> None:
    """
    Save a Python object to disk as a pickle.

    Parameters
    ----------
    obj : Any
        Python object to serialize and write.
    path : str
        Full path to the output pickle file.
    logger : logging.Logger
        Logger used to report what this helper is doing.
    """
    # Log which file we are writing.
    logger.info("Writing pickle to '%s'.", path)
    # Open the pickle file in binary write mode and serialize.
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def attach_pixels_to_all_pairs(
    force_recompute: bool = False,
    sensor_config: SensorConfig = ACTIVE_SENSOR,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Attach pixel coordinates and on-detector masks to all observer–target pairs.

    This function is the single entry point you should call from higher-
    level scripts. It orchestrates the full chain:

        1. Ensures that upstream ICRS pair products exist by calling
           NEBULA_ICRS_PAIR_PICKLER.attach_icrs_to_all_pairs(). This, in
           turn, guarantees that flux, LOS+illum+LOS-flux, and pointing
           products are also up to date (thanks to the upstream pickler
           chaining).

        2. Uses NEBULA_SENSOR_PROJECTION to build WCS objects for all
           observers and to project each observer–target ICRS LOS onto
           the sensor pixel grid, populating per-observer fields:

               by_observer[obs_name]["pix_x"]
               by_observer[obs_name]["pix_y"]
               by_observer[obs_name]["on_detector"]

           and, when available:

               by_observer[obs_name]["on_detector_and_visible"]
               by_observer[obs_name]["on_detector_visible_sunlit"]

        3. Writes the resulting observer and target tracks to dedicated
           pickles under NEBULA_OUTPUT/PIXEL_SatPickles for later reuse.

    If pixel-augmented pickles already exist and force_recompute is
    False, this function will simply reload and return them without
    recomputing the pixel geometry.

    Parameters
    ----------
    force_recompute : bool, optional
        If False (default) and both observer_tracks_with_pixels.pkl and
        target_tracks_with_pixels.pkl already exist in PIXEL_SatPickles,
        those pickles will be loaded and returned directly. If True, the
        entire pipeline from ICRS pair attachment through pixel
        projection will be rerun and the pixel pickles will be
        overwritten.
    sensor_config : SensorConfig, optional
        Sensor configuration describing the detector geometry and FOV.
        Defaults to ACTIVE_SENSOR.
    logger : logging.Logger or None, optional
        Logger to use for status and diagnostic messages. If None, a
        default console logger is created.

    Returns
    -------
    obs_tracks_out : dict
        Dictionary mapping observer names to observer track dictionaries,
        augmented with any additional fields upstream picklers may have
        attached (e.g., pointing).
    tar_tracks_out : dict
        Dictionary mapping target names to target track dictionaries,
        with per-observer pixel coordinates and masks attached under:

            target["by_observer"][obs_name]["pix_x"]
            target["by_observer"][obs_name]["pix_y"]
            target["by_observer"][obs_name]["on_detector"]

        and, when applicable:

            target["by_observer"][obs_name]["on_detector_and_visible"]
            target["by_observer"][obs_name]["on_detector_visible_sunlit"]
    """
    # Build a default console logger if the caller did not provide one.
    if logger is None:
        logger = _build_default_logger()

    # Ensure that the base output directory and pixel subdirectory exist.
    _ensure_output_dirs(logger=logger)

    # Check whether pixel pickles already exist on disk.
    pixel_pickles_exist = (
        os.path.isfile(OBS_PIXEL_PICKLE_PATH)
        and os.path.isfile(TAR_PIXEL_PICKLE_PATH)
    )

    # Fast path: reuse existing pixel pickles if allowed.
    if pixel_pickles_exist and not force_recompute:
        logger.info(
            "NEBULA_PIXEL_PICKLER: Reusing existing pixel pickles from '%s'.",
            PIXEL_TRACKS_DIR,
        )
        obs_tracks = _load_pickle(OBS_PIXEL_PICKLE_PATH, logger=logger)
        tar_tracks = _load_pickle(TAR_PIXEL_PICKLE_PATH, logger=logger)
        return obs_tracks, tar_tracks

    # ------------------------------------------------------------------
    # Recompute path: ensure upstream ICRS pair products exist
    # ------------------------------------------------------------------

    logger.info(
        "NEBULA_PIXEL_PICKLER: Computing pixel geometry; "
        "calling NEBULA_ICRS_PAIR_PICKLER.attach_icrs_to_all_pairs().",
    )

    # This call cascades upstream (LOS+illum+flux, pointing, ICRS pairs).
    obs_tracks, tar_tracks = NEBULA_ICRS_PAIR_PICKLER.attach_icrs_to_all_pairs(
        force_recompute=force_recompute,
        logger=logger,
    )

    # ------------------------------------------------------------------
    # Pixel projection: build WCS for observers and attach per-pair pixels
    # ------------------------------------------------------------------

    logger.info(
        "NEBULA_PIXEL_PICKLER: Attaching pixel geometry to all observer–target pairs.",
    )

    # Build WCS objects (static or time-varying) for each observer.
    wcs_map = NEBULA_SENSOR_PROJECTION._build_wcs_for_all_observers(  # type: ignore[attr-defined]
        obs_tracks=obs_tracks,
        sensor_config=sensor_config,
        logger=logger,
    )

    # Prepare the iterable over targets for the main pixel-attachment loop.
    target_items = tar_tracks.items()
    # Wrap in tqdm if available for a progress bar over targets.
    if tqdm is not None:
        target_items = tqdm(
            target_items,
            total=len(tar_tracks),
            desc="Attaching pixel geometry",
        )

    # Loop over each target (optionally with tqdm progress bar).
    for tar_name, tar_track in target_items:
        # Get the per-observer dictionary for this target, if any.
        by_observer = tar_track.get("by_observer", None)
        # If there is no per-observer data, skip this target.
        if not by_observer:
            logger.debug(
                "NEBULA_PIXEL_PICKLER: target '%s' has no by_observer; skipping.",
                tar_name,
            )
            continue

        # Loop over each observer associated with this target.
        for obs_name in by_observer.keys():
            # Skip if this observer has no WCS entry (should not normally happen).
            if obs_name not in wcs_map:
                logger.warning(
                    "NEBULA_PIXEL_PICKLER: observer '%s' not found in WCS map; "
                    "skipping target '%s' for this observer.",
                    obs_name,
                    tar_name,
                )
                continue

            # Retrieve the WCS object or list for this observer.
            nebula_wcs_entry = wcs_map[obs_name]

            # Attach pixels and masks for this single observer–target pair.
            NEBULA_SENSOR_PROJECTION._project_single_pair_to_pixels_and_masks(  # type: ignore[attr-defined]
                observer_track=obs_tracks[obs_name],
                target_track=tar_track,
                obs_name=obs_name,
                nebula_wcs_entry=nebula_wcs_entry,
                sensor_config=sensor_config,
                logger=logger,
            )

    # ------------------------------------------------------------------
    # Save the updated tracks with pixel information to disk
    # ------------------------------------------------------------------

    logger.info(
        "NEBULA_PIXEL_PICKLER: Writing pixel-augmented tracks to '%s'.",
        PIXEL_TRACKS_DIR,
    )

    _save_pickle(obs_tracks, OBS_PIXEL_PICKLE_PATH, logger=logger)
    _save_pickle(tar_tracks, TAR_PIXEL_PICKLE_PATH, logger=logger)

    # Return the updated observer and target tracks to the caller.
    return obs_tracks, tar_tracks


# ---------------------------------------------------------------------------
# Script entry point for one-off regeneration from the command line / IDE
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    If this module is executed as a script, run the full pixel attachment
    pipeline with force_recompute=True. This is primarily intended for
    debugging and one-off regeneration of the pixel pickles from the
    command line or from within an IDE like Spyder.
    """
    # Build a default logger for console output.
    _logger = _build_default_logger()
    # Run the attachment pipeline with force_recompute=True to ensure that
    # all upstream and pixel pickles are freshly generated.
    attach_pixels_to_all_pairs(force_recompute=True, logger=_logger)
