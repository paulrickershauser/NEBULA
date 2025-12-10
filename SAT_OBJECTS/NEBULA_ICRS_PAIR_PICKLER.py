# NEBULA_ICRS_PAIR_PICKLER.py
# ---------------------------------------------------------------------------
# Attach ICRS observer–target geometry to all NEBULA tracks and pickle them
# ---------------------------------------------------------------------------
"""
NEBULA_ICRS_PAIR_PICKLER
========================

High-level post-processing helper that sits *on top of*
NEBULA_SCHEDULE_PICKLER and NEBULA_OBS_TAR_ICRS.

Pipeline layering
-----------------

1. NEBULA_FLUX_PICKLER
   - Computes intrinsic flux / magnitude vs time for all targets.

2. NEBULA_LOS_FLUX_PICKLER
   - Applies LOS visibility gates and attaches LOS-gated flux arrays.

3. NEBULA_SCHEDULE_PICKLER
   - Ensures LOS+illum+flux pickles exist (via NEBULA_LOS_FLUX_PICKLER).
   - Attaches pointing information to all observer tracks.

4. NEBULA_ICRS_PAIR_PICKLER  (this module)
   - Uses NEBULA_OBS_TAR_ICRS to convert TEME state vectors to ICRS.
   - Computes observer–target ICRS line-of-sight (LOS) geometry:
        * obs_icrs_x/y/z_km
        * tar_icrs_x/y/z_km
        * los_icrs_ra_deg, los_icrs_dec_deg
        * los_icrs_range_km
        * los_icrs_unit_vec (3-component)
   - Retrofits existing observer-relative fields so they are stored under
     a per-observer dictionary on each target:
        target["by_observer"][observer_name][field_name]
   - Persists the augmented observer/target tracks to dedicated pickles
     for later reuse by frame-generation and sensor-model code.

Public API
----------

attach_icrs_to_all_pairs(force_recompute=False, logger=None)
    Main entry point. Ensures upstream pickles exist by calling
    NEBULA_SCHEDULE_PICKLER.attach_pointing_to_all_observers(), then
    computes and attaches ICRS geometry for every observer–target pair.

    Returns (observer_tracks, target_tracks), where both dicts contain
    the augmented tracks.

Notes
-----
- This module **does not** change how LOS / flux / pointing are computed;
  it only adds ICRS positions and observer-relative ICRS LOS geometry.
- Existing flat fields on the target tracks (e.g., "los_visible",
  "rad_flux_g_w_m2") are left in place for backward compatibility,
  while their values are also stored under:

      target["by_observer"][observer_name][field_name]

  so that future code can cleanly support multiple observers.
"""

# Import typing helpers for type hints.
from typing import Any, Dict, Tuple, Optional

# Import standard-library modules for file paths, directories, and pickling.
import os
import pickle
import logging
# Compact text progress bars while looping over observer–target pairs
from tqdm.auto import tqdm
# Import NEBULA configuration for the base output directory.
from Configuration.NEBULA_PATH_CONFIG import NEBULA_OUTPUT_DIR

# Import the upstream schedule pickler to ensure pointing + LOS+flux exist.
from Utility.SAT_OBJECTS import NEBULA_SCHEDULE_PICKLER

# Import the observer–target ICRS geometry helpers.
from Utility.SENSOR import NEBULA_OBS_TAR_ICRS


# ---------------------------------------------------------------------------
# Output paths for ICRS-augmented pickles
# ---------------------------------------------------------------------------

# Directory where ICRS-augmented tracks will be written, under NEBULA_OUTPUT.
ICRS_TRACKS_DIR = os.path.join(NEBULA_OUTPUT_DIR, "ICRS_SatPickles")

# Path for observer tracks that include absolute ICRS state.
OBS_ICRS_PICKLE = os.path.join(
    ICRS_TRACKS_DIR, "observer_tracks_with_icrs.pkl"
)

# Path for target tracks that include absolute ICRS state and per-observer
# ICRS line-of-sight geometry plus retrofitted relative fields.
TAR_ICRS_PICKLE = os.path.join(
    ICRS_TRACKS_DIR, "target_tracks_with_icrs_pairs.pkl"
)


# List of field names on the *target* that are observer-relative and should
# be stored under target["by_observer"][obs_name] in addition to the flat
# top-level fields for backward compatibility.
OBSERVER_RELATIVE_ARRAY_FIELDS = [
    # LOS geometry fields from NEBULA_VISIBILITY / LOS picklers
    "los_visible",
    "los_h",
    "los_regime",
    "los_fallback",
    # Illumination fields from NEBULA_SKYFIELD_ILLUMINATION
    "illum_is_sunlit",
    "illum_phase_angle_rad",
    "illum_fraction_illuminated",
    # Radiometry arrays from NEBULA_FLUX / NEBULA_LOS_FLUX_PICKLER
    "rad_range_obs_sat_m",
    "rad_lambert_phase_function",
    "rad_flux_g_w_m2",
    "rad_photon_flux_g_m2_s",
    "rad_app_mag_g",
    "rad_flux_g_w_m2_los_only",
    "rad_photon_flux_g_m2_s_los_only",
    "rad_app_mag_g_los_only",
]

# Scalar radiometry fields on the target that are also observer-dependent
# (e.g., which sensor and efficiency were assumed in the flux model).
OBSERVER_RELATIVE_SCALAR_FIELDS = [
    "radiometry_sensor_name",
    "radiometry_eta_eff",
]


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
    # Get a logger instance for this module.
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
        # Add the handler to the logger.
        logger.addHandler(handler)
        # Set the default logging level to INFO.
        logger.setLevel(logging.INFO)

    # Return the configured logger.
    return logger


def _ensure_output_dir_exists(logger: logging.Logger) -> None:
    """
    Ensure that NEBULA_OUTPUT_DIR exists on disk.

    Parameters
    ----------
    logger : logging.Logger
        Logger used to report what this helper is doing.
    """
    # Check whether the base NEBULA_OUTPUT_DIR exists as a directory.
    if not os.path.isdir(NEBULA_OUTPUT_DIR):
        # Log that we are about to create the top-level output directory.
        logger.info(
            "Creating NEBULA_OUTPUT_DIR at '%s' for ICRS pair pickles.",
            NEBULA_OUTPUT_DIR,
        )
        # Recursively create the directory.
        os.makedirs(NEBULA_OUTPUT_DIR, exist_ok=True)


def _load_pickle(path: str, logger: logging.Logger) -> Any:
    """
    Load a Python object from a pickle file.

    Parameters
    ----------
    path : str
        Full path to the pickle file.
    logger : logging.Logger
        Logger used to report what this helper is doing.

    Returns
    -------
    Any
        The deserialized Python object contained in the pickle.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    """
    # Log that we are about to load a pickle from the given path.
    logger.info("Loading pickle from '%s'", path)

    # Open the file in binary read mode.
    with open(path, "rb") as f:
        # Use pickle.load to deserialize the object and return it.
        obj = pickle.load(f)

    # Return the loaded object to the caller.
    return obj


def _save_pickle(obj: Any, path: str, logger: logging.Logger) -> None:
    """
    Save a Python object to disk as a pickle.

    Parameters
    ----------
    obj : Any
        Object to serialize (typically a dict of tracks).
    path : str
        Full path to the output pickle file.
    logger : logging.Logger
        Logger used to report what this helper is doing.
    """
    # Log that we are about to write a pickle to the given path.
    logger.info("Writing pickle to '%s'", path)

    # Open the file in binary write mode.
    with open(path, "wb") as f:
        # Use pickle.dump to serialize the object into the file.
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


# ---------------------------------------------------------------------------
# Helper to copy existing observer-relative fields under by_observer
# ---------------------------------------------------------------------------

def _ensure_by_observer_dict(target_track: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure that a target track has a 'by_observer' dictionary.

    Parameters
    ----------
    target_track : dict
        Target track dictionary to modify in-place if needed.

    Returns
    -------
    by_observer : dict
        The 'by_observer' dictionary on the target track (existing or new).
    """
    # If the 'by_observer' key does not exist on the target track...
    if "by_observer" not in target_track:
        # ...then create an empty dictionary for it.
        target_track["by_observer"] = {}

    # Return the dictionary stored under 'by_observer'.
    return target_track["by_observer"]


def _copy_relative_fields_for_observer(
    target_track: Dict[str, Any],
    observer_name: str,
    geom: Dict[str, Any],
    logger: logging.Logger,
) -> None:
    """
    Attach ICRS LOS geometry under target['by_observer'][observer_name].

    This helper is responsible for *only* one thing:
        - ensuring that the target track has a per-observer dictionary
          at target_track["by_observer"][observer_name], and
        - writing the ICRS line-of-sight (LOS) geometry for the given
          observer–target pair into that dictionary.

    All other observer-relative quantities (LOS visibility flags,
    illumination flags, radiometry arrays, etc.) are now handled
    upstream by the LOS / illumination / flux picklers and stored
    directly under:

        target_track["by_observer"][observer_name]

    **Important design choice**

    We intentionally do *not* read from or write to any top-level
    observer-relative fields on the target track (for example:
    "los_visible", "illum_is_sunlit", "rad_flux_g_w_m2_los_only").
    This keeps the data model clean and fully multi-observer:

        - absolute, non-relative quantities (orbit, absolute ICRS
          position) live at the top level of the target track
        - observer-relative quantities (LOS, illumination, flux, ICRS
          LOS geometry) live under `by_observer[observer_name]`.

    Parameters
    ----------
    target_track : dict
        Target track dictionary to augment in-place. This is one entry
        from the target_tracks dict, keyed by target name. It is
        expected to already contain geometric / orbital fields such as
        "name", "r_eci_km", "times", etc., and may already contain a
        "by_observer" dictionary created by upstream picklers.

    observer_name : str
        Name of the observer satellite (for example, "SBSS (USA 216)").
        This string is used as the key under target_track["by_observer"]
        where the per-observer data is stored.

    geom : dict
        Dictionary returned by
        NEBULA_OBS_TAR_ICRS.compute_observer_target_icrs_geometry().
        For this helper we use the following keys:
            - "los_icrs_ra_deg"   : line-of-sight right ascension [deg]
            - "los_icrs_dec_deg"  : line-of-sight declination [deg]
            - "los_icrs_range_km" : observer→target range [km]
            - "los_icrs_unit_vec" : 3-element unit vector in ICRS

    logger : logging.Logger
        Logger used to report what this helper is doing. At the default
        INFO level, this function is quiet; it only emits DEBUG messages
        for fine-grained tracing when needed.

    Returns
    -------
    None
        The function modifies `target_track` in-place and does not
        return anything.
    """
    # ------------------------------------------------------------------
    # Step 1: ensure the target has a 'by_observer' dictionary.
    # ------------------------------------------------------------------

    # Call the helper that either returns the existing 'by_observer'
    # dict on the target track or, if missing, creates an empty dict and
    # attaches it at target_track["by_observer"].
    by_observer = _ensure_by_observer_dict(target_track)

    # ------------------------------------------------------------------
    # Step 2: obtain (or create) the per-observer entry.
    # ------------------------------------------------------------------

    # Try to look up the dictionary for this specific observer name.
    obs_dict = by_observer.get(observer_name)

    # If there is no entry yet for this observer...
    if obs_dict is None:
        # ...create an empty dictionary to hold all observer-relative
        # fields (LOS, illumination, flux, ICRS geometry, etc.).
        obs_dict = {}

        # Store this newly created dictionary back into the
        # by_observer container under the observer's name.
        by_observer[observer_name] = obs_dict

    # ------------------------------------------------------------------
    # Step 3: emit an optional debug message (no wall-of-text at INFO).
    # ------------------------------------------------------------------

    # At DEBUG level, report which target and observer we are attaching
    # ICRS LOS geometry for. This is intentionally a DEBUG log so that
    # normal runs are quiet while still allowing deep debugging when
    # needed.
    logger.debug(
        "Attaching ICRS LOS geometry under by_observer[%s] for target '%s'",
        observer_name,
        target_track.get("name", "<unknown>"),
    )

    # ------------------------------------------------------------------
    # Step 4: attach / overwrite the ICRS LOS geometry for this pair.
    # ------------------------------------------------------------------

    # Store the line-of-sight right ascension in degrees as seen by
    # this observer for this target.
    obs_dict["los_icrs_ra_deg"] = geom["los_icrs_ra_deg"]

    # Store the line-of-sight declination in degrees.
    obs_dict["los_icrs_dec_deg"] = geom["los_icrs_dec_deg"]

    # Store the line-of-sight range in kilometers from observer to target.
    obs_dict["los_icrs_range_km"] = geom["los_icrs_range_km"]

    # Store the 3-component line-of-sight unit vector in the ICRS frame.
    obs_dict["los_icrs_unit_vec"] = geom["los_icrs_unit_vec"]



# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def attach_icrs_to_all_pairs(
    force_recompute: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Attach ICRS observer–target geometry to all NEBULA tracks.

    This high-level function orchestrates the ICRS geometry workflow:

        1. Ensures LOS+illum+flux + pointing pickles exist by calling
           NEBULA_SCHEDULE_PICKLER.attach_pointing_to_all_observers().

        2. If ICRS-augmented pickles already exist and `force_recompute`
           is False, reloads them from disk and returns.

        3. Otherwise:
            a. Converts observer and target TEME state vectors to
               absolute ICRS Cartesian coordinates.
            b. For each observer–target pair, computes the line-of-sight
               ICRS RA/Dec, range, and unit vector.
            c. Attaches absolute ICRS state to each track:
                   obs_track["icrs_x_km"], etc.
                   tar_track["icrs_x_km"], etc.
            d. Retrofits existing observer-relative fields into
               target_track["by_observer"][observer_name] and adds the
               new ICRS LOS fields.
            e. Writes the augmented observer and target dictionaries to
               the ICRS pickles in NEBULA_OUTPUT/ICRS_SatPickles.

    Parameters
    ----------
    force_recompute : bool, optional
        If False (default) and ICRS-augmented pickles already exist,
        they are loaded from disk and returned without recomputing
        geometry.  If True, ICRS geometry is recomputed even if the
        pickles are present.

    logger : logging.Logger or None, optional
        Logger instance to use for messages.  If None, a default
        console logger is created.

    Returns
    -------
    observer_tracks : dict
        Dictionary mapping observer name → augmented observer track.

    target_tracks : dict
        Dictionary mapping target name → augmented target track.
    """
    # If caller did not provide a logger, build a default one.
    if logger is None:
        logger = _build_default_logger()

    # Ensure the base NEBULA output directory exists.
    _ensure_output_dir_exists(logger)

    # Ensure the ICRS-specific subdirectory exists.
    if not os.path.isdir(ICRS_TRACKS_DIR):
        # Log that we are going to create the ICRS output directory.
        logger.info(
            "Creating ICRS_TRACKS_DIR at '%s' for ICRS-augmented pickles.",
            ICRS_TRACKS_DIR,
        )
        # Recursively create the directory.
        os.makedirs(ICRS_TRACKS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Fast path: reuse existing ICRS pickles if allowed.
    # ------------------------------------------------------------------

    # Check if recomputation is disabled and both ICRS pickles exist.
    if (
        not force_recompute
        and os.path.exists(OBS_ICRS_PICKLE)
        and os.path.exists(TAR_ICRS_PICKLE)
    ):
        # Log that we are reusing the existing ICRS pickles.
        logger.info(
            "Reusing existing ICRS pickles '%s' and '%s' (force_recompute=False).",
            OBS_ICRS_PICKLE,
            TAR_ICRS_PICKLE,
        )

        # Load the ICRS-augmented observer tracks from disk.
        obs_tracks = _load_pickle(OBS_ICRS_PICKLE, logger)

        # Load the ICRS-augmented target tracks from disk.
        tar_tracks = _load_pickle(TAR_ICRS_PICKLE, logger)

        # Return the cached observer and target dictionaries.
        return obs_tracks, tar_tracks

    # ------------------------------------------------------------------
    # Step 1: ensure upstream LOS+flux + pointing products exist.
    # ------------------------------------------------------------------

    logger.info(
        "NEBULA_ICRS_PAIR_PICKLER: calling SCHEDULE_PICKLER to "
        "ensure LOS+flux and pointing products exist."
    )

    # Call the schedule pickler to obtain observer and target tracks.
    # This call will internally:
    #   - build base tracks if missing,
    #   - compute LOS+illum+flux and LOS-gated flux,
    #   - attach pointing arrays to observer tracks.
    obs_tracks, tar_tracks = NEBULA_SCHEDULE_PICKLER.attach_pointing_to_all_observers(
        force_recompute=force_recompute,
        logger=logger,
    )

    # ------------------------------------------------------------------
    # Step 2: loop over all observer–target pairs and compute ICRS geometry.
    # ------------------------------------------------------------------

    num_obs = len(obs_tracks)
    num_tar = len(tar_tracks)

    logger.info(
        "NEBULA_ICRS_PAIR_PICKLER: computing ICRS geometry for %d observers "
        "and %d targets.",
        num_obs,
        num_tar,
    )

    if num_obs == 0 or num_tar == 0:
        logger.warning(
            "NEBULA_ICRS_PAIR_PICKLER: no observers (%d) or targets (%d); "
            "skipping ICRS geometry.",
            num_obs,
            num_tar,
        )
    else:
        # Loop over each observer track in the dictionary.
        for obs_idx, (obs_name, obs_track) in enumerate(
            obs_tracks.items(), start=1
        ):
            # Coarse progress for logs (one line per observer only).
            logger.info(
                "Computing ICRS geometry for observer %d / %d: %s",
                obs_idx,
                num_obs,
                obs_name,
            )

            # Compute absolute ICRS state for the observer track once.
            obs_icrs_state = NEBULA_OBS_TAR_ICRS.convert_track_teme_to_icrs(
                obs_track
            )

            # Attach the observer ICRS state arrays to the observer track.
            obs_track["icrs_x_km"] = obs_icrs_state["icrs_x_km"]
            obs_track["icrs_y_km"] = obs_icrs_state["icrs_y_km"]
            obs_track["icrs_z_km"] = obs_icrs_state["icrs_z_km"]

            # Progress bar over *all* targets for this observer.
            for tar_name, tar_track in tqdm(
                tar_tracks.items(),
                desc=f"ICRS: {obs_name}",
                unit="target",
            ):
                # Optional fine-grained debug (off at INFO level).
                logger.debug(
                    "  Observer %s: processing target '%s'",
                    obs_name,
                    tar_name,
                )

                # Compute observer–target ICRS geometry for this pair.
                geom = NEBULA_OBS_TAR_ICRS.compute_observer_target_icrs_geometry(
                    obs_track,
                    tar_track,
                )

                # Attach absolute ICRS state to the target if not already done.
                if "icrs_x_km" not in tar_track:
                    tar_track["icrs_x_km"] = geom["tar_icrs_x_km"]
                    tar_track["icrs_y_km"] = geom["tar_icrs_y_km"]
                    tar_track["icrs_z_km"] = geom["tar_icrs_z_km"]

                # Populate the per-observer 'by_observer' dictionary on the
                # target with ICRS LOS geometry (no top-level writes).
                _copy_relative_fields_for_observer(
                    target_track=tar_track,
                    observer_name=obs_name,
                    geom=geom,
                    logger=logger,
                )


    # ------------------------------------------------------------------
    # Step 3: write the augmented observer and target tracks to pickles.
    # ------------------------------------------------------------------

    # Save the ICRS-augmented observer tracks to disk.
    _save_pickle(obs_tracks, OBS_ICRS_PICKLE, logger)

    # Save the ICRS-augmented target tracks to disk.
    _save_pickle(tar_tracks, TAR_ICRS_PICKLE, logger)

    # Log that the ICRS pair pickler has completed successfully.
    logger.info(
        "NEBULA_ICRS_PAIR_PICKLER: completed ICRS geometry attachment "
        "for %d observers and %d targets.",
        len(obs_tracks),
        len(tar_tracks),
    )

    # Return the augmented observer and target track dictionaries.
    return obs_tracks, tar_tracks


# ---------------------------------------------------------------------------
# Optional: simple CLI entry point for manual testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    If this module is executed as a script, run the full ICRS attachment
    pipeline with force_recompute=True.  This is primarily intended for
    debugging and one-off regeneration of the ICRS pickles from the
    command line or from within an IDE like Spyder.
    """
    # Build a default logger for console output.
    _logger = _build_default_logger()

    # Run the attachment pipeline with force_recompute=True.
    attach_icrs_to_all_pairs(force_recompute=True, logger=_logger)
