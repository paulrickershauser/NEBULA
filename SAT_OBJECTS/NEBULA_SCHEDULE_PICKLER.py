"""
NEBULA_SCHEDULE_PICKLER
=======================

High-level post-processing helper that sits *on top of*
NEBULA_LOS_FLUX_PICKLER and attaches pointing information to all
observer tracks.

Pipeline layering
-----------------

1. NEBULA_FLUX_PICKLER
   - Computes intrinsic flux / magnitude vs time for all targets.

2. NEBULA_LOS_FLUX_PICKLER
   - Combines LOS visibility with flux and writes
       - observer_tracks_with_los_illum_flux_los.pkl
       - target_tracks_with_los_illum_flux_los.pkl
     containing:
       - observer_tracks : dict[name -> track dict]
       - target_tracks   : dict[name -> track dict]
     where each target track now has LOS-gated flux arrays. :contentReference[oaicite:2]{index=2}

3. NEBULA_SCHEDULE_PICKLER  (this module)
   - Calls NEBULA_LOS_FLUX_PICKLER.attach_los_flux_to_all_targets(...)
     to obtain the observer and target tracks (reusing or recomputing
     LOS-gated flux as needed).
   - For each observer track, calls
       Utility.SCHEDULING.NEBULA_POINTING_DISPATCHER.build_pointing_schedule(...)
     which:
       - uses the anti-Sun pointing implementation in
         Utility.SCHEDULING.NEBULA_POINTING_ANTISUN, 
       - attaches per-timestep boresight and environment flags back onto
         the observer track as 'pointing_*' fields
         (boresight RA/Dec, Earth-block mask, sensor-sunlit, etc.).
   - Writes a new pickle:
       - observer_tracks_with_pointing.pkl
     containing the updated observer tracks (with all previous LOS/flux
     info preserved, plus the new pointing arrays).

The targets are *not* modified by this module; they are passed through
exactly as returned from NEBULA_LOS_FLUX_PICKLER.  This keeps the API
simple and object-oriented: caller scripts work exclusively with the
observer/target track objects.

Public usage
------------

Typical usage pattern in a script (e.g., Sim test.py):

    from Utility.SAT_OBJECTS import NEBULA_SCHEDULE_PICKLER

    obs_tracks, tar_tracks = (
        NEBULA_SCHEDULE_PICKLER.attach_pointing_to_all_observers(
            force_recompute=False
        )
    )

After this call:

    - obs_tracks[name] has additional arrays:
        'pointing_boresight_ra_deg'
        'pointing_boresight_dec_deg'
        'pointing_earth_blocked'
        'pointing_sensor_is_sunlit'
        'pointing_sun_angle_deg'
        'pointing_sun_excluded'
        'pointing_valid_for_projection'

    - tar_tracks is the same dict of target tracks you would get from
      NEBULA_LOS_FLUX_PICKLER (with LOS-gated flux already attached).

We intentionally do *not* expose the PointingSchedule dataclasses from
the pointing implementation here; all relevant information is carried by
the observer track objects themselves.
"""

import os
import pickle
import logging
from typing import Any, Dict, Tuple, Optional
# Compact progress bars for multi-observer processing
from tqdm.auto import tqdm
# ---------------------------------------------------------------------------
# NEBULA configuration: output directory
# ---------------------------------------------------------------------------

# Import the base NEBULA output directory.
from Configuration.NEBULA_PATH_CONFIG import NEBULA_OUTPUT_DIR

# ---------------------------------------------------------------------------
# Upstream LOS+flux pickler and its paths
# ---------------------------------------------------------------------------

# Import the LOS+flux pickler so we can call it directly.
from Utility.SAT_OBJECTS import NEBULA_LOS_FLUX_PICKLER

# We reuse the LOS+flux pickler's knowledge of where the base observer /
# target pickles live, but we *do not* edit those files here.  Instead,
# we produce a *new* pickle for observer tracks with pointing attached.
# (See NEBULA_LOS_FLUX_PICKLER for details.) :contentReference[oaicite:4]{index=4}

# ---------------------------------------------------------------------------
# Pointing configuration and dispatcher
# ---------------------------------------------------------------------------

# Import the pointing configuration dataclass + default config.
from Configuration.NEBULA_POINTING_CONFIG import (
    PointingConfig,
    DEFAULT_POINTING_CONFIG,
)

# Import the high-level pointing dispatcher.  This will internally call
# NEBULA_POINTING_ANTISUN for the anti-Sun mode and attach 'pointing_*'
# arrays onto the observer track. 
from Utility.SCHEDULING import NEBULA_POINTING_DISPATCHER


# ---------------------------------------------------------------------------
# New pickle path for observer tracks with pointing attached
# ---------------------------------------------------------------------------

# Directory where observer tracks with pointing attached will be stored
POINTING_TRACKS_DIR = os.path.join(NEBULA_OUTPUT_DIR, "POINT_SatPickles")

# Path for the observer tracks that now include *both* LOS+illum+flux
# (from NEBULA_LOS_FLUX_PICKLER) and pointing arrays (from this module).
OBS_WITH_POINTING_PICKLE = os.path.join(
    POINTING_TRACKS_DIR, "observer_tracks_with_pointing.pkl"
)



# ---------------------------------------------------------------------------
# Internal helpers: logging and pickle I/O
# ---------------------------------------------------------------------------

def _build_default_logger() -> logging.Logger:
    """
    Build a simple console logger if the caller does not supply one.

    Returns
    -------
    logging.Logger
        Logger instance with INFO level and a basic stream handler.
    """
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def _ensure_output_dir_exists(logger: logging.Logger) -> None:
    """
    Ensure that NEBULA_OUTPUT_DIR exists on disk.

    Parameters
    ----------
    logger : logging.Logger
        Logger used to report what this helper is doing.
    """
    if not os.path.isdir(NEBULA_OUTPUT_DIR):
        logger.info(
            "Creating NEBULA_OUTPUT_DIR at '%s' for schedule pickles.",
            NEBULA_OUTPUT_DIR,
        )
        os.makedirs(NEBULA_OUTPUT_DIR, exist_ok=True)


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
        Deserialized Python object.
    """
    logger.info("Loading pickle from '%s'.", path)
    with open(path, "rb") as f:
        obj = pickle.load(f)
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
    logger.info("Writing pickle to '%s'.", path)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def attach_pointing_to_all_observers(
    force_recompute: bool = False,
    config: PointingConfig = DEFAULT_POINTING_CONFIG,
    ephemeris_path: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Attach pointing schedules to all observers and return tracks.

    This function is the single entry point you should call from higher-
    level scripts.  It orchestrates the full chain:

        1. Calls NEBULA_LOS_FLUX_PICKLER.attach_los_flux_to_all_targets()
           to obtain observer and target tracks that already include
           LOS visibility, illumination, and LOS-gated flux arrays.
           This step itself has its own caching behavior controlled by
           `force_recompute`.

        2. Optionally reuses existing pointing pickles produced by this
           module (observer_tracks_with_pointing.pkl) if they exist and
           `force_recompute` is False.

        3. Otherwise, for each observer track, calls
           NEBULA_POINTING_DISPATCHER.build_pointing_schedule(...),
           which:
               - computes the pointing geometry according to `config`,
               - attaches 'pointing_*' arrays onto the observer track.

        4. Writes the updated observer tracks (with pointing fields) to
           observer_tracks_with_pointing.pkl.

    Targets are *not* modified here: they are returned exactly as from
    NEBULA_LOS_FLUX_PICKLER (with LOS-gated flux already attached).

    Parameters
    ----------
    force_recompute : bool, optional
        Controls recomputation of BOTH the LOS+flux step and the
        pointing step:

            - If False (default):
                - NEBULA_LOS_FLUX_PICKLER will reuse its existing
                  LOS-gated flux pickles if present.
                - If observer_tracks_with_pointing.pkl exists, this
                  function will reload that file and return it (while
                  still calling NEBULA_LOS_FLUX_PICKLER to get the
                  current target tracks).

            - If True:
                - NEBULA_LOS_FLUX_PICKLER will recompute LOS-gated flux
                  from its base flux pickles.
                - This function will recompute pointing from scratch
                  and overwrite observer_tracks_with_pointing.pkl.

    config : PointingConfig, optional
        Pointing configuration to use.  DEFAULT_POINTING_CONFIG selects
        the anti-Sun stare mode with your chosen limb margin / Sun
        exclusion parameters.  You can pass a modified config to test
        alternative pointing behaviors.
    ephemeris_path : str or None, optional
        Optional explicit path to "de440s.bsp".  If None, the pointing
        dispatcher and underlying Skyfield code will use their default
        project-relative ephemeris lookup (via NEBULA_SKYFIELD_ILLUMINATION). :contentReference[oaicite:6]{index=6}
    logger : logging.Logger or None, optional
        Logger to use.  If None, a default console logger is created.

    Returns
    -------
    observer_tracks : dict[str, Any]
        Mapping from observer name to track dict.  Each track includes:
            - fields from NEBULA_FLUX_PICKLER and NEBULA_LOS_FLUX_PICKLER
              (orbits, LOS flags, illumination, LOS-gated flux, etc.),
            - plus additional 'pointing_*' arrays attached by the
              pointing dispatcher (boresight RA/Dec, masks, etc.).
    target_tracks : dict[str, Any]
        Mapping from target name to track dict, as returned by
        NEBULA_LOS_FLUX_PICKLER.attach_los_flux_to_all_targets().  These
        tracks are *not* modified by this function.

    Raises
    ------
    FileNotFoundError
        If the upstream LOS+flux pickler has never been run and its
        base pickles are missing.  In that case, run
        NEBULA_FLUX_PICKLER and NEBULA_LOS_FLUX_PICKLER first.
    """
    # Build or reuse a logger.
    if logger is None:
        logger = _build_default_logger()

    # Ensure the output directory exists for reading/writing pickles.
    _ensure_output_dir_exists(logger)

    # Ensure the pointing subdirectory exists (e.g., NEBULA_OUTPUT/POINT_SatPickles).
    pointing_dir = os.path.dirname(OBS_WITH_POINTING_PICKLE)
    if not os.path.isdir(pointing_dir):
        logger.info(
            "Creating pointing output directory '%s'.",
            pointing_dir,
        )
        os.makedirs(pointing_dir, exist_ok=True)


    # ------------------------------------------------------------------
    # Step 1: ensure LOS+flux products exist by calling the upstream
    #         pickler.  This will either reuse existing pickles or
    #         recompute them based on `force_recompute`.
    # ------------------------------------------------------------------
    logger.info("NEBULA_SCHEDULE_PICKLER: calling LOS+flux pickler first.")
    obs_los_flux, tar_los_flux = (
        NEBULA_LOS_FLUX_PICKLER.attach_los_flux_to_all_targets(
            force_recompute=force_recompute,
            logger=logger,
        )
    )

    # obs_los_flux now contains observer tracks with LOS + illumination
    # + LOS-gated flux information attached.  tar_los_flux contains the
    # corresponding target tracks.

    # ------------------------------------------------------------------
    # Step 2: if we are allowed to reuse existing pointing pickles and
    #         they exist, load them and return them (while still
    #         returning the *current* target tracks from LOS+flux).
    # ------------------------------------------------------------------
    if not force_recompute and os.path.exists(OBS_WITH_POINTING_PICKLE):
        logger.info(
            "Reusing existing pointing pickle '%s' (force_recompute=False).",
            OBS_WITH_POINTING_PICKLE,
        )
        obs_with_pointing = _load_pickle(OBS_WITH_POINTING_PICKLE, logger)
        # Return the cached observer tracks (with pointing) and the
        # freshly-loaded target tracks from LOS+flux.
        return obs_with_pointing, tar_los_flux

    # ------------------------------------------------------------------
    # Step 3: recompute pointing for each observer track.
    #         (multi-observer aware, with a compact tqdm progress bar.)
    # ------------------------------------------------------------------
    total_obs = len(obs_los_flux)
    logger.info(
        "Recomputing pointing for %d observer(s) "
        "(force_recompute=%s or no pointing pickle found).",
        total_obs,
        force_recompute,
    )

    # Prepare container for updated observer tracks.
    obs_tracks_out: Dict[str, Any] = {}

    if total_obs == 0:
        # Nothing to do; pass the (empty) dict through.
        logger.warning(
            "NEBULA_SCHEDULE_PICKLER: no observers found in LOS+flux tracks; "
            "nothing to attach pointing to."
        )
    else:
        # Iterate over all observers with a progress bar instead of
        # one INFO line per observer (avoids wall-of-text logs).
        for name, track in tqdm(
            obs_los_flux.items(),
            desc="POINTING",
            unit="obs",
        ):
            # For each observer, call the high-level pointing dispatcher.
            # This will:
            #   - compute the pointing schedule according to `config`,
            #   - attach 'pointing_*' arrays onto `track` if
            #     store_on_observer=True (we set that here),
            #   - return a PointingSchedule dataclass that we do *not* need
            #     to keep around (the important data lives on `track`).
            _ = NEBULA_POINTING_DISPATCHER.build_pointing_schedule(
                observer_track=track,
                config=config,
                eph=None,
                ephemeris_path=ephemeris_path,
                store_on_observer=True,
                logger=logger,
            )

            # Store the updated track in the output dict, keyed by
            # observer name.  This is fully multi-observer: no notion of
            # a "primary" observer, every track is treated identically.
            obs_tracks_out[name] = track


    # ------------------------------------------------------------------
    # Step 4: write the augmented observer tracks to their own pickle.
    # ------------------------------------------------------------------
    _save_pickle(obs_tracks_out, OBS_WITH_POINTING_PICKLE, logger)

    logger.info(
        "NEBULA_SCHEDULE_PICKLER: completed pointing attachment for %d observers.",
        len(obs_tracks_out),
    )

    # Return the updated observer tracks (with pointing) and the
    # unmodified target tracks from LOS+flux.
    return obs_tracks_out, tar_los_flux


# ---------------------------------------------------------------------------
# CLI / dev entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Default behavior: only recompute pointing if the pointing pickle
    # does not exist yet.  LOS+flux behavior is governed by its own
    # caching inside NEBULA_LOS_FLUX_PICKLER.
    attach_pointing_to_all_observers(force_recompute=False)
