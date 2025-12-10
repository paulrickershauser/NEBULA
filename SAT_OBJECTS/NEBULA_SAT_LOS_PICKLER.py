# -*- coding: utf-8 -*-
"""
NEBULA_SAT_LOS_PICKLER.py

High-level helper for attaching line-of-sight (LOS) visibility arrays to
NEBULA SatelliteTrack objects and writing out augmented pickle files.

Design goal
-----------
Provide a *single* function that you can call from your simulation code
to go from "raw propagated tracks" → "tracks with LOS visibility fields",
without having to manually orchestrate:

    - Reading TLEs,
    - Propagating orbits and building SatelliteTrack objects,
    - Looping through every observer–target pair to evaluate LOS,
    - Serializing the updated tracks back to disk.

This module is intentionally modeled after NEBULA_SAT_PICKLER and the
Sim_VIS_test.py script, but packaged as a reusable utility that lives
in:

    Utility/SAT_OBJECTS/NEBULA_SAT_LOS_PICKLER.py

Typical usage (from a top-level sim script)
-------------------------------------------
In a script like `Sim_VIS_test.py` or your future end-to-end simulation,
you will be able to write:

    from Utility.SAT_OBJECTS import NEBULA_SAT_LOS_PICKLER

    obs_with_los, tar_with_los = NEBULA_SAT_LOS_PICKLER.attach_los_to_all_targets()

After this call:

    - `obs_with_los`  is a dict loaded from 'observer_tracks_with_los.pkl'
    - `tar_with_los`  is a dict loaded from 'target_tracks_with_los.pkl'

Each entry contains the original serialised SatelliteTrack fields
(e.g. 'times', 'r_eci_km', etc.) plus the additional LOS fields:

    - 'los_visible' : 1D array of bool or 0/1 ints (LOS vs time)
    - 'los_h'       : 1D array of float LOS altitudes [km]
    - 'los_regime'  : 1D array of strings with Cinelli regime labels
    - 'los_fallback': 1D array of bool indicating generic-geometry fallback

Implementation details
----------------------
- This module *delegates* all LOS math and regime selection to:

      Utility.LOS.NEBULA_VIS_PICKLE
      Utility.LOS.NEBULA_VISIBILITY_DISPATCHER
      Utility.LOS.NEBULA_VISIBILITY

  so it does not duplicate any geometry or Cinelli logic.

- It relies on NEBULA_SAT_PICKLER to either:
    (a) load existing pickled tracks from NEBULA_OUTPUT/BASE_SatPickles, or
    (b) build and pickle them from scratch if they are not present.

- We now support:
    - one or more observers,
    - LOS evaluated between **every** observer and **every** target track,
    - results stored per-target and per-observer under
      ``target['by_observer'][obs_name]['los_*']``,
    - default blocking radius R_BLOCK and default regime tolerances.

You can always extend this later with:
    - custom visibility field names,
    - custom Rb or tolerance overrides,
    - selective subsets of observers or targets.

"""

import logging
from pathlib import Path
import pickle
from typing import Dict, Tuple, Any

import numpy as np

# Optional progress bar for long LOS loops
try:
    from tqdm.auto import tqdm  # type: ignore
except ImportError:
    tqdm = None  # graceful fallback if tqdm is not installed


# NEBULA configuration: path management
from Configuration.NEBULA_PATH_CONFIG import ensure_output_directory
# NEBULA pickling helpers: build or load SatelliteTrack objects
from Utility.SAT_OBJECTS import NEBULA_SAT_PICKLER
# NEBULA LOS wrapper: snapshot → timeseries
from Utility.LOS import NEBULA_VIS_PICKLE


def attach_los_to_all_targets(
    *,
    visibility_field: str = "los_visible",
    h_field: str = "los_h",
    regime_field: str = "los_regime",
    fallback_field: str = "los_fallback",
    force_recompute: bool = False,
    logger: logging.Logger | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:

    """
    High-level entry point: ensure tracks exist, attach LOS arrays to
    every target, write augmented pickles, and reload them.

    Parameters
    ----------
    visibility_field : str, optional
        Name of the attribute / pickle field that will store the LOS
        visibility flag array on each target track.  Default: "los_visible".

    h_field : str, optional
        Name of the attribute / pickle field that will store the LOS
        minimum Earth-center distance h(t) [km] for each timestep.
        Default: "los_h".

    regime_field : str, optional
        Name of the attribute / pickle field that will store the Cinelli
        regime label (string) used at each timestep.  Default: "los_regime".

    fallback_field : str, optional
        Name of the attribute / pickle field that will store a boolean
        array indicating whether the generic geometric fallback was used
        at each timestep.  Default: "los_fallback".

    logger : logging.Logger or None, optional
        Optional logger.  If None, a module-level logger named
        "NEBULA_SAT_LOS_PICKLER" is created and used.

    Returns
    -------
    observer_tracks_with_los : dict
        Dictionary loaded from 'observer_tracks_with_los.pkl'.  Keys are
        observer satellite names; values are plain serialised dicts of
        their Track fields.  For now, we do not modify observers with LOS
        attributes, but we still re-serialise them for completeness.

    target_tracks_with_los : dict
        Dictionary loaded from 'target_tracks_with_los.pkl'.  Keys are
        target satellite names; values are plain serialised dicts of
        their Track fields, *including* the new LOS arrays under the
        configured field names (e.g. "los_visible").
    """
    # ------------------------------------------------------------------
    # 1) Prepare logger
    # ------------------------------------------------------------------
    if logger is None:
        logger = logging.getLogger("NEBULA_SAT_LOS_PICKLER")
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            )
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    logger.info("Starting LOS attachment for all targets.")

    # ------------------------------------------------------------------
    # 2) Ensure output directory and LOS_SatPickles exist
    # ------------------------------------------------------------------
    out_dir: Path = ensure_output_directory()
    tracks_dir: Path = out_dir / "LOS_SatPickles"
    tracks_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Using NEBULA output directory: %s", out_dir)
    logger.info("Looking for LOS pickles in: %s", tracks_dir)


    # ------------------------------------------------------------------
    # 2a) If LOS pickles already exist and we are NOT forcing recompute,
    #     simply load and return them.
    # ------------------------------------------------------------------
    vis_obs_path = tracks_dir / "observer_tracks_with_los.pkl"
    vis_tar_path = tracks_dir / "target_tracks_with_los.pkl"

    if (not force_recompute
            and vis_obs_path.exists()
            and vis_tar_path.exists()):
        logger.info(
            "NEBULA_SAT_LOS_PICKLER: reusing existing LOS pickles "
            "(force_recompute=%s).",
            force_recompute,
        )

        with vis_obs_path.open("rb") as f_obs:
            observer_tracks_with_los = pickle.load(f_obs)
        with vis_tar_path.open("rb") as f_tar:
            target_tracks_with_los = pickle.load(f_tar)

        return observer_tracks_with_los, target_tracks_with_los


    # ------------------------------------------------------------------
    # 3) Obtain base SatelliteTrack objects (with caching at base level)
    # ------------------------------------------------------------------
    observer_tracks, target_tracks = NEBULA_SAT_PICKLER.sat_object_pickler(
        force_recompute=force_recompute
    )
    logger.info(
        "Obtained base observer/target tracks via sat_object_pickler(force_recompute=%s).",
        force_recompute,
    )

    if not observer_tracks:
        raise RuntimeError("NEBULA_SAT_LOS_PICKLER: No observer tracks available.")
    
    # List of all observer names (for logging / diagnostics).
    observer_names = list(observer_tracks.keys())
    
    logger.info("Using %d observers for LOS: %s", len(observer_names), observer_names)
    logger.info("Number of targets to process: %d", len(target_tracks))

    # ------------------------------------------------------------------
    # 4) Loop over each observer *and* each target, and populate
    #    target['by_observer'][obs_name]['los_*'] for every pair.
    # ------------------------------------------------------------------
    for obs_name, observer_track in observer_tracks.items():
        logger.info("Computing LOS with observer '%s'.", obs_name)

        # Optional tqdm progress bar over targets for this observer
        target_items = list(target_tracks.items())
        n_targets = len(target_items)
        if tqdm is not None:
            target_iter = tqdm(
                target_items,
                total=n_targets,
                desc=f"LOS: {obs_name}",
            )
        else:
            target_iter = target_items

        for idx, (tar_name, target_track) in enumerate(target_iter, start=1):
            # If tqdm is not available, fall back to per-target INFO logging
            if tqdm is None:
                logger.info(
                    "[%s] [%3d/%3d] Computing LOS for target '%s'.",
                    obs_name,
                    idx,
                    n_targets,
                    tar_name,
                )

            # Call the LOS helper – this both writes LOS fields onto the
            # target_track and returns them in a result dict.
            result = NEBULA_VIS_PICKLE.attach_los_visibility_to_target(
                observer_track=observer_track,
                target_track=target_track,
                visibility_field=visibility_field,
                h_field=h_field,
                regime_field=regime_field,
                fallback_field=fallback_field,
                custom_tolerances=None,
                logger=logger,
            )

            # Ensure the per-observer container exists on this target.
            if isinstance(target_track, dict):
                by_obs = target_track.get("by_observer")
            else:
                # SatelliteTrack or other object-like track
                by_obs = getattr(target_track, "by_observer", None)

            if by_obs is None:
                by_obs = {}
                if isinstance(target_track, dict):
                    target_track["by_observer"] = by_obs
                else:
                    setattr(target_track, "by_observer", by_obs)

            # Ensure the dict for this specific observer exists.
            obs_entry = by_obs.get(obs_name)
            if obs_entry is None:
                obs_entry = {}
                by_obs[obs_name] = obs_entry


            # Store LOS results under this observer’s entry.
            # Cast to arrays so everything is consistent downstream.
            obs_entry["los_visible"] = np.asarray(result["visible"], dtype=bool)
            obs_entry["los_h"] = np.asarray(result["h"], dtype=float)
            obs_entry["los_regime"] = np.asarray(result["regime"])
            obs_entry["los_fallback"] = np.asarray(result["fallback"], dtype=bool)

            # Quick human-readable summary for this observer–target pair.
            frac_visible = float(obs_entry["los_visible"].mean())
            if tqdm is None:
                logger.info(
                    "[%s] Target '%s' visible for %5.1f%% of timesteps.",
                    obs_name,
                    tar_name,
                    frac_visible * 100.0,
                )


    # ------------------------------------------------------------------
    # 5) Serialize updated tracks back to new pickle files
    # ------------------------------------------------------------------
    # Convert updated SatelliteTrack objects back to plain dicts (including
    # new LOS attributes) using the same helper that NEBULA_SAT_PICKLER uses.
    obs_serial = {
        name: NEBULA_SAT_PICKLER._track_to_dict(track)
        for name, track in observer_tracks.items()
    }
    tar_serial = {
        name: NEBULA_SAT_PICKLER._track_to_dict(track)
        for name, track in target_tracks.items()
    }

    vis_obs_path = tracks_dir / "observer_tracks_with_los.pkl"
    vis_tar_path = tracks_dir / "target_tracks_with_los.pkl"

    with vis_obs_path.open("wb") as f_obs:
        pickle.dump(obs_serial, f_obs)
    with vis_tar_path.open("wb") as f_tar:
        pickle.dump(tar_serial, f_tar)

    logger.info("Saved augmented observer tracks → %s", vis_obs_path)
    logger.info("Saved augmented target tracks   → %s", vis_tar_path)

    # ------------------------------------------------------------------
    # 6) Reload the new *_with_los pickles so they appear cleanly in Spyder
    # ------------------------------------------------------------------
    with vis_obs_path.open("rb") as f_obs:
        observer_tracks_with_los = pickle.load(f_obs)
    with vis_tar_path.open("rb") as f_tar:
        target_tracks_with_los = pickle.load(f_tar)

    logger.info("Reloaded augmented observer tracks from: %s", vis_obs_path)
    logger.info("Reloaded augmented target tracks   from: %s", vis_tar_path)
    logger.info(
        "LOS attachment complete. Each target entry now has a 'by_observer' "
        "mapping with per-observer LOS arrays (fields '%s', '%s', '%s', '%s').",
        visibility_field,
        h_field,
        regime_field,
        fallback_field,
    )

    return observer_tracks_with_los, target_tracks_with_los

if __name__ == "__main__":
    # If you run this module directly from Spyder with:
    #
    #   %runfile '.../Utility/SAT_OBJECTS/NEBULA_SAT_LOS_PICKLER.py' --wdir
    #
    # then the augmented dictionaries will be created as top-level
    # variables in the workspace for interactive inspection.
    observer_tracks_with_los, target_tracks_with_los = attach_los_to_all_targets(
        force_recompute=True
    )
