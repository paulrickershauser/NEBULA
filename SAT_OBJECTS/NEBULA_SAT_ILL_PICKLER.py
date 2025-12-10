# -*- coding: utf-8 -*-
"""
NEBULA_SAT_ILL_PICKLER.py

High-level helper for attaching *illumination* arrays to NEBULA satellite
track objects that already have LOS information.

Design goal
-----------
Provide a single "behind-the-scenes" function that you can call from your
simulation code to go from:

    "tracks with LOS fields"  →  "tracks with LOS + Skyfield illumination"

without manually orchestrating:

    - Calling NEBULA_SAT_LOS_PICKLER,
    - Looping over every observer–target pair,
    - Calling the Skyfield-based illumination routine, and
    - Writing updated pickles.

This module lives in:

    Utility/SAT_OBJECTS/NEBULA_SAT_ILL_PICKLER.py

and is intentionally modeled after NEBULA_SAT_LOS_PICKLER and the earlier
illumination_test.py script.

Typical usage
-------------
From a top-level sim driver, you can write:

    from Utility.SAT_OBJECTS import NEBULA_SAT_ILL_PICKLER

    obs_ill, tar_ill = NEBULA_SAT_ILL_PICKLER.attach_illum_to_all_targets()

After this call:

    - `obs_ill` is a dict loaded from
          'observer_tracks_with_los_illum.pkl'
    - `tar_ill` is a dict loaded from
          'target_tracks_with_los_illum.pkl'

Each *target* entry contains, in addition to the original serialised
track fields and LOS arrays, these illumination fields:

    - 'illum_is_sunlit'             : bool array, True if target is sunlit
    - 'illum_phase_angle_rad'       : float array, Sun–target–observer angle [rad]
    - 'illum_fraction_illuminated'  : float array, Lambertian lit fraction

Implementation details
----------------------
- All orbit propagation and track construction is delegated to
  `NEBULA_SAT_LOS_PICKLER`, which itself uses NEBULA_SAT_PICKLER and
  NEBULA_VIS_PICKLE.

- All illumination math is delegated to
  `Utility.RADIOMETRY.NEBULA_SKYFIELD_ILLUMINATION`, which:

    * loads the DE440s JPL ephemeris,
    * builds Skyfield TEME→ICRF positions,
    * calls `ICRF.is_sunlit(eph)` for Earth-shadow tests,
    * computes Sun–target–observer phase angle α(t),
    * and evaluates the Lambertian illuminated fraction:

          f(t) = (1 + cos α(t)) / 2

  (same expression used by Skyfield's `fraction_illuminated()` helper).

- This module simply:
    1) ensures LOS-enhanced tracks exist,
    2) chooses one observer (by name or first in dict),
    3) runs the illumination step for every target, and
    4) serialises the updated dicts to new pickles.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple, Any, Optional
import pickle

import numpy as np

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


# NEBULA path management (for NEBULA_OUTPUT / track_debug)
from Configuration.NEBULA_PATH_CONFIG import ensure_output_directory

# NEBULA LOS helper: builds tracks (if needed) and attaches LOS arrays
from Utility.SAT_OBJECTS import NEBULA_SAT_LOS_PICKLER

# Skyfield-based illumination utilities (already working in your test)
from Utility.RADIOMETRY.NEBULA_SKYFIELD_ILLUMINATION import (
    compute_illumination_timeseries_for_pair,
    load_de440s_ephemeris,
)



def attach_illum_to_all_targets(
    *,
    observer_name: Optional[str] = None,
    ephemeris_path: Optional[str] = None,
    force_recompute: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    High-level entry point: ensure LOS-enhanced tracks exist, attach
    Skyfield illumination arrays to every target, write augmented
    pickles, and reload them.

    Parameters
    ----------
    observer_name : str or None, optional
        Name of the observer satellite to use for illumination geometry.
        If None, the first observer in the dictionary returned by
        NEBULA_SAT_LOS_PICKLER.attach_los_to_all_targets() is used.

    ephemeris_path : str or None, optional
        Explicit path to the JPL DE440s binary ephemeris (de440s.bsp).
        If None, the default path defined inside
        NEBULA_SKYFIELD_ILLUMINATION.load_de440s_ephemeris() is used.

    force_recompute : bool, optional
        If False (default) and the illumination pickles
        ``observer_tracks_with_los_illum.pkl`` and
        ``target_tracks_with_los_illum.pkl`` already exist in the
        NEBULA_OUTPUT/track_debug directory, those files are simply
        loaded and returned (no new illumination computation is run).
        If True, illumination is recomputed for all targets and the
        pickles are overwritten.

    logger : logging.Logger or None, optional
        Optional logger for informational messages. If None, this
        function creates a logger named "NEBULA_SAT_ILL_PICKLER" with a
        basic StreamHandler.


    Returns
    -------
    observer_tracks_with_illum : dict
        Dictionary loaded from
            NEBULA_OUTPUT/track_debug/observer_tracks_with_los_illum.pkl
        Keys are observer names; values are *serialised dicts* of
        track fields.  At present, observers are not modified with
        illumination attributes, but they are re-serialised for
        completeness.

    target_tracks_with_illum : dict
        Dictionary loaded from
            NEBULA_OUTPUT/track_debug/target_tracks_with_los_illum.pkl
        Keys are target names; values are *serialised dicts* of track
        fields, including LOS arrays and the newly attached illumination
        arrays:

            - 'illum_is_sunlit'            : numpy.ndarray[bool]
            - 'illum_phase_angle_rad'      : numpy.ndarray[float]
            - 'illum_fraction_illuminated' : numpy.ndarray[float]
    """
    # ----------------------------------------------------------------------
    # 1) Prepare logger
    # ----------------------------------------------------------------------
    if logger is None:
        logger = logging.getLogger("NEBULA_SAT_ILL_PICKLER")
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            )
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    logger.info("Starting illumination attachment for all targets.")

    # ----------------------------------------------------------------------
    # 2) Ensure NEBULA output directory and ILLUM_SatPickles folder exist
    # ----------------------------------------------------------------------
    out_dir: Path = ensure_output_directory()
    tracks_dir: Path = out_dir / "ILLUM_SatPickles"
    tracks_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Using NEBULA output directory: %s", out_dir)
    logger.info("Illumination tracks directory: %s", tracks_dir)


    # ----------------------------------------------------------------------
    # 2a) If LOS+illumination pickles already exist and we are NOT forcing
    #     recomputation, simply load and return them.
    # ----------------------------------------------------------------------
    obs_illum_path = tracks_dir / "observer_tracks_with_los_illum.pkl"
    tgt_illum_path = tracks_dir / "target_tracks_with_los_illum.pkl"

    if (not force_recompute
            and obs_illum_path.exists()
            and tgt_illum_path.exists()):
        logger.info(
            "NEBULA_SAT_ILL_PICKLER: reusing existing LOS+illum pickles "
            "(force_recompute=%s).",
            force_recompute,
        )

        with obs_illum_path.open("rb") as f_obs:
            observer_tracks_with_illum = pickle.load(f_obs)
        with tgt_illum_path.open("rb") as f_tgt:
            target_tracks_with_illum = pickle.load(f_tgt)

        return observer_tracks_with_illum, target_tracks_with_illum


    # ----------------------------------------------------------------------
    # 3) Call LOS pickler to get tracks with LOS fields
    # ----------------------------------------------------------------------
    logger.info(
        "Calling NEBULA_SAT_LOS_PICKLER.attach_los_to_all_targets() "
        "to ensure LOS fields are present."
    )

    observer_tracks, target_tracks = NEBULA_SAT_LOS_PICKLER.attach_los_to_all_targets(
        force_recompute=force_recompute,
        logger=logger,
    )


    if not observer_tracks:
        raise RuntimeError("NEBULA_SAT_ILL_PICKLER: No observer tracks available.")

    # ----------------------------------------------------------------------
    # 4) Choose which observer(s) to use for illumination geometry
    # ----------------------------------------------------------------------
    if observer_name is not None:
        # Single-observer mode: restrict to the requested observer.
        if observer_name not in observer_tracks:
            raise KeyError(
                f"Requested observer_name='{observer_name}' not found in "
                f"observer_tracks keys: {list(observer_tracks.keys())}"
            )
        observer_names = [observer_name]
    else:
        # Multi-observer mode: use all observers returned by the LOS pickler.
        observer_names = list(observer_tracks.keys())

    logger.info(
        "Using %d observer(s) for illumination computations: %s",
        len(observer_names),
        observer_names,
    )
    logger.info("Number of targets to process: %d", len(target_tracks))

    # ----------------------------------------------------------------------
    # 5) Load DE440s ephemeris once (reuse for all observers / targets)
    # ----------------------------------------------------------------------
    eph = load_de440s_ephemeris(ephemeris_path=ephemeris_path, logger=logger)

    # ----------------------------------------------------------------------
    # 6) Loop over each observer and each target; attach illumination
    #     fields and cache them in target.by_observer[obs_name].
    # ----------------------------------------------------------------------
    n_targets = len(target_tracks)

    for obs_name in observer_names:
        observer_track = observer_tracks[obs_name]
        logger.info("Computing illumination with observer '%s'.", obs_name)

        target_iter = target_tracks.items()
        if tqdm is not None:
            target_iter = tqdm(
                target_iter,
                total=n_targets,
                desc=f"ILLUM: {obs_name}",
                unit="target",
            )

        for idx, (tgt_name, tgt_track) in enumerate(target_iter, start=1):
            # Fallback: if tqdm is not available, emit per-target INFO logs.
            if tqdm is None:
                logger.info(
                    "[%s] [%3d/%3d] Computing illumination for target '%s'.",
                    obs_name,
                    idx,
                    n_targets,
                    tgt_name,
                )
        
            try:
                # Compute illumination *without* writing anything onto tgt_track.
                illum_result = compute_illumination_timeseries_for_pair(
                    observer_track=observer_track,
                    target_track=tgt_track,
                    eph=eph,
                    store_on_target=False,
                    logger=logger,
                )
        
                # Pull arrays directly from the result container.
                is_sunlit = illum_result.is_sunlit
                phase = illum_result.phase_angle_rad
                frac = illum_result.fraction_illuminated
        
                # Ensure a per-observer container exists on this target.
                if isinstance(tgt_track, dict):
                    by_obs = tgt_track.get("by_observer")
                else:
                    by_obs = getattr(tgt_track, "by_observer", None)
        
                if by_obs is None:
                    by_obs = {}
                    if isinstance(tgt_track, dict):
                        tgt_track["by_observer"] = by_obs
                    else:
                        setattr(tgt_track, "by_observer", by_obs)
        
                # Ensure the dict for this specific observer exists.
                obs_entry = by_obs.get(obs_name)
                if obs_entry is None:
                    obs_entry = {}
                    by_obs[obs_name] = obs_entry
        
                # Store illumination results under this observer’s entry.
                obs_entry["illum_is_sunlit"] = np.asarray(is_sunlit, dtype=bool)
                obs_entry["illum_phase_angle_rad"] = np.asarray(phase, dtype=float)
                obs_entry["illum_fraction_illuminated"] = np.asarray(frac, dtype=float)

                # Optional sanity log for the first target per observer
                # when tqdm is not available (to avoid console spam).
                if idx == 1 and len(is_sunlit) > 0 and tqdm is None:
                    logger.info(
                        "[%s] Example for first target '%s': "
                        "first is_sunlit=%s, phase[deg]=%.2f, frac_illum=%.3f",
                        obs_name,
                        tgt_name,
                        bool(is_sunlit[0]),
                        float(np.degrees(phase[0])),
                        float(frac[0]),
                    )

            except Exception as exc:  # pragma: no cover
                # If a particular target fails illumination for this observer,
                # log and move on to the next one.
                logger.exception(
                    "[%s] Error computing illumination for target '%s': %s",
                    obs_name,
                    tgt_name,
                    exc,
                )

    logger.info(
        "Finished attaching illumination fields for all observers and targets."
    )


    # ----------------------------------------------------------------------
    # 7) Serialize updated tracks to new pickle files
    # ----------------------------------------------------------------------
    obs_path = tracks_dir / "observer_tracks_with_los_illum.pkl"
    tgt_path = tracks_dir / "target_tracks_with_los_illum.pkl"

    with obs_path.open("wb") as f_obs:
        pickle.dump(observer_tracks, f_obs)

    with tgt_path.open("wb") as f_tgt:
        pickle.dump(target_tracks, f_tgt)

    logger.info("Saved augmented observer tracks → %s", obs_path)
    logger.info("Saved augmented target tracks   → %s", tgt_path)

    # ----------------------------------------------------------------------
    # 8) Reload the new pickles so they appear cleanly in the workspace
    # ----------------------------------------------------------------------
    with obs_path.open("rb") as f_obs:
        observer_tracks_with_illum = pickle.load(f_obs)

    with tgt_path.open("rb") as f_tgt:
        target_tracks_with_illum = pickle.load(f_tgt)

    logger.info(
        "Reloaded augmented tracks. "
        "Each target now includes LOS fields plus "
        "'illum_is_sunlit', 'illum_phase_angle_rad', "
        "and 'illum_fraction_illuminated'."
    )

    return observer_tracks_with_illum, target_tracks_with_illum


if __name__ == "__main__":
    # Running this file directly from Spyder (with wdir set to NEBULA root),
    # e.g. using:
    #
    #   %runfile "Utility/SAT_OBJECTS/NEBULA_SAT_ILL_PICKLER.py" --wdir
    #
    # will execute the illumination pipeline and leave the reloaded dicts
    # in the workspace as:
    #
    #   observer_tracks_with_illum
    #   target_tracks_with_illum
    #
    # Fast path: reuse existing LOS+illum pickles if they exist
    # observer_tracks_with_illum, target_tracks_with_illum = attach_illum_to_all_targets(
    #     force_recompute=False
    # )
    #
    # Clean path: recompute LOS and illumination from current tracks
    observer_tracks_with_illum, target_tracks_with_illum = attach_illum_to_all_targets(
        force_recompute=True
    )

    print("\n=== NEBULA_SAT_ILL_PICKLER completed ===")
    print("Observers with LOS + illum:", len(observer_tracks_with_illum))
    print("Targets   with LOS + illum:", len(target_tracks_with_illum))

