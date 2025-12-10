"""
NEBULA_SKY_SELECTOR
===================

Attach per-window *sky footprints* (RA/Dec center + radius) to the
ranked photon-only target windows produced by NEBULA_TARGET_PHOTONS.

This module sits on top of:

- Utility.FRAMES.NEBULA_TARGET_PHOTONS
    (which writes TARGET_PHOTON_FRAMES/obs_target_frames_ranked.pkl)
- Utility.SAT_OBJECTS.NEBULA_SCHEDULE_PICKLER
    (which attaches pointing to observer tracks and exposes boresight RA/Dec)

For each observer and each window we:

1. Use the window's [start_index, end_index] to select the corresponding
   coarse time range in the observer's track.
2. Read the *boresight* pointing RA/Dec arrays from the observer track
   (e.g. "boresight_ra_deg" / "boresight_dec_deg", or "pointing_ra_deg"
   / "pointing_dec_deg" as a fallback).
3. Compute a "center" direction (RA/Dec) and a radius that covers:
      - the sensor half-FOV, plus
      - any slew of the boresight within the window, plus
      - a small safety margin in degrees.
4. Store these as:
      window["sky_center_ra_deg"]
      window["sky_center_dec_deg"]
      window["sky_radius_deg"]
   along with a few diagnostic fields.

The resulting structure is intended as the **input** to later Gaia
query stages (e.g. NEBULA_STAR_QUERY), which can perform cone
searches on these footprints and attach Gaia stars per window.

Typical usage (from Spyder/IPython)
-----------------------------------

    from Utility.STARS import NEBULA_SKY_SELECTOR as NSS

    logger = NSS.get_logger()
    NSS.main(logger=logger)

This will:

    - load TARGET_PHOTON_FRAMES/obs_target_frames_ranked.pkl
    - call NEBULA_SCHEDULE_PICKLER to ensure pointing-attached tracks exist
    - attach sky footprints per window
    - write TARGET_PHOTON_FRAMES/obs_target_frames_ranked_with_sky.pkl
"""

from __future__ import annotations

import logging
import math
import os
import pickle
from typing import Any, Dict, Tuple

import numpy as np

from Configuration.NEBULA_PATH_CONFIG import NEBULA_OUTPUT_DIR
from Configuration.NEBULA_SENSOR_CONFIG import ACTIVE_SENSOR
from Utility.SAT_OBJECTS import NEBULA_SCHEDULE_PICKLER  # type: ignore
from Configuration.NEBULA_STAR_CONFIG import get_safety_margin_deg


# ---------------------------------------------------------------------------
# Logger helper
# ---------------------------------------------------------------------------

def get_logger(name: str = "NEBULA_SKY_SELECTOR") -> logging.Logger:
    """
    Create or retrieve a simple console logger for this module.

    Parameters
    ----------
    name : str, optional
        Logger name. Default is "NEBULA_SKY_SELECTOR".

    Returns
    -------
    logger : logging.Logger
        Logger configured with a basic stream handler if one does not
        already exist.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# ---------------------------------------------------------------------------
# Small geometry helper: great-circle separation on the sphere
# ---------------------------------------------------------------------------

def _angular_sep_deg(
    ra1_deg: float,
    dec1_deg: float,
    ra2_deg: np.ndarray,
    dec2_deg: np.ndarray,
) -> np.ndarray:
    """
    Compute great-circle angular separation(s) between (ra1, dec1) and
    arrays (ra2, dec2) in degrees.

    Parameters
    ----------
    ra1_deg, dec1_deg : float
        Scalar reference direction in degrees.
    ra2_deg, dec2_deg : array_like
        Arrays of directions in degrees.

    Returns
    -------
    sep_deg : np.ndarray
        Angular separation(s) in degrees, same shape as ra2_deg/dec2_deg.
    """
    ra1_rad = np.deg2rad(ra1_deg)
    dec1_rad = np.deg2rad(dec1_deg)
    ra2_rad = np.deg2rad(ra2_deg)
    dec2_rad = np.deg2rad(dec2_deg)

    sin_dec1 = math.sin(dec1_rad)
    cos_dec1 = math.cos(dec1_rad)
    sin_dec2 = np.sin(dec2_rad)
    cos_dec2 = np.cos(dec2_rad)
    delta_ra = ra2_rad - ra1_rad

    # Spherical law of cosines
    cos_gamma = sin_dec1 * sin_dec2 + cos_dec1 * cos_dec2 * np.cos(delta_ra)
    # Guard against numerical noise
    cos_gamma = np.clip(cos_gamma, -1.0, 1.0)

    gamma_rad = np.arccos(cos_gamma)
    return np.rad2deg(gamma_rad)


# ---------------------------------------------------------------------------
# Core function: attach sky footprints per observer / window
# ---------------------------------------------------------------------------

def attach_sky_footprints(
    obs_target_frames: Dict[str, Any],
    obs_tracks: Dict[str, Any],
    fov_half_angle_deg: float | None = None,
    safety_margin_deg: float = None,
    logger: logging.Logger | None = None,
) -> Dict[str, Any]:
    """
    Attach sky footprints (center RA/Dec + radius) to each window in
    `obs_target_frames` based on pointing in `obs_tracks`.

    Parameters
    ----------
    obs_target_frames : dict
        Dictionary keyed by observer name, as produced by
        NEBULA_TARGET_PHOTONS.build_obs_target_frames_for_all_observers()
        and optionally culled/ranked.

        Structure per observer:

            obs_target_frames[obs_name] = {
                "observer_name": obs_name,
                "rows": int,
                "cols": int,
                "dt_frame_s": float,
                "windows": [
                    {
                        "window_index": int,
                        "start_index": int,
                        "end_index": int,
                        "start_time": any,
                        "end_time": any,
                        "n_frames": int,
                        "n_targets": int,
                        "targets": {...},  # per-target time series
                    },
                    ...
                ],
            }

    obs_tracks : dict
        Dictionary keyed by observer name, typically returned by
        NEBULA_SCHEDULE_PICKLER.attach_pointing_to_all_observers().
        Each entry must include boresight pointing arrays in degrees,
        e.g.:

            obs_tracks[obs_name]["boresight_ra_deg"]
            obs_tracks[obs_name]["boresight_dec_deg"]

        or, as a fallback:

            obs_tracks[obs_name]["pointing_ra_deg"]
            obs_tracks[obs_name]["pointing_dec_deg"]

        with length equal to the number of coarse timesteps.

    fov_half_angle_deg : float or None, optional
        Half-angle of the sensor field-of-view in degrees. If None,
        this is derived as 0.5 * ACTIVE_SENSOR.fov_deg.

    safety_margin_deg : float, optional
        Additional margin added to the radius (in degrees) to ensure
        we do not clip stars due to small numerical differences or
        future modeling changes. Default is 0.2 deg.

    logger : logging.Logger or None, optional
        Logger for informational messages. If None, a default logger
        is created via get_logger().

    Returns
    -------
    updated : dict
        The same `obs_target_frames` dictionary with each window
        augmented in-place with:

            "sky_center_ra_deg"      : float
            "sky_center_dec_deg"     : float
            "sky_radius_deg"         : float
            "sky_selector_max_slew_deg" : float
            "sky_selector_fov_half_deg" : float
            "sky_selector_margin_deg"   : float
            "sky_selector_status"       : str

        The function returns the dict for convenience/chaining.
    """
    if logger is None:
        logger = get_logger()
        
    if safety_margin_deg is None:
        safety_margin_deg = get_safety_margin_deg()

    # Derive half-FOV if not explicitly provided
    if fov_half_angle_deg is None:
        try:
            fov_half_angle_deg = 0.5 * float(ACTIVE_SENSOR.fov_deg)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Unable to derive fov_half_angle_deg from ACTIVE_SENSOR.fov_deg. "
                "Provide fov_half_angle_deg explicitly or ensure ACTIVE_SENSOR "
                "has a 'fov_deg' attribute."
            ) from exc

    logger.info(
        "NEBULA_SKY_SELECTOR: attaching sky footprints using "
        "fov_half_angle_deg=%.3f, safety_margin_deg=%.3f.",
        fov_half_angle_deg,
        safety_margin_deg,
    )

    # Helper: choose RA/Dec array keys from an observer track dict
    def _get_ra_dec_arrays(track: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract pointing RA/Dec arrays from an observer track.

        We try keys in the following order:

            1) ("pointing_boresight_ra_deg", "pointing_boresight_dec_deg")
               - current naming used by NEBULA_SCHEDULE_PICKLER /
                 pointing dispatcher / NEBULA_WCS.

            2) ("boresight_ra_deg", "boresight_dec_deg")
               - legacy naming from older pointing modules.

            3) ("pointing_ra_deg", "pointing_dec_deg")
               - another legacy variant.

        Raises
        ------
        KeyError
            If none of the expected key pairs are found on the track.
        """
        candidate_pairs = [
            ("pointing_boresight_ra_deg", "pointing_boresight_dec_deg"),
            ("boresight_ra_deg", "boresight_dec_deg"),
            ("pointing_ra_deg", "pointing_dec_deg"),
        ]

        for ra_key, dec_key in candidate_pairs:
            if ra_key in track and dec_key in track:
                ra = np.asarray(track[ra_key], dtype=float)
                dec = np.asarray(track[dec_key], dtype=float)

                if ra.shape != dec.shape:
                    raise KeyError(
                        f"Mismatched RA/Dec shapes for keys '{ra_key}'/'{dec_key}': "
                        f"{ra.shape} vs {dec.shape}"
                    )

                return ra, dec

        raise KeyError(
            "Observer track is missing expected pointing keys. "
            f"Tried {candidate_pairs}."
        )


    # Iterate over observers in the photon-frame structure
    for obs_name, obs_entry in obs_target_frames.items():
        if obs_name not in obs_tracks:
            logger.warning(
                "Observer '%s' not found in obs_tracks; skipping sky footprints "
                "for this observer.",
                obs_name,
            )
            continue

        track = obs_tracks[obs_name]
        try:
            ra_arr, dec_arr = _get_ra_dec_arrays(track)
        except KeyError as exc:
            logger.error(
                "Observer '%s' is missing pointing arrays in obs_tracks: %s",
                obs_name,
                exc,
            )
            continue

        n_samples = len(ra_arr)
        windows = obs_entry.get("windows", [])

        logger.info(
            "Observer '%s': attaching sky footprints to %d windows "
            "(n_samples=%d).",
            obs_name,
            len(windows),
            n_samples,
        )

        for w in windows:
            start_idx = int(w.get("start_index", -1))
            end_idx = int(w.get("end_index", -1))

            # Basic index sanity checks
            if start_idx < 0 or end_idx < 0 or end_idx < start_idx:
                w["sky_selector_status"] = "invalid_indices"
                logger.warning(
                    "Observer '%s', window %s has invalid indices "
                    "[start_index=%d, end_index=%d]; skipping.",
                    obs_name,
                    w.get("window_index", -1),
                    start_idx,
                    end_idx,
                )
                continue

            # Clip to available samples
            start_idx_clamped = max(0, min(start_idx, n_samples - 1))
            end_idx_clamped = max(0, min(end_idx, n_samples - 1))

            if end_idx_clamped < start_idx_clamped:
                w["sky_selector_status"] = "clamped_empty"
                logger.warning(
                    "Observer '%s', window %s indices clamped to empty range; "
                    "skipping.",
                    obs_name,
                    w.get("window_index", -1),
                )
                continue

            idx_range = np.arange(start_idx_clamped, end_idx_clamped + 1, dtype=int)
            if idx_range.size == 0:
                w["sky_selector_status"] = "empty_range"
                logger.warning(
                    "Observer '%s', window %s has empty index range after "
                    "clamping; skipping.",
                    obs_name,
                    w.get("window_index", -1),
                )
                continue

            # For sidereal stare, boresight is (almost) constant; for slews we
            # handle the variation explicitly via max separation.
            mid_idx = idx_range[len(idx_range) // 2]
            center_ra_deg = float(ra_arr[mid_idx])
            center_dec_deg = float(dec_arr[mid_idx])

            # Compute max angular separation between boresight(t) in this
            # window and the chosen center direction.
            ra_win = ra_arr[idx_range]
            dec_win = dec_arr[idx_range]
            sep_deg = _angular_sep_deg(center_ra_deg, center_dec_deg, ra_win, dec_win)
            max_slew_deg = float(np.max(sep_deg))

            # Final footprint radius: half-FOV + slew span + safety margin
            sky_radius_deg = float(
                fov_half_angle_deg + max_slew_deg + safety_margin_deg
            )

            w["sky_center_ra_deg"] = center_ra_deg
            w["sky_center_dec_deg"] = center_dec_deg
            w["sky_radius_deg"] = sky_radius_deg
            w["sky_selector_max_slew_deg"] = max_slew_deg
            w["sky_selector_fov_half_deg"] = float(fov_half_angle_deg)
            w["sky_selector_margin_deg"] = float(safety_margin_deg)
            w["sky_selector_status"] = "ok"

    return obs_target_frames


# ---------------------------------------------------------------------------
# Main driver: load ranked windows, attach footprints, save new pickle
# ---------------------------------------------------------------------------

def main(logger: logging.Logger | None = None) -> None:
    """
    Script entry point for NEBULA_SKY_SELECTOR.

    Steps
    -----
    1. Load ranked photon-only windows:
           TARGET_PHOTON_FRAMES/obs_target_frames_ranked.pkl
    2. Ensure observer tracks with pointing exist by calling
           NEBULA_SCHEDULE_PICKLER.attach_pointing_to_all_observers()
       with force_recompute=False.
    3. Attach sky footprints per observer/window via `attach_sky_footprints`.
    4. Save the updated dictionary to:
           TARGET_PHOTON_FRAMES/obs_target_frames_ranked_with_sky.pkl
    """
    if logger is None:
        logger = get_logger()

    # Directory holding the target photon frames pickles
    frames_dir = os.path.join(NEBULA_OUTPUT_DIR, "TARGET_PHOTON_FRAMES")
    os.makedirs(frames_dir, exist_ok=True)

    ranked_path = os.path.join(frames_dir, "obs_target_frames_ranked.pkl")
    if not os.path.exists(ranked_path):
        raise FileNotFoundError(
            f"Ranked target photon frames pickle not found at '{ranked_path}'. "
            "Run NEBULA_TARGET_PHOTONS.main() first to generate it."
        )

    logger.info(
        "Loading ranked target photon frames from '%s'.",
        ranked_path,
    )
    with open(ranked_path, "rb") as f:
        obs_target_frames_ranked: Dict[str, Any] = pickle.load(f)

    # Get observer tracks with pointing attached.
    # This call will:
    #   - reuse existing pickles if available (force_recompute=False), or
    #   - recompute LOS+flux+pointing as needed.
    logger.info(
        "Calling NEBULA_SCHEDULE_PICKLER.attach_pointing_to_all_observers("
        "force_recompute=False) to obtain observer tracks with pointing."
    )
    obs_tracks, _tar_tracks = NEBULA_SCHEDULE_PICKLER.attach_pointing_to_all_observers(
        force_recompute=False,
        logger=logger,
    )

    logger.info(
        "Attaching sky footprints per observer/window based on pointing arrays."
    )
    updated_frames = attach_sky_footprints(
        obs_target_frames=obs_target_frames_ranked,
        obs_tracks=obs_tracks,
        fov_half_angle_deg=None,        # derive from ACTIVE_SENSOR
        safety_margin_deg=get_safety_margin_deg(),          # tweak via config later if desired
        logger=logger,
    )

    out_path = os.path.join(frames_dir, "obs_target_frames_ranked_with_sky.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(updated_frames, f)

    # Compact summary
    for obs_name, obs_entry in updated_frames.items():
        windows = obs_entry.get("windows", [])
        n_with_sky = sum(
            1 for w in windows if w.get("sky_selector_status", "") == "ok"
        )
        logger.info(
            "Observer '%s': %d windows total, %d with sky footprints attached.",
            obs_name,
            len(windows),
            n_with_sky,
        )

    logger.info(
        "NEBULA_SKY_SELECTOR complete. "
        "Updated ranked frames with sky footprints written to '%s'.",
        out_path,
    )


if __name__ == "__main__":
    main()
