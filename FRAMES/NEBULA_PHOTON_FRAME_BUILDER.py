"""
NEBULA_PHOTON_FRAME_BUILDER
===========================

Purpose
-------
Photon-focused version of NEBULA_FRAME_BUILDER.

This module organizes NEBULA's coarse-time satellite products into
*frame catalogs* that carry only **photon flux** information, without
converting to electrons or using sensor-specific radiometric quantities
like collecting area, optical throughput, or quantum efficiency.

It answers:

    "At each frame time for observer X, which targets are on the detector,
     sunlit, and what is their photon flux at the aperture, and where
     do they fall in pixel coordinates?"

Key ideas
---------
- We rely entirely on existing NEBULA picklers:

    NEBULA_PIXEL_PICKLER.attach_pixels_to_all_pairs()

  which guarantees that upstream LOS, illumination, flux, and pointing
  products have already been attached.

- Frame existence is decided *only* by the observer-level flag:

    obs["pointing_valid_for_projection"][i] == True

- Source inclusion inside a frame is decided *only* by the per-target,
  per-observer flag:

    entry["on_detector_visible_sunlit"][i] == True

- For each included source we *do not* convert photon flux to electrons.
  Instead we store:

    * phi_ph_m2_s        : LOS-gated photon flux at the aperture
                           [photons m^-2 s^-1]
    * flux_ph_m2_frame   : phi_ph_m2_s * t_exp_s
                           [photons m^-2 per frame]

Outputs
-------
For a single observer, build_frames_for_observer_photon(...) returns:

    {
      "observer_name": str,
      "sensor_name":   str,
      "rows":          int,
      "cols":          int,
      "dt_frame_s":    float,
      "frames": [
        {
          "coarse_index": int,
          "t_utc":        datetime,
          "t_exp_s":      float,
          "sources": [
            {
              "source_id":          str,
              "source_type":        "target",
              "x_pix":              float,
              "y_pix":              float,
              "phi_ph_m2_s":        float,
              "flux_ph_m2_frame":   float,
              "app_mag_g":          float,
              "range_km":           float,
            },
            ...
          ],
        },
        ...
      ],
    }

The multi-observer, window-grouped version
build_frames_by_observer_and_window_photon(...) returns the same
structure but with an added "windows" list keyed by observer.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

# We still use ACTIVE_SENSOR for geometry (rows, cols), but we do NOT use
# any radiometric fields (area, throughput, QE) in this photon-only builder.
from Configuration.NEBULA_SENSOR_CONFIG import ACTIVE_SENSOR  # type: ignore

# Pixel pickler (sits on top of all upstream picklers)
from Utility.SAT_OBJECTS import NEBULA_PIXEL_PICKLER  # type: ignore


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_logger(name: str = "NEBULA_PHOTON_FRAME_BUILDER") -> logging.Logger:
    """
    Create or return a simple console logger for this module.

    Parameters
    ----------
    name : str, optional
        Name of the logger to create or retrieve.

    Returns
    -------
    logger : logging.Logger
        Logger instance configured with a basic stream handler.
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


def get_frame_time_info_for_observer(
    obs_track: Dict[str, Any],
    *,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Determine which coarse timesteps will be treated as frames.

    Parameters
    ----------
    obs_track : dict
        One observer track dictionary from obs_tracks[obs_name], as returned
        by NEBULA_PIXEL_PICKLER.attach_pixels_to_all_pairs(). Must contain:
            - "times" (list/array of datetime objects)
            - "pointing_valid_for_projection" (array of bool)
    logger : logging.Logger, optional
        For status messages. If None, a default logger is created.

    Returns
    -------
    info : dict
        Dictionary with keys:
            - "frame_indices" : np.ndarray[int]
            - "frame_times"   : np.ndarray[datetime]
            - "dt_frame_s"    : float
    """
    if logger is None:
        logger = _get_logger()

    times = np.asarray(obs_track["times"])
    mask_valid = np.asarray(
        obs_track["pointing_valid_for_projection"], dtype=bool
    )

    if times.shape[0] != mask_valid.shape[0]:
        raise ValueError(
            "get_frame_time_info_for_observer: length mismatch between "
            f"times ({times.shape[0]}) and pointing_valid_for_projection "
            f"({mask_valid.shape[0]})."
        )

    # Indices where the observer is in a valid pointing configuration.
    frame_indices = np.where(mask_valid)[0]
    frame_times = times[frame_indices]

    # Estimate dt_frame_s from the coarse time grid. For now, use the
    # median difference in seconds between consecutive coarse samples.
    if len(times) > 1:
        dt_seconds = np.median(
            [
                (times[i + 1] - times[i]).total_seconds()
                for i in range(len(times) - 1)
            ]
        )
    else:
        dt_seconds = 0.0

    logger.info(
        "Photon frame builder: observer '%s' has %d coarse timesteps, "
        "%d with pointing_valid_for_projection=True. Estimated dt=%.3f s.",
        obs_track.get("name", "UNKNOWN"),
        len(times),
        len(frame_indices),
        dt_seconds,
    )

    return {
        "frame_indices": frame_indices,
        "frame_times": frame_times,
        "dt_frame_s": float(dt_seconds),
    }


def split_frame_indices_into_windows(
    frame_indices: np.ndarray,
    frame_times: np.ndarray,
    *,
    logger: Optional[logging.Logger] = None,
) -> List[Dict[str, Any]]:
    """
    Group contiguous frame indices into pointing *windows*.

    A new window starts whenever there is a gap in the coarse index sequence,
    i.e. frame_indices[i] != frame_indices[i-1] + 1. This effectively
    groups together runs of `pointing_valid_for_projection == True`.

    Parameters
    ----------
    frame_indices : np.ndarray[int]
        Indices where the observer is valid for projection.
    frame_times : np.ndarray[datetime]
        Times corresponding to `frame_indices`, same length.
    logger : logging.Logger, optional
        For status messages.

    Returns
    -------
    windows : list of dict
        Each entry has:
            - "window_index"  : int (0, 1, 2, ...)
            - "frame_indices" : np.ndarray[int]
            - "frame_times"   : np.ndarray[datetime]
            - "start_index"   : int
            - "end_index"     : int
            - "start_time"    : datetime
            - "end_time"      : datetime
    """
    if logger is None:
        logger = _get_logger()

    windows: List[Dict[str, Any]] = []

    if frame_indices.size == 0:
        logger.info(
            "Photon frame builder: no valid pointing timesteps; 0 windows."
        )
        return windows

    start = 0
    win_idx = 0

    # Walk through the indices and start a new window on any gap.
    for i in range(1, len(frame_indices)):
        if frame_indices[i] != frame_indices[i - 1] + 1:
            # Close out previous window [start, i)
            idx_slice = frame_indices[start:i]
            time_slice = frame_times[start:i]
            windows.append(
                {
                    "window_index": win_idx,
                    "frame_indices": idx_slice,
                    "frame_times": time_slice,
                    "start_index": int(idx_slice[0]),
                    "end_index": int(idx_slice[-1]),
                    "start_time": time_slice[0],
                    "end_time": time_slice[-1],
                }
            )
            win_idx += 1
            start = i

    # Final window [start, end]
    idx_slice = frame_indices[start:]
    time_slice = frame_times[start:]
    windows.append(
        {
            "window_index": win_idx,
            "frame_indices": idx_slice,
            "frame_times": time_slice,
            "start_index": int(idx_slice[0]),
            "end_index": int(idx_slice[-1]),
            "start_time": time_slice[0],
            "end_time": time_slice[-1],
        }
    )

    logger.info(
        "Photon frame builder: split %d valid timesteps into %d windows.",
        len(frame_indices),
        len(windows),
    )

    return windows


# ---------------------------------------------------------------------------
# Photon-only source builder
# ---------------------------------------------------------------------------

def build_frame_sources_for_index_photon(
    idx: int,
    observer_name: str,
    tar_tracks: Dict[str, Any],
    *,
    t_exp_s: float,
    logger: Optional[logging.Logger] = None,
) -> List[Dict[str, Any]]:
    """
    Build a list of *photon-flux* source entries for a single frame time index.

    This applies the *per-target* inclusion mask:

        entry["on_detector_visible_sunlit"][idx] == True

    and for each included target carries forward the photon flux at the
    aperture without converting to electrons.

    Specifically, per source we store:

        - phi_ph_m2_s       : LOS-gated photon flux at aperture
                              [photons m^-2 s^-1]
        - flux_ph_m2_frame  : phi_ph_m2_s * t_exp_s
                              [photons m^-2 per frame]

    No collecting area, optical throughput, or quantum efficiency is
    applied here.

    Parameters
    ----------
    idx : int
        Coarse time index (must already satisfy pointing_valid_for_projection).
    observer_name : str
        Name of the observer, used to select the correct by_observer block
        inside each target track.
    tar_tracks : dict
        Target track dictionary returned by NEBULA_PIXEL_PICKLER.
    t_exp_s : float
        Exposure time for this frame (seconds). For now, typically equal
        to the coarse dt_frame_s.
    logger : logging.Logger, optional
        Logger for debug messages.

    Returns
    -------
    sources : list of dict
        One dictionary per included target, with keys:
            - "source_id"
            - "source_type"
            - "x_pix", "y_pix"
            - "phi_ph_m2_s"
            - "flux_ph_m2_frame"
            - "app_mag_g"
            - "range_km"
    """
    if logger is None:
        logger = _get_logger()

    sources: List[Dict[str, Any]] = []

    # Loop over all targets and build per-source entries.
    for tar_name, tar_track in tar_tracks.items():
        by_obs = tar_track.get("by_observer", {})
        if observer_name not in by_obs:
            # This target was never paired with this observer; skip.
            continue

        entry = by_obs[observer_name]

        # Safety: ensure the LOS-gated + on-detector arrays exist and are long enough.
        if "on_detector_visible_sunlit" not in entry:
            continue
        if idx >= len(entry["on_detector_visible_sunlit"]):
            continue

        if not bool(entry["on_detector_visible_sunlit"][idx]):
            # Target is not on the detector + sunlit at this frame time.
            continue

        # Extract pixel coordinates (float) for this timestep.
        x_pix = float(entry["pix_x"][idx])
        y_pix = float(entry["pix_y"][idx])

        # LOS-gated photon flux at the aperture [photons m^-2 s^-1].
        # If the key is missing, skip the target.
        if "rad_photon_flux_g_m2_s_los_only" not in entry:
            continue
        phi_ph_m2_s = float(entry["rad_photon_flux_g_m2_s_los_only"][idx])

        # Apparent magnitude in G band when LOS + visible; may be +inf when
        # there is no meaningful brightness. If missing, default to NaN.
        app_mag_g = float(
            entry.get("rad_app_mag_g_los_only", [np.nan])[idx]
        )

        # Range [km] (optional but useful for diagnostics).
        range_km = float(
            entry.get("los_icrs_range_km", [np.nan])[idx]
        )

        # Integrated photon flux per frame [photons m^-2]
        flux_ph_m2_frame = phi_ph_m2_s * t_exp_s

        source_entry: Dict[str, Any] = {
            "source_id": tar_name,
            "source_type": "target",
            "x_pix": x_pix,
            "y_pix": y_pix,
            "phi_ph_m2_s": phi_ph_m2_s,
            "flux_ph_m2_frame": flux_ph_m2_frame,
            "app_mag_g": app_mag_g,
            "range_km": range_km,
        }

        sources.append(source_entry)

    return sources


# ---------------------------------------------------------------------------
# Public API: single-observer photon frames
# ---------------------------------------------------------------------------

def build_frames_for_observer_photon(
    observer_name: str,
    *,
    max_frames: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Build a per-frame photon-flux catalog for one observer.

    This is the photon-only analogue of build_frames_for_observer() in
    NEBULA_FRAME_BUILDER. It:

        1. Calls NEBULA_PIXEL_PICKLER.attach_pixels_to_all_pairs(...) to load
           (or build) the pixel-augmented observer and target tracks.

        2. Uses get_frame_time_info_for_observer(...) to find all coarse
           timesteps where 'pointing_valid_for_projection' is True and uses
           those timesteps as frame times.

        3. For each frame index, calls build_frame_sources_for_index_photon(...)
           to build a list of contributing targets with photon fluxes.

        4. Packages the results into an 'ObserverFramesPhoton' dictionary that
           can later be pickled or passed to an optics / sensor simulator.

    Parameters
    ----------
    observer_name : str
        Name of the observer to process, e.g. "SBSS (USA 216)".
    max_frames : int or None, optional
        Optional cap on the number of frames to build (useful for quick
        experiments). If None, all valid frames are built.
    logger : logging.Logger, optional
        Logger to use. If None, a default console logger is created.

    Returns
    -------
    observer_frames : dict
        Dictionary with keys:
            - "observer_name"
            - "sensor_name"
            - "rows"
            - "cols"
            - "dt_frame_s"
            - "frames" : list of per-frame dicts (photon flux only)
    """
    if logger is None:
        logger = _get_logger()

    # 1) Load pixel-augmented tracks for all observers and targets.
    obs_tracks, tar_tracks = NEBULA_PIXEL_PICKLER.attach_pixels_to_all_pairs(
        force_recompute=False,
        sensor_config=ACTIVE_SENSOR,
        logger=logger,
    )

    if observer_name not in obs_tracks:
        raise KeyError(
            f"Observer '{observer_name}' not found in obs_tracks. "
            f"Available observers: {list(obs_tracks.keys())}"
        )

    obs_track = obs_tracks[observer_name]

    # 2) Determine which coarse indices become frames.
    time_info = get_frame_time_info_for_observer(
        obs_track,
        logger=logger,
    )
    frame_indices: np.ndarray = time_info["frame_indices"]
    frame_times: np.ndarray = time_info["frame_times"]
    dt_frame_s: float = time_info["dt_frame_s"]

    if max_frames is not None and max_frames < len(frame_indices):
        frame_indices = frame_indices[:max_frames]
        frame_times = frame_times[:max_frames]
        logger.info(
            "Photon frame builder: restricting to first %d frames for observer '%s'.",
            len(frame_indices),
            observer_name,
        )

    # 3) Build per-frame source lists (photon flux only).
    frames: List[Dict[str, Any]] = []

    for idx, t_utc in zip(frame_indices, frame_times):
        sources = build_frame_sources_for_index_photon(
            idx=idx,
            observer_name=observer_name,
            tar_tracks=tar_tracks,
            t_exp_s=dt_frame_s,
            logger=logger,
        )

        frame_entry: Dict[str, Any] = {
            "coarse_index": int(idx),
            "t_utc": t_utc,
            "t_exp_s": dt_frame_s,
            "sources": sources,
        }

        frames.append(frame_entry)

    logger.info(
        "Photon frame builder: constructed %d frames for observer '%s'.",
        len(frames),
        observer_name,
    )

    observer_frames: Dict[str, Any] = {
        "observer_name": observer_name,
        "sensor_name": "EVK4",
        "rows": ACTIVE_SENSOR.rows,
        "cols": ACTIVE_SENSOR.cols,
        "dt_frame_s": dt_frame_s,
        "frames": frames,
    }

    return observer_frames


# ---------------------------------------------------------------------------
# Public API: all observers, grouped by window (photon-only)
# ---------------------------------------------------------------------------

def build_frames_by_observer_and_window_photon(
    *,
    max_frames_per_window: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Build photon-flux frame catalogs for *all* observers, grouped by window.

    This is the photon-only analogue of build_frames_by_observer_and_window()
    in NEBULA_FRAME_BUILDER. It:

        1. Calls NEBULA_PIXEL_PICKLER.attach_pixels_to_all_pairs(...) once
           to load (or build) the pixel-augmented observer + target tracks.

        2. For each observer, uses get_frame_time_info_for_observer(...) to
           find the coarse indices where 'pointing_valid_for_projection' is
           True.

        3. Splits those indices into contiguous windows using
           split_frame_indices_into_windows(...).

        4. For each window and each frame index inside that window, calls
           build_frame_sources_for_index_photon(...) to build the per-frame
           source list with photon flux quantities only.

    Parameters
    ----------
    max_frames_per_window : int or None, optional
        Optional cap on the number of frames *per window* (useful for quick
        experiments). If None, all valid frames in each window are built.
    logger : logging.Logger, optional
        Logger to use. If None, a default console logger is created.

    Returns
    -------
    frames_by_observer : dict
        Dictionary keyed by observer_name. Each value has:

            {
              "observer_name": str,
              "sensor_name":   "EVK4",
              "rows":          int,
              "cols":          int,
              "dt_frame_s":    float,
              "windows": [
                {
                  "window_index":        int,
                  "start_index":         int,
                  "end_index":           int,
                  "start_time":          datetime,
                  "end_time":            datetime,
                  "n_frames":            int,
                  "n_unique_targets":    int,
                  "unique_target_ids":   list[str],
                  "frames": [
                    {
                      "coarse_index": int,
                      "t_utc":        datetime,
                      "t_exp_s":      float,
                      "sources":      [... photon-only source dicts ...]
                    },
                    ...
                  ],
                },
                ...
              ],
            }
    """
    if logger is None:
        logger = _get_logger()

    # 1) Load pixel-augmented tracks once for all observers and targets.
    obs_tracks, tar_tracks = NEBULA_PIXEL_PICKLER.attach_pixels_to_all_pairs(
        force_recompute=False,
        sensor_config=ACTIVE_SENSOR,
        logger=logger,
    )

    frames_by_observer: Dict[str, Any] = {}

    logger.info(
        "Photon frame builder: constructing frames for %d observers.",
        len(obs_tracks),
    )

    for obs_name, obs_track in obs_tracks.items():
        logger.info(
            "Photon frame builder: processing observer '%s'.", obs_name
        )

        # 2) Find all valid frame indices for this observer.
        time_info = get_frame_time_info_for_observer(
            obs_track,
            logger=logger,
        )
        frame_indices: np.ndarray = time_info["frame_indices"]
        frame_times: np.ndarray = time_info["frame_times"]
        dt_frame_s: float = time_info["dt_frame_s"]

        # 3) Split those indices into contiguous pointing windows.
        windows_meta = split_frame_indices_into_windows(
            frame_indices,
            frame_times,
            logger=logger,
        )

        windows_out: List[Dict[str, Any]] = []

        for win in windows_meta:
            win_indices = win["frame_indices"]
            win_times = win["frame_times"]

            # Optional cap per window.
            if (
                max_frames_per_window is not None
                and len(win_indices) > max_frames_per_window
            ):
                logger.info(
                    "Photon frame builder: window %d for observer '%s' has %d frames; "
                    "restricting to first %d for inspection.",
                    win["window_index"],
                    obs_name,
                    len(win_indices),
                    max_frames_per_window,
                )
                win_indices = win_indices[:max_frames_per_window]
                win_times = win_times[:max_frames_per_window]

            frames: List[Dict[str, Any]] = []

            # 4) Build per-frame photon-flux source lists inside this window.
            for idx, t_utc in zip(win_indices, win_times):
                sources = build_frame_sources_for_index_photon(
                    idx=int(idx),
                    observer_name=obs_name,
                    tar_tracks=tar_tracks,
                    t_exp_s=dt_frame_s,
                    logger=logger,
                )

                frame_entry: Dict[str, Any] = {
                    "coarse_index": int(idx),
                    "t_utc": t_utc,
                    "t_exp_s": dt_frame_s,
                    "sources": sources,
                }
                frames.append(frame_entry)

            # Window-level summary of targets.
            unique_target_ids = sorted(
                {
                    src["source_id"]
                    for frame in frames
                    for src in frame.get("sources", [])
                }
            )
            n_unique_targets = len(unique_target_ids)
            n_frames = len(frames)

            window_out: Dict[str, Any] = {
                "window_index": int(win["window_index"]),
                "start_index": int(win["start_index"]),
                "end_index": int(win["end_index"]),
                "start_time": win["start_time"],
                "end_time": win["end_time"],
                "n_frames": n_frames,
                "n_unique_targets": n_unique_targets,
                "unique_target_ids": unique_target_ids,
                "frames": frames,
            }
            windows_out.append(window_out)

        total_frames = sum(len(w["frames"]) for w in windows_out)
        logger.info(
            "Photon frame builder: observer '%s' has %d windows with %d total frames.",
            obs_name,
            len(windows_out),
            total_frames,
        )

        frames_by_observer[obs_name] = {
            "observer_name": obs_name,
            "sensor_name": "EVK4",
            "rows": ACTIVE_SENSOR.rows,
            "cols": ACTIVE_SENSOR.cols,
            "dt_frame_s": dt_frame_s,
            "windows": windows_out,
        }

    logger.info(
        "Photon frame builder: built windowed photon frame catalogs for %d observers.",
        len(frames_by_observer),
    )

    return frames_by_observer


# ---------------------------------------------------------------------------
# Script-style entry point (manual debug)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger = _get_logger()
    logger.info(
        "NEBULA_PHOTON_FRAME_BUILDER: building photon frames for ALL observers, "
        "grouped by pointing windows."
    )

    all_photon_frames = build_frames_by_observer_and_window_photon(
        max_frames_per_window=None,
        logger=logger,
    )

    for obs_name, obs_data in all_photon_frames.items():
        n_windows = len(obs_data.get("windows", []))
        n_frames = sum(len(w.get("frames", [])) for w in obs_data.get("windows", []))
        logger.info(
            "NEBULA_PHOTON_FRAME_BUILDER: observer '%s' -> %d windows, %d frames.",
            obs_name,
            n_windows,
            n_frames,
        )
