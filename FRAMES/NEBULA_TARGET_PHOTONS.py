"""
NEBULA_TARGET_PHOTONS
=====================

Photon-centered counterpart to NEBULA_PHOTUTILS_WORKFLOW.

This module sits *on top of* NEBULA_PHOTON_FRAME_BUILDER and organizes
its per-frame, per-source photon flux into **per-target time series**
that can be fed directly into downstream sensor/circuit simulations.

Key properties:

- Uses NEBULA_PHOTON_FRAME_BUILDER.build_frames_by_observer_and_window_photon()
  as its only upstream dependency.
- Operates purely in the photon domain:
    * phi_ph_m2_s      [photons m^-2 s^-1]    (at the optical aperture)
    * flux_ph_m2_frame [photons m^-2 frame^-1]
- Does NOT apply:
    * collecting area
    * optical throughput
    * quantum efficiency
    * read noise, dark current, etc.

High-level usage
----------------
Typical manual usage from Spyder/IPython:

    from Utility.FRAMES import NEBULA_TARGET_PHOTONS as NTP

    logger = NTP.get_logger()

    # 1) Build photon frames and pick one "best" window for an observer
    obs_name, obs_data, best_window = NTP.build_photon_frames_and_pick_best_window(
        max_frames_per_window=None,
        observer_name=None,  # or "SBSS (USA 216)", "SAPPHIRE", etc.
        logger=logger,
    )

    # 2) Build per-target photon time series for that window
    photon_catalog = NTP.build_target_photon_timeseries_for_window(
        obs_name,
        obs_data,
        best_window,
        logger=logger,
    )

    # 3) Inspect/hand off photon_catalog["targets"][target_id] to your
    #    circuit-level sensor simulator.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

# Photon-only frame builder (no sensor-specific radiometry)
from Utility.FRAMES import NEBULA_PHOTON_FRAME_BUILDER as PFB  # type: ignore

import os
import pickle

from Configuration.NEBULA_PATH_CONFIG import NEBULA_OUTPUT_DIR

# Progress bar import 
from tqdm import tqdm 

# ---------------------------------------------------------------------------
# Logger helper
# ---------------------------------------------------------------------------

def get_logger(name: str = "NEBULA_TARGET_PHOTONS") -> logging.Logger:
    """
    Create or retrieve a simple console logger for this module.

    Parameters
    ----------
    name : str, optional
        Name of the logger to create or retrieve. Default is
        "NEBULA_TARGET_PHOTONS".

    Returns
    -------
    logger : logging.Logger
        Logger instance configured with a basic stream handler if one
        does not already exist.
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
# Step 1: Build photon frames and select a "best" window (optional)
# ---------------------------------------------------------------------------

def build_photon_frames_and_pick_best_window(
    max_frames_per_window: int | None = None,
    observer_name: str | None = None,
    logger: logging.Logger | None = None,
) -> tuple[str, Dict[str, Any], Dict[str, Any]]:
    """
    Build photon frames for all observers, then select a "best" window
    for a single observer.

    This is a convenience wrapper for interactive work. It mirrors the
    logic in NEBULA_PHOTUTILS_WORKFLOW.build_frames_and_pick_best_window,
    but operates on photon-only frames provided by
    NEBULA_PHOTON_FRAME_BUILDER.

    Observer selection
    ------------------
    * If `observer_name` is given, that observer must exist in the
      frames_by_observer dictionary, otherwise a RuntimeError is raised.
    * If `observer_name` is None, the first observer name in sorted
      order is used. This keeps the function generic and avoids any
      mission-specific hard-coding.

    Window selection
    ----------------
    For the chosen observer, the "best" window is defined as the one
    that maximizes the pair:

        (n_unique_targets, n_frames)

    using lexicographic ordering. In other words, windows with more
    unique targets are preferred, and ties are broken by the number of
    frames.

    Parameters
    ----------
    max_frames_per_window : int or None, optional
        If not None, cap the number of frames per window when calling
        PFB.build_frames_by_observer_and_window_photon(). This is
        primarily useful for quick tests on large scenarios.
    observer_name : str or None, optional
        Name of the observer to use. If None, the first observer in
        sorted(frames_by_observer.keys()) is selected.
    logger : logging.Logger or None, optional
        Optional logger instance. If None, a default console logger is
        created via get_logger().

    Returns
    -------
    obs_name : str
        Name of the observer whose window was selected.
    obs_data : dict
        Observer data dictionary from NEBULA_PHOTON_FRAME_BUILDER
        containing metadata and the list of windows.
    best_window : dict
        The chosen window dictionary. It contains fields such as:
          - "window_index"
          - "n_unique_targets"
          - "n_frames"
          - "frames" (list of per-frame photon dicts).
    """
    if logger is None:
        logger = get_logger()

    logger.info(
        "Building photon frames for all observers via "
        "NEBULA_PHOTON_FRAME_BUILDER."
    )

    frames_by_observer = PFB.build_frames_by_observer_and_window_photon(
        max_frames_per_window=max_frames_per_window,
        logger=logger,
    )

    if not frames_by_observer:
        raise RuntimeError("No observers found in photon frames_by_observer.")

    # ------------------------------------------------------------------
    # Observer choice: generic, no mission-specific hard-coding
    # ------------------------------------------------------------------
    if observer_name is None:
        # Default to the first observer in sorted order
        obs_name = sorted(frames_by_observer.keys())[0]
        logger.info(
            "No observer_name provided; using first observer '%s' "
            "from sorted list.",
            obs_name,
        )
    else:
        if observer_name not in frames_by_observer:
            raise RuntimeError(
                f"Requested observer '{observer_name}' not found in "
                f"frames_by_observer keys: {sorted(frames_by_observer.keys())}"
            )
        obs_name = observer_name

    obs_data = frames_by_observer[obs_name]
    windows = obs_data.get("windows", [])

    if not windows:
        raise RuntimeError(f"Observer '{obs_name}' has no photon windows.")

    # ------------------------------------------------------------------
    # Window choice: lexicographically maximize (#unique targets, #frames)
    # ------------------------------------------------------------------
    best_window = max(
        windows,
        key=lambda w: (w.get("n_unique_targets", 0), w.get("n_frames", 0)),
    )

    logger.info(
        "Selected photon window %d for observer '%s': %d unique targets, %d frames.",
        best_window.get("window_index", -1),
        obs_name,
        best_window.get("n_unique_targets", 0),
        best_window.get("n_frames", 0),
    )

    return obs_name, obs_data, best_window


# ---------------------------------------------------------------------------
# Step 2: Build per-target photon time series for a given window
# ---------------------------------------------------------------------------

def build_target_photon_timeseries_for_window(
    obs_name: str,
    obs_data: Dict[str, Any],
    window: Dict[str, Any],
    logger: logging.Logger | None = None,
) -> Dict[str, Any]:
    """
    Rearrange per-frame photon sources into per-target photon time series
    for a single observer + window.

    Inputs
    ------
    This function expects the photon-frame structure produced by
    NEBULA_PHOTON_FRAME_BUILDER.build_frames_by_observer_and_window_photon(),
    specifically for one observer:

        obs_data = frames_by_observer[obs_name]
        window   = one entry from obs_data["windows"]

    Each frame in window["frames"] is expected to contain:

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
        }

    Outputs
    -------
    The returned dictionary has the following structure:

        {
          "observer_name": str,
          "window_index":  int,
          "n_frames":      int,
          "rows":          int,
          "cols":          int,
          "targets": {
            source_id: {
              "source_id":        str,
              "source_type":      str,
              "coarse_indices":   np.ndarray[int],
              "t_utc":            np.ndarray[datetime],
              "t_exp_s":          np.ndarray[float],
              "x_pix":            np.ndarray[float],
              "y_pix":            np.ndarray[float],
              "phi_ph_m2_s":      np.ndarray[float],
              "flux_ph_m2_frame": np.ndarray[float],
              "app_mag_g":        np.ndarray[float],
              "range_km":         np.ndarray[float],
            },
            ...
          },
        }

    This is intended to be the main hand-off structure for downstream
    sensor / circuit simulators that operate in the photon domain.

    Parameters
    ----------
    obs_name : str
        Name of the observer whose window is being processed.
    obs_data : dict
        Observer photon-frame dictionary from
        NEBULA_PHOTON_FRAME_BUILDER.build_frames_by_observer_and_window_photon().
        Used for detector geometry and metadata.
    window : dict
        Single window dictionary for this observer (one element from
        obs_data["windows"]).
    logger : logging.Logger or None, optional
        Optional logger instance. If None, a default console logger is
        created via get_logger().

    Returns
    -------
    photon_catalog : dict
        Photon-centric catalog as described above, containing one entry
        per unique target in the window.
    """
    if logger is None:
        logger = get_logger()

    frames = window.get("frames", [])
    if not frames:
        raise RuntimeError(
            f"Observer '{obs_name}' window {window.get('window_index', -1)} "
            "has no frames in build_target_photon_timeseries_for_window()."
        )

    logger.debug(
        "Building per-target photon time series for observer '%s', window %s "
        "(%d frames).",
        obs_name,
        window.get("window_index", -1),
        len(frames),
    )

    # Accumulator: source_id -> dict of lists
    targets: Dict[str, Dict[str, List[Any]]] = {}

    for frame in frames:
        coarse_idx = int(frame.get("coarse_index", -1))
        t_utc = frame.get("t_utc", None)
        t_exp_s = float(frame.get("t_exp_s", np.nan))

        for src in frame.get("sources", []):
            source_id = str(src.get("source_id", "UNKNOWN"))
            source_type = str(src.get("source_type", "target"))

            if source_id not in targets:
                # Initialize the per-target lists on first encounter
                targets[source_id] = {
                    "source_id": source_id,
                    "source_type": source_type,
                    "coarse_indices": [],
                    "t_utc": [],
                    "t_exp_s": [],
                    "x_pix": [],
                    "y_pix": [],
                    "phi_ph_m2_s": [],
                    "flux_ph_m2_frame": [],
                    "app_mag_g": [],
                    "range_km": [],
                }

            rec = targets[source_id]

            rec["coarse_indices"].append(coarse_idx)
            rec["t_utc"].append(t_utc)
            rec["t_exp_s"].append(t_exp_s)
            rec["x_pix"].append(float(src["x_pix"]))
            rec["y_pix"].append(float(src["y_pix"]))
            rec["phi_ph_m2_s"].append(float(src["phi_ph_m2_s"]))
            rec["flux_ph_m2_frame"].append(float(src["flux_ph_m2_frame"]))
            rec["app_mag_g"].append(float(src["app_mag_g"]))
            rec["range_km"].append(float(src["range_km"]))

    # Convert lists -> numpy arrays for each target
    for source_id, rec in targets.items():
        rec["coarse_indices"] = np.asarray(rec["coarse_indices"], dtype=int)
        rec["t_utc"] = np.asarray(rec["t_utc"], dtype=object)
        rec["t_exp_s"] = np.asarray(rec["t_exp_s"], dtype=float)
        rec["x_pix"] = np.asarray(rec["x_pix"], dtype=float)
        rec["y_pix"] = np.asarray(rec["y_pix"], dtype=float)
        rec["phi_ph_m2_s"] = np.asarray(rec["phi_ph_m2_s"], dtype=float)
        rec["flux_ph_m2_frame"] = np.asarray(rec["flux_ph_m2_frame"], dtype=float)
        rec["app_mag_g"] = np.asarray(rec["app_mag_g"], dtype=float)
        rec["range_km"] = np.asarray(rec["range_km"], dtype=float)

    n_targets = len(targets)

    logger.debug(
        "Completed per-target photon time series: %d unique targets "
        "for observer '%s', window %s.",
        n_targets,
        obs_name,
        window.get("window_index", -1),
    )

    photon_catalog: Dict[str, Any] = {
        "observer_name": obs_name,
        "window_index": int(window.get("window_index", -1)),
        "n_frames": len(frames),
        "rows": int(obs_data.get("rows", 0)),
        "cols": int(obs_data.get("cols", 0)),
        "targets": targets,
    }

    return photon_catalog

# ---------------------------------------------------------------------------
# Step 3: Build observer target frames for all observers
# ---------------------------------------------------------------------------

def build_obs_target_frames_for_all_observers(
    max_frames_per_window: int | None = None,
    logger: logging.Logger | None = None,
) -> Dict[str, Any]:
    """
    Build per-observer, per-window target photon time series for *all* observers.

    This is a thin wrapper around NEBULA_PHOTON_FRAME_BUILDER plus the existing
    `build_target_photon_timeseries_for_window()` logic.

    Parameters
    ----------
    max_frames_per_window : int or None
        If not None, NEBULA_PHOTON_FRAME_BUILDER is allowed to truncate very long
        windows to at most this many frames. If None, all frames are kept.
    logger : logging.Logger or None
        Optional logger. If None, a default logger is created.

    Returns
    -------
    obs_target_frames : dict
        Dictionary keyed by observer name. For each observer:

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
                        "targets": {...}  # exactly as returned by
                                          # build_target_photon_timeseries_for_window()
                    },
                    ...
                ],
            }

        Note
        ----
        Windows with zero targets are still included here with "n_targets" == 0.
        A later helper will cull and sort windows per observer.
    """
    if logger is None:
        logger = get_logger()

    logger.info(
        "Building photon frames for all observers via NEBULA_PHOTON_FRAME_BUILDER."
    )

    # Ask the photon frame builder for the per-observer, per-window frame catalogs
    frames_by_observer: Dict[str, Any] = (
        PFB.build_frames_by_observer_and_window_photon(
            max_frames_per_window=max_frames_per_window,
            logger=logger,
        )
    )

    obs_target_frames: Dict[str, Any] = {}

    # Total number of windows across all observers, for the global progress bar
    total_windows = sum(
        len(obs_data.get("windows", [])) for obs_data in frames_by_observer.values()
    )

    # Loop over each observer in the frames_by_observer structure,
    # wrapped in a single global tqdm bar over all windows.
    with tqdm(
        total=total_windows,
        desc="Target photon time series",
        unit="window",
    ) as pbar:

        for obs_name, obs_data in frames_by_observer.items():
            # Basic sensor metadata for this observer
            rows = int(obs_data.get("rows", 0))
            cols = int(obs_data.get("cols", 0))
            dt_frame_s = float(obs_data.get("dt_frame_s", float("nan")))

            windows = obs_data.get("windows", [])
            n_windows = len(windows)

            logger.info(
                "Observer '%s': %d windows found (rows=%d, cols=%d, dt_frame_s=%.3f).",
                obs_name,
                n_windows,
                rows,
                cols,
                dt_frame_s,
            )

            obs_entry: Dict[str, Any] = {
                "observer_name": obs_name,
                "rows": rows,
                "cols": cols,
                "dt_frame_s": dt_frame_s,
                "windows": [],
            }

            # Per-window loop: convert frames into per-target photon time series
            for window in windows:
                photon_catalog = build_target_photon_timeseries_for_window(
                    obs_name=obs_name,
                    obs_data=obs_data,
                    window=window,
                    logger=logger,
                )

                n_targets = len(photon_catalog.get("targets", {}))

                window_record: Dict[str, Any] = {
                    "window_index": int(
                        photon_catalog.get(
                            "window_index", window.get("window_index", -1)
                        )
                    ),
                    "start_index": int(window.get("start_index", -1)),
                    "end_index": int(window.get("end_index", -1)),
                    "start_time": window.get("start_time"),
                    "end_time": window.get("end_time"),
                    "n_frames": int(
                        photon_catalog.get(
                            "n_frames", len(window.get("frames", []))
                        )
                    ),
                    "n_targets": n_targets,
                    "targets": photon_catalog.get("targets", {}),
                }

                obs_entry["windows"].append(window_record)

                # Advance the global progress bar by one window
                pbar.update(1)

            obs_target_frames[obs_name] = obs_entry

        obs_target_frames[obs_name] = obs_entry

        logger.info(
            "Observer '%s': built target photon catalogs for %d windows "
            "(%d total targets across all windows, including zeros).",
            obs_name,
            len(obs_entry["windows"]),
            sum(w["n_targets"] for w in obs_entry["windows"]),
        )

    logger.info(
        "Finished building obs_target_frames for %d observers with %d total windows.",
        len(obs_target_frames),
        total_windows,
    )

    return obs_target_frames

# ---------------------------------------------------------------------------
# Step 4: Cull windows with no targets
# ---------------------------------------------------------------------------

def cull_and_rank_obs_target_frames(
    obs_target_frames: Dict[str, Any],
    logger: logging.Logger | None = None,
) -> Dict[str, Any]:
    """
    For each observer:

    * Drop windows with zero targets (n_targets == 0).
    * Sort remaining windows in descending order of (n_targets, n_frames).

    This produces a new dictionary with the same structure as `obs_target_frames`
    but with windows culled and ordered from "most targets" to "fewest" per observer.

    Parameters
    ----------
    obs_target_frames : dict
        Output of `build_obs_target_frames_for_all_observers()`.
    logger : logging.Logger or None
        Optional logger. If None, a default logger is created.

    Returns
    -------
    ranked : dict
        Dictionary keyed by observer name with the same metadata, but with
        `windows` filtered and sorted.
    """
    if logger is None:
        logger = get_logger()

    ranked: Dict[str, Any] = {}

    total_before = 0
    total_after = 0

    # Loop over all observers
    for obs_name, obs_entry in obs_target_frames.items():
        windows = list(obs_entry.get("windows", []))
        total_before += len(windows)

        # Keep only windows that actually have at least one target
        non_empty = [w for w in windows if w.get("n_targets", 0) > 0]
        total_after += len(non_empty)

        # Sort by descending (n_targets, n_frames) so "richest" windows come first
        non_empty.sort(
            key=lambda w: (w.get("n_targets", 0), w.get("n_frames", 0)),
            reverse=True,
        )

        new_entry = dict(obs_entry)
        new_entry["windows"] = non_empty
        ranked[obs_name] = new_entry

        if non_empty:
            logger.info(
                "Observer '%s': %d -> %d windows after culling; "
                "top window %d has %d targets over %d frames.",
                obs_name,
                len(windows),
                len(non_empty),
                non_empty[0].get("window_index", -1),
                non_empty[0].get("n_targets", 0),
                non_empty[0].get("n_frames", 0),
            )
        else:
            logger.warning(
                "Observer '%s': all %d windows had zero targets; nothing remains "
                "after culling.",
                obs_name,
                len(windows),
            )

    logger.info(
        "Culled %d windows with zero targets across all observers; %d windows remain.",
        total_before - total_after,
        total_after,
    )

    return ranked

# ---------------------------------------------------------------------------
# Step 5: Save Frames as pickles
# ---------------------------------------------------------------------------
def save_obs_target_frames_pickle(
    obs_target_frames: Dict[str, Any],
    filename: str,
    logger: logging.Logger | None = None,
) -> str:
    """
    Save an `obs_target_frames`-style dictionary to a pickle under NEBULA_OUTPUT_DIR.

    Parameters
    ----------
    obs_target_frames : dict
        Dictionary keyed by observer name.
    filename : str
        Filename for the pickle, e.g. 'obs_target_frames_raw.pkl'.
    logger : logging.Logger or None
        Optional logger. If None, a default logger is created.

    Returns
    -------
    out_path : str
        Full path to the written pickle file.
    """
    if logger is None:
        logger = get_logger()

    # Place these under a dedicated subdirectory beneath NEBULA_OUTPUT_DIR
    out_dir = os.path.join(NEBULA_OUTPUT_DIR, "TARGET_PHOTON_FRAMES")
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, filename)

    with open(out_path, "wb") as f:
        pickle.dump(obs_target_frames, f)

    logger.info(
        "Saved obs_target_frames dictionary with %d observers to '%s'.",
        len(obs_target_frames),
        out_path,
    )

    return out_path

# ----------------------------------------------------------------------
# Entry point: build, save, and summarize obs_target_frames
# ----------------------------------------------------------------------
def main() -> None:
    """
    Driver for NEBULA_TARGET_PHOTONS when run as a script.

    Steps:
        1. Use NEBULA_PHOTON_FRAME_BUILDER to build photon frames for
           *all* observers and *all* windows.
        2. Convert those frames into per-target photon time series per
           observer & window (obs_target_frames).
        3. Save a 'raw' pickle containing all windows, including those
           with zero targets.
        4. Cull & rank windows per observer and save a second pickle
           where windows are sorted by number of targets and empty
           windows are removed.
    """
     
    logger = get_logger()

    # 1. Build per-observer, per-window photon time series dictionaries
    obs_target_frames = build_obs_target_frames_for_all_observers(
        max_frames_per_window=None,
        logger=logger,
    )

    # 2. Save the raw version (all windows, including zero-target ones)
    raw_path = save_obs_target_frames_pickle(
        obs_target_frames,
        filename="obs_target_frames_raw.pkl",
        logger=logger,
    )

    # 3. Cull empty windows and rank by #targets, then save that version
    ranked_frames = cull_and_rank_obs_target_frames(
        obs_target_frames,
        logger=logger,
    )

    ranked_path = save_obs_target_frames_pickle(
        ranked_frames,
        filename="obs_target_frames_ranked.pkl",
        logger=logger,
    )

    # 4. Log a compact summary so you can sanity-check in the console
    for obs_name, obs_entry in ranked_frames.items():
        windows = obs_entry.get("windows", [])
        logger.info(
            "Observer '%s': %d non-empty windows after culling.",
            obs_name,
            len(windows),
        )

        # Show a quick view of the richest few windows
        for w in windows[:3]:
            logger.info(
                "  Window %d: n_frames=%d, n_targets=%d, coarse_index=[%d..%d]",
                w.get("window_index", -1),
                w.get("n_frames", -1),
                w.get("n_targets", 0),
                w.get("start_index", -1),
                w.get("end_index", -1),
            )

    logger.info(
        "NEBULA_TARGET_PHOTONS complete. Raw windows: '%s'. "
        "Culled/ranked windows: '%s'.",
        raw_path,
        ranked_path,
    )


if __name__ == "__main__":
    main()

