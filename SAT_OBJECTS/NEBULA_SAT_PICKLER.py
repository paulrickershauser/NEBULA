# NEBULA_SAT_PICKLER.py
# ---------------------------------------------------------------------------
# Build NEBULA SatelliteTrack objects from TLEs and persist them as pickles
# ---------------------------------------------------------------------------
"""
NEBULA_SAT_PICKLER

Purpose
-------
Utility script/module that builds rich satellite objects (SatelliteTrack)
for all configured observers and targets, then saves them to pickle files
for fast reuse in other NEBULA components.

Compared to NEBULA_MINI_MAIN, this script is focused on:

    1. Running the end-to-end pipeline to create SatelliteTrack dictionaries
       for observers and targets.
    2. Persisting those dictionaries under NEBULA_OUTPUT/track_debug/.
    3. Providing a helper function to load the pickled tracks back into
       the workspace (e.g., in Spyder) for quick interactive use.

Requirements
------------
- NEBULA configuration modules:

    Configuration/
        NEBULA_PATH_CONFIG.py
        NEBULA_TIME_CONFIG.py

- NEBULA utility modules:

    Utility/
        NEBULA_TIME_PARSER.py
        NEBULA_TIMESTEP_PLANNER.py
        NEBULA_TLE_READER.py
        NEBULA_PROPAGATOR.py
        NEBULA_COE.py
        NEBULA_SATOBJ_CREATION.py

- The sgp4 library installed (for Satrec propagation).
#If you want to force recompute pickles change line 205
"""

# --- Standard library imports ------------------------------------------------
from pathlib import Path
from typing import Dict, Tuple

import logging
import pickle
import inspect

# --- Third-party imports -----------------------------------------------------
from sgp4.api import Satrec

# --- NEBULA configuration imports -------------------------------------------
# Paths: root, TLE locations, output, logging.
from Configuration.NEBULA_PATH_CONFIG import (
    OBS_TLE_FILE,
    TAR_TLE_FILE,
    ensure_output_directory,
    configure_logging,
)

# Time window + base propagation step settings.
from Configuration.NEBULA_TIME_CONFIG import (
    DEFAULT_TIME_WINDOW,
    DEFAULT_PROPAGATION_STEPS,
)

# --- NEBULA utility imports --------------------------------------------------
# Time parsing and time-grid construction.
from Utility.PROPAGATION.NEBULA_TIME_PARSER import parse_time_window_config
from Utility.PROPAGATION.NEBULA_TIMESTEP_PLANNER import build_time_grid, summarize_time_grid

# TLE → Satrec map.
from Utility.PROPAGATION.NEBULA_TLE_READER import read_tle_file

# Satrec + time grid → SatelliteTrack objects.
from Utility.SAT_OBJECTS.NEBULA_SATOBJ_CREATION import (
    SatelliteTrack,
    build_satellite_tracks,
)


# ---------------------------------------------------------------------------
# Helper: summarize one SatelliteTrack in a couple of lines
# ---------------------------------------------------------------------------
def summarize_track(name: str, track: SatelliteTrack) -> str:
    """Build a small human-readable summary string for a SatelliteTrack."""
    # Number of time samples.
    n_samples = len(track.times)

    if n_samples == 0:
        return f"{name}: (no samples)"

    # First and last timestamps.
    t0 = track.times[0]
    t1 = track.times[-1]

    # Norms of the first and last position vectors (to get a feel for orbit).
    import numpy as np  # local import to keep the top clean

    r0_norm = float(np.linalg.norm(track.r_eci_km[0, :]))
    r1_norm = float(np.linalg.norm(track.r_eci_km[-1, :]))

    # First and last true anomalies (in degrees for readability).
    nu0_deg = float(track.nu_rad[0] * 180.0 / np.pi)
    nu1_deg = float(track.nu_rad[-1] * 180.0 / np.pi)

    lines = [
        f"Satellite '{name}':",
        f"  samples:   {n_samples}",
        f"  t[0]:      {t0.isoformat()}",
        f"  t[-1]:     {t1.isoformat()}",
        f"  |r(0)|:    {r0_norm:9.3f} km",
        f"  |r(end)|:  {r1_norm:9.3f} km",
        f"  ν(0):      {nu0_deg:9.3f} deg",
        f"  ν(end):    {nu1_deg:9.3f} deg",
        f"  a:         {track.a_km:9.3f} km",
        f"  e:         {track.e:9.6f}",
        f"  i:         {track.inc_rad * 180.0 / np.pi:9.3f} deg",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers for robust (de)serialization
# ---------------------------------------------------------------------------
def _track_to_dict(track: SatelliteTrack) -> dict:
    """Convert a SatelliteTrack instance to a plain-JSON/dict structure.

    This avoids pickling the class object itself (which can break under
    Spyder's module reloader) and instead stores only primitive/numpy data
    which pickle can always handle.

    We also strip out any observer-relative fields (LOS / illumination)
    that are meant to live only under the per-observer container
    ``track.by_observer[obs_name]``.
    """
    # First build a raw dict from the track object.
    if hasattr(track, "_asdict") and callable(getattr(track, "_asdict")):
        data = dict(track._asdict())
    elif hasattr(track, "__dict__"):
        data = dict(track.__dict__)
    else:
        raise TypeError(
            f"SatelliteTrack of type {type(track)!r} is not serializable by _track_to_dict"
        )

    # Drop per-observer-only fields; these are stored under by_observer.
    per_observer_only = {
        "los_visible",
        "los_h",
        "los_regime",
        "los_fallback",
        "illum_is_sunlit",
        "illum_phase_angle_rad",
        "illum_fraction_illuminated",
    }
    for key in per_observer_only:
        data.pop(key, None)

    return data



def _dict_to_track(data: dict) -> SatelliteTrack:
    """Rebuild a SatelliteTrack from a dict of fields.

    We filter the dict so that only keys that appear in the SatelliteTrack
    __init__ signature are passed, to be robust to any extra cached attributes.
    """
    sig = inspect.signature(SatelliteTrack)
    allowed = sig.parameters.keys()
    filtered = {k: v for k, v in data.items() if k in allowed}
    return SatelliteTrack(**filtered)


# ---------------------------------------------------------------------------
# Helper: load pickled tracks (logic from pickle_test, but generalized)
# ---------------------------------------------------------------------------
def load_pickled_tracks(
    tracks_dir: Path,
) -> Tuple[Dict[str, SatelliteTrack], Dict[str, SatelliteTrack]]:
    """Load observer and target SatelliteTrack dictionaries from pickle files.

    This now expects the pickle files to contain *plain dicts* for each track,
    which are converted back into SatelliteTrack instances on load.
    """
    obs_path = tracks_dir / "observer_tracks.pkl"
    tar_path = tracks_dir / "target_tracks.pkl"

    if not obs_path.exists():
        raise FileNotFoundError(f"Observer tracks pickle not found: {obs_path}")
    if not tar_path.exists():
        raise FileNotFoundError(f"Target tracks pickle not found: {tar_path}")

    with obs_path.open("rb") as f_obs:
        observer_data = pickle.load(f_obs)

    with tar_path.open("rb") as f_tar:
        target_data = pickle.load(f_tar)

    # Rebuild SatelliteTrack objects using the *current* class definition
    # from Utility.NEBULA_SATOBJ_CREATION, avoiding the Spyder reload trap.
    observer_tracks = {name: _dict_to_track(d) for name, d in observer_data.items()}
    target_tracks = {name: _dict_to_track(d) for name, d in target_data.items()}

    logging.info(
        "Deserialized %d observer track(s) from %s",
        len(observer_tracks),
        obs_path,
    )
    logging.info(
        "Deserialized %d target track(s) from %s",
        len(target_tracks),
        tar_path,
    )

    return observer_tracks, target_tracks

def sat_object_pickler(
    force_recompute: bool = False,  #change this if you want to recompute 
) -> Tuple[Dict[str, SatelliteTrack], Dict[str, SatelliteTrack]]:
    """
    Convenience wrapper to obtain observer and target SatelliteTrack
    dictionaries, with optional caching.

    If base-track pickles already exist and `force_recompute` is False,
    this function simply loads them from disk using
    :func:`load_pickled_tracks`.

    If `force_recompute` is True, or the base-track pickles are missing,
    it calls :func:`pickle_sats` to rebuild the tracks from the TLE files.

    Parameters
    ----------
    force_recompute : bool, optional
        If False (default) and ``observer_tracks.pkl`` /
        ``target_tracks.pkl`` exist in the NEBULA_OUTPUT/track_debug
        directory, those pickles are loaded and returned.
        If True, base tracks are recomputed from the TLEs via
        :func:`pickle_sats`, and the pickles are overwritten.

    Returns
    -------
    observer_tracks : Dict[str, SatelliteTrack]
        Dictionary mapping observer name → SatelliteTrack instance.

    target_tracks : Dict[str, SatelliteTrack]
        Dictionary mapping target name → SatelliteTrack instance.
    """
    # Determine where the base-track pickles should live.
    out_dir: Path = ensure_output_directory()
    tracks_dir: Path = out_dir / "BASE_SatPickles"
    tracks_dir.mkdir(parents=True, exist_ok=True)

    obs_path = tracks_dir / "observer_tracks.pkl"
    tar_path = tracks_dir / "target_tracks.pkl"

    # If we're allowed to reuse and both pickles exist, just load them.
    if not force_recompute and obs_path.exists() and tar_path.exists():
        logging.info(
            "NEBULA_SAT_PICKLER.sat_object_pickler: reusing existing base "
            "track pickles from %s and %s (force_recompute=%s).",
            obs_path,
            tar_path,
            force_recompute,
        )
        return load_pickled_tracks(tracks_dir)

    # Otherwise, (re)run the full base-track pipeline.
    logging.info(
        "NEBULA_SAT_PICKLER.sat_object_pickler: building base tracks from "
        "TLEs (force_recompute=%s).",
        force_recompute,
    )
    return pickle_sats()



# ---------------------------------------------------------------------------
# Main: build and pickle SatelliteTrack dictionaries
# ---------------------------------------------------------------------------
def pickle_sats() -> Tuple[Dict[str, SatelliteTrack], Dict[str, SatelliteTrack]]:
    """Run the NEBULA_SAT_PICKLER pipeline.

    This will:

    1. Ensure the output directory exists and configure logging.
    2. Parse the default time window and build a time grid.
    3. Read observer and target TLE files.
    4. Build SatelliteTrack objects for all observers and targets.
    5. Save the resulting dictionaries as pickle files under
       ``NEBULA_OUTPUT/track_debug/`` (as *plain dict* representations).
    6. Reload the pickles (using :func:`load_pickled_tracks`) and return the
       deserialized SatelliteTrack dictionaries.
    """
    # 1) Ensure the output directory exists and set up logging.
    out_dir: Path = ensure_output_directory()
    log_path: Path = configure_logging(level=logging.INFO)

    logging.info("NEBULA_SAT_PICKLER: output dir  = %s", out_dir)
    logging.info("NEBULA_SAT_PICKLER: log file    = %s", log_path)

    # 2) Parse the default time window and build a time grid.
    start_dt, end_dt = parse_time_window_config(DEFAULT_TIME_WINDOW)
    logging.info(
        "Time window: %s  →  %s",
        start_dt.isoformat(),
        end_dt.isoformat(),
    )

    # Base timestep from configuration (no adaptation yet).
    base_dt_s = DEFAULT_PROPAGATION_STEPS.base_dt_s
    logging.info("Base propagation step: %.3f s", base_dt_s)

    # Build the time grid using the timestep planner.
    times = build_time_grid(
        DEFAULT_TIME_WINDOW,
        DEFAULT_PROPAGATION_STEPS,
    )

    # Log a compact summary of the time grid.
    logging.info(summarize_time_grid(times))

    # 3) Read observer and target TLE files into {name: Satrec} dicts.
    logging.info("Reading observer TLEs from %s", OBS_TLE_FILE)
    observer_sats: Dict[str, Satrec] = read_tle_file(OBS_TLE_FILE)

    logging.info("Reading target TLEs from %s", TAR_TLE_FILE)
    target_sats: Dict[str, Satrec] = read_tle_file(TAR_TLE_FILE)

    logging.info(
        "Loaded %d observer Satrec(s) and %d target Satrec(s)",
        len(observer_sats),
        len(target_sats),
    )

    # 4) Build SatelliteTrack objects for observers and targets.
    logging.info("Building SatelliteTrack objects for observers...")
    observer_tracks: Dict[str, SatelliteTrack] = build_satellite_tracks(
        observer_sats,
        times,
    )

    logging.info("Building SatelliteTrack objects for targets...")
    target_tracks: Dict[str, SatelliteTrack] = build_satellite_tracks(
        target_sats,
        times,
    )

    logging.info(
        "Built %d observer track(s) and %d target track(s)",
        len(observer_tracks),
        len(target_tracks),
    )

    # 5) Persist tracks to disk for later inspection / reuse.
    #    We save them as pickle files under NEBULA_OUTPUT/BASE_SatPickles/.
    tracks_dir = out_dir / "BASE_SatPickles"
    tracks_dir.mkdir(parents=True, exist_ok=True)

    obs_path = tracks_dir / "observer_tracks.pkl"
    tar_path = tracks_dir / "target_tracks.pkl"

    # Convert SatelliteTrack objects → plain dicts to avoid pickling the class.
    observer_serial = {
        name: _track_to_dict(track) for name, track in observer_tracks.items()
    }
    target_serial = {
        name: _track_to_dict(track) for name, track in target_tracks.items()
    }

    with obs_path.open("wb") as f_obs:
        pickle.dump(observer_serial, f_obs)

    with tar_path.open("wb") as f_tar:
        pickle.dump(target_serial, f_tar)

    logging.info(
        "Serialized observer tracks → %s and target tracks → %s",
        obs_path,
        tar_path,
    )

    # 6) Reload the pickles so that callers (and Spyder users) can be sure
    #    they're getting the deserialized objects.
    observer_tracks_loaded, target_tracks_loaded = load_pickled_tracks(tracks_dir)

    # 7) Print short summaries of a few tracks so we can visually inspect them.
    print("\n=== NEBULA_SAT_PICKLER: Observer tracks (first 3) ===")
    if not observer_tracks_loaded:
        print("  [none]")
    else:
        for name, track in list(observer_tracks_loaded.items())[:3]:
            print(summarize_track(name, track))
            print("-" * 60)

    print("\n=== NEBULA_SAT_PICKLER: Target tracks (first 3) ===")
    if not target_tracks_loaded:
        print("  [none]")
    else:
        for name, track in list(target_tracks_loaded.items())[:3]:
            print(summarize_track(name, track))
            print("-" * 60)

    logging.info("NEBULA_SAT_PICKLER completed successfully.")

    return observer_tracks_loaded, target_tracks_loaded


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # When run as a script (e.g., from Spyder), explicitly recompute the
    # base tracks from TLEs so you always see a fresh propagation.
    observer_tracks, target_tracks = sat_object_pickler(force_recompute=True)
