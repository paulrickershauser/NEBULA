"""
NEBULA_LOS_FLUX_PICKLER
=======================

Post-processing helper that takes the LOS + illumination + flux pickles produced
by NEBULA_FLUX_PICKLER and attaches *LOS-gated* radiometric arrays to each
target track.

Motivation
----------
NEBULA_FLUX computes the Gaia G-band flux, photon flux, and apparent magnitude
for every timestep in the track, assuming an unobstructed line between the
observer and the target (i.e., it does not consider Earth occultation).

The LOS pipeline (Cinelli + geometric fallback + poliastro checks) independently
adds a 'los_visible' array (0/1 or bool) that encodes whether the target is
geometrically visible from the observer at each timestep.

This module combines those two pieces of information by constructing new arrays
that represent the radiometry *only when there is actual line-of-sight*:

    - rad_flux_g_w_m2_los_only
    - rad_photon_flux_g_m2_s_los_only
    - rad_app_mag_g_los_only

Outside of LOS (los_visible == 0):
    - flux arrays are set to 0.0
    - magnitudes are set to +inf (no measurable brightness)

The original flux arrays from NEBULA_FLUX are left untouched, so you can still
inspect "intrinsic" brightness versus "actually observable" brightness.
"""

import os
import pickle
import logging
from typing import Any, Dict, Tuple

import numpy as np
# Optional progress bar
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

# ---------------------------------------------------------------------------
# NEBULA configuration: output directory
# ---------------------------------------------------------------------------

from Configuration.NEBULA_PATH_CONFIG import NEBULA_OUTPUT_DIR
from Utility.SAT_OBJECTS import NEBULA_FLUX_PICKLER

# Directory where LOS + illumination + flux pickles live
# (must match NEBULA_FLUX_PICKLER)
FLUX_TRACKS_DIR = os.path.join(NEBULA_OUTPUT_DIR, "FLUX_SatPickles")

# Base pickles produced by NEBULA_FLUX_PICKLER
OBS_LOS_ILL_FLUX_PICKLE = os.path.join(
    FLUX_TRACKS_DIR, "observer_tracks_with_los_illum_flux.pkl"
)
TAR_LOS_ILL_FLUX_PICKLE = os.path.join(
    FLUX_TRACKS_DIR, "target_tracks_with_los_illum_flux.pkl"
)

# Directory where LOS-gated flux pickles will be stored
LOS_FLUX_TRACKS_DIR = os.path.join(NEBULA_OUTPUT_DIR, "LOSFLUX_SatPickles")

# New pickles that will contain LOS-gated flux arrays
OBS_LOS_ILL_FLUX_LOS_PICKLE = os.path.join(
    LOS_FLUX_TRACKS_DIR, "observer_tracks_with_los_illum_flux_los.pkl"
)
TAR_LOS_ILL_FLUX_LOS_PICKLE = os.path.join(
    LOS_FLUX_TRACKS_DIR, "target_tracks_with_los_illum_flux_los.pkl"
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
            "Creating NEBULA_OUTPUT_DIR at '%s' for LOS-gated flux pickles.",
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

def attach_los_flux_to_all_targets(
    force_recompute: bool = False,
    logger: logging.Logger | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Attach LOS-gated flux arrays to all targets and write new pickles.

    This function expects that NEBULA_FLUX_PICKLER has already been run, so that
    the following base files exist:

        - observer_tracks_with_los_illum_flux.pkl
        - target_tracks_with_los_illum_flux.pkl

    Each target track is expected to have a per-observer container::

        target['by_observer'][obs_name]

    where, for each observer name ``obs_name``, the entry includes:

        - 'los_visible'              (LOS flag per timestep: 0/1 or bool)
        - 'rad_flux_g_w_m2'          (Gaia G-band irradiance at observer [W m^-2])
        - 'rad_photon_flux_g_m2_s'   (G-band photon flux at observer [photons m^-2 s^-1])
        - 'rad_app_mag_g'            (G-band apparent magnitude as seen by observer)

    This function loops over **all observers** and constructs LOS-gated
    versions of those arrays, stored back under the same per-observer entries:

        - 'rad_flux_g_w_m2_los_only'
        - 'rad_photon_flux_g_m2_s_los_only'
        - 'rad_app_mag_g_los_only'


    With the following behavior:

        - Where los_visible == 1 (True):
            the LOS-gated arrays equal the original flux / mag values.

        - Where los_visible == 0 (False):
            flux arrays are set to 0.0,
            magnitude array is set to +inf (no measurable brightness).

    Parameters
    ----------
    force_recompute : bool, optional
        If False (default) and LOS-gated flux pickles already exist on disk,
        this function will simply reload and return them.

        If True, the function will ignore any existing LOS-gated flux pickles,
        recompute the LOS-gated arrays from the base flux pickles, overwrite
        the LOS-gated pickles on disk, and then return the updated tracks.
    logger : logging.Logger or None, optional
        Logger to use. If None, a default console logger is created.

    Returns
    -------
    observer_tracks : dict
        Dictionary of observer track objects (usually unchanged by this step).
    target_tracks : dict
        Dictionary of target track objects, augmented with the LOS-gated
        flux and magnitude arrays.
    """
    # If caller did not provide a logger, build a default one.
    if logger is None:
        logger = _build_default_logger()

    # Make sure NEBULA_OUTPUT_DIR exists for reading/writing pickles.
    _ensure_output_dir_exists(logger)

    # Ensure the LOS-flux subdirectory exists, e.g. NEBULA_OUTPUT/LOSFLUX_SatPickles.
    if not os.path.isdir(LOS_FLUX_TRACKS_DIR):
        logger.info(
            "Creating LOS-flux output directory '%s'.",
            LOS_FLUX_TRACKS_DIR,
        )
        os.makedirs(LOS_FLUX_TRACKS_DIR, exist_ok=True)

    # Fast path: reuse existing LOS-gated flux pickles if they exist and the
    # user did not request recomputation.
    if (
        not force_recompute
        and os.path.exists(OBS_LOS_ILL_FLUX_LOS_PICKLE)
        and os.path.exists(TAR_LOS_ILL_FLUX_LOS_PICKLE)
    ):
        logger.info(
            "Reusing existing LOS-gated flux pickles (force_recompute=False)."
        )
        obs_tracks = _load_pickle(OBS_LOS_ILL_FLUX_LOS_PICKLE, logger)
        tar_tracks = _load_pickle(TAR_LOS_ILL_FLUX_LOS_PICKLE, logger)
        return obs_tracks, tar_tracks

    # Slow path: ensure base LOS+illum+flux pickles exist, then recompute
    # LOS-gated flux arrays from them.

    if not (
        os.path.exists(OBS_LOS_ILL_FLUX_PICKLE)
        and os.path.exists(TAR_LOS_ILL_FLUX_PICKLE)
    ):
        logger.info(
            "Base LOS+illum+flux pickles are missing; calling "
            "NEBULA_FLUX_PICKLER.attach_flux_to_all_targets() to build them "
            "(force_recompute=%s).",
            force_recompute,
        )

        # This call will internally:
        #   - ensure LOS pickles exist,
        #   - attach Skyfield illumination,
        #   - compute Lambertian flux for all targets,
        #   - and write the *_with_los_illum_flux.pkl pickles.
        # We do not need the returned dicts here; we just rely on the pickles.
        NEBULA_FLUX_PICKLER.attach_flux_to_all_targets(
            force_recompute=force_recompute,
            logger=logger,
        )

    logger.info(
        "Recomputing LOS-gated flux arrays from base flux pickles "
        "(force_recompute=%s or no LOS-gated pickles found).",
        force_recompute,
    )
    observer_tracks = _load_pickle(OBS_LOS_ILL_FLUX_PICKLE, logger)
    target_tracks = _load_pickle(TAR_LOS_ILL_FLUX_PICKLE, logger)


    # For now, observer tracks typically do not carry flux arrays; we simply
    # pass them through unchanged. If you later attach sensor-background
    # models to the observer, you can extend this logic.
    obs_tracks_out: Dict[str, Any] = observer_tracks

    # We will gate flux in-place on the target_tracks dict and then write it out.
    tar_tracks_out: Dict[str, Any] = target_tracks

    observer_names = list(observer_tracks.keys())
    n_targets = len(target_tracks)

    logger.info(
        "Computing LOS-gated flux for %d observers and %d targets.",
        len(observer_names),
        n_targets,
    )

    for obs_name in observer_names:
        n_processed = 0
        n_skipped_type = 0
        n_skipped_no_entry = 0
        n_skipped_no_los = 0
        n_skipped_no_flux = 0

        # Progress bar over targets for this observer
        if tqdm is not None:
            target_iter = tqdm(
                target_tracks.items(),
                total=n_targets,
                desc=f"LOSFLUX: {obs_name}",
                unit="target",
            )
        else:
            target_iter = target_tracks.items()

        for name, track in target_iter:
            # We only know how to handle dict-like tracks here.
            if not isinstance(track, dict):
                n_skipped_type += 1
                continue

            by_obs = track.get("by_observer")
            if not by_obs or obs_name not in by_obs:
                # No per-observer entry for this target/observer pair
                n_skipped_no_entry += 1
                continue

            obs_entry = by_obs[obs_name]

            # LOS mask for this observer–target pair
            los = obs_entry.get("los_visible", None)
            if los is None:
                n_skipped_no_los += 1
                continue

            # Base radiometric arrays for this observer–target pair
            flux_g = obs_entry.get("rad_flux_g_w_m2", None)
            photon_flux_g = obs_entry.get("rad_photon_flux_g_m2_s", None)
            mag_g = obs_entry.get("rad_app_mag_g", None)

            if flux_g is None or photon_flux_g is None or mag_g is None:
                n_skipped_no_flux += 1
                continue

            los_mask = np.asarray(los, dtype=bool)
            flux_g = np.asarray(flux_g, dtype=float)
            photon_flux_g = np.asarray(photon_flux_g, dtype=float)
            mag_g = np.asarray(mag_g, dtype=float)

            # Defensive: align lengths if needed (quietly truncate to min length).
            n = min(len(los_mask), len(flux_g), len(photon_flux_g), len(mag_g))
            if not (
                len(los_mask) == len(flux_g) == len(photon_flux_g) == len(mag_g)
            ):
                los_mask = los_mask[:n]
                flux_g = flux_g[:n]
                photon_flux_g = photon_flux_g[:n]
                mag_g = mag_g[:n]

            # Construct LOS-gated arrays:
            #   - flux, photon flux → zeroed outside LOS
            #   - magnitude → +inf outside LOS
            los_mask_float = los_mask.astype(float)

            flux_g_los = flux_g * los_mask_float
            photon_flux_g_los = photon_flux_g * los_mask_float
            mag_g_los = np.where(los_mask, mag_g, np.inf)

            # Attach the new arrays to the *per-observer* entry.
            # No observer-relative arrays are written at the top level.
            obs_entry["rad_flux_g_w_m2_los_only"] = flux_g_los
            obs_entry["rad_photon_flux_g_m2_s_los_only"] = photon_flux_g_los
            obs_entry["rad_app_mag_g_los_only"] = mag_g_los

            n_processed += 1

        # Compact per-observer summary; no per-target wall-of-text logs.
        logger.info(
            "Observer '%s': LOS-gated flux computed for %d/%d targets "
            "(skipped: type=%d, no by_observer entry=%d, missing LOS=%d, "
            "missing flux=%d).",
            obs_name,
            n_processed,
            n_targets,
            n_skipped_type,
            n_skipped_no_entry,
            n_skipped_no_los,
            n_skipped_no_flux,
        )

    # Write out the augmented observer and target tracks.
    _save_pickle(obs_tracks_out, OBS_LOS_ILL_FLUX_LOS_PICKLE, logger)
    _save_pickle(tar_tracks_out, TAR_LOS_ILL_FLUX_LOS_PICKLE, logger)

    logger.info(
        "NEBULA_LOS_FLUX_PICKLER: completed LOS-gated flux attachment for "
        "%d targets and %d observers.",
        len(tar_tracks_out),
        len(observer_names),
    )

    return obs_tracks_out, tar_tracks_out



# ---------------------------------------------------------------------------
# CLI / dev entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Default behavior: only recompute if LOS-gated flux pickles do not exist.
    attach_los_flux_to_all_targets(force_recompute=False)
