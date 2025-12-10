# -*- coding: utf-8 -*-
"""
NEBULA_FLUX_PICKLER
===================

High-level helper to attach Lambertian radiometry / flux information
to all observer–target track objects and cache the result as pickles.

Design goals
------------
- Build on top of the LOS + illumination pickles produced by
  NEBULA_SAT_ILL_PICKLER:
    * observer_tracks_with_los_illum.pkl
    * target_tracks_with_los_illum.pkl

- For each target track, call:
    NEBULA_FLUX.attach_lambertian_radiometry_to_target(...)
  which:
    * Reads illumination geometry and ranges already attached to the track.
    * Uses Lambertian-sphere radiometry in the Gaia G band
      (see NEBULA_FLUX for detailed equations & literature references).
    * Writes per-timestep fluxes and apparent magnitudes directly into
      the target track dictionary (in-place mutation).

- Cache the fully-augmented objects to:
    * observer_tracks_with_los_illum_flux.pkl
    * target_tracks_with_los_illum_flux.pkl

- Provide a simple user-facing API:
    attach_flux_to_all_targets(...)
  so that calling code (e.g., a top-level simulation driver) can do:

    from Utility.SAT_OBJECTS import NEBULA_FLUX_PICKLER
    obs_flux, tar_flux = NEBULA_FLUX_PICKLER.attach_flux_to_all_targets()

  and immediately receive observer / target dictionaries that now contain:
    * Orbital states (from propagation)
    * LOS flags (from visibility module)
    * Illumination geometry (from Skyfield)
    * Radiometric flux & magnitude time series (from NEBULA_FLUX)

Inputs
------
- Existing LOS + illumination pickles:
    observer_tracks_with_los_illum.pkl
    target_tracks_with_los_illum.pkl

  These are created by NEBULA_SAT_ILL_PICKLER.attach_illum_to_all_targets().

- Configuration:
    * NEBULA_OUTPUT_DIR (output directory root)
      and ensure_output_directory() from NEBULA_PATH_CONFIG.
    * Radiometry details (solar G-band zero-point, albedo, radius, etc.)
      are handled internally by NEBULA_FLUX and the configuration modules
      it imports (NEBULA_SAT_OPTICAL_CONFIG, NEBULA_SENSOR_CONFIG, etc.).

Outputs
-------
- Two new pickle files in NEBULA_OUTPUT_DIR:
    * observer_tracks_with_los_illum_flux.pkl
    * target_tracks_with_los_illum_flux.pkl

- In-memory Python objects returned to caller:
    * observer_tracks: Dict[str, dict]
        - Typically one observer keyed by its name (e.g., "SBSS (USA 216)").
    * target_tracks: Dict[str, dict]
        - All target GEO satellites keyed by their catalog names.

Each target track dict is augmented in-place with fields created by
NEBULA_FLUX (e.g. G-band fluxes and apparent magnitudes vs. time).

Usage patterns
--------------
1) Fast path: reuse existing flux pickles if they exist

    from Utility.SAT_OBJECTS import NEBULA_FLUX_PICKLER
    obs_flux, tar_flux = NEBULA_FLUX_PICKLER.attach_flux_to_all_targets()

2) Force recompute: ignore any existing flux pickles and recompute from
   LOS + illumination tracks (which themselves may be recomputed if
   NEBULA_SAT_ILL_PICKLER is also called with force_recompute=True):

    obs_flux, tar_flux = NEBULA_FLUX_PICKLER.attach_flux_to_all_targets(
        force_recompute=True
    )

Author
------
NEBULA radiometry pipeline, Lambertian sphere approximation for GEO targets.
"""

from __future__ import annotations

# Standard library imports
import logging  # For status / debug messages
import os       # For filesystem paths and existence checks
import pickle   # For reading / writing track pickles

from typing import Dict, Tuple, Optional  # For type annotations

import numpy as np

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

# NEBULA configuration: output directory and helper to ensure it exists
from Configuration.NEBULA_PATH_CONFIG import NEBULA_OUTPUT_DIR, ensure_output_directory

# NEBULA pickling helpers: generic load routine for pickled track dicts
from Utility.SAT_OBJECTS import NEBULA_SAT_PICKLER

# NEBULA illumination pickler: produces LOS + illumination pickles
from Utility.SAT_OBJECTS import NEBULA_SAT_ILL_PICKLER

# NEBULA radiometry core: Lambertian flux computation for one observer/target pair
from Utility.RADIOMETRY import NEBULA_FLUX

# NEBULA sensor configuration: default EVK4 event camera
from Configuration.NEBULA_SENSOR_CONFIG import ACTIVE_SENSOR


# ---------------------------------------------------------------------------
# Module-level constants: filenames for LOS + illumination + flux pickles
# ---------------------------------------------------------------------------

# Directory where LOS + illumination pickles live (produced by NEBULA_SAT_ILL_PICKLER)
ILLUM_TRACKS_DIR = os.path.join(NEBULA_OUTPUT_DIR, "ILLUM_SatPickles")

# Name of the LOS + illumination pickles (produced by NEBULA_SAT_ILL_PICKLER)
OBS_LOS_ILL_PICKLE = os.path.join(
    ILLUM_TRACKS_DIR,
    "observer_tracks_with_los_illum.pkl",
)
TAR_LOS_ILL_PICKLE = os.path.join(
    ILLUM_TRACKS_DIR,
    "target_tracks_with_los_illum.pkl",
)

# Directory where LOS + illumination + flux pickles will be stored
FLUX_TRACKS_DIR = os.path.join(NEBULA_OUTPUT_DIR, "FLUX_SatPickles")

# Name of the final LOS + illumination + flux pickles (produced by THIS module)
OBS_LOS_ILL_FLUX_PICKLE = os.path.join(
    FLUX_TRACKS_DIR,
    "observer_tracks_with_los_illum_flux.pkl",
)
TAR_LOS_ILL_FLUX_PICKLE = os.path.join(
    FLUX_TRACKS_DIR,
    "target_tracks_with_los_illum_flux.pkl",
)



# ---------------------------------------------------------------------------
# Helper: build a logger if the caller does not supply one
# ---------------------------------------------------------------------------

def _build_default_logger() -> logging.Logger:
    """
    Build a simple console logger for NEBULA_FLUX_PICKLER.

    Returns
    -------
    logging.Logger
        Logger instance configured to output INFO-level messages to stdout.
    """
    # Create or retrieve a logger with a specific name for this module
    logger = logging.getLogger("NEBULA_FLUX_PICKLER")

    # If the logger has no handlers yet, configure a basic console handler
    if not logger.handlers:
        # Set the overall logging level for this logger
        logger.setLevel(logging.INFO)

        # Create a stream handler that writes to standard output
        ch = logging.StreamHandler()

        # Set the handler's own logging level
        ch.setLevel(logging.INFO)

        # Define a simple log message format including module name and level
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Attach the formatter to the handler
        ch.setFormatter(formatter)

        # Attach the handler to the logger
        logger.addHandler(ch)

    # Return the configured logger
    return logger


# ---------------------------------------------------------------------------
# Core API: attach flux to all observer–target pairs and cache as pickles
# ---------------------------------------------------------------------------

def attach_flux_to_all_targets(
    observer_name: Optional[str] = None,
    sensor=None,
    force_recompute: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Dict[str, dict], Dict[str, dict]]:

    """
    Attach Lambertian radiometric flux / magnitude time series to all targets.

    This high-level function:
        1) Ensures the NEBULA output directory exists.
        2) If *not* force_recompute and the final flux pickles already exist,
           simply reloads them and returns immediately (fast path).
        3) Otherwise, calls NEBULA_SAT_ILL_PICKLER.attach_illum_to_all_targets()
           to obtain LOS + illumination-augmented tracks.
        4) Selects a single observer track (by name if provided; otherwise the
           first and only observer).
        5) For each target, calls NEBULA_FLUX.attach_lambertian_radiometry_to_target()
           which:
               - Reads target illumination geometry and ranges.
               - Uses a Lambertian-sphere approximation in the Gaia G band.
               - Writes radiometric fields (flux, photon rates, apparent magnitudes)
                 into the target track dictionary in-place.
        6) Writes the augmented observer / target track dictionaries back to disk
           as:
               observer_tracks_with_los_illum_flux.pkl
               target_tracks_with_los_illum_flux.pkl
        7) Reloads these new pickles for safety and returns them.

    Parameters
    ----------
    observer_name : str or None, optional
        Name (key) of the observer track to use. If None, and there is exactly
        one observer in the dictionary, that observer is used automatically.
        If there are multiple observers and observer_name is not provided,
        a ValueError is raised.

    sensor : object or None, optional
        Sensor configuration object describing the optical system and event
        camera used to detect the reflected G-band light. This is expected to
        be one of the sensor objects defined in NEBULA_SENSOR_CONFIG (for
        example EVK4_SENSOR), providing at least:
            - sensor.optical_throughput   (dimensionless, 0–1)
            - sensor.quantum_efficiency   (dimensionless, 0–1)
        If None, this function defaults to ACTIVE_SENSOR. Internally, an
        effective band-integrated efficiency factor η is computed as:
            η_eff = optical_throughput × quantum_efficiency

    force_recompute : bool, optional
        If False (default):
            - If the final flux pickles already exist, they are simply reloaded
              and returned, skipping all recomputation.
        If True:
            - LOS + illumination tracks are recomputed via
              NEBULA_SAT_ILL_PICKLER.attach_illum_to_all_targets(...),
              and then the flux is recomputed for all targets and cached.

    logger : logging.Logger or None, optional
        Logger for progress and debug messages. If None, a default logger
        specific to this module is created.

    Returns
    -------
    observer_tracks : Dict[str, dict]
        Dictionary mapping observer name -> observer track dictionary.
        The observer tracks are currently not augmented with per-target flux,
        but they are kept for completeness and future extensions.

    target_tracks : Dict[str, dict]
        Dictionary mapping target name -> target track dictionary.
        Each target track has been augmented in-place with radiometric fields
        added by NEBULA_FLUX.attach_lambertian_radiometry_to_target(), such as:
            - G-band flux at the observer aperture vs. time.
            - G-band photon rates vs. time.
            - Apparent magnitude in Gaia G band vs. time.

    Notes
    -----
    - All heavy radiometric math (Lambertian sphere, phase function, G-band
      zero-point, photon energies, etc.) is implemented in NEBULA_FLUX.
      This pickler is only responsible for orchestration and caching.

    - This function assumes that NEBULA_SAT_ILL_PICKLER and NEBULA_FLUX are
      correctly configured to read:
          * Geometric / optical properties from NEBULA_SAT_OPTICAL_CONFIG
          * Sensor properties (aperture, efficiency, etc.) from
            NEBULA_SENSOR_CONFIG, if used inside NEBULA_FLUX.
    """
    # If the caller did not provide a logger, build a default one
    if logger is None:
        logger = _build_default_logger()

    # Ensure that NEBULA_OUTPUT_DIR exists on disk.
    # This helper uses the global NEBULA_OUTPUT_DIR and returns the Path.
    output_dir = ensure_output_directory()
    logger.info("Ensured NEBULA output directory exists at: %s", output_dir)

    # Ensure the flux subdirectory exists (e.g., NEBULA_OUTPUT/FLUX_SatPickles).
    os.makedirs(FLUX_TRACKS_DIR, exist_ok=True)
    logger.info("Flux pickles will be written to: %s", FLUX_TRACKS_DIR)

    # ----------------------------------------------------------------------
    # Fast path: if flux pickles exist and we are not forcing recomputation,
    #            simply reload them and return.
    # ----------------------------------------------------------------------
    if (not force_recompute
            and os.path.exists(OBS_LOS_ILL_FLUX_PICKLE)
            and os.path.exists(TAR_LOS_ILL_FLUX_PICKLE)):
        logger.info(
            "NEBULA_FLUX_PICKLER: Found existing LOS+illum+flux pickles; "
            "loading from disk (force_recompute=False)."
        )

        # Reload observer tracks with full LOS + illumination + flux information.
        # These pickles were written by this module as plain dictionaries, so a
        # direct pickle.load is sufficient.
        with open(OBS_LOS_ILL_FLUX_PICKLE, "rb") as f_obs:
            observer_tracks = pickle.load(f_obs)

        # Reload target tracks with full LOS + illumination + flux information.
        with open(TAR_LOS_ILL_FLUX_PICKLE, "rb") as f_tar:
            target_tracks = pickle.load(f_tar)

        # Return the reloaded dictionaries directly.
        return observer_tracks, target_tracks


    # ----------------------------------------------------------------------
    # Slow path: either flux pickles do not exist, or the user explicitly
    #            requested recomputation. Start from LOS + illumination.
    # ----------------------------------------------------------------------
    logger.info(
        "NEBULA_FLUX_PICKLER: Computing flux for all targets "
        "(force_recompute=%s).",
        force_recompute,
    )

    # Step 1: obtain LOS + illumination tracks using the dedicated pickler
    #         This returns dictionaries of track dicts keyed by name.
    observer_tracks, target_tracks = NEBULA_SAT_ILL_PICKLER.attach_illum_to_all_targets(
        observer_name=observer_name,
        force_recompute=force_recompute,
        logger=logger,
    )

    # ----------------------------------------------------------------------
    # Step 2: determine which observer(s) to use for flux.
    #         - If observer_name was explicitly provided, restrict to that one.
    #         - Otherwise, use all observers returned by the illumination pickler.
    # ----------------------------------------------------------------------
    if observer_name is not None:
        if observer_name not in observer_tracks:
            raise ValueError(
                f"Requested observer_name='{observer_name}' not found in "
                f"observer_tracks keys: {list(observer_tracks.keys())}"
            )
        observer_names = [observer_name]
    else:
        observer_names = list(observer_tracks.keys())

    logger.info(
        "Flux computation will use %d observer(s): %s",
        len(observer_names),
        observer_names,
    )

    # ----------------------------------------------------------------------
    # Step 2b: choose a sensor configuration and derive effective efficiency.
    #          If no sensor is supplied, default to ACTIVE_SENSOR.
    # ----------------------------------------------------------------------
    if sensor is None:
        sensor = ACTIVE_SENSOR
        logger.info("No sensor supplied; defaulting to ACTIVE_SENSOR.")

    # Compute effective band-integrated efficiency η_eff.
    # If quantum_efficiency is None, interpret this as "photon-flux only"
    # and use η_eff = 1.0 so NEBULA_FLUX returns flux at the entrance pupil.
    try:
        optical_throughput = getattr(sensor, "optical_throughput")
        quantum_efficiency = getattr(sensor, "quantum_efficiency")
    except AttributeError as exc:
        raise AttributeError(
            "Sensor object passed to attach_flux_to_all_targets() is missing "
            "'optical_throughput' or 'quantum_efficiency' attributes."
        ) from exc

    if quantum_efficiency is None:
        # Photon-flux-only mode (no instrument efficiency applied)
        eta_eff = 1.0
        logger.info(
            "Sensor '%s' has quantum_efficiency=None; "
            "using eta_eff=1.0 (photon-flux only, at entrance pupil).",
            getattr(sensor, "name", "UNKNOWN"),
        )
    else:
        # Standard mode: apply throughput × QE. If throughput is None,
        # treat it as 1.0 so QE alone sets the efficiency.
        if optical_throughput is None:
            eta_eff = float(quantum_efficiency)
        else:
            eta_eff = float(optical_throughput) * float(quantum_efficiency)


    logger.info(
        "Computing Lambertian flux for %d targets with sensor '%s' "
        "(eta_eff=%.3f).",
        len(target_tracks),
        getattr(sensor, "name", "UNKNOWN"),
        eta_eff,
    )

    # Optionally store basic sensor info in each target for traceability
    sensor_name = getattr(sensor, "name", "UNKNOWN")
    n_targets = len(target_tracks)

    # ------------------------------------------------------------------
    # Step 3: loop over each observer and each target with tqdm,
    #         caching flux per observer under target["by_observer"].
    # ------------------------------------------------------------------
    for obs_name in observer_names:
        observer_track = observer_tracks[obs_name]
        logger.info(
            "Computing Lambertian flux for %d targets with observer '%s' "
            "and sensor '%s' (eta_eff=%.3f).",
            n_targets,
            obs_name,
            sensor_name,
            eta_eff,
        )

        target_iter = target_tracks.items()
        if tqdm is not None:
            target_iter = tqdm(
                target_iter,
                total=n_targets,
                desc=f"FLUX: {obs_name}",
                unit="target",
            )

        for idx, (target_name, target_track) in enumerate(target_iter, start=1):
            # If tqdm is not available, fall back to per-target INFO logging.
            if tqdm is None:
                logger.info(
                    "[%s] Flux computation: processing target %d / %d: %s",
                    obs_name,
                    idx,
                    n_targets,
                    target_name,
                )

            # Record basic sensor metadata for traceability (same for all observers)
            target_track["radiometry_sensor_name"] = sensor_name
            target_track["radiometry_eta_eff"] = eta_eff

            try:
                # ------------------------------------------------------------------
                # 1) Get per-observer illumination for this target.
                #    NEBULA_SAT_ILL_PICKLER has already populated:
                #       target_track["by_observer"][obs_name]["illum_*"]
                # ------------------------------------------------------------------
                by_obs = target_track.get("by_observer")
                if by_obs is None or obs_name not in by_obs:
                    raise KeyError(
                        f"Target '{target_name}' has no by_observer entry for "
                        f"observer '{obs_name}'"
                    )

                obs_entry = by_obs[obs_name]

                illum_is_sunlit = np.asarray(
                    obs_entry.get("illum_is_sunlit", None),
                    dtype=bool,
                )
                phase_angle_rad = np.asarray(
                    obs_entry.get("illum_phase_angle_rad", None),
                    dtype=float,
                )

                if illum_is_sunlit.size == 0 or phase_angle_rad.size == 0:
                    raise KeyError(
                        f"Illumination arrays missing for target '{target_name}' "
                        f"and observer '{obs_name}'"
                    )

                frac_illum = obs_entry.get("illum_fraction_illuminated", None)
                if frac_illum is not None:
                    frac_illum = np.asarray(frac_illum, dtype=float)

                # ------------------------------------------------------------------
                # 2) Build a minimal target view for radiometry using THIS
                #    observer's illumination, then call the single-pair solver.
                # ------------------------------------------------------------------
                flux_target = {
                    "times": np.asarray(target_track["times"]),
                    "r_eci_km": np.asarray(target_track["r_eci_km"]),
                    "illum_is_sunlit": illum_is_sunlit,
                    "illum_phase_angle_rad": phase_angle_rad,
                    "illum_fraction_illuminated": frac_illum,
                }

                # Carry over any per-target optical properties (not observer-specific)
                if "optical_radius_m" in target_track:
                    flux_target["optical_radius_m"] = target_track["optical_radius_m"]
                if "optical_geometric_albedo_g" in target_track:
                    flux_target["optical_geometric_albedo_g"] = target_track[
                        "optical_geometric_albedo_g"
                    ]

                # Run Lambertian radiometry for this observer–target pair.
                result = NEBULA_FLUX.compute_lambertian_radiometry_for_pair(
                    observer_track=observer_track,
                    target_track=flux_target,
                    eta_eff=eta_eff,
                    logger=logger,
                )

                # ------------------------------------------------------------------
                # 3) Attach radiometry to the per-observer entry on the ORIGINAL
                #    target track. No observer-relative fields are written to the
                #    top level; everything lives in by_observer[obs_name].
                # ------------------------------------------------------------------
                obs_entry["rad_range_obs_sat_m"] = np.asarray(
                    result.range_obs_sat_m
                )
                obs_entry["rad_lambert_phase_function"] = np.asarray(
                    result.lambert_phase_function
                )
                obs_entry["rad_flux_g_w_m2"] = np.asarray(
                    result.flux_g_w_m2
                )
                obs_entry["rad_photon_flux_g_m2_s"] = np.asarray(
                    result.photon_flux_g_m2_s
                )
                obs_entry["rad_app_mag_g"] = np.asarray(
                    result.app_mag_g
                )

            except Exception as exc:
                # Log an error but continue with the remaining targets
                logger.error(
                    "[%s] Error computing flux for target '%s': %s",
                    obs_name,
                    target_name,
                    exc,
                    exc_info=True,
                )


    # ----------------------------------------------------------------------
    # Step 4: write the fully augmented tracks back to disk as new pickles.
    # ----------------------------------------------------------------------
    logger.info(
        "Writing augmented observer/target tracks with flux to:\n"
        "  %s\n"
        "  %s",
        OBS_LOS_ILL_FLUX_PICKLE,
        TAR_LOS_ILL_FLUX_PICKLE,
    )

    # Serialize observer tracks (currently unchanged by flux, but kept for
    # symmetry and possible future per-observer radiometric fields).
    with open(OBS_LOS_ILL_FLUX_PICKLE, "wb") as f_obs:
        pickle.dump(observer_tracks, f_obs, protocol=pickle.HIGHEST_PROTOCOL)

    # Serialize target tracks with all radiometric fields now attached
    with open(TAR_LOS_ILL_FLUX_PICKLE, "wb") as f_tar:
        pickle.dump(target_tracks, f_tar, protocol=pickle.HIGHEST_PROTOCOL)

    # ----------------------------------------------------------------------
    # Step 5: reload the new pickles for safety and return them.
    # ----------------------------------------------------------------------

    logger.info(
        "NEBULA_FLUX_PICKLER: Completed flux attachment for %d targets.",
        len(target_tracks),
    )

    # Return the reloaded dictionaries to the caller
    return observer_tracks, target_tracks


# ---------------------------------------------------------------------------
# Optional command-line entry point for quick manual tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # If this file is run directly, perform a small test:
    #   - Use default observer (assumes there is only one).
    #   - Use default sensor (ACTIVE_SENSOR).
    #   - Do NOT force recompute (reuse flux pickles if present).
    log = _build_default_logger()
    log.info("Running NEBULA_FLUX_PICKLER test harness (__main__).")

    # Example with explicit sensor (EVK4); you can omit sensor=... to rely on
    # the same default.
    obs_tracks, tar_tracks = attach_flux_to_all_targets(
        observer_name=None,
        sensor=ACTIVE_SENSOR,
        force_recompute=False,
        logger=log,
    )


    log.info("Loaded %d observer(s) and %d target(s) with flux attached.",
             len(obs_tracks), len(tar_tracks))

    # Optionally, print a short summary of one target for sanity checking
    if tar_tracks:
        example_name = next(iter(tar_tracks.keys()))
        example_track = tar_tracks[example_name]
        log.info("Example target: %s", example_name)
        # We only check that some of the expected radiometric keys exist;
        # detailed numeric validation is handled elsewhere.
        for key in [
            "flux_G_band_W_m2",
            "flux_G_photons_m2_s",
            "flux_G_photons_s",
            "apparent_mag_G",
        ]:
            log.info("  Has radiometry field '%s'? %s",
                     key, key in example_track)
