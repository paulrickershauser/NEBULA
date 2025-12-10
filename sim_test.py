# sim_test.py
# ---------------------------------------------------------------------------
# High-level driver script to run the NEBULA pipeline through pixel projection
# and inspect basic summary info about the resulting tracks.
# ---------------------------------------------------------------------------

"""
sim_test
========

This script is a simple entry point for exercising the NEBULA pipeline
up through the pixel layer. It:

    1. Configures a basic logger for console output.

    2. Calls NEBULA_PIXEL_PICKLER.attach_pixels_to_all_pairs(), which
       internally cascades through the upstream picklers:

           - NEBULA_FLUX_PICKLER
           - NEBULA_LOS_FLUX_PICKLER
           - NEBULA_SCHEDULE_PICKLER
           - NEBULA_ICRS_PAIR_PICKLER
           - NEBULA_SENSOR_PROJECTION (via NEBULA_PIXEL_PICKLER)

       and ensures that both observer_tracks_with_pixels.pkl and
       target_tracks_with_pixels.pkl are up to date.

    3. Prints a brief summary of the number of observers and targets,
       and shows example per-observer keys for one target. This gives a
       quick check that the pixel-level fields (pix_x, pix_y,
       on_detector, etc.) are present and shaped correctly.

Usage
-----

From your NEBULA root directory in Spyder or a terminal:

    %run sim_test.py

or

    python sim_test.py

You can control whether to force a full recompute of all upstream and
pixel pickles by toggling the FORCE_RECOMPUTE flag below.
"""

# ---------------------------------------------------------------------------
# Standard-library imports
# ---------------------------------------------------------------------------

from typing import Dict, Any, Optional
import logging
import numpy as np
import os

# ---------------------------------------------------------------------------
# NEBULA imports
# ---------------------------------------------------------------------------

# Import the sensor configuration (if you want to override the default).
from Configuration.NEBULA_SENSOR_CONFIG import ACTIVE_SENSOR
from Configuration import NEBULA_PATH_CONFIG
from Configuration.NEBULA_STAR_CONFIG import NEBULA_STAR_CATALOG

# Import the pixel pickler that runs the full chain and attaches pixel data.
from Utility.SAT_OBJECTS import NEBULA_PIXEL_PICKLER

# Photon-domain per-target time series + frames
from Utility.FRAMES import NEBULA_TARGET_PHOTONS as NTP

# Sky footprints (attach RA/Dec/radius to TARGET_PHOTON_FRAMES windows)
from Utility.STARS import NEBULA_SKY_SELECTOR as NSS

# Gaia cone queries over those sky footprints
from Utility.STARS import NEBULA_QUERY_GAIA as NQG

# ----------------------------------------------------------------------
# Star-field utilities: projections + photon time series
# ----------------------------------------------------------------------
from Utility.STARS import NEBULA_STAR_PROJECTION
from Utility.STARS import NEBULA_STAR_SLEW_PROJECTION
from Utility.STARS import NEBULA_STAR_PHOTONS


# ---------------------------------------------------------------------------
# Configuration flags
# ---------------------------------------------------------------------------

# Flag that controls whether to recompute the entire upstream and pixel
# pipeline. Set to True if you have changed core code and want to
# regenerate all pickles. Set to False to reuse existing pickles.
FORCE_RECOMPUTE: bool = False

# Flag that controls whether to build photon-domain per-target time series
# and save obs_target_frames pickles (raw + ranked) via NEBULA_TARGET_PHOTONS.
BUILD_TARGET_PHOTON_FRAMES: bool = False

# Flag that controls whether to run the sky-footprint and Gaia catalog
# pipeline after TARGET_PHOTON_FRAMES are available. If True, this will:
#   1) Call NEBULA_SKY_SELECTOR.main(logger=...)
#   2) Call NEBULA_QUERY_GAIA.main(mag_limit_sensor_G=None, logger=...)
RUN_GAIA_PIPELINE: bool = False
# Control whether to run the star-field pipeline (projection + photons)
RUN_STAR_PIPELINE: bool = True



# ---------------------------------------------------------------------------
# Helper to configure logging
# ---------------------------------------------------------------------------

def configure_logging() -> logging.Logger:
    """
    Configure and return a logger for the sim_test script.

    Returns
    -------
    logger : logging.Logger
        Logger instance configured to log INFO-level messages to the
        console with a simple timestamped format.
    """
    # Get a logger specific to this script using its module name.
    logger = logging.getLogger("sim_test")

    # If the logger has no handlers yet, configure a basic stream handler.
    if not logger.handlers:
        # Create a stream handler that writes log messages to stderr.
        handler = logging.StreamHandler()
        # Define a simple format with time, name, level, and message.
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        # Attach the formatter to the handler so messages are formatted.
        handler.setFormatter(formatter)
        # Add the handler to the logger so it becomes active.
        logger.addHandler(handler)
        # Set the logger to INFO level to show standard progress messages.
        logger.setLevel(logging.INFO)

    # Return the configured logger to the caller.
    return logger


# ---------------------------------------------------------------------------
# Simple summary helpers
# ---------------------------------------------------------------------------

def summarize_tracks(
    obs_tracks: Dict[str, Any],
    tar_tracks: Dict[str, Any],
    logger: logging.Logger,
) -> None:
    """
    Print a brief summary of observer and target tracks, focusing on
    pixel-level fields for a quick sanity check.

    Parameters
    ----------
    obs_tracks : dict
        Dictionary mapping observer names to observer track dictionaries.
    tar_tracks : dict
        Dictionary mapping target names to target track dictionaries.
    logger : logging.Logger
        Logger used to emit summary information.
    """
    # Log the number of observers that were processed.
    logger.info("Number of observers: %d", len(obs_tracks))
    # Log the number of targets that were processed.
    logger.info("Number of targets:   %d", len(tar_tracks))

    # If there are no targets at all, there is nothing more to summarize.
    if not tar_tracks:
        logger.warning("No targets found in tar_tracks; nothing to summarize.")
        return

    # Pick an arbitrary target name (e.g., the first key) for inspection.
    example_tar_name = next(iter(tar_tracks.keys()))
    # Retrieve the corresponding target track dictionary.
    example_tar_track = tar_tracks[example_tar_name]

    # Log which target we are using as an example.
    logger.info("Example target: '%s'", example_tar_name)

    # Extract the per-observer dictionary for this example target, if present.
    by_observer = example_tar_track.get("by_observer", None)

    # If this target has no by_observer entry, log and return early.
    if not by_observer:
        logger.warning(
            "Example target '%s' has no by_observer entry; "
            "no per-observer pixel fields to summarize.",
            example_tar_name,
        )
        return

    # Pick an arbitrary observer name that views this example target.
    example_obs_name = next(iter(by_observer.keys()))
    # Retrieve the corresponding per-observer sub-dictionary.
    example_by_obs = by_observer[example_obs_name]

    # Log which observer we are using for per-observer field inspection.
    logger.info("Example observer for that target: '%s'", example_obs_name)

    # Build a sorted list of keys for this observer–target pair.
    obs_keys = sorted(example_by_obs.keys())
    # Log the available keys so we can verify the presence of pixel fields.
    logger.info(
        "Per-observer keys for [target='%s', observer='%s']:\n  %s",
        example_tar_name,
        example_obs_name,
        ", ".join(obs_keys),
    )

    # Try to log the lengths of the main pixel fields if they exist.
    for field in ("pix_x", "pix_y", "on_detector", "on_detector_visible_sunlit"):
        # Check if this field is present for the example observer–target pair.
        if field in example_by_obs:
            # Attempt to get the length of the array (works for lists/ndarrays).
            try:
                field_len = len(example_by_obs[field])
            except TypeError:
                field_len = -1  # Use -1 if length cannot be determined.
            # Log the field name and its length.
            logger.info("  Field '%s' length: %d", field, field_len)
        else:
            # If the field is not present, log that it is missing.
            logger.info("  Field '%s' not present for this pair.", field)

def annotate_windows_with_tracking_mode(
    obs_tracks: Dict[str, Any],
    ranked_target_frames: Dict[str, Any],
    pixel_scale_rad: float,
    pix_threshold: float,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Annotate each window in ranked_target_frames with a tracking mode.

    This is a STRICT / FAIL-HARD helper.

    For each observer and each window, this function:
        1. Reads boresight pointing from obs_tracks (preferring
           'pointing_boresight_ra_deg' / 'pointing_boresight_dec_deg',
           falling back to 'boresight_ra_deg' / 'boresight_dec_deg').
        2. Converts RA/Dec time series to unit vectors on the celestial sphere.
        3. Computes angular motion between consecutive samples.
        4. Converts that angular motion to pixel motion using pixel_scale_rad.
        5. For each window, examines the per-step pixel drift over coarse indices
           [start_index .. end_index] and sets:

               window["tracking_mode"] = "sidereal"
                   if max_step_pix < pix_threshold
               window["tracking_mode"] = "slew"
                   otherwise

           It also stores:
               window["total_drift_pix"] : float
                   Sum of per-step drifts (integrated drift over the window).
               window["max_step_pix"]    : float
                   Maximum per-step drift within the window.

    STRICT behavior:
        - Any missing or malformed data that prevents a reliable classification
          results in a RuntimeError. This includes:
              * missing obs_track for an observer
              * missing or empty RA/Dec arrays
              * RA/Dec length mismatch
              * fewer than 2 coarse samples
              * missing start_index / end_index
              * out-of-bounds indices
              * end_index <= start_index
              * empty drift slice for a window

    Parameters
    ----------
    obs_tracks : dict
        Per-observer track dictionaries containing boresight pointing arrays.
    ranked_target_frames : dict
        Per-observer frames-with-windows structure produced by
        NEBULA_TARGET_PHOTONS / NEBULA_PHOTON_FRAME_BUILDER.
        This function modifies each window entry in-place.
    pixel_scale_rad : float
        Sensor plate scale [radians per pixel]. Must be > 0.
    pix_threshold : float
        Threshold on per-frame drift in pixels. If max_step_pix is less than
        this value, the window is classified as 'sidereal'; otherwise 'slew'.
    logger : logging.Logger, optional
        Logger for informational messages. If None, a default 'sim_test' logger
        is used.

    Raises
    ------
    RuntimeError
        If any observer or window has inconsistent or insufficient data to
        compute tracking mode.
    ValueError
        If pixel_scale_rad is not positive.
    """
    if logger is None:
        logger = logging.getLogger("sim_test")

    if pixel_scale_rad <= 0.0:
        raise ValueError(
            f"annotate_windows_with_tracking_mode: pixel_scale_rad must be "
            f"positive, got {pixel_scale_rad!r}."
        )

    for obs_name, frames_entry in ranked_target_frames.items():
        obs_track = obs_tracks.get(obs_name)
        if obs_track is None:
            raise RuntimeError(
                f"annotate_windows_with_tracking_mode: no obs_track for "
                f"observer '{obs_name}'."
            )

        # --- Safe selection of RA / Dec arrays (no boolean `or` on numpy arrays) ---
        if "pointing_boresight_ra_deg" in obs_track:
            ra_data = obs_track["pointing_boresight_ra_deg"]
        elif "boresight_ra_deg" in obs_track:
            ra_data = obs_track["boresight_ra_deg"]
        else:
            raise RuntimeError(
                f"annotate_windows_with_tracking_mode: observer '{obs_name}' "
                f"has no 'pointing_boresight_ra_deg' or 'boresight_ra_deg'."
            )

        if "pointing_boresight_dec_deg" in obs_track:
            dec_data = obs_track["pointing_boresight_dec_deg"]
        elif "boresight_dec_deg" in obs_track:
            dec_data = obs_track["boresight_dec_deg"]
        else:
            raise RuntimeError(
                f"annotate_windows_with_tracking_mode: observer '{obs_name}' "
                f"has no 'pointing_boresight_dec_deg' or 'boresight_dec_deg'."
            )

        ra_deg = np.asarray(ra_data, dtype=float)
        dec_deg = np.asarray(dec_data, dtype=float)

        if ra_deg.size == 0 or dec_deg.size == 0:
            raise RuntimeError(
                f"annotate_windows_with_tracking_mode: empty RA/Dec arrays "
                f"for observer '{obs_name}'."
            )

        if ra_deg.shape != dec_deg.shape:
            raise RuntimeError(
                f"annotate_windows_with_tracking_mode: RA/Dec length mismatch "
                f"for observer '{obs_name}' (ra={ra_deg.size}, dec={dec_deg.size})."
            )

        # --- Convert boresight RA/Dec to unit vectors on the celestial sphere ---
        ra_rad = np.deg2rad(ra_deg)
        dec_rad = np.deg2rad(dec_deg)

        cos_dec = np.cos(dec_rad)
        bx = np.cos(ra_rad) * cos_dec
        by = np.sin(ra_rad) * cos_dec
        bz = np.sin(dec_rad)
        b = np.stack((bx, by, bz), axis=1)  # shape (N, 3), N coarse samples

        if b.shape[0] < 2:
            raise RuntimeError(
                f"annotate_windows_with_tracking_mode: fewer than 2 time samples "
                f"for observer '{obs_name}'; cannot compute drift."
            )

        # Angular motion between consecutive coarse samples: Δθ_j = arccos(b_j · b_{j+1})
        dot = np.einsum("ij,ij->i", b[:-1], b[1:])
        dot = np.clip(dot, -1.0, 1.0)  # numerical safety
        delta_theta_rad = np.arccos(dot)  # shape (N-1,)

        # Convert angular motion to approximate pixel motion using the plate scale.
        # Assumes pixel_scale_rad is [radians per pixel].
        delta_pix = delta_theta_rad / pixel_scale_rad  # pixels per step

        windows = frames_entry.get("windows", [])
        for w in windows:
            start_idx = w.get("start_index")
            end_idx = w.get("end_index")
            w_index = w.get("window_index", None)

            if start_idx is None or end_idx is None:
                raise RuntimeError(
                    f"annotate_windows_with_tracking_mode: window (index={w_index}) "
                    f"for observer '{obs_name}' missing start_index or end_index."
                )

            if not (0 <= start_idx < ra_deg.size) or not (0 <= end_idx < ra_deg.size):
                raise RuntimeError(
                    f"annotate_windows_with_tracking_mode: window (index={w_index}) "
                    f"indices [{start_idx}..{end_idx}] out of bounds for observer "
                    f"'{obs_name}' (N={ra_deg.size})."
                )

            if end_idx <= start_idx:
                raise RuntimeError(
                    f"annotate_windows_with_tracking_mode: window (index={w_index}) "
                    f"for observer '{obs_name}' has non-increasing indices "
                    f"[start_index={start_idx}, end_index={end_idx}]."
                )

            # delta_pix[j] is the drift from coarse index j -> j+1.
            # For a window covering coarse indices [start_idx..end_idx], we want
            # steps j = start_idx .. (end_idx-1) inclusive, which is exactly
            # delta_pix[start_idx:end_idx].
            step_slice = slice(start_idx, end_idx)
            window_delta = delta_pix[step_slice]

            if window_delta.size == 0:
                raise RuntimeError(
                    f"annotate_windows_with_tracking_mode: window (index={w_index}) "
                    f"for observer '{obs_name}' produced an empty drift slice "
                    f"from indices [{start_idx}..{end_idx}]."
                )

            total_drift_pix = float(np.sum(window_delta))
            max_step_pix = float(np.max(window_delta))

            # Use per-step drift to decide sidereal vs slew.
            # Rationale: even if tiny numerical noise accumulates over a long window,
            # what matters physically is whether stars smear significantly between frames.
            mode = "sidereal" if max_step_pix < pix_threshold else "slew"

            w["tracking_mode"] = mode
            w["total_drift_pix"] = total_drift_pix
            w["max_step_pix"] = max_step_pix

        logger.info(
            "annotate_windows_with_tracking_mode: observer '%s' -> annotated %d windows.",
            obs_name,
            len(windows),
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Main entry point for sim_test.

    This function:

        1. Configures logging.
        2. Runs NEBULA_PIXEL_PICKLER.attach_pixels_to_all_pairs() to
           ensure all upstream and pixel pickles are up to date.
        3. Prints a brief summary of the resulting tracks, focusing on
           pixel-level fields for a quick sanity check.
        4. Optionally (BUILD_TARGET_PHOTON_FRAMES=True), runs the
           photon-domain pipeline via NEBULA_TARGET_PHOTONS to build
           per-target photon time series for all observers and windows,
           and saves both "raw" and "ranked" pickles under
           NEBULA_OUTPUT/TARGET_PHOTON_FRAMES.
        5. Optionally (RUN_GAIA_PIPELINE=True), runs the sky-footprint
           and Gaia catalog pipeline:

           - NEBULA_SKY_SELECTOR.main(logger=...)
           - NEBULA_QUERY_GAIA.main(mag_limit_sensor_G=None, logger=...)
    """
    # Expose these in the interactive namespace (Spyder variable explorer).
    global obs_tracks, tar_tracks
    global obs_target_frames, ranked_target_frames
    global gaia_cache
    global obs_star_projections, obs_star_slew_projections, obs_star_photons


    # Configure a logger for this script.
    logger = configure_logging()

    # Log that the sim_test pixel pipeline is starting.
    logger.info(
        "sim_test: Starting NEBULA pipeline through NEBULA_PIXEL_PICKLER "
        "(force_recompute=%s).",
        FORCE_RECOMPUTE,
    )

    # 1) Pixel pipeline (this cascades through all upstream picklers as needed)
    obs_tracks, tar_tracks = NEBULA_PIXEL_PICKLER.attach_pixels_to_all_pairs(
        force_recompute=FORCE_RECOMPUTE,
        sensor_config=ACTIVE_SENSOR,
        logger=logger,
    )

    # 2) Brief pixel-level summary
    summarize_tracks(
        obs_tracks=obs_tracks,
        tar_tracks=tar_tracks,
        logger=logger,
    )

    logger.info("sim_test: Completed NEBULA pixel pipeline successfully.")

    # 3) Optional photon-domain pipeline (all observers, all windows)
    if BUILD_TARGET_PHOTON_FRAMES:
        logger.info(
            "sim_test: Building per-target photon time series for all "
            "observers via NEBULA_TARGET_PHOTONS."
        )

        # Build per-observer, per-window photon catalogs
        obs_target_frames = NTP.build_obs_target_frames_for_all_observers(
            max_frames_per_window=None,
            logger=logger,
        )

        # Save "raw" version (includes windows with zero targets)
        raw_path = NTP.save_obs_target_frames_pickle(
            obs_target_frames,
            filename="obs_target_frames_raw.pkl",
            logger=logger,
        )

        # Cull empty windows and rank by (n_targets, n_frames)
        ranked_target_frames = NTP.cull_and_rank_obs_target_frames(
            obs_target_frames,
            logger=logger,
        )
        
        # Annotate windows with tracking mode.
        #
        # Plate scale from the active sensor (radians per pixel).
        pixel_scale_rad = ACTIVE_SENSOR.pixel_scale_rad

        # Per-frame drift threshold in pixels, expressed as a fraction of the PSF FWHM.
        # Rationale:
        #   - Treat boresight drift during a single frame as a small smear added in quadrature
        #     to the intrinsic PSF.
        #   - Keeping drift ≲ 0.3 * FWHM inflates the effective FWHM by only ~2% for a
        #     Gaussian PSF convolved with a short uniform trail, which is well below typical
        #     tolerances in astronomical imaging and consistent with guiding-error heuristics
        #     that tracking RMS should be ≲ 0.25–0.3 * FWHM.
        psf_fwhm_pix = ACTIVE_SENSOR.psf_fwhm_pix or 1.0
        pix_threshold = 0.3 * psf_fwhm_pix

        annotate_windows_with_tracking_mode(
            obs_tracks=obs_tracks,
            ranked_target_frames=ranked_target_frames,
            pixel_scale_rad=pixel_scale_rad,
            pix_threshold=pix_threshold,
            logger=logger,
        )


        # Save the culled/sorted version
        ranked_path = NTP.save_obs_target_frames_pickle(
            ranked_target_frames,
            filename="obs_target_frames_ranked.pkl",
            logger=logger,
        )

        # Compact summary per observer
        for obs_name, obs_entry in ranked_target_frames.items():
            windows = obs_entry.get("windows", [])
            logger.info(
                "sim_test: Observer '%s' has %d non-empty photon windows "
                "after culling.",
                obs_name,
                len(windows),
            )
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
            "sim_test: Photon-target frame pipeline complete. "
            "Raw='%s', ranked='%s'.",
            raw_path,
            ranked_path,
        )
    else:
        logger.info(
            "sim_test: BUILD_TARGET_PHOTON_FRAMES=False; skipping photon "
            "time-series and TARGET_PHOTON_FRAMES pickles."
        )

    # 4) Optional sky-footprint + Gaia catalog pipeline
    if RUN_GAIA_PIPELINE:
        logger.info(
            "sim_test: Running NEBULA_SKY_SELECTOR to attach sky footprints "
            "to TARGET_PHOTON_FRAMES windows."
        )
        NSS.main(logger=logger)

        logger.info(
            "sim_test: Running NEBULA_QUERY_GAIA to query Gaia for each "
            "eligible window."
        )
        # mag_limit_sensor_G=None => use ACTIVE_SENSOR.mag_limit internally
        gaia_cache = NQG.main(
            mag_limit_sensor_G=None,
            logger=logger,
        )

        # Brief per-observer summary from the cache
        for obs_name, obs_entry in gaia_cache.items():
            run_meta = obs_entry.get("run_meta", {})
            counts = run_meta.get("window_counts", {})
            logger.info(
                "sim_test: Gaia cache summary for observer '%s' -> "
                "queried=%d, ok=%d, error=%d, skipped_zero_targets=%d, "
                "skipped_bad_sky=%d, skipped_broken=%d, total_stars_ok=%d",
                obs_name,
                counts.get("queried", 0),
                counts.get("ok", 0),
                counts.get("error", 0),
                counts.get("skipped_zero_targets", 0),
                counts.get("skipped_bad_sky", 0),
                counts.get("skipped_broken", 0),
                run_meta.get("total_stars_ok_windows", 0),
            )
    else:
        logger.info(
            "sim_test: RUN_GAIA_PIPELINE=False; skipping sky-footprint and "
            "Gaia catalog queries."
        )
    # 5) Optional star-field pipeline: projection (sidereal + slew) + photons
    #
    # This stage:
    #   - Uses ranked_target_frames (with tracking_mode already annotated),
    #   - Uses the Gaia cones/cache products written by NEBULA_QUERY_GAIA,
    #   - Builds per-window star projections (sidereal + slew),
    #   - Converts those into per-frame star photon time series.
    #
    # All three modules (projection, slew projection, photons) are written as
    # standalone pipelines that read the necessary pickles from NEBULA_OUTPUT
    # and Configuration.* and then write their own outputs back under
    # NEBULA_OUTPUT/FRAMES / NEBULA_OUTPUT/STARS (depending on how you've
    # configured them).
    if RUN_STAR_PIPELINE:
        logger.info(
            "sim_test: Running star-field pipeline "
            "(NEBULA_STAR_PROJECTION + NEBULA_STAR_SLEW_PROJECTION + NEBULA_STAR_PHOTONS)."
        )

        # Sidereal star projections: builds obs_star_projections for windows
        # with tracking_mode='sidereal' (and/or all windows, depending on the
        # internal logic of NEBULA_STAR_PROJECTION).
        #
        # Resolve default paths used by the star-field pipeline so we can
        # pass them explicitly to downstream modules that do not infer
        # locations internally.
        star_projection_path = NEBULA_STAR_PROJECTION._resolve_default_output_path()
        obs_tracks_path = NEBULA_STAR_PROJECTION._resolve_default_obs_tracks_path()
        star_slew_output_path = os.path.join(
            NEBULA_PATH_CONFIG.NEBULA_OUTPUT_DIR,
            "STARS",
            getattr(NEBULA_STAR_CATALOG, "name", "UNKNOWN_CATALOG"),
            "obs_star_slew_projections.pkl",
        )

        # main(...) is expected to:
        #   - Load ranked_target_frames + Gaia cones + obs_tracks from disk,
        #   - Build per-window star projection products,
        #   - Write obs_star_projections.pkl,
        #   - Return the in-memory obs_star_projections dict (or None).
        obs_star_projections = NEBULA_STAR_PROJECTION.main(logger=logger)

        # Slew star projections: builds obs_star_slew_projections for windows
        # with tracking_mode='slew', using per-frame WCS to follow the stars
        # across the detector during the motion.
        #
        # main(...) is expected to:
        #   - Load the same ranked_target_frames + Gaia cones + obs_tracks,
        #   - Build per-frame star positions for slewing windows only,
        #   - Write obs_star_slew_projections.pkl,
        #   - Return the in-memory obs_star_slew_projections dict (or None).
        obs_star_slew_projections = NEBULA_STAR_SLEW_PROJECTION.main(
            star_projection_path=star_projection_path,
            obs_tracks_path=obs_tracks_path,
            output_path=star_slew_output_path,
            logger=logger,
        )

        # Star photons: consumes both the sidereal and slew projection
        # products and builds per-star, per-frame photon time series
        # aligned with the target photon frames.
        #
        # main(...) is expected to:
        #   - Load obs_target_frames_ranked.pkl (for windows & frame timing),
        #   - Load obs_star_projections.pkl (sidereal),
        #   - Load obs_star_slew_projections.pkl (slew),
        #   - Build obs_star_photons[obs_name]["windows"][i]["stars"][...],
        #   - Write obs_star_photons.pkl,
        #   - Return the in-memory obs_star_photons dict (or None).
        obs_star_photons = NEBULA_STAR_PHOTONS.run_star_photons_pipeline_from_pickles(
            frames_with_sky_path=NEBULA_STAR_PROJECTION._resolve_default_frames_path(),
            star_projection_sidereal_path=star_projection_path,
            star_projection_slew_path=star_slew_output_path,
            logger=logger,
        )

        # Provide a compact log summary if we actually got an in-memory dict.
        if isinstance(obs_star_photons, dict):
            logger.info(
                "sim_test: Star-photon pipeline produced entries for %d observers.",
                len(obs_star_photons),
            )
            for obs_name, entry in obs_star_photons.items():
                n_windows = len(entry.get("windows", []))
                logger.info(
                    "  Observer '%s': %d star-photon windows.",
                    obs_name,
                    n_windows,
                )
    else:
        logger.info(
            "sim_test: RUN_STAR_PIPELINE=False; skipping star projection and "
            "star-photon pipelines."
        )



# ---------------------------------------------------------------------------
# Script guard
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # If this file is executed as a script, call the main() function.
    main()
