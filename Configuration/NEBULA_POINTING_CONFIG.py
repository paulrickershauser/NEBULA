"""
NEBULA_POINTING_CONFIG.py

Pointing-mode configuration for the NEBULA (Neuromorphic Event-Based
Luminance Asset-tracking) simulation framework.

This module is responsible ONLY for describing high-level pointing
strategies and their tunable parameters.  It does not perform any
geometry, coordinate transforms, or line-of-sight intersection tests.
Those behaviors are implemented in runtime utilities such as
NEBULA_POINTING.py, NEBULA_VISIBILITY.py, and any WCS / sensor
projection helpers.

In practice, this file provides:

  - An enumeration of supported pointing modes (PointingMode),
    including:
        * A fixed ICRS RA/Dec stare with Earth-avoidance gating
          (FIXED_ICRS_EARTH_AVOID).
        * An "anti-Sun stare" mode for dawn–dusk sun-synchronous
          LEO platforms that stare toward the midnight / anti-Sun
          sector (ANTI_SUN_STARE).  In this mode, the actual
          boresight RA/Dec are computed at runtime from the Sun
          ephemeris and are NOT hard-coded here.

  - A PointingConfig dataclass that collects the mode selection and
    a small set of tunable parameters that downstream runtime code
    can interpret when computing the boresight direction and rolling
    the focal plane.

  - A DEFAULT_POINTING_CONFIG instance for use when no pointing
    configuration is supplied explicitly.

  - A describe_pointing_config() helper for logging / debugging,
    mirroring the pattern used in NEBULA_TIME_CONFIG.

By design, this module keeps imports minimal (standard-library only)
and avoids heavy dependencies such as astropy or skyfield.  That makes
it safe to import in any environment where configuration is needed,
without triggering geometry or ephemeris machinery.

Typical usage
-------------

A runtime pointing helper (e.g. NEBULA_POINTING.py) might:

  1. Import this module and read DEFAULT_POINTING_CONFIG (or a
     user-provided PointingConfig instance).

  2. Branch on the config.mode field to determine which pointing law
     to apply (fixed RA/Dec vs anti-Sun stare, etc.).

  3. Use the remaining fields (roll_deg, earth_avoid_margin_deg,
     require_sensor_sunlit, sun_exclusion_angle_deg, etc.) as knobs
     when computing the boresight RA/Dec, checking Earth occlusion,
     and building a WCS for sensor projection.

This separation keeps the configuration declarative while leaving the
actual numerical logic in dedicated utility modules.
"""

from dataclasses import dataclass  # Import dataclass for defining simple configuration containers.
from enum import Enum              # Import Enum for defining pointing mode identifiers.
from typing import Final           # Import Final to mark module-level defaults as constants.


class PointingMode(str, Enum):
    """
    Enumeration of supported high-level pointing modes in NEBULA.

    The enumeration values are string-based so that they are easy to
    serialize (e.g., into JSON), log, or use as keys in configuration
    dictionaries.

    Modes
    -----
    FIXED_ICRS_EARTH_AVOID :
        The sensor boresight is held at a fixed right ascension and
        declination in the ICRS frame.  At runtime, a pointing helper
        will:
          * Convert the stored (boresight_ra_deg, boresight_dec_deg)
            into a unit vector in the same inertial frame used for
            the observer state.
          * Evaluate whether the corresponding line-of-sight passes
            through the Earth, using the global blocking radius
            R_BLOCK from NEBULA_ENV_CONFIG.
          * Optionally enforce an additional angular "limb margin"
            given by earth_avoid_margin_deg to keep the boresight
            away from the bright Earth limb.

    ANTI_SUN_STARE :
        A higher-level mode that represents an "anti-Sun stare"
        geometry for a dawn–dusk sun-synchronous LEO platform
        observing GEO targets near the midnight sector.

        In this mode, the actual boresight RA/Dec are NOT specified
        directly in this configuration file.  Instead, a runtime
        pointing helper will:
          * Compute the instantaneous anti-Sun direction in ICRS
            from the Sun ephemeris (e.g., RA_sun(t) + 180 deg).
          * Optionally apply a small offset to better align with the
            GEO belt.
          * Apply Earth-avoidance logic using the global R_BLOCK and
            earth_avoid_margin_deg.
          * Optionally enforce a Sun exclusion angle given by
            sun_exclusion_angle_deg, if non-zero.

        The configuration fields in PointingConfig still apply to
        this mode, but the boresight_ra_deg and boresight_dec_deg
        fields are interpreted as "unused" (or as optional offsets),
        depending on how NEBULA_POINTING.py chooses to implement
        the anti-Sun stare law.
        
    ANTI_SUN_GEO_BELT :
        A sidereal (fixed-ICRS) stare optimized for GEO observations
        from a dawn–dusk sun-synchronous LEO platform.

        In this mode, a runtime pointing helper will:
          * Choose a single reference time within the simulation
            window (e.g., the midpoint of the time grid) that
            represents the "midnight sector" / Earth-shadow geometry,
            following the dawn–dusk + anti-Sun logic discussed in the
            pointing strategy paper. :contentReference[oaicite:6]{index=6}
          * Compute the anti-Sun right ascension at that reference
            time, α_ref = α_sun(t_ref) + 180 deg (mod 360).
          * Set the boresight declination to δ_ref = 0 deg so that the
            boresight lies on the GEO belt (Earth's equatorial plane),
            effectively "snapping" the anti-Sun direction onto the GEO
            ring.
          * Hold (α_ref, δ_ref) fixed for the entire observation
            window so that stars remain stationary in the field of
            view (sidereal stare) while GEO objects drift at ~15″/s.
          * Apply Earth-avoidance, sensor-sunlit, and Sun-exclusion
            logic using the same configuration fields as the other
            modes (earth_avoid_margin_deg, require_sensor_sunlit,
            sun_exclusion_angle_deg).

        The boresight_ra_deg and boresight_dec_deg fields in
        PointingConfig are not used to *define* this mode; instead,
        the runtime implementation derives a fixed boresight from the
        Sun ephemeris as described above.
    """

    # Define the fixed-ICRS mode enumeration value as a string.
    FIXED_ICRS_EARTH_AVOID = "fixed_icrs_earth_avoid"
    # Define the anti-Sun stare mode enumeration value as a string.
    ANTI_SUN_STARE = "anti_sun_stare"
    
    ANTI_SUN_GEO_BELT = "anti_sun_geo_belt"



@dataclass(frozen=True)
class PointingConfig:
    """
    High-level pointing configuration for NEBULA.

    This dataclass collects both the pointing mode selection and the
    small set of tunable parameters that downstream runtime modules
    use to compute the sensor boresight, enforce Earth-avoidance
    constraints, and (optionally) enforce Sun-exclusion constraints.

    Fields
    ------
    mode :
        The high-level pointing strategy to use, represented as a
        PointingMode enumeration value.  Downstream code can branch
        on this value to decide whether to:
          * Use the fixed ICRS RA/Dec fields directly
            (FIXED_ICRS_EARTH_AVOID), or
          * Derive the boresight direction from the anti-Sun
            geometry (ANTI_SUN_STARE).

    boresight_ra_deg :
        Right ascension of the desired boresight direction in the
        ICRS frame, in degrees.  This field is primarily used when
        mode == FIXED_ICRS_EARTH_AVOID.  For the ANTI_SUN_STARE mode,
        downstream code may ignore this field or treat it as an
        optional RA offset to be applied to the anti-Sun vector.

    boresight_dec_deg :
        Declination of the desired boresight direction in the ICRS
        frame, in degrees.  As with boresight_ra_deg, this is mainly
        used for the FIXED_ICRS_EARTH_AVOID mode.  In the
        ANTI_SUN_STARE mode, it may be ignored or treated as a Dec
        offset, depending on the chosen implementation.

    roll_deg :
        Rotation of the focal plane about the boresight, in degrees.
        A roll of 0 degrees corresponds to a convention where the
        detector +y axis aligns with increasing Declination in the
        local tangent plane of the WCS.  Positive roll is a
        right-hand rotation about the boresight.  This field applies
        to all pointing modes.

    earth_avoid_margin_deg :
        Additional angular clearance, in degrees, by which the
        boresight must clear the Earth's apparent limb before runtime
        pointing code will consider a direction usable.  At runtime,
        this margin is combined with the geometric blocking radius
        R_BLOCK from NEBULA_ENV_CONFIG.  A value of 0.0 means "allow
        pointing right up to the geometric limb defined by R_BLOCK."

    require_sensor_sunlit :
        If True, downstream runtime logic may choose to treat frames
        when the observer (sensor platform) is in Earth's shadow as
        "non-science" or "not valid," reflecting operational power
        constraints of real missions (e.g., preferring operations
        only when the spacecraft is sunlit).  If False, eclipse vs
        sunlit status is ignored by the pointing configuration, and
        all timesteps are potentially usable from a geometry
        standpoint.

    sun_exclusion_angle_deg :
        Optional angular exclusion radius around the Sun direction,
        in degrees.  If set to a positive value, runtime pointing
        code can choose to:
          * Reject boresight directions that lie within this angle
            of the Sun (for hardware safety), or
          * Flag such frames separately for "high dynamic range" or
            "torture test" simulations.
        A value of 0.0 disables Sun-exclusion constraints at the
        configuration level (runtime code may still impose its own
        safety limits if desired).

    Notes
    -----
    This configuration object is intentionally "dumb": it encodes
    only parameters and does not know how to interpret them.  The
    actual behavior of each pointing mode (e.g. how to compute the
    anti-Sun vector, how to check Earth limb clearance, how to
    enforce Sun exclusion) is left to NEBULA_POINTING.py and the
    rest of the runtime geometry pipeline.
    """

    # Store which pointing strategy to use (fixed ICRS vs anti-Sun stare).
    mode: PointingMode = PointingMode.ANTI_SUN_STARE

    # Store the ICRS right ascension of the boresight in degrees.
    # Used directly when mode == FIXED_ICRS_EARTH_AVOID; may be ignored
    # or used as an offset when mode == ANTI_SUN_STARE.
    boresight_ra_deg: float = 0.0

    # Store the ICRS declination of the boresight in degrees.
    # Used directly when mode == FIXED_ICRS_EARTH_AVOID; may be ignored
    # or used as an offset when mode == ANTI_SUN_STARE.
    boresight_dec_deg: float = 0.0

    # Store the roll angle of the focal plane about the boresight, in degrees.
    # This is interpreted consistently by downstream WCS / projection code
    # regardless of pointing mode.
    roll_deg: float = 0.0

    # Store an extra angular clearance from the Earth's limb, in degrees.
    # This supplements the geometric blocking radius R_BLOCK from
    # NEBULA_ENV_CONFIG.  A value of 0.0 means no additional clearance
    # beyond R_BLOCK.
    earth_avoid_margin_deg: float = 0.0

    # Flag indicating whether the sensor platform should be required to be
    # sunlit for a timestep to be considered "valid" for science.  Runtime
    # code can interpret this as an operational realism switch.
    require_sensor_sunlit: bool = False

    # Store the Sun-exclusion angle in degrees.  If > 0, runtime code may
    # choose to reject or specially handle boresight directions that lie
    # within this angular distance of the Sun direction.
    sun_exclusion_angle_deg: float = 0.0


# Define the default pointing configuration that NEBULA will use when a
# more specific configuration is not supplied elsewhere.  Here we take
# the default to be an anti-Sun stare mode, with zero limb margin and
# no Sun-exclusion enforced at the configuration level.
DEFAULT_POINTING_CONFIG: Final[PointingConfig] = PointingConfig(
    # Use the anti-Sun stare mode by default, matching the geometry
    # described in typical dawn–dusk sun-synchronous LEO + GEO
    # observation scenarios.
    mode=PointingMode.ANTI_SUN_GEO_BELT,
    # For ANTI_SUN_STARE, boresight_ra_deg is not used directly; we set
    # it to 0.0 as a neutral placeholder.
    boresight_ra_deg=0.0,
    # For ANTI_SUN_STARE, boresight_dec_deg is also not used directly;
    # we set it to 0.0 as a neutral placeholder.
    boresight_dec_deg=0.0,
    # Set the default roll of the focal plane to 0 degrees.  This
    # corresponds to a natural alignment of the detector axes with the
    # RA/Dec axes in the local tangent plane.
    roll_deg=0.0,
    # Set the additional Earth limb-avoidance margin to 0.0 degrees.
    # Earth blocking will still be enforced via R_BLOCK in the runtime
    # geometry, but no extra angular clearance is required here.
    earth_avoid_margin_deg=0.0,
    # Do not require the sensor platform to be sunlit by default.  This
    # keeps the default configuration focused on pure geometry; users
    # can enable this flag if they want to reflect power/ops constraints.
    require_sensor_sunlit=False,
    # Set the Sun-exclusion angle to 0.0 degrees by default, meaning
    # that this configuration does not impose any Sun-avoidance
    # constraint.  Runtime code may still impose safety limits if needed.
    sun_exclusion_angle_deg=0.0,
)


def describe_pointing_config(config: PointingConfig = DEFAULT_POINTING_CONFIG) -> str:
    """
    Build a human-readable multi-line string summarizing a PointingConfig.

    This helper mirrors the "describe_*" pattern used in
    NEBULA_TIME_CONFIG and is intended for logging, debugging, or
    configuration summaries.  It does not perform any computations;
    it simply formats the contents of a PointingConfig into a
    convenient string.

    Parameters
    ----------
    config : PointingConfig, optional
        The pointing configuration to describe.  If omitted, the
        module-level DEFAULT_POINTING_CONFIG is used.

    Returns
    -------
    str
        A multi-line string summarizing the pointing mode and its
        key tunable parameters.
    """
    # Build the first line of the description, indicating that this
    # text summarizes a pointing configuration.
    header = "Pointing configuration:\n"
    # Build a line describing the pointing mode as a string value.
    mode_line = f"  mode:                   {config.mode.value}\n"
    # Build a line summarizing the boresight ICRS right ascension.
    ra_line = f"  boresight_ra_deg:       {config.boresight_ra_deg:.6f}\n"
    # Build a line summarizing the boresight ICRS declination.
    dec_line = f"  boresight_dec_deg:      {config.boresight_dec_deg:.6f}\n"
    # Build a line summarizing the focal-plane roll angle.
    roll_line = f"  roll_deg:               {config.roll_deg:.3f}\n"
    # Build a line summarizing the Earth limb-avoidance margin.
    earth_margin_line = (
        f"  earth_avoid_margin_deg: {config.earth_avoid_margin_deg:.3f}\n"
    )
    # Build a line summarizing whether sensor sunlit status is required.
    sensor_sunlit_line = (
        f"  require_sensor_sunlit:  {bool(config.require_sensor_sunlit)}\n"
    )
    # Build a line summarizing the Sun-exclusion angle.
    sun_exclusion_line = (
        f"  sun_exclusion_angle_deg:{config.sun_exclusion_angle_deg:.3f}\n"
    )
    # Concatenate all of the individual lines into a single multi-line
    # string that can be returned to the caller.
    description = (
        header
        + mode_line
        + ra_line
        + dec_line
        + roll_line
        + earth_margin_line
        + sensor_sunlit_line
        + sun_exclusion_line
    )
    # Return the assembled description string to the caller.
    return description
