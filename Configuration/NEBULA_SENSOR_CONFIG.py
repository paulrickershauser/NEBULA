"""
NEBULA_SENSOR_CONFIG.py

Sensor configuration definitions for the NEBULA
(Neuromorphic Event-Based Luminance Asset-tracking) simulation framework.

This module defines a `SensorConfig` dataclass that describes the geometric,
radiometric, and neuromorphic properties of a camera (pixel grid, field of
view, pixel pitch, aperture, contrast threshold, etc.), along with a single
predefined instance for the EVK4 event-based sensor used in NEBULA.

Other modules should import and use these objects rather than hard-coding
sensor parameters directly.
"""

from dataclasses import dataclass
from typing import Optional
import math


@dataclass
class SensorConfig:
    """
    Configuration parameters for a single sensor + optical system.

    This class is intentionally focused on *sensor properties* and provides
    a few convenience properties (e.g., pixel scale, collecting area) so that
    higher-level simulation code can query the sensor directly instead of
    re-deriving these every time.

    The radiometry-related attributes (aperture, throughput, quantum
    efficiency) are kept simple and band-averaged. Detailed wavelength-
    dependent modeling (QE(λ), T_opt(λ)) should live in a more advanced
    instrument model if needed.
    """

    # ------------------------------------------------------------------
    # Core geometry
    # ------------------------------------------------------------------

    rows: int
    """Number of detector pixel rows (vertical dimension)."""

    cols: int
    """Number of detector pixel columns (horizontal dimension)."""

    pixel_pitch: float
    """
    Physical size of one detector pixel in meters.

    For the Prophesee/Sony IMX636 sensor used in EVK4, the pixel size is
    4.86 x 4.86 µm.【turn35search5†L1-L7】【turn35search17†L8-L13】
    """

    focal_length: float
    """
    Effective focal length of the optical system in meters.

    This describes the combined telescope + relay optics, not just the
    sensor package itself.
    """

    fov_deg: float
    """
    Horizontal field-of-view in degrees.

    This is the on-sky FOV produced by the combination of focal length,
    sensor size, and any additional optics. It is used to derive the
    approximate angular pixel scale.
    """

    # ------------------------------------------------------------------
    # Detection / sensitivity (photometric-level)
    # ------------------------------------------------------------------

    mag_limit: float
    """
    Limiting apparent magnitude for reliable detections in the chosen band.

    This is an *approximate* performance figure that depends on telescope
    aperture, sky brightness, tracking, and processing. For event-based
    cameras used with small SSA telescopes, recent studies report limiting
    magnitudes around G~9.7–10 for point sources.【turn35search6†L1-L6】【turn35search20†L1-L3】
    """

    # ------------------------------------------------------------------
    # Radiometry hooks (aperture + efficiency)
    # ------------------------------------------------------------------

    aperture_diameter_m: Optional[float] = None
    """
    Diameter of the entrance pupil / telescope primary, in meters.

    If provided, this is used to derive the collecting area via:

        A_collect = π * (aperture_diameter_m / 2)^2

    If both `aperture_diameter_m` and `aperture_area_m2` are None, the
    radiometry code must supply an effective collecting area from elsewhere.
    """

    aperture_area_m2: Optional[float] = None
    """
    Effective collecting area in square meters.

    If this is set directly, it takes precedence over `aperture_diameter_m`
    when computing the collecting area. This can be useful if obstructions
    (secondary mirrors, baffles) significantly reduce the geometric area.
    """

    optical_throughput: Optional[float] = None
    """
    Band-averaged optical throughput (0–1).

    This lumps together mirror/lens transmission, filter losses, and any
    other optical inefficiencies. Typical small telescopes with a few
    elements often have an overall throughput in the 0.3–0.6 range.
    """

    quantum_efficiency: Optional[float] = None
    """
    Band-averaged detector quantum efficiency (0–1).

    For modern back-illuminated CMOS sensors, QE near 600–700 nm can reach
    ~0.6–0.8. When set, this value will be used by radiometry utilities to
    convert incident photon flux [photons/s/m^2] into detected electrons or
    events per second.
    """

    # ------------------------------------------------------------------
    # Neuromorphic / event-based parameters (optional)
    # ------------------------------------------------------------------

    contrast_threshold: Optional[float] = None
    """
    Minimum *relative* log-intensity change required to trigger an event.

    For EVK4 (IMX636), the nominal contrast threshold is specified as
    ~25% intensity change in the datasheet.【turn35search17†L8-L13】

    If this is None, the sensor can be treated as a conventional
    frame-based device in simulations that do not model events.
    """

    refractory_period_us: Optional[float] = None
    """
    Minimum time between events on the same pixel, in microseconds.

    This models pixel-level dead-time / saturation behavior. For EVK4, a
    value on the order of a few hundred microseconds is a reasonable
    starting point (comparable to the camera's quoted latency).【turn35search17†L8-L13】

    If this is None, no refractory dead-time is modeled.
    """
    psf_fwhm_pix: Optional[float] = None
    """
    Assumed PSF FWHM in detector pixels for point-like sources.

    Used by image-generation utilities (e.g. NEBULA_PHOTUTILS_WORKFLOW)
    to spread each source over a 2D Gaussian PSF. If None, a delta-function
    (single-pixel) PSF is used.
    """

    # ------------------------------------------------------------------
    # Derived convenience properties (object-oriented "behavior")
    # ------------------------------------------------------------------

    @property
    def aspect_ratio(self) -> float:
        """
        Image aspect ratio, defined as width / height.

        Useful for projection logic or plotting code that needs to
        know how "wide" the sensor is relative to its height.
        """
        return self.cols / self.rows

    @property
    def fov_rad(self) -> float:
        """
        Horizontal field-of-view in radians.

        This is simply the degree FOV converted to radians and is
        handy for rate calculations and small-angle approximations.
        """
        return math.radians(self.fov_deg)

    @property
    def pixel_scale_rad(self) -> float:
        """
        Approximate horizontal angular pixel scale in radians per pixel.

        For small fields of view, this is well approximated by:

            pixel_scale ≈ fov_rad / cols

        This is useful for converting on-sky angular rates (deg/s,
        rad/s) into approximate pixel rates.
        """
        return self.fov_rad / self.cols

    @property
    def pixel_scale_deg(self) -> float:
        """
        Approximate horizontal angular pixel scale in degrees per pixel.

        Convenience wrapper around `fov_deg / cols` so that calling
        code doesn't have to duplicate this logic.
        """
        return self.fov_deg / self.cols

    @property
    def collecting_area_m2(self) -> Optional[float]:
        """
        Effective collecting area in m^2, combining aperture geometry.

        Priority:
            1. If `aperture_area_m2` is set, return it directly.
            2. Else if `aperture_diameter_m` is set, compute π (D/2)^2.
            3. Else return None and let higher-level code decide.
        """
        if self.aperture_area_m2 is not None:
            return self.aperture_area_m2
        if self.aperture_diameter_m is not None:
            radius = 0.5 * self.aperture_diameter_m
            return math.pi * radius * radius
        return None

    @property
    def is_event_based(self) -> bool:
        """
        True if this sensor is configured as a neuromorphic / event-based device.

        This is a simple check that both the contrast threshold and
        refractory period are set, which is often useful for switching
        behavior in downstream simulation code.
        """
        return (
            self.contrast_threshold is not None
            and self.refractory_period_us is not None
        )


# ----------------------------------------------------------------------
# EVK4 sensor configuration
# ----------------------------------------------------------------------

# Prophesee Metavision EVK4 HD with Sony IMX636 event sensor.
# Specs (public sources):
#   • Resolution: 1280 x 720 pixels【turn35search5†L1-L7】【turn35search9†L1-L4】
#   • Pixel size: 4.86 x 4.86 µm【turn35search5†L1-L7】【turn35search17†L8-L13】
#   • Nominal contrast threshold: ~25% intensity change【turn35search17†L8-L13】
#   • Dynamic range: >86 dB【turn35search17†L8-L13】
#
# The focal_length, fov_deg, aperture, throughput, and QE are *system-level*
# parameters that depend on the telescope and optics you pair with EVK4.
# The values below are reasonable starting points for a small SSA telescope
# and can be tuned to match your actual hardware.

EVK4_SENSOR = SensorConfig(
    # Geometry tied to the sensor silicon itself
    rows=720,
    cols=1280,
    pixel_pitch=4.86e-6,   # 4.86 µm pixels

    # Optical system (telescope) parameters
    focal_length=1.0,      # [m] placeholder: 1 m effective focal length
    fov_deg=6.0,           # [deg] horizontal FOV used in NEBULA sims

    # Photometric / sensitivity level
    mag_limit=10.0,        # Approx. G-band limiting magnitude with small aperture【turn35search6†L1-L6】【turn35search20†L1-L3】

    # Aperture + efficiency (tunable; these are reasonable first guesses)
    aperture_diameter_m=0.3,     # 30 cm class telescope (adjust as needed)
    aperture_area_m2=None,       # derived from diameter unless overridden
    optical_throughput=0.5,      # 50% band-averaged throughput (mirrors + optics)
    quantum_efficiency=0.6,      # 60% band-averaged QE in visible (approx.)
    
    # New: PSF width (tunable)
    psf_fwhm_pix=2.0,   # ~2-pixel FWHM Gaussian PSF (adjust later)
   # If later you want to tie this to a physical angular resolution, you can compute:
    # plate scale
    # ≈
    # pixel pitch
    # 𝑓
    # eff
    #  
    # [rad/pix]
    # plate scale≈
    # f
    # eff
    # 	​
    
    # pixel pitch
    # 	​
    
    # [rad/pix]
    
    # and then
    # FWHM_pix = FWHM_arcsec / plate_scale_arcsec_per_pix, where plate_scale_arcsec_per_pix = plate_scale_rad_per_pix * 206265.

    # Event-based behavior
    contrast_threshold=0.25,     # 25% nominal intensity change【turn35search17†L8-L13】
    refractory_period_us=300.0,  # dead-time ~ few 10^2 µs (tunable)
)
# ----------------------------------------------------------------------
# Gen-3 VGA-CD sensor configuration (McMahon-Crabtree & Monet 2021)
# ----------------------------------------------------------------------
# Prophesee Gen-3 VGA-CD used in [McMahon-Crabtree & Monet 2021] with
# an 85 mm f/1.4 lens for star-field tests.
#
#   • Resolution: 640 x 480
#   • Pixel size: 15 µm
#   • Lens: 85 mm, f/1.4
#   • Limiting mag (dark sky, this setup): ~9.8
#
# Derived geometry:
#   sensor_width  = 640 * 15 µm = 9.6 mm
#   sensor_height = 480 * 15 µm = 7.2 mm
#   FOV_horiz ≈ 2 arctan( (9.6 mm / 2) / 85 mm ) ≈ 6.5 deg

GEN3_VGA_CD_SENSOR = SensorConfig(
    # Geometry tied to Gen-3 VGA-CD silicon
    rows=480,
    cols=640,
    pixel_pitch=15.0e-6,     # 15 µm pixels

    # Optical system: 85 mm f/1.4 lens
    focal_length=0.085,      # [m]
    # Horizontal FOV computed from sensor size and focal length
    fov_deg=6.5,             # [deg] ≈ 2 * atan(9.6 mm / (2 * 85 mm))

    # Photometric / sensitivity level for this setup
    mag_limit=9.8,           # limiting mag from on-sky tests

    # Aperture + efficiency
    aperture_diameter_m=0.085 / 1.4,  # ≈ 0.0607 m clear aperture
    aperture_area_m2=None,            # derived from diameter
    optical_throughput=0.5,           # placeholder; tune vs data

    # For pure photon-flux work, leave QE None and set eta_eff=1.0
    # in NEBULA_FLUX. Later, if you want electrons, you can set QE
    # and use eta_eff = throughput * QE.
    quantum_efficiency=None,

    # PSF: extremely undersampled; a 1-pixel FWHM Gaussian is a
    # reasonable spot model for synthetic CCD-like frames.
    psf_fwhm_pix=1.0,

    # Event-camera behavior (approximate; tune if needed)
    contrast_threshold=0.12,          # ~12% contrast (Gen-3 docs)
    refractory_period_us=300.0,
)

# ----------------------------------------------------------------------
# Default "active" sensor
# ----------------------------------------------------------------------
# Downstream code can import ACTIVE_SENSOR instead of a specific
# device if you want to switch sensors in one place.
ACTIVE_SENSOR = GEN3_VGA_CD_SENSOR

def log_active_sensor(logger):
    logger.info(
        "ACTIVE_SENSOR: %s (%d x %d, pixel_pitch=%.2e m, f=%.3f m, FOV=%.2f deg)",
        ACTIVE_SENSOR.name,
        ACTIVE_SENSOR.rows,
        ACTIVE_SENSOR.cols,
        ACTIVE_SENSOR.pixel_pitch,
        ACTIVE_SENSOR.focal_length,
        ACTIVE_SENSOR.fov_deg,
    )
