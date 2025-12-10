"""
NEBULA_WCS.py

Astropy-based World Coordinate System (WCS) helpers for the NEBULA
(Neuromorphic Event-Based Luminance Asset-tracking) simulation framework.

This module provides a thin wrapper around `astropy.wcs.WCS` for
turning pointing information (boresight RA/Dec + focal-plane roll)
and sensor geometry (rows, cols, FOV) into a self-consistent World
Coordinate System for projecting celestial coordinates (RA, Dec) onto
detector pixel coordinates.

In particular, this module defines:

Classes
-------
NebulaWCS
    A small convenience wrapper that holds an `astropy.wcs.WCS` instance
    along with the sensor shape (rows, cols) and provides methods for:
        - world_to_pixel(ra, dec) -> (x_pix, y_pix)
        - world_to_pixel_skycoord(coords) -> (x_pix, y_pix)
        - pixel_to_world(x_pix, y_pix) -> astropy.coordinates.SkyCoord

Functions
---------
make_tan_wcs_from_pointing(boresight_ra_deg, boresight_dec_deg,
                           roll_deg, sensor_config) -> NebulaWCS
    Build a 2D TAN-projection WCS centered on the given boresight
    direction with a specified focal-plane roll, using the geometry
    from a NEBULA SensorConfig (rows, cols, FOV).

build_wcs_for_observer(observer_track, sensor_config) -> NebulaWCS | list[NebulaWCS]
    Construct one or more NebulaWCS objects for a given observer track
    produced by NEBULA_SCHEDULE_PICKLER, using that track's stored
    pointing fields:
        - pointing_boresight_ra_deg
        - pointing_boresight_dec_deg
        - roll_deg

project_radec_to_pixels(wcs_obj, ra_deg, dec_deg) -> (x_pix, y_pix)
    Convenience function that forwards to NebulaWCS.world_to_pixel for
    code that prefers a functional style.

Conventions
-----------
- Celestial coordinates are ICRS right ascension (RA) and declination (Dec)
  expressed in degrees.

- The WCS is a standard "RA---TAN", "DEC--TAN" gnomonic projection.

- The reference world coordinate (CRVAL1, CRVAL2) is the boresight RA/Dec.

- The reference pixel (CRPIX1, CRPIX2) is the center of the detector:
      CRPIX1 = (cols + 1) / 2
      CRPIX2 = (rows + 1) / 2

- The pixel scale is derived from the sensor's horizontal field of view
  and number of columns:
      pixel_scale_deg = sensor_config.pixel_scale_deg

- The CD matrix is chosen such that:
      * Right ascension increases to the LEFT in image display
        (CD1_1 < 0 at zero roll),
      * Declination increases UP in the world coordinate sense
        (CD2_2 > 0 at zero roll),
      * A positive `roll_deg` corresponds to a right-hand rotation of
        the focal plane about the boresight, consistent with the
        PointingConfig documentation.

- Pixel coordinates follow the Astropy / FITS convention where
  (x_pix, y_pix) are zero-based when using origin=0 with
  WCS.all_world2pix / all_pix2world.  The mapping between these pixel
  coordinates and NumPy array indices is handled by calling code
  (typically: frame[int(y_pix), int(x_pix)]).
"""

# Import __future__ annotations so type hints can refer to classes defined later.
from __future__ import annotations

# Import dataclass for defining small, structured containers.
from dataclasses import dataclass

# Import typing helpers for clearer function signatures.
from typing import Any, List, Sequence, Tuple, Union

# Import NumPy for vectorized array handling and math utilities.
import numpy as np

# Import Astropy WCS for FITS-style world-to-pixel transformations.
from astropy.wcs import WCS

# Import SkyCoord and units for high-level celestial coordinate handling.
from astropy.coordinates import SkyCoord
import astropy.units as u

# Import the NEBULA sensor configuration dataclass and default EVK4 instance.
from Configuration.NEBULA_SENSOR_CONFIG import SensorConfig, ACTIVE_SENSOR


# --------------------------------------------------------------------------- #
# Internal helpers for working with NEBULA "track" objects
# --------------------------------------------------------------------------- #

def _get_track_field(track: Any, field_name: str, required: bool = True) -> Any:
    """
    Safely extract a field from a NEBULA "track" object.

    This helper mirrors the behavior of the similarly named function
    in NEBULA_SKYFIELD_ILLUMINATION but is redefined here to avoid
    heavy dependencies on radiometry and ephemeris loading.

    Parameters
    ----------
    track : Any
        A NEBULA track object, which may be:
          * a dict-like object with string keys, or
          * an object with attributes corresponding to field names.

    field_name : str
        Name of the field to extract from the track.

    required : bool, optional
        If True (default), raise an error if the field is missing.
        If False, return None when the field is not found.

    Returns
    -------
    Any
        The value associated with the requested field.

    Raises
    ------
    KeyError
        If `track` is dict-like and the field is missing and required=True.

    AttributeError
        If `track` is object-like and the field is missing and required=True.
    """
    # Check if the track is a dict-like object.
    if isinstance(track, dict):
        # If the field is present in the dict, return it.
        if field_name in track:
            return track[field_name]
        # If the field is required and missing, raise a KeyError.
        if required:
            raise KeyError(f"Track dict missing required field '{field_name}'")
        # If not required and missing, return None.
        return None

    # If the track is not a dict, check for an attribute with the given name.
    if hasattr(track, field_name):
        # If the attribute exists, return its value.
        return getattr(track, field_name)

    # If the attribute is missing and required, raise an AttributeError.
    if required:
        raise AttributeError(f"Track object missing required field '{field_name}'")

    # If the attribute is not required and missing, return None.
    return None


# --------------------------------------------------------------------------- #
# NebulaWCS wrapper class
# --------------------------------------------------------------------------- #

@dataclass
class NebulaWCS:
    """
    Thin wrapper around an Astropy `WCS` instance for NEBULA.

    This class stores:
        - the underlying `astropy.wcs.WCS` object,
        - the detector dimensions (rows, cols),

    and provides convenience methods for transforming between celestial
    coordinates (RA, Dec) and detector pixel coordinates (x, y).

    Attributes
    ----------
    wcs : astropy.wcs.WCS
        The underlying Astropy WCS object configured with TAN projection
        and RA/Dec axes.

    rows : int
        Number of detector rows (vertical dimension of the sensor).

    cols : int
        Number of detector columns (horizontal dimension of the sensor).

    Notes
    -----
    - All RA and Dec values are in degrees in the ICRS frame.
    - Pixel coordinates are zero-based and intended to be compatible
      with NumPy array indexing when using origin=0 in WCS calls.
    """

    # Store the underlying Astropy WCS object.
    wcs: WCS

    # Store the number of sensor rows (vertical dimension).
    rows: int

    # Store the number of sensor columns (horizontal dimension).
    cols: int

    def world_to_pixel(
        self,
        ra: Union[float, Sequence[float], np.ndarray, SkyCoord],
        dec: Union[float, Sequence[float], np.ndarray, None] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert celestial coordinates (RA, Dec) to detector pixel coordinates.

        This method accepts either:
            - a SkyCoord object (ICRS RA/Dec), or
            - separate RA and Dec values in degrees.

        Parameters
        ----------
        ra : float, array-like, or astropy.coordinates.SkyCoord
            If a SkyCoord, must contain ICRS RA/Dec values.  In this case,
            the `dec` argument is ignored.  If a float or array-like, it is
            interpreted as right ascension in degrees.

        dec : float or array-like, optional
            Declination in degrees.  This must be provided if `ra` is not a
            SkyCoord.  If `ra` is a SkyCoord, this argument is ignored.

        Returns
        -------
        x_pix : numpy.ndarray
            Array of pixel x-coordinates (zero-based, columns).

        y_pix : numpy.ndarray
            Array of pixel y-coordinates (zero-based, rows).

        Notes
        -----
        - Internally uses the Astropy low-level WCS method `all_world2pix`
          with `origin=0` to match NumPy-style zero-based indexing.
        - The output arrays can be used directly to index a 2D frame via
          `frame[y_pix_int, x_pix_int]` after rounding or flooring as
          appropriate.
        """
        # If `ra` is provided as a SkyCoord, extract RA/Dec in degrees.
        if isinstance(ra, SkyCoord):
            # Extract right ascension in degrees from the SkyCoord.
            ra_deg = ra.ra.to_value(u.deg)
            # Extract declination in degrees from the SkyCoord.
            dec_deg = ra.dec.to_value(u.deg)
        else:
            # Convert RA input to a NumPy array of floats.
            ra_deg = np.asarray(ra, dtype=float)
            # If `dec` is None in this branch, the caller forgot to supply it.
            if dec is None:
                raise ValueError(
                    "dec must be provided when ra is not a SkyCoord instance."
                )
            # Convert Dec input to a NumPy array of floats.
            dec_deg = np.asarray(dec, dtype=float)

        # Call the low-level Astropy WCS transformation from world to pixel.
        # Use origin=0 to obtain zero-based pixel coordinates suitable for
        # direct use with NumPy arrays.
        x_pix, y_pix = self.wcs.all_world2pix(ra_deg, dec_deg, 0)

        # Ensure the outputs are NumPy arrays (even if scalars were input).
        x_pix = np.asarray(x_pix, dtype=float)
        y_pix = np.asarray(y_pix, dtype=float)

        # Return the pair of pixel coordinate arrays.
        return x_pix, y_pix

    def world_to_pixel_skycoord(
        self,
        coords: SkyCoord,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convenience wrapper: convert SkyCoord positions to pixel coordinates.

        Parameters
        ----------
        coords : astropy.coordinates.SkyCoord
            Celestial coordinates (typically ICRS RA/Dec) of sources to be
            projected onto the sensor.

        Returns
        -------
        x_pix : numpy.ndarray
            Array of pixel x-coordinates (zero-based, columns).

        y_pix : numpy.ndarray
            Array of pixel y-coordinates (zero-based, rows).

        Notes
        -----
        - This simply forwards to `world_to_pixel` with the SkyCoord input.
        """
        # Delegate the work to the `world_to_pixel` method, which already
        # handles SkyCoord inputs and unit conversion.
        return self.world_to_pixel(coords)

    def pixel_to_world(
        self,
        x_pix: Union[float, Sequence[float], np.ndarray],
        y_pix: Union[float, Sequence[float], np.ndarray],
    ) -> SkyCoord:
        """
        Convert detector pixel coordinates (x, y) back to celestial coordinates.

        Parameters
        ----------
        x_pix : float or array-like
            Pixel x-coordinate(s), using zero-based indexing (columns).

        y_pix : float or array-like
            Pixel y-coordinate(s), using zero-based indexing (rows).

        Returns
        -------
        coords : astropy.coordinates.SkyCoord
            SkyCoord object containing the corresponding celestial
            coordinates (ICRS RA/Dec in degrees).

        Notes
        -----
        - Internally uses the Astropy low-level WCS method `all_pix2world`
          with `origin=0` to match NumPy-style zero-based indexing.
        - The returned SkyCoord can be used directly with other Astropy
          coordinate utilities or converted to RA/Dec arrays using
          `.ra.deg` and `.dec.deg`.
        """
        # Convert x pixel coordinates to a NumPy array of floats.
        x_arr = np.asarray(x_pix, dtype=float)
        # Convert y pixel coordinates to a NumPy array of floats.
        y_arr = np.asarray(y_pix, dtype=float)

        # Use the low-level Astropy WCS method to transform pixel coordinates
        # back to world coordinates (RA, Dec in degrees).  Set origin=0 for
        # zero-based pixel coordinates.
        ra_deg, dec_deg = self.wcs.all_pix2world(x_arr, y_arr, 0)

        # Wrap the resulting RA/Dec arrays in a SkyCoord in the ICRS frame.
        coords = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")

        # Return the SkyCoord object with celestial coordinates.
        return coords


# --------------------------------------------------------------------------- #
# WCS builder: from pointing config + sensor geometry
# --------------------------------------------------------------------------- #

def make_tan_wcs_from_pointing(
    boresight_ra_deg: float,
    boresight_dec_deg: float,
    roll_deg: float,
    sensor_config: SensorConfig,
) -> NebulaWCS:
    """
    Build a 2D TAN-projection WCS for a NEBULA sensor from pointing parameters.

    This function constructs a simple celestial WCS with:
        - Projection: "RA---TAN", "DEC--TAN" (gnomonic).
        - World reference point (CRVAL): boresight RA/Dec.
        - Pixel reference point (CRPIX): center of the detector.
        - Pixel scale: derived from sensor_config.pixel_scale_deg.
        - Orientation: RA-left and Dec-up at zero roll, with a focal-plane
          roll specified by `roll_deg`.

    Parameters
    ----------
    boresight_ra_deg : float
        Right ascension of the sensor boresight in the ICRS frame, in degrees.

    boresight_dec_deg : float
        Declination of the sensor boresight in the ICRS frame, in degrees.

    roll_deg : float
        Focal-plane roll angle about the boresight, in degrees.
        The convention matches PointingConfig:
        - roll_deg = 0 means the detector +y axis aligns with increasing
          Declination in the local tangent plane.
        - Positive roll is a right-hand rotation about the boresight.

    sensor_config : SensorConfig
        NEBULA SensorConfig instance describing the sensor geometry,
        particularly:
            - sensor_config.rows : number of detector rows,
            - sensor_config.cols : number of detector columns,
            - sensor_config.pixel_scale_deg : horizontal angular pixel
              scale in degrees per pixel.

    Returns
    -------
    nebula_wcs : NebulaWCS
        A NebulaWCS instance containing the configured Astropy WCS and the
        sensor dimensions (rows, cols).

    Notes
    -----
    - The CD matrix is built using a standard separation of scale and
      rotation:
          * base pixel scales: CDELT1 = -scale, CDELT2 = +scale,
          * rotation matrix based on roll_deg,
          * CD = diag(CDELT1, CDELT2) @ PC, where PC is a 2x2 rotation.

      This yields RA increasing to the left and Dec increasing up at
      zero roll, with roll_deg implementing an additional focal-plane
      rotation.
    """
    # Extract the number of sensor rows (vertical dimension).
    rows = int(sensor_config.rows)
    # Extract the number of sensor columns (horizontal dimension).
    cols = int(sensor_config.cols)

    # Convert the boresight right ascension to a float.
    ra0 = float(boresight_ra_deg)
    # Convert the boresight declination to a float.
    dec0 = float(boresight_dec_deg)
    # Convert the roll angle in degrees to a float.
    roll = float(roll_deg)

    # Obtain the approximate horizontal pixel scale in degrees per pixel
    # from the sensor configuration (fov_deg / cols).
    pixel_scale_deg = float(sensor_config.pixel_scale_deg)

    # Define the base CDELT values for each axis.
    # CDELT1 < 0 enforces RA increasing to the left at zero roll.
    cdelt1 = -pixel_scale_deg
    # CDELT2 > 0 enforces Dec increasing upward at zero roll.
    cdelt2 = +pixel_scale_deg

    # Convert the roll angle from degrees to radians for trigonometric functions.
    theta = np.deg2rad(roll)

    # Build the 2x2 rotation matrix for the PC terms using the standard
    # convention where a positive angle corresponds to a right-hand
    # rotation.  For small angles, this behaves like a standard 2D
    # rotation in the local tangent plane.
    pc11 = np.cos(theta)
    pc12 = np.sin(theta)
    pc21 = -np.sin(theta)
    pc22 = np.cos(theta)

    # Combine the scale (CDELTi) and rotation (PC) into a CD matrix.
    # CD = diag(CDELT1, CDELT2) @ PC
    cd11 = cdelt1 * pc11
    cd12 = cdelt1 * pc12
    cd21 = cdelt2 * pc21
    cd22 = cdelt2 * pc22

    # Create a new 2D Astropy WCS object.
    w = WCS(naxis=2)

    # Set the coordinate types for each axis to RA/Dec with TAN projection.
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    # Set the world coordinate units to degrees for both axes.
    w.wcs.cunit = ["deg", "deg"]

    # Set the reference world coordinate (boresight RA/Dec) at the reference pixel.
    w.wcs.crval = [ra0, dec0]

    # Place the reference pixel at the center of the detector.
    w.wcs.crpix = [(cols + 1.0) / 2.0, (rows + 1.0) / 2.0]

    # Set the CD matrix elements to encode scale + rotation.
    w.wcs.cd = np.array([[cd11, cd12],
                         [cd21, cd22]])

    # Optionally, annotate the reference frame as ICRS for clarity.
    w.wcs.radesys = "ICRS"

    # Create the NebulaWCS wrapper with the configured WCS and sensor size.
    nebula_wcs = NebulaWCS(wcs=w, rows=rows, cols=cols)

    # Return the constructed NebulaWCS object.
    return nebula_wcs


# --------------------------------------------------------------------------- #
# WCS builder for an entire observer track
# --------------------------------------------------------------------------- #

def build_wcs_for_observer(
    observer_track: Any,
    sensor_config: SensorConfig = ACTIVE_SENSOR,
) -> Union[NebulaWCS, List[NebulaWCS]]:
    """
    Build one or more NebulaWCS objects for a single NEBULA observer track.

    This function inspects the pointing fields attached to an observer
    track (typically produced by NEBULA_SCHEDULE_PICKLER and one of
    the pointing dispatcher functions), and constructs WCS objects that
    can be used to project sources into detector pixel coordinates.

    The observer_track is expected to carry, at minimum, the following
    fields (either as dict keys or object attributes):

        - "pointing_boresight_ra_deg"
        - "pointing_boresight_dec_deg"
        - "roll_deg"

    These may be scalars (for a fixed pointing) or arrays matching the
    length of observer_track.times (for time-varying pointing).

    Parameters
    ----------
    observer_track : Any
        NEBULA observer track object produced by NEBULA_SAT_PICKLER and
        NEBULA_SCHEDULE_PICKLER.  Must support either dict-like or
        attribute-style access for the pointing fields.

    sensor_config : SensorConfig, optional
        Sensor configuration to use when building the WCS.  Defaults to
        the ACTIVE_SENSOR instance imported from NEBULA_SENSOR_CONFIG.

    Returns
    -------
    wcs_out : NebulaWCS or list[NebulaWCS]
        If the boresight RA/Dec and roll are constant in time, a single
        NebulaWCS instance is returned.  If they vary with time, a list
        of NebulaWCS objects is returned, one per timestep, aligned with
        observer_track.times.

    Raises
    ------
    ValueError
        If the pointing fields have inconsistent lengths or incompatible
        shapes that prevent per-timestep WCS construction.
    """
    # Extract the boresight RA field from the observer track.
    ra_field = _get_track_field(observer_track, "pointing_boresight_ra_deg")
    # Extract the boresight Dec field from the observer track.
    dec_field = _get_track_field(observer_track, "pointing_boresight_dec_deg")
    # Extract the focal-plane roll field from the observer track.
    roll_field = _get_track_field(observer_track, "roll_deg")

    # Convert the RA, Dec, and roll fields to NumPy arrays for inspection.
    ra_arr = np.asarray(ra_field)
    dec_arr = np.asarray(dec_field)
    roll_arr = np.asarray(roll_field)

    # Determine whether the fields are time-varying by checking for
    # non-scalar shapes.
    ra_is_scalar = ra_arr.ndim == 0
    dec_is_scalar = dec_arr.ndim == 0
    roll_is_scalar = roll_arr.ndim == 0

    # If all three fields are scalar, build a single static WCS.
    if ra_is_scalar and dec_is_scalar and roll_is_scalar:
        # Build a single NebulaWCS using the scalar pointing values.
        wcs_single = make_tan_wcs_from_pointing(
            boresight_ra_deg=float(ra_arr),
            boresight_dec_deg=float(dec_arr),
            roll_deg=float(roll_arr),
            sensor_config=sensor_config,
        )
        # Return the static NebulaWCS.
        return wcs_single

    # At least one field is array-like; we will build per-timestep WCS objects.

    # Determine the "target" shape for time-varying pointing fields:
    # use the shape of the first non-scalar among (RA, Dec, roll).
    target_shape = None
    for arr, is_scalar in ((ra_arr, ra_is_scalar),
                           (dec_arr, dec_is_scalar),
                           (roll_arr, roll_is_scalar)):
        if not is_scalar:
            target_shape = arr.shape
            break

    if target_shape is None:
        # This should be impossible because the all-scalar case returned above,
        # but keep a defensive check.
        raise ValueError(
            "build_wcs_for_observer: could not infer target_shape for "
            "time-varying WCS construction."
        )

    # Broadcast any scalar pointing fields to the target_shape so we can
    # treat everything uniformly as arrays.
    if ra_is_scalar:
        ra_arr = np.full(target_shape, float(ra_arr))
    if dec_is_scalar:
        dec_arr = np.full(target_shape, float(dec_arr))
    if roll_is_scalar:
        roll_arr = np.full(target_shape, float(roll_arr))

    # Now all three should have the same shape; if not, raise an error.
    if not (ra_arr.shape == dec_arr.shape == roll_arr.shape):
        raise ValueError(
            "Inconsistent shapes for pointing fields on observer_track: "
            f"ra shape={ra_arr.shape}, dec shape={dec_arr.shape}, "
            f"roll shape={roll_arr.shape}"
        )


    # Flatten the arrays to 1D for iteration while preserving ordering.
    ra_flat = ra_arr.ravel()
    dec_flat = dec_arr.ravel()
    roll_flat = roll_arr.ravel()

    # Initialize a list to hold per-timestep NebulaWCS objects.
    wcs_list: List[NebulaWCS] = []

    # Iterate over each timestep index and build a WCS.
    for ra_val, dec_val, roll_val in zip(ra_flat, dec_flat, roll_flat):
        # For each timestep, build a NebulaWCS using the scalar pointing.
        wcs_t = make_tan_wcs_from_pointing(
            boresight_ra_deg=float(ra_val),
            boresight_dec_deg=float(dec_val),
            roll_deg=float(roll_val),
            sensor_config=sensor_config,
        )
        # Append the per-timestep WCS to the list.
        wcs_list.append(wcs_t)

    # Return the list of NebulaWCS objects, one per timestep.
    return wcs_list


# --------------------------------------------------------------------------- #
# Convenience functional interface
# --------------------------------------------------------------------------- #

def project_radec_to_pixels(
    nebula_wcs: NebulaWCS,
    ra_deg: Union[float, Sequence[float], np.ndarray],
    dec_deg: Union[float, Sequence[float], np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project RA/Dec coordinates (in degrees) to pixel coordinates using NebulaWCS.

    This is a convenience wrapper for code that prefers a functional
    style rather than calling methods directly on the NebulaWCS object.

    Parameters
    ----------
    nebula_wcs : NebulaWCS
        NebulaWCS instance that encapsulates the underlying Astropy WCS
        and sensor geometry.

    ra_deg : float or array-like
        Right ascension(s) in degrees (ICRS frame).

    dec_deg : float or array-like
        Declination(s) in degrees (ICRS frame).

    Returns
    -------
    x_pix : numpy.ndarray
        Pixel x-coordinate(s) corresponding to the input RA/Dec, zero-based.

    y_pix : numpy.ndarray
        Pixel y-coordinate(s) corresponding to the input RA/Dec, zero-based.

    Notes
    -----
    - Internally delegates to NebulaWCS.world_to_pixel.
    """
    # Delegate the RA/Dec to pixel projection to the NebulaWCS instance.
    x_pix, y_pix = nebula_wcs.world_to_pixel(ra_deg, dec_deg)

    # Return the resulting pixel coordinate arrays.
    return x_pix, y_pix
