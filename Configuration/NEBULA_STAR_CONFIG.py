"""
NEBULA_STAR_CONFIG.py

This configuration module defines how the NEBULA simulation chooses and
describes its stellar *catalog* and how it typically *queries* that
catalog around a field of view.

The intent is very narrow on purpose:

  - This file answers: "Which stars are we calling upon, and how do we
    usually ask for them?"

  - It does NOT configure:
      * diffuse sky background models,
      * telescope collecting area,
      * optical throughput,
      * detector noise.

    Those topics live in other NEBULA_* configuration modules
    (for example, a future NEBULA_BACKGROUND_CONFIG).

NEBULA code that needs star catalog information should import from this
file rather than hard-coding catalog names, band labels, or query
parameters.  This helps keep the "Gaia wiring" in one place.
"""

# ---------------------------------------------------------------------------
# Standard library imports
# ---------------------------------------------------------------------------

# Import the dataclass decorator so we can define small configuration
# classes that behave like lightweight objects with named fields instead
# of ad-hoc dictionaries.  This gives type hints and immutability for
# star catalog/query policy.
from dataclasses import dataclass

# Import typing helpers for optional types and lists / dicts so we can
# declare exactly which fields are optional and what the helper functions
# return.
from typing import Optional, List, Dict


# ---------------------------------------------------------------------------
# Star catalog configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StarCatalogConfig:
    """
    Describe a single stellar catalog + band that NEBULA can use.

    This class is a simple "configuration object" that groups together
    the metadata needed to understand a catalog:

      - a short internal name (e.g. "GAIA_DR3_G"),
      - a human-readable description,
      - the underlying table or dataset identifier (e.g. a Gaia TAP table),
      - which column holds the magnitude in the chosen band,
      - the band name itself ("G", "V", "r", etc.),
      - the catalog release/version ("DR2", "DR3", ...),
      - the astrometric reference epoch (e.g. 2016.0 for Gaia DR3),
      - optionally, which columns hold magnitudes in auxiliary bands
        (e.g. Gaia BP/RP).

    It does NOT fetch any data by itself; it is just a structured
    description that other NEBULA modules (e.g. NEBULA_QUERY_GAIA) can
    read when building queries or interpreting catalog results.
    """

    # Short internal name for this catalog configuration.
    # This is what you might log, or use in filenames, to tag which
    # star catalog + band was used for a particular NEBULA run.
    name: str

    # Human-readable description of the catalog and band.
    # Intended for logs, plots, or documentation so that a future reader
    # can immediately understand what "name" refers to in scientific terms.
    description: str

    # Identifier for the catalog table or dataset.
    # For Gaia this is typically a TAP table name like "gaiadr3.gaia_source".
    # For a local catalog it might be a filename template or database key.
    table: str

    # Name of the column that holds the apparent magnitude in the
    # chosen photometric band.
    #
    # For Gaia DR3 G-band this is "phot_g_mean_mag".  NEBULA modules
    # that convert magnitudes to photon flux will assume this column
    # contains the appropriate bandpass magnitudes.
    mag_column: str

    # Short band label (e.g. "G", "V", "r").
    # Used mainly for plot labels, log messages, and for building
    # human-readable strings like "Gaia DR3 G".
    band: str

    # Catalog release or version string (e.g. "DR3").
    # Keeping this explicit is important for reproducibility; star
    # positions, magnitudes, and error models can change between
    # releases.
    release: str

    # Astrometric reference epoch of this catalog, expressed as a decimal
    # year (e.g. 2016.0 for Gaia DR3).
    #
    # Gaia sky positions and proper motions are defined relative to this
    # epoch.  If NEBULA propagates stars from the catalog epoch to an
    # observation epoch, this is the "t0" in that propagation.
    reference_epoch: float

    # Optional column name for Gaia BP magnitudes, if available.
    # For Gaia DR3 this is typically "phot_bp_mean_mag" and corresponds
    # to a blue-ish broad band used for color information.
    bp_mag_column: Optional[str] = None

    # Optional column name for Gaia RP magnitudes, if available.
    # For Gaia DR3 this is typically "phot_rp_mean_mag" and corresponds
    # to a red-ish broad band used for color information.
    rp_mag_column: Optional[str] = None


# Define the specific catalog NEBULA will use for point-source stars
# by default.  Right now that is "Gaia DR3, G-band".
NEBULA_STAR_CATALOG = StarCatalogConfig(
    # Short internal name for this catalog configuration.  This string
    # will appear in logs and output filenames under NEBULA_OUTPUT/STARS.
    name="GAIA_DR3_G",

    # Human-facing description of what this catalog represents.
    # This is not used in the code logic but is very helpful for
    # debugging, logging, and figure captions.
    description="Gaia Data Release 3, G-band point-source catalog",

    # Underlying table or dataset identifier.  For astroquery.gaia this
    # is the TAP table name to query; for local catalogs it would be
    # the table/file you have mirrored.
    table="gaiadr3.gaia_source",

    # Column name containing the G-band magnitude in Gaia DR3.
    # This is the column NEBULA will treat as "mag_G" for flux
    # conversions in the radiometry pipeline.
    mag_column="phot_g_mean_mag",

    # Short band label for this configuration (e.g. G band).
    band="G",

    # Catalog release identifier.  Used purely as metadata for
    # reproducibility and sanity-checking against cached results.
    release="DR3",

    # Gaia DR3 reference epoch (positions / proper motions are referred
    # to 2016.0 in the Gaia documentation).
    reference_epoch=2016.0,

    # Optional auxiliary bands: Gaia BP and RP mean magnitudes.
    # Keeping these here allows NEBULA to request color information
    # from Gaia without hard-coding column names in multiple files.
    bp_mag_column="phot_bp_mean_mag",
    rp_mag_column="phot_rp_mean_mag",
)


# ---------------------------------------------------------------------------
# Star catalog query configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StarQueryConfig:
    """
    Default tuning parameters for how NEBULA queries the star catalog.

    This is another configuration object.  It does not perform any
    queries itself; it just encodes *policy*:

      - how far beyond the strict field-of-view we search (angular padding),
      - how much deeper than the sensor's limiting magnitude we ask
        the catalog to go (magnitude buffer),
      - a maximum row limit (to avoid unexpectedly huge queries),
      - whether or not NEBULA should attempt to use catalog proper
        motions when computing star positions at the observation epoch,
      - which extra columns we want (BP/RP, parallax, RUWE, variability),
      - which query engine we intend to use ("astroquery" vs "local").

    NEBULA_QUERY_GAIA (or any future catalog module) will import this
    config and apply these defaults when building Gaia queries.
    """

    # Extra angular padding around the strict FOV cone, in degrees.
    #
    # Example: if the half-angle of the field-of-view footprint for a
    # window is 3 deg and cone_padding_deg is 0.3, the query module
    # might ask for stars out to 3.3 deg, then later cull them exactly
    # using the projected sensor footprint on a per-frame basis.
    cone_padding_deg: float

    # Additional magnitude depth beyond the sensor's limiting magnitude.
    #
    # Example: if the sensor limit is mag 9.8 and mag_buffer is 0.5,
    # the query module could request stars down to mag 10.3 from the
    # catalog (sensor_limit + mag_buffer) to give some guard-band for
    # modeling or calibration before trimming faint stars.
    mag_buffer: float

    # Maximum number of catalog rows to accept from a single query.
    #
    # A value of -1 can be used to mean "no explicit limit" at the
    # configuration level, with the understanding that:
    #   - the remote TAP service may still enforce its own limit, and
    #   - NEBULA query code may add its own safety checks.
    max_rows: int

    # Flag indicating whether NEBULA should attempt to use proper-motion
    # information from the catalog when computing star positions at the
    # observation time.
    #
    # If True, catalog query modules are expected to request proper
    # motion columns (pmra, pmdec) and to propagate positions from the
    # catalog reference epoch (e.g. 2016.0) to t_obs.
    use_proper_motion: bool

    # Query mode: "astroquery" for remote Gaia TAP, "local" for local
    # mirrored Gaia shards.  NEBULA_QUERY_GAIA uses this to decide
    # which backend implementation to call.
    mode: str = "astroquery"

    # Whether to request BP/RP columns in Gaia queries.
    # If True, the query will ask for the columns named in
    # NEBULA_STAR_CATALOG.bp_mag_column and rp_mag_column (if they are
    # not None), so later flux computations can use color information.
    include_bp_rp: bool = True

    # Whether to request parallax in Gaia queries.
    # Parallax is not needed for pure sky-projected brightness, but
    # may be useful if you later care about distances or want to
    # filter out poorly measured stars.
    include_parallax: bool = False

    # Whether to request RUWE (Renormalised Unit Weight Error), a Gaia
    # astrometric quality indicator.  RUWE can be used to filter out
    # dubious astrometric solutions if desired.
    include_ruwe: bool = False

    # Whether to request the photometric variability flag.
    # This column indicates whether Gaia considers a source variable.
    # NEBULA uses VARIABLE_FLAG_MAPPING to compress the string values
    # into small integer codes if this flag is enabled.
    include_variability_flag: bool = False


# Default star-query policy for NEBULA.
# These values are based on what you previously used successfully in
# your Gaia query experiments (pad_deg = 0.3, mag_buffer = 0.5, etc.).
NEBULA_STAR_QUERY = StarQueryConfig(
    # Angular padding around the field of view [deg].  This inflates
    # the geometric cone used to query Gaia relative to the tight
    # sky footprint computed by NEBULA_SKY_SELECTOR.
    cone_padding_deg=0.3,

    # Extra depth to request in magnitude space (beyond the sensor's
    # own limiting magnitude) when building catalog queries.  The
    # actual query limit used by NEBULA_QUERY_GAIA is:
    #   mag_limit_G = sensor_star_mag_limit_G + mag_buffer
    mag_buffer=0.5,

    # Maximum number of catalog rows to accept from a query.  -1 here
    # indicates "no explicit config-level cap"; NEBULA_QUERY_GAIA may
    # still enforce a practical limit via Gaia.ROW_LIMIT, etc.
    max_rows=-1,

    # Whether NEBULA intends to use proper-motion information from the
    # catalog when computing star positions at the observation epoch.
    # If True, we will request pmra/pmdec and later propagate using
    # NEBULA_STAR_CATALOG.reference_epoch as the starting point.
    use_proper_motion=True,

    # Use remote Gaia TAP via astroquery by default.  If you later
    # mirror Gaia locally, you can change this to "local" and adjust
    # NEBULA_QUERY_GAIA accordingly.
    mode="astroquery",

    # Request BP/RP; you want these for future color-dependent flux
    # conversions (e.g. using Gaia G + BP-RP color to infer spectral
    # shape for sensor bandpass).
    include_bp_rp=True,

    # Start simple: do not request parallax / RUWE / variability
    # columns by default.  You can flip these to True when you need
    # them without touching NEBULA_QUERY_GAIA again.
    include_parallax=False,
    include_ruwe=False,
    include_variability_flag=False,
)


# ---------------------------------------------------------------------------
# Variability flag mapping for Gaia (phot_variable_flag)
# ---------------------------------------------------------------------------

# Map Gaia DR3 phot_variable_flag values (strings) to small integer codes
# for compact storage in NEBULA pickles.
#
# The exact string values are defined in the Gaia DR3 documentation; here
# we only care about a coarse encoding that distinguishes "no info",
# "variable", and "constant".  You can extend or refine this mapping
# later if you want more detailed variability classes in the simulation.
VARIABLE_FLAG_MAPPING: Dict[str, int] = {
    # No variability information / not classified.  Gaia uses several
    # NOT_AVAILABLE* variants; we compress all of them to code 0.
    "NOT_AVAILABLE": 0,
    "NOT_AVAILABLE_QUIESCENT": 0,
    "NOT_AVAILABLE_VARIABLE": 0,

    # Classified as variable (generic variability flag).
    "VARIABLE": 1,

    # Classified as quiescent / non-variable (explicitly constant).
    "CONSTANT": 2,
}


# ---------------------------------------------------------------------------
# Column selection helper for Gaia queries
# ---------------------------------------------------------------------------

def get_gaia_query_columns(use_proper_motion: bool = True) -> List[str]:
    """
    Return the canonical list of Gaia columns NEBULA should request.

    Parameters
    ----------
    use_proper_motion : bool, optional
        If True, include proper motion columns (pmra, pmdec) in the
        returned list. This should normally match NEBULA_STAR_QUERY.use_proper_motion.

    Returns
    -------
    List[str]
        List of Gaia column names suitable for use in a TAP query or
        local subset. The list always includes the core astrometric /
        photometric columns required by NEBULA_QUERY_GAIA.

    Notes
    -----
    - This function isolates all Gaia column-name decisions in one
      place so that modules like NEBULA_QUERY_GAIA do not hard-code
      column lists.
    - The returned list may contain optional columns (BP/RP, parallax,
      RUWE, variability) depending on NEBULA_STAR_QUERY flags.
    """
    # Shorthand handles to the catalog and query configs so we do not
    # repeatedly type the global names.
    cat = NEBULA_STAR_CATALOG
    q = NEBULA_STAR_QUERY

    # Start with the absolute minimum columns we always need:
    # - source_id: unique identifier for the Gaia source,
    # - ra, dec: sky position in ICRS at the reference epoch,
    # - cat.mag_column: band used for flux modeling (e.g. phot_g_mean_mag).
    columns: List[str] = [
        "source_id",
        "ra",
        "dec",
        cat.mag_column,
    ]

    # Optionally request BP/RP mean magnitudes if NEBULA_STAR_QUERY says
    # we care about color and the catalog knows which columns to use.
    if q.include_bp_rp:
        if cat.bp_mag_column is not None:
            columns.append(cat.bp_mag_column)
        if cat.rp_mag_column is not None:
            columns.append(cat.rp_mag_column)

    # Optionally request proper motions if both the argument and the
    # global query config agree that we intend to use proper motion.
    if use_proper_motion and q.use_proper_motion:
        columns.extend(["pmra", "pmdec"])

    # Optionally request parallax if enabled.  This is not required for
    # pure sky-projected brightness but may be useful for more advanced
    # modeling or filtering of problematic sources.
    if q.include_parallax:
        columns.append("parallax")

    # Optionally request RUWE (astrometric quality indicator) if enabled.
    if q.include_ruwe:
        columns.append("ruwe")

    # Optionally request the photometric variability flag if enabled.
    if q.include_variability_flag:
        columns.append("phot_variable_flag")

    # De-duplicate and drop any accidental Nones / empty strings to avoid
    # sending redundant or invalid column names to Gaia TAP.
    seen = set()
    cleaned: List[str] = []
    for col in columns:
        if col and col not in seen:
            seen.add(col)
            cleaned.append(col)

    # Return the cleaned, order-preserving list of column names.
    return cleaned


# ---------------------------------------------------------------------------
# Convenience helper for logging / debugging
# ---------------------------------------------------------------------------

def describe_star_configuration() -> str:
    """
    Construct a multi-line string summarizing the current star catalog
    and query configuration.

    Returns
    -------
    str
        Human-readable description of catalog + query policy, suitable
        for logging at NEBULA startup or when debugging.
    """
    # Local handles for brevity.
    cat = NEBULA_STAR_CATALOG
    q = NEBULA_STAR_QUERY

    # Build a line for auxiliary bands if we have BP/RP columns configured.
    bp_rp_line = ""
    if cat.bp_mag_column is not None or cat.rp_mag_column is not None:
        bp_rp_line = "  aux bands:   "
        if cat.bp_mag_column is not None:
            bp_rp_line += f"BP column: {cat.bp_mag_column}  "
        if cat.rp_mag_column is not None:
            bp_rp_line += f"RP column: {cat.rp_mag_column}\n"
        # Ensure trailing newline so the catalog_info string looks clean.
        if not bp_rp_line.endswith("\n"):
            bp_rp_line += "\n"

    # Build the catalog summary string with name, description, table, band,
    # and release.  This is primarily for logs / debugging.
    catalog_info = (
        "Star catalog:\n"
        f"  name:        {cat.name}\n"
        f"  description: {cat.description}\n"
        f"  table:       {cat.table}\n"
        f"  band:        {cat.band} (mag column: {cat.mag_column})\n"
        f"  release:     {cat.release}\n"
        f"  ref epoch:   {cat.reference_epoch:.1f}\n"
        f"{bp_rp_line}"
    )


    # Build the query-policy summary, showing angular padding, magnitude
    # buffer, row limit, and whether proper motion is intended to be used.
    query_info = (
        "Star query defaults:\n"
        f"  cone padding:      {q.cone_padding_deg:.2f} deg\n"
        f"  magnitude buffer:  {q.mag_buffer:.2f} mag\n"
        f"  max rows:          {q.max_rows}\n"
        f"  use proper motion: {q.use_proper_motion}\n"
        f"  mode:              {q.mode}\n"
        f"  include BP/RP:     {q.include_bp_rp}\n"
        f"  include parallax:  {q.include_parallax}\n"
        f"  include RUWE:      {q.include_ruwe}\n"
        f"  include var flag:  {q.include_variability_flag}\n"
    )

    # Return a single string containing both catalog and query info.
    return catalog_info + query_info


# How much to inflate the FOV radius when defining Gaia cone-search
# footprints (deg).  This safety margin is used in NEBULA_SKY_SELECTOR
# when building sky_radius_deg for each window, before any additional
# cone_padding_deg is applied at the catalog-query stage.
SAFETY_MARGIN_DEG: float = 0.2


def get_safety_margin_deg() -> float:
    """
    Return the default safety margin (deg) to inflate the FOV radius
    when defining Gaia cone-search footprints.

    This helper exists so that NEBULA_SKY_SELECTOR (and any other code)
    uses a single, centralized value for how much to "pad" the
    geometric field-of-view when building sky footprints for Gaia
    queries.
    """
    return SAFETY_MARGIN_DEG
