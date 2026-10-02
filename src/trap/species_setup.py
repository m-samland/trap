"""Point ``species`` at an explicit database directory without changing the cwd.

``species`` locates its configuration through the ``SPECIES_CONFIG`` environment
variable, falling back to ``species_config.ini`` in the current working directory
when that variable is unset. TRAP used to satisfy that fallback by chdir'ing into
the species database directory and never chdir'ing back, which silently
reinterpreted every relative path a caller had configured, most visibly by writing
the whole ``template_matching/`` tree under the species directory
(https://github.com/m-samland/trap/issues/39).

Setting the variable once, from an absolute config path, removes the cwd from the
picture for the entire species surface TRAP touches: ``Database``, ``ReadModel``,
``ReadFilter``, ``SyntheticPhotometry`` and ``plot_spectrum`` all read it.

The one thing the cwd used to provide for free was the location of the data folder.
``SpeciesInit`` has no ``data_folder`` argument, writes the relative ``./data/``
into a config it creates, and then creates that folder relative to the cwd. The
config it writes is used verbatim by every downloader afterwards. So the config
here is written *before* ``SpeciesInit`` runs, with an absolute ``data_folder``,
which is what the old chdir achieved by accident.
"""

import logging
import os
from configparser import ConfigParser
from configparser import Error as ConfigParserError
from pathlib import Path

from species import SpeciesInit

logger = logging.getLogger(__name__)

# Defaults copied from SpeciesInit, so a config written here is one it accepts.
_DEFAULT_DATABASE_NAME = "species_database.hdf5"
_DEFAULT_DATA_FOLDER = "./data/"
_DEFAULT_VEGA_MAG = "0.03"


def configure_species(species_database_directory) -> Path:
    """Point ``species`` at *species_database_directory*, initializing if needed.

    Exports ``SPECIES_CONFIG`` and leaves the process working directory untouched.
    An existing database is adopted as it stands, including a config that names a
    database somewhere else; ``SpeciesInit`` runs only when there is nothing to
    adopt, because it rewrites the HDF5's ``configuration`` group and so counts as
    a writer against a database other processes may be reading. Safe to call
    repeatedly.

    Parameters
    ----------
    species_database_directory : str or Path
        Directory holding ``species_config.ini`` and ``species_database.hdf5``.

    Returns
    -------
    Path
        Absolute path to the species database directory.

    Raises
    ------
    ValueError
        If *species_database_directory* is None.
    """
    if species_database_directory is None:
        raise ValueError("Need to specify species database directory.")

    directory = Path(species_database_directory).expanduser().absolute()
    directory.mkdir(parents=True, exist_ok=True)

    config_file = directory / "species_config.ini"
    database_file, data_folder = _write_anchored_config(config_file, directory)

    if not database_file.is_file():
        # str, not Path: before species 0.11.0, SpeciesInit exported
        # SPECIES_CONFIG only for a str config_file, and every later
        # Database()/ReadFilter() then fell back to the cwd.
        SpeciesInit(config_file=str(config_file), database_file=str(database_file))

    # Downloaders write into this folder without creating it: add_model_grid does a
    # parentless mkdir of a subdirectory, and the SVO filter fetch urlretrieves
    # straight into it.
    data_folder.mkdir(parents=True, exist_ok=True)

    os.environ["SPECIES_CONFIG"] = str(config_file)

    return directory


def _write_anchored_config(config_file: Path, directory: Path) -> tuple[Path, Path]:
    """Ensure *config_file* exists and its paths are absolute.

    Returns the absolute ``(database, data_folder)``. Both are used verbatim by
    species (``Database`` hands its ``database`` straight to ``h5py.File``), and a
    config written by an earlier chdir-based run holds relative values that only
    resolved because the cwd was the species directory. Anchoring them to
    *directory* is what lets such a database keep working. A value that is already
    absolute is left alone, so a config deliberately pointing somewhere else
    survives.
    """
    config = _read_config(config_file)

    if not config.has_section("species"):
        config = ConfigParser(allow_no_value=True)
        config.add_section("species")

    section = config["species"]
    changed = False

    if "vega_mag" not in section:
        section["vega_mag"] = _DEFAULT_VEGA_MAG
        changed = True

    anchored = {}
    for key, default in (
        ("database", _DEFAULT_DATABASE_NAME),
        ("data_folder", _DEFAULT_DATA_FOLDER),
    ):
        value = Path(section.get(key, default))
        if not value.is_absolute():
            value = (directory / value).absolute()
            section[key] = str(value)
            changed = True
        anchored[key] = value

    if changed:
        _write_config(config_file, config)

    return anchored["database"], anchored["data_folder"]


def _read_config(config_file: Path) -> ConfigParser:
    """Parse *config_file*, treating an unreadable one as absent.

    A run killed mid-write leaves a truncated config behind. Rewriting it is
    recoverable; letting configparser raise from deep inside a reduction is not.
    """
    config = ConfigParser(allow_no_value=True)

    if not config_file.is_file():
        return config

    try:
        config.read(config_file)
    except ConfigParserError:
        logger.warning(
            "Could not parse %s; rewriting it from defaults.", config_file
        )
        return ConfigParser(allow_no_value=True)

    return config


def _write_config(config_file: Path, config: ConfigParser) -> None:
    """Write *config* to *config_file* atomically.

    Via a temporary file in the same directory so a reader never sees the
    truncated window that a plain ``open(..., "w")`` opens up.
    """
    temporary = config_file.with_name(f"{config_file.name}.{os.getpid()}.tmp")

    with open(temporary, "w", encoding="utf-8") as file_obj:
        config.write(file_obj)

    os.replace(temporary, config_file)
