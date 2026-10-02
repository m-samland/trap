"""Tests for pointing species at a database directory without changing the cwd.

The bug these guard against (m-samland/trap#39) was invisible in isolation: species
was configured by chdir'ing into its database directory, so every relative path the
calling pipeline had configured silently changed meaning for the rest of the
process. The assertions therefore care as much about what the process cwd is
*after* the call, and about what did *not* appear in it, as about the files that
get created.
"""

import os
from configparser import ConfigParser
from pathlib import Path

import pytest

from trap import species_setup
from trap.species_setup import configure_species


def _write_config(species_dir: Path, database: str, data_folder: str = "./data/") -> Path:
    """Write a species config by hand, as an earlier chdir-based run would have."""
    species_dir.mkdir(parents=True, exist_ok=True)
    config_file = species_dir / "species_config.ini"
    config = ConfigParser()
    config.add_section("species")
    config.set("species", "database", database)
    config.set("species", "data_folder", data_folder)
    config.set("species", "vega_mag", "0.03")
    with open(config_file, "w", encoding="utf-8") as file_obj:
        config.write(file_obj)
    return config_file


def _read_config(species_dir: Path) -> ConfigParser:
    config = ConfigParser()
    config.read(species_dir / "species_config.ini")
    return config


@pytest.fixture(autouse=True)
def _isolated_species_config(monkeypatch):
    """Keep SPECIES_CONFIG changes from leaking between tests or into the session."""
    monkeypatch.delenv("SPECIES_CONFIG", raising=False)


class TestConfigureSpecies:
    def test_leaves_the_working_directory_untouched(self, tmp_path, monkeypatch):
        workdir = tmp_path / "pipeline_cwd"
        workdir.mkdir()
        monkeypatch.chdir(workdir)

        configure_species(tmp_path / "species")

        assert Path.cwd() == workdir

    def test_creates_the_database_in_the_requested_directory(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        species_dir = tmp_path / "species"

        returned = configure_species(species_dir)

        assert returned == species_dir
        assert (species_dir / "species_config.ini").is_file()
        assert (species_dir / "species_database.hdf5").is_file()
        # The cwd is where the old chdir-based version would have put them.
        assert not (tmp_path / "species_config.ini").exists()

    def test_exports_the_config_path_for_later_species_calls(self, tmp_path):
        species_dir = tmp_path / "species"

        configure_species(species_dir)

        # Database(), ReadFilter() and friends read this variable and only fall back
        # to the cwd when it is unset.
        assert os.environ["SPECIES_CONFIG"] == str(species_dir / "species_config.ini")

    def test_is_idempotent(self, tmp_path):
        species_dir = tmp_path / "species"

        configure_species(species_dir)
        first = (species_dir / "species_config.ini").read_text()
        configure_species(species_dir)

        assert (species_dir / "species_config.ini").read_text() == first

    def test_skips_reinitialization_once_configured(self, tmp_path, monkeypatch):
        """SpeciesInit prints a banner, queries PyPI, and rewrites the HDF5."""
        species_dir = tmp_path / "species"
        configure_species(species_dir)

        calls = []
        monkeypatch.setattr(species_setup, "SpeciesInit", lambda **kwargs: calls.append(kwargs))
        configure_species(species_dir)

        assert calls == []

    def test_reinitializes_for_a_different_directory(self, tmp_path, monkeypatch):
        configure_species(tmp_path / "species")

        calls = []
        monkeypatch.setattr(species_setup, "SpeciesInit", lambda **kwargs: calls.append(kwargs))
        configure_species(tmp_path / "other_species")

        assert len(calls) == 1

    def test_resolves_a_relative_directory_against_the_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        returned = configure_species("species")

        assert returned == tmp_path / "species"
        assert returned.is_absolute()

    def test_expands_user(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

        returned = configure_species("~/species")

        assert returned == tmp_path / "species"

    def test_rejects_a_missing_directory_argument(self):
        with pytest.raises(ValueError, match="species database directory"):
            configure_species(None)


class TestDataFolder:
    """The data folder is what the chdir used to place correctly by accident.

    SpeciesInit has no ``data_folder`` argument: it writes the relative ``./data/``
    into the config and then creates that path relative to the cwd. Downloaders
    read the value back verbatim and do not create it, so an unanchored config
    means both a stray folder in the caller's cwd and a FileNotFoundError on the
    first download.
    """

    def test_anchors_the_data_folder_so_downloads_ignore_the_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        species_dir = tmp_path / "species"

        configure_species(species_dir)

        data_folder = Path(_read_config(species_dir)["species"]["data_folder"])

        assert data_folder.is_absolute()
        assert data_folder == species_dir / "data"

    def test_creates_the_data_folder_it_points_at(self, tmp_path):
        """add_model_grid mkdirs a subdirectory of this without ``parents=True``."""
        species_dir = tmp_path / "species"

        configure_species(species_dir)

        assert (species_dir / "data").is_dir()

    def test_creates_no_data_folder_in_the_working_directory(self, tmp_path, monkeypatch):
        workdir = tmp_path / "pipeline_cwd"
        workdir.mkdir()
        monkeypatch.chdir(workdir)

        configure_species(tmp_path / "species")

        assert list(workdir.iterdir()) == []

    def test_anchors_an_existing_relative_data_folder(self, tmp_path):
        """An existing database keeps its layout, it just stops depending on the cwd."""
        species_dir = tmp_path / "species"
        _write_config(species_dir, database="species_database.hdf5")

        configure_species(species_dir)

        assert Path(_read_config(species_dir)["species"]["data_folder"]) == species_dir / "data"

    def test_keeps_a_data_folder_that_is_already_absolute(self, tmp_path):
        species_dir = tmp_path / "species"
        shared = tmp_path / "shared_data"
        _write_config(species_dir, database="species_database.hdf5", data_folder=str(shared))

        configure_species(species_dir)

        assert Path(_read_config(species_dir)["species"]["data_folder"]) == shared
        assert shared.is_dir()


class TestExistingDatabase:
    """A database written by an earlier run must be adopted, not re-initialized.

    SpeciesInit rewrites the HDF5's ``configuration`` group, so running it against a
    populated database makes every process a writer against a file other processes
    may be reading.
    """

    def test_adopts_an_existing_database_without_reinitializing(self, tmp_path, monkeypatch):
        species_dir = tmp_path / "species"
        _write_config(species_dir, database="species_database.hdf5")
        (species_dir / "species_database.hdf5").write_bytes(b"pre-existing")

        calls = []
        monkeypatch.setattr(species_setup, "SpeciesInit", lambda **kw: calls.append(kw))
        configure_species(species_dir)

        assert calls == []
        assert (species_dir / "species_database.hdf5").read_bytes() == b"pre-existing"
        assert os.environ["SPECIES_CONFIG"] == str(species_dir / "species_config.ini")

    def test_keeps_a_database_the_config_points_elsewhere(self, tmp_path, monkeypatch):
        """The config, not the directory layout, decides which database is used."""
        species_dir = tmp_path / "species"
        elsewhere = tmp_path / "shared" / "species_database.hdf5"
        elsewhere.parent.mkdir(parents=True)
        elsewhere.write_bytes(b"shared")
        _write_config(species_dir, database=str(elsewhere))

        calls = []
        monkeypatch.setattr(species_setup, "SpeciesInit", lambda **kw: calls.append(kw))
        configure_species(species_dir)

        assert calls == []
        assert _read_config(species_dir)["species"]["database"] == str(elsewhere)
        assert not (species_dir / "species_database.hdf5").exists()

    def test_anchors_a_relative_database_path(self, tmp_path, monkeypatch):
        """A chdir-era config names the database relatively.

        ``Database`` hands the value straight to ``h5py.File``, so without the cwd
        being the species directory a relative path resolves against whatever the
        reduction happens to be running in.
        """
        species_dir = tmp_path / "species"
        _write_config(species_dir, database="species_database.hdf5")
        (species_dir / "species_database.hdf5").write_bytes(b"pre-existing")

        reduction_cwd = tmp_path / "reduction"
        reduction_cwd.mkdir()
        monkeypatch.chdir(reduction_cwd)

        calls = []
        monkeypatch.setattr(species_setup, "SpeciesInit", lambda **kw: calls.append(kw))
        configure_species(species_dir)

        database = Path(_read_config(species_dir)["species"]["database"])
        assert database == species_dir / "species_database.hdf5"
        assert database.is_file()
        # Still adopted, not re-initialized, despite the rewritten path.
        assert calls == []

    def test_initializes_when_the_config_names_a_missing_database(self, tmp_path):
        """A config without its database is not a usable install; initialize it."""
        species_dir = tmp_path / "species"
        _write_config(species_dir, database="species_database.hdf5")

        configure_species(species_dir)

        assert (species_dir / "species_database.hdf5").is_file()

    def test_recovers_from_an_unparseable_config(self, tmp_path):
        """A run killed mid-write leaves a truncated config; rewrite it, do not raise."""
        species_dir = tmp_path / "species"
        species_dir.mkdir()
        (species_dir / "species_config.ini").write_text("not an ini file\n")

        configure_species(species_dir)

        assert _read_config(species_dir)["species"]["database"]
        assert (species_dir / "species_database.hdf5").is_file()
