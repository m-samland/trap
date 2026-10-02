"""Build the 51 Eri b IRDIS DB_K12 test data from a spherical reduction.

The output, ``51eri_irdis_db_k12_2015-09-24_K1.fits`` (or ``_K2`` with ``--channel 1``),
is a cut-out of one channel of the VLT/SPHERE-IRDIS DB_K12 sequence of 51 Eridani from
the night of 2015-09-24, all 256 frames, taken from the spherical reference reduction of
that night (spherical ``tests/regression/run_51eri_irdis_reference.py``). It carries
everything spherical passes to TRAP for this observation, with the leading wavelength
axis kept so the arrays go to TRAP unchanged:

================  =========================  ================================================
Extension         Shape                      Content
================  =========================  ================================================
``SCI``           (1, 256, size, size) f4    coronagraphic cube (``coro_cube.fits``)
``IVAR``          (1, 256, size, size) f4    inverse variance (``coro_ivar_cube.fits``)
``BADPIX``        (1, size, size) u1         bad-pixel map, 1 = bad
``CENTERS``       (1, 256, 2) f8             per-frame star centre (x, y) in the cut-out
``PSF``           (1, 57, 57) f4             unsaturated flux PSF, mean over flux frames
``DEROT_ANGLE``   (256,) f8                  ``DEROT ANGLE`` of ``frames_info_coro.csv``
``WAVELENGTH``    (1,) f8                    channel wavelength in nm
``TRANSMISSION``  (N, 2) f8                  coronagraph transmission: separation (mas),
                                             throughput
================  =========================  ================================================

The crop follows spherical's convention (``apply_crop`` in
``spherical/pipeline/steps/irdis_preprocess.py``): one fixed box for the whole sequence,
with origin ``round(star) - size // 2`` around the median star position and an odd
``size``, so the star lies within about half a pixel of the pixel TRAP treats as the
image centre. The centres are shifted into the cut-out, not re-sampled: the data are
cut, never interpolated. ``size`` leaves room for TRAP's own crop for a search region
out to 43 px (133 px here, including the centre drift), so TRAP never clamps it.

This script reads spherical's output files only and does not import spherical::

    python examples/test_data/make_51eri_irdis_db_k12.py \\
        ~/data/sphere/reduction/IRDIS/observation/'*_51_Eri'/DB_K12/2015-09-24/converted \\
        ~/data/sphere/reduction/IRDIS/trap/'*_51_Eri'/DB_K12/2015-09-24/provenance.json \\
        ../spherical/src/spherical/pipeline/data/N_ALC_JYH_S-IRDIS_H23-transmission.txt

``provenance.json`` is the record the reference driver writes; its package versions
and commits go into the primary header, and the script refuses products that were not
written during that run.
"""

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

CROP_SIZE = 137
CHANNEL_NAMES = ("K1", "K2")
OUTPUT_PATTERN = "51eri_irdis_db_k12_2015-09-24_{}.fits"
PROVENANCE_PACKAGES = ("spherical", "trap", "charis")
PRODUCTS = (
    "coro_cube.fits",
    "coro_ivar_cube.fits",
    "badpixel_map.fits",
    "image_centers_fitted_robust.fits",
    "psf_cube_for_postprocessing.fits",
    "frames_info_coro.csv",
    "wavelengths.fits",
)


def check_products(converted, provenance):
    """Exit unless every product was written during the run that ``provenance`` records."""
    start = datetime.datetime.fromisoformat(provenance["started_utc"]).timestamp()
    end = datetime.datetime.fromisoformat(provenance["finished_utc"]).timestamp()
    stale = [name for name in PRODUCTS if not start <= (converted / name).stat().st_mtime <= end]
    if stale:
        sys.exit(
            "These products were not written by the run in provenance.json, so its "
            f"versions would not describe them: {', '.join(stale)}"
        )


def provenance_cards(provenance):
    """Header cards with the reference run's time span and package versions."""
    cards = [
        ("RUNSTART", provenance["started_utc"], "spherical reference run start (UTC)"),
        ("RUNEND", provenance["finished_utc"], "spherical reference run end (UTC)"),
    ]
    for name in PROVENANCE_PACKAGES:
        record = provenance["packages"][name]
        key = name[:5].upper()
        commit = (record.get("vcs_info") or {}).get("commit_id", "")
        cards.append((f"{key}VER", record["version"], f"{name} version"))
        cards.append((f"{key}SHA", commit, f"{name} commit"))
    return cards


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("converted", type=Path, help="spherical 'converted' directory")
    parser.add_argument("provenance", type=Path, help="provenance.json of the reference run")
    parser.add_argument("transmission", type=Path, help="N_ALC_JYH_S-IRDIS_H23-transmission.txt")
    parser.add_argument("--channel", type=int, choices=(0, 1), default=0, help="0 = K1, 1 = K2")
    parser.add_argument("--output", type=Path, help="default: next to this script")
    args = parser.parse_args()
    channel = args.channel
    output = args.output or Path(__file__).parent / OUTPUT_PATTERN.format(CHANNEL_NAMES[channel])

    provenance = json.loads(args.provenance.read_text())
    check_products(args.converted, provenance)

    centers_full = fits.getdata(args.converted / "image_centers_fitted_robust.fits")[channel].astype("f8")
    x0, y0 = np.round(np.nanmedian(centers_full, axis=0)).astype(int) - CROP_SIZE // 2
    centers = centers_full - [x0, y0]

    def cut(name, dtype):
        with fits.open(args.converted / name, memmap=True) as hdul:
            box = hdul[0].data[channel, ..., y0:y0 + CROP_SIZE, x0:x0 + CROP_SIZE]
            return np.array(box, dtype=dtype)[None]

    sci = cut("coro_cube.fits", "f4")
    ivar = cut("coro_ivar_cube.fits", "f4")
    badpix = cut("badpixel_map.fits", "u1")
    psf = np.nanmean(fits.getdata(args.converted / "psf_cube_for_postprocessing.fits")[channel], axis=0)
    derot = pd.read_csv(args.converted / "frames_info_coro.csv")["DEROT ANGLE"].to_numpy("f8")
    wavelength = fits.getdata(args.converted / "wavelengths.fits").astype("f8")[channel]
    transmission = np.loadtxt(args.transmission)

    if sci.shape[-1] != CROP_SIZE or not (sci.shape[1] == centers.shape[0] == derot.size):
        sys.exit(
            f"shapes disagree: cube {sci.shape}, centres {centers.shape}, angles {derot.shape}; "
            "is the crop inside the frame?"
        )

    primary = fits.PrimaryHDU()
    header = primary.header
    header["OBJECT"] = ("51 Eri", "target")
    header["INSTRUME"] = ("SPHERE-IRDIS", "instrument")
    header["FILTER"] = ("DB_K12", "dual-band filter")
    header["CHANNEL"] = (CHANNEL_NAMES[channel], f"channel {channel} of DB_K12")
    header["NIGHT"] = ("2015-09-24", "night start")
    header["PIXSCALE"] = (0.01225, "arcsec per pixel")
    header["DEROT"] = ("PUPIL", "derotator mode (ADI)")
    header["CROPSIZE"] = (CROP_SIZE, "side of the cut-out (px)")
    header["CROPX"] = (int(x0), "cut-out origin x in the 1024 px frame")
    header["CROPY"] = (int(y0), "cut-out origin y in the 1024 px frame")
    for key, value, comment in provenance_cards(provenance):
        header[key] = (value, comment)
    header["BUILDER"] = ("make_51eri_irdis_db_k12.py", "script in examples/test_data")

    fits.HDUList([
        primary,
        fits.ImageHDU(sci, name="SCI"),
        fits.ImageHDU(ivar, name="IVAR"),
        fits.ImageHDU(badpix, name="BADPIX"),
        fits.ImageHDU(centers[None], name="CENTERS"),
        fits.ImageHDU(psf[None].astype("f4"), name="PSF"),
        fits.ImageHDU(derot, name="DEROT_ANGLE"),
        fits.ImageHDU(np.array([wavelength]), name="WAVELENGTH"),
        fits.ImageHDU(transmission, name="TRANSMISSION"),
    ]).writeto(output, overwrite=True)
    print(f"wrote {output} ({output.stat().st_size / 2**20:.1f} MiB)")


if __name__ == "__main__":
    main()
