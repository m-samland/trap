import copy
import os

import numpy as np
from astropy import units as u
from trap.embed_shell import ipsh

import species
from species.read.read_filter import ReadFilter


class SpectralTemplate(object):
    def __init__(
            self, name, instrument, companion_modelbox, stellar_modelbox,
            wavelength_indices=None,
            correct_transmission=False,
            fit_offset=False,
            fit_slope=False,
            number_of_pca_regressors=0,
            use_spectral_correlation=True,
            species_database_directory=None):

        # Check if necessary information is available
        if instrument.instrument_type == 'ifu' and \
                (instrument.spectral_resolution is None or instrument.wavelengths is None):
            raise ValueError('Instrument spectral resolution or wavelength not set.')
        if instrument.instrument_type == 'photometry' and \
                (instrument.filters is None or instrument.wavelengths is None):
            raise ValueError('Instrument filter names not provided')

        self.name = name
        self.instrument = instrument
        self.wavelength_indices = wavelength_indices
        self.correct_transmission = correct_transmission
        self.fit_offset = fit_offset
        self.fit_slope = fit_slope
        self.number_of_pca_regressors = number_of_pca_regressors
        self.use_spectral_correlation = use_spectral_correlation
        self.species_database_directory = species_database_directory

        self.companion_modelbox = copy.deepcopy(companion_modelbox)

        if stellar_modelbox is None:
            # Produce flat spectrum
            stellar_modelbox = copy.deepcopy(self.companion_modelbox)
            stellar_modelbox.flux = np.ones(len(stellar_modelbox.wavelength))
        self.stellar_modelbox = copy.deepcopy(stellar_modelbox)

        wavelengths = instrument.wavelengths.to(u.micron)
        if wavelength_indices is not None:
            wavelengths = wavelengths[wavelength_indices]

        if instrument.instrument_type == 'ifu':
            if not correct_transmission:
                self.stellar_modelbox.smooth_spectrum(
                    spec_res=instrument.spectral_resolution)
                self.stellar_modelbox.resample_spectrum(
                    wavel_resample=wavelengths.value)
                self.companion_modelbox.smooth_spectrum(
                    spec_res=instrument.spectral_resolution)
                self.companion_modelbox.resample_spectrum(
                    wavel_resample=wavelengths.value)

                self.contrast_modelbox = copy.deepcopy(self.companion_modelbox)
                self.contrast_modelbox.flux = self.companion_modelbox.flux / self.stellar_modelbox.flux
            else:
                #     star_modelbox_lowres = star_read_model.get_model(
                #         model_param=star_model_param, spec_res=spec_res,  wavel_resample=wavelengths.value, smooth=True)
                #     flux_amp_file = '/home/masa4294/science/charis_sphere_reductions/IFS/observation/*_51_Eri/OBS_H/2015-09-25/extracted/fixed/lstsq/converted/flux_amplitude_calibrated.fits'
                #     # flux_amp_file = '/home/masa4294/science/charis_sphere_reductions/IFS/observation/HD_4113/OBS_YJ/2016-07-20/extracted/lstsq/converted/flux_amplitude_calibrated.fits'
                #     flux_amplitudes = fits.getdata(flux_amp_file)[analysis.wavelength_indices]
                #
                #     # For transmission curve, flux amplitudes should be normalized first and then median combined IMO, todo later
                #     flux_amplitude_median = np.median(flux_amplitudes, axis=1)
                #
                #     # median_flux_amplitudes = np.median(flux_amplitudes, axis=1)
                #     trans = flux_amplitude_median / star_modelbox_lowres.flux
                #     trans = trans / np.max(trans)
                #
                #     trans_func = interp1d(wavelengths, trans, kind='cubic', fill_value='extrapolate')
                #     wavelengths_interp = np.linspace(wavelengths[0], wavelengths[-1], 300)
                #
                #     plt.scatter(wavelengths, trans, label='overall transmission')
                #     plt.plot(wavelengths_interp, trans_func(wavelengths_interp), label='cubic spline')
                #     plt.xlabel("Wavelength (micron)")
                #     plt.ylabel("Transmission")
                #     plt.legend()
                #     plt.show()
                #
                #     transmission = trans_func(planet_modelbox.wavelength)
                #     transmission[transmission < 0] = 0.
                # else:
                #     transmission = np.ones_like(planet_modelbox.flux)
                #
                # contrast_model = (planet_modelbox.flux / star_modelbox.flux) * transmission
                raise NotImplementedError("Tranmission correction not yet implemented")

        if instrument.instrument_type == 'photometry':
            if wavelength_indices is not None:
                filters = np.array(self.instrument.filters)[wavelength_indices]
            else:
                filters = self.instrument.filters
            if not correct_transmission:
                # species.read.read_filter.ReadFilter(filter_name: str)

                self.stellar_modelbox.smooth_spectrum(spec_res=55)
                self.stellar_modelbox.resample_spectrum(
                    wavel_resample=np.linspace(0.9, 2.4, 200))
                self.companion_modelbox.smooth_spectrum(spec_res=55)
                self.companion_modelbox.resample_spectrum(
                    wavel_resample=np.linspace(0.9, 2.4, 200))

                self.contrast_modelbox = copy.deepcopy(companion_modelbox)
                self.contrast_modelbox.flux = self.companion_modelbox.flux / self.stellar_modelbox.flux
                contrasts = []
                wavelengths = []
                for filter_name in filters:
                    synphot = species.SyntheticPhotometry(filter_name)
                    wavelengths.append(
                        ReadFilter(filter_name).mean_wavelength())
                    star_phot, _ = synphot.spectrum_to_flux(
                        wavelength=self.stellar_modelbox.wavelength,
                        flux=self.stellar_modelbox.flux)
                    companion_phot, _ = synphot.spectrum_to_flux(
                        wavelength=self.companion_modelbox.wavelength,
                        flux=self.companion_modelbox.flux)
                    contrasts.append(companion_phot / star_phot)
                self.contrast_modelbox.flux = np.array(contrasts)
                self.contrast_modelbox.wavelength = np.array(wavelengths)

            else:
                raise NotImplementedError("Tranmission correction not yet implemented")

        self.mean_normalized_contrast_value = np.mean(self.contrast_modelbox.flux)
        self.normalized_contrast_modelbox = copy.deepcopy(self.contrast_modelbox)
        self.normalized_contrast_modelbox.flux = self.contrast_modelbox.flux / self.mean_normalized_contrast_value

    def plot_template(self, species_database_directory=None, output_path=None, plot_normalized=True):
        if plot_normalized:
            modelbox = self.normalized_contrast_modelbox
        else:
            modelbox = self.contrast_modelbox

        if self.instrument.instrument_type == 'photometry':
            filters = self.instrument.filters
            os.chdir(self.species_database_directory)
            xlim = None
            ylim = None
        else:
            xlim = (float(modelbox.wavelength[0]), float(modelbox.wavelength[-1]))
            ylim = (np.min(modelbox.flux), np.max(modelbox.flux))
            filters = None

        print(filters)
        species.plot_spectrum(
            boxes=[modelbox],
            filters=filters,
            offset=(-0.08, -0.04),
            scale=None,  # ('log', 'log'),
            # None,  # (np.min(modelbox.wavelength), np.max(modelbox.wavelength)),  # (0.8, 5.),
            xlim=xlim,
            ylim=ylim,
            title=f'{self.name}',
            legend={'loc': 'lower right', 'frameon': False, 'fontsize': 12.},
            output=output_path)
