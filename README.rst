trap
====
Detection of exoplanets in direct imaging data by causal regression of temporal systematics
-------------------------------------------------------------------------------------------

TRAP is novel algorithm used to detect exoplanets in high-contrast imaging data, which is based on building a causal temporal systematic noise model to remove the stellar contamination obscuring a faint planet signal. Traditionally this has been done using image-based approaches instead of working
in the time-domain. The main benefit of this new approach is that it works significantly better for companion signals very close to the central star.
For a detailed description please refer to `Samland et al. 2021 <https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..24S/abstract>`_.

TRAP can currently only be installed from source and requires at least Python 3.5 or higher. We recommend using the "pip install" or "pip install -e ." command (if you want to make changes to the code base) in the directory containing the "setup.py" file.
TRAP is using Ray for multiprocessing, we recommend installing it via 'pip install -U "ray[default]"' to get access to the dashboard for monitoring utilities.

The package has been tested only on Linux/iOS with the packages provided in any recent version of Conda.
Please provide feedback if there are issues installing the package on your system.

A detailed documentation website is not available at the moment. Please refer to the in-code documentation and the tutorial notebook containing a simple example based on the test data included in the package.

Dependencies
------------
TRAP requires the following packages in a reasonably up-to-date version
to function:

- 'numpy', 'scipy', 'matplotlib', 'scikit-learn', 'numba', 'seaborn', 'tqdm', 'bottleneck', 'ray', 'natsorted'
- 'astropy', 'photutils'


Contributing
------------

This is the first (hopefully stable) release, but there is plenty to do! As the code grew over a lengthy development period and is mostly written by a single author, it still has many rough edges that could and should be refactored to be easier to understand and maintain. Please excuse if it still lacks elegance. It should be very easy to use however!

Please open a new issue or new pull request for bugs, feedback, or new features you would like to see.   If there is an issue you would like to work on, please leave a comment and we will be happy to assist.   New contributions and contributors are very welcome!

New to github or open source projects?  If you are unsure about where to start or haven't used github before, please feel free to email `@m-samland`_.

Feedback and feature requests?  Is there something missing you would like to see?  Please open an issue or send an email to  `@m-samland`_.

Acknowledgements
----------------

If you have found TRAP useful to your research, please cite Samland et al. 2021.

.. _@m-samland: https://github.com/m-samland
