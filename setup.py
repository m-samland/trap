import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="trap",
    version="1.0.0",
    author="Matthias Samland",
    author_email="m.samland@mailbox.org",
    description="Detection of exoplanets in direct imaging data by causal regression of temporal systematics",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/m-samland/trap",
    packages=setuptools.find_packages(),
    # data_files=[('test_data', ['test_data/science_cube.fits',
    #                            'test_data/psf_model.fits',
    #                            'test_data/parallactic_angles.fits'])],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
    ],
    python_requires='>=3.5',
    install_requires=['numpy', 'scipy', 'matplotlib', 'numba', 'pandas',
                      'scikit-learn', 'astropy', 'photutils', 'seaborn', 'tqdm',
                      'ray', 'bottleneck']
)
