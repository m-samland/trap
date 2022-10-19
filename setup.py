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
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
    ],
    packages=setuptools.find_packages(),
    package_data={"": ["test_data/*.fits"]},
    include_package_data=True,
    python_requires='>=3.8',
    install_requires=['numpy', 'scipy', 'matplotlib', 'numba', 'pandas',
                      'scikit-learn', 'astropy', 'photutils', 'seaborn', 'tqdm',
                      'ray', 'bottleneck', 'natsort', 'species']
)
