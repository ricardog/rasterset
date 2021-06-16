from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="rasterset",
    version="0.3.1",
    author="Ricardo E. Gonzalez",
    author_email="ricardog@itinerisinc.com",
    description="Operate on groups of rasters as as a set",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.7",
    url="https://github.com/ricardog/rasterset",
    project_urls={
        "Bug Tracker": "https://github.com/ricardog/rasterset/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    install_requires=[
        "asciitree",
        "blosc",
        "cftime",
        "Click",
        "dask[distributed]",
        "lz4",
        "numpy",
        "pandas",
        "r2py @ git+https://github.com/ricardog/r2py.git",
        "rasterio",
        "rioxarray",
        "setuptools",
        "tqdm",
    ],
    extras_require={
        "dev": ["black", "flake8", "pytest"],
        "test": ["fiona"]
    },
)
