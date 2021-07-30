import setuptools
from distutils.core import setup

with open("README.md", "r") as f:
    long_description = f.read()

with open("lsframe/version.py", "r") as f:
    # Define __version__
    exec(f.read())

setup(
    name="lsframe",
    packages=setuptools.find_packages(),
    version=__version__,
    license="MIT",
    description="lsframe - map, classify, frame",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Thomas Casey",
    url="https://github.com/thcasey3/lsframe",
    download_url="https://github.com/thcasey3/lsframe/",
    project_urls={
        "Homepage": "https://thcasey3.github.io/lsframe/",
        "Documentation": "https://thcasey3.github.io/lsframe/",
        "Source": "https://github.com/thcasey3/lsframe/",
    },
    keywords=["classify language recognition"],
    python_requires=">=3.6",
    install_requires=[
        "python-dateutil>=2.6.1",
        "numpy>=1.13.3",
        "pandas>=1.0.4",
        "seaborn>=0.10",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    entry_points=dict(console_scripts=[]),
)
