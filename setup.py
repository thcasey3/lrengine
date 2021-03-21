import setuptools
from distutils.core import setup

with open("README.md", "r") as f:
    long_description = f.read()

with open("lrengine/version.py", "r") as f:
    # Define __version__
    exec(f.read())

setup(
    name="lrengine",
    packages=setuptools.find_packages(),
    version=__version__,
    license="MIT",
    description="lrengine - make sense of file or folder names",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Thomas Casey",
    url="",
    download_url="",
    project_urls={},
    keywords=["machine learning", "language recognition"],
    python_requires=">=3.6",
    install_requires=[
        "dateutils>=0.6.12",
        "numpy>=1.20.0",
        "scipy>=1.6.1",
        "matplotlib>=3.3.4",
        "pandas>=1.2.3",
        "scikit-learn>=0.24",
        "seaborn>=0.11.1",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    entry_points=dict(
        console_scripts=[
            # "hydrationGUI=dnplab.hydrationGUI:main_func",
        ]
    ),
)
