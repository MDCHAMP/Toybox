from setuptools import setup, find_packages


# The text of the README file
with open('README.md') as f:
    rm = f.read()

# This call to setup() does all the work
setup(
    name="MDOF-toybox",
    version="0.1.1",
    description="Tooling to generate toy MDOF dynamics data for arbitrary linear and nonlinear systems in python",
    long_description=rm,
    long_description_content_type="text/markdown",
    url="https://github.com/MDCHAMP/Toybox",
    author="Max Champneys",
    author_email="max.champneys@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
    ],
    packages=['toybox'],
    package_dir={'':'src'},
    include_package_data=False,
    install_requires=[
        "numpy"
    ],
)