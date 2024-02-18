from setuptools import setup
__version__ = "1.0"



setup(
    name="polyfit",
    version=__version__,
    author="Robert Reischke",
    author_email="reischke@posteo.net",
    # url="",
    description="Fitting reaction ratios using non-linear least squares",
    # long_description="",
    install_requires=['numpy','matplotlib','scipy'],
    zip_safe=False,
)
