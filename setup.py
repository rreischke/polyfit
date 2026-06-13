from setuptools import setup, find_packages

__version__ = "1.0"

setup(
    name="polyfit",
    version=__version__,
    author="Robert Reischke",
    author_email="reischke@posteo.net",
    # url="",
    description="Fitting reaction ratios using non-linear least squares",
    # long_description="",
    packages=find_packages(),
    py_modules=["gui"],
    install_requires=["numpy", "matplotlib", "scipy", "tqdm"],
    entry_points={
        # gui_scripts suppresses the terminal window on Windows;
        # on macOS/Linux it behaves identically to console_scripts.
        "gui_scripts": [
            "polyfit-gui = gui:main",
        ],
    },
    zip_safe=False,
)
