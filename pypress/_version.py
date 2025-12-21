"""Version."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("pypress")
except PackageNotFoundError:
    # Package is not installed
    __version__ = "0.0.0+unknown"
