import logging

from . import _version

logging.getLogger("trap").addHandler(logging.NullHandler())

try:
    __version__ = _version.version
except Exception:
    __version__ = "dev"