"""
KiwiCalc - Modular Mathematical Engine
Backward compatibility entry point. The monolith has been refactored into the `kiwicalc/` package.
"""
import warnings
from kiwicalc import *

warnings.warn(
    "Importing from single-file kiwicalc.py is deprecated. Use the modular 'kiwicalc' package instead.",
    DeprecationWarning,
    stacklevel=2
)
