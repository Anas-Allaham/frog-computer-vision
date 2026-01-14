"""
detectors/__init__.py

Exports all detector classes for Zuma game.
"""

from .window_detector import ZumaWindowDetector as WindowDetector
from .zuma_frog_detector import ZumaLauncherDetector
from .balls_detectors import ZumaBallDetector

__all__ = [
    'WindowDetector',
    'ZumaLauncherDetector',
    'ZumaBallDetector'
]