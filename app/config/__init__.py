"""
Navigator config package.
Public interface: import get_settings from here, nowhere else.
"""

from app.config.settings import Settings, get_settings

__all__ = ["Settings", "get_settings"]
