from enum import Enum


class LengthEnum(str, Enum):
    LONG = 'long',
    MEDIUM = 'medium',
    SHORT = 'short',
    AUTO = 'auto',

class VolumeEnum(str, Enum):
    """
      A General Enum for the for high, medium, low and auto
    """
    HIGH = 'HIGH',
    MEDIUM = 'medium',
    LOW = 'low',
    AUTO = 'auto',


class TextFormat(str, Enum):
    PARAGRAPHS = 'paragraph',
    BULLETS = 'bullets',
    AUTO = 'auto',