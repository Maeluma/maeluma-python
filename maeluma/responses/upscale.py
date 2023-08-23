



from typing import Any, Dict, Iterator, List, Optional
from dataclasses import dataclass, field

from maeluma.responses.base import MaelumaObject

@dataclass
class UpscaledImage(MaelumaObject):
    id: str
    url: str
    meta: Optional[dict]