from typing import Any, Dict, Iterator, List, Optional
from enum import Enum
from maeluma.responses.base import MaelumaObject



class VoiceOption(str, Enum):
    SHIRLEY = 'shirley'

class SpeechGeneration(MaelumaObject):
    def __init__(
        self,
        generations: List[str],
        voice: Optional[VoiceOption] = VoiceOption.SHIRLEY,
    ) -> None:
        self.generations = generations
        self.voice = voice
    
    @classmethod
    def from_response(cls, response, **kwargs):
        return cls(
            generations=response.get("output_uris", []),
            voice=response.get("voice", VoiceOption.SHIRLEY.value),
        )