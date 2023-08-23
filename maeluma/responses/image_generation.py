import json
from collections import UserList
from typing import Any, Dict, Generator, List, NamedTuple, Optional

import requests

from maeluma.responses.base import MaelumaObject, _df_html

TokenLikelihood = NamedTuple("TokenLikelihood", [("token", str), ("likelihood", float)])

TOKEN_COLORS = [
    (-2, "#FFECE2"),
    (-4, "#FFD6BC"),
    (-6, "#FFC59A"),
    (-8, "#FFB471"),
    (-10, "#FFA745"),
    (-12, "#FE9F00"),
    (-1e9, "#E18C00"),
]


class ImageGeneration(MaelumaObject, str):
    def __new__(cls, text: str, *_, **__):
        return str.__new__(cls, text)

    def __init__(
        self,
        prompt: str,
        width: int,
        height: int,
        negative_prompt: str,
        url: str,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.prompt = prompt
        self.width = width
        self.height = height
        self.negative_prompt = negative_prompt
        self.url = url

    @classmethod
    def from_response(cls, response, prompt=None, **kwargs):
        image_metadata = response.get("meta")
        
        return cls(
            prompt=response.get("prompt"),
            width=image_metadata.get("width", 512),
            height=image_metadata.get('height', 512),
            negative_prompt=response.get('negative_prompt', ''),
            id=image_metadata.get("file_prefix", response.get("output")[0]),
            url=response.get("output")[0],
            **kwargs,
        )


class ImageGenerations(UserList, MaelumaObject):
    def __init__(self, generations,  meta: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(generations)
        self.meta = meta

    @classmethod
    def from_dict(cls, response: Dict[str, Any]) -> List[ImageGeneration]:
        generations: List[ImageGeneration] = []
        for gen in response["generations"]:
            generations.append(
                ImageGeneration(
                    gen["text"],
                    likelihood,
                    token_likelihoods,
                    prompt=response.get("prompt"),
                    id=gen["id"],
                    finish_reason=gen.get("finish_reason"),
                )
            )

        return cls(generations, response.get("meta"))

    @property
    def generations(self) -> List[ImageGeneration]:  # backward compatibility
        return self.data

    @property
    def prompt(self) -> str:
        """Returns the prompt used as input"""
        return self[0].prompt  # should all be the same


# ("likelihood", Optional[float])]) not supported
StreamingText = NamedTuple("StreamingText", [("index", Optional[int]), ("text", str), ("is_finished", bool)])


class StreamingImageGenerations(MaelumaObject):
    def __init__(self, response):
        self.response = response
        self.id = None
        self.generations = None
        self.finish_reason = None
        self.texts = []

    def _make_response_item(self, line) -> Optional[StreamingText]:
        streaming_item = json.loads(line)
        is_finished = streaming_item.get("is_finished")

        if not is_finished:
            index = streaming_item.get("index", 0)
            text = streaming_item.get("text")
            while len(self.texts) <= index:
                self.texts.append("")
            if text is None:
                return None
            self.texts[index] += text
            return StreamingText(text=text, is_finished=is_finished, index=index)

        self.finish_reason = streaming_item.get("finish_reason")
        generation_response = streaming_item.get("response")

        if generation_response is None:
            return None

        self.id = generation_response.get("id")
        # likelihoods not supported in streaming currently
        self.generations = ImageGeneration.from_dict(generation_response)
        return None

    def __iter__(self) -> Generator[StreamingText, None, None]:
        if not isinstance(self.response, requests.Response):
            raise ValueError("For AsyncClient, use `async for` to iterate through the `StreamingGenerations`")

        for line in self.response.iter_lines():
            item = self._make_response_item(line)
            if item is not None:
                yield item

    async def __aiter__(self) -> Generator[StreamingText, None, None]:
        async for line in self.response.content:
            item = self._make_response_item(line)
            if item is not None:
                yield item