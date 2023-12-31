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


class TextGeneration(MaelumaObject, str):
    def __new__(cls, text: str, *_, **__):
        return str.__new__(cls, text)

    def __init__(
        self,
        text: str,
        likelihood: float,
        token_likelihoods: List[TokenLikelihood] = [],
        prompt: str = None,
        finish_reason: str = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.prompt = prompt
        self.text = text
        self.likelihood = likelihood
        self.finish_reason = finish_reason
        self.token_likelihoods = token_likelihoods

    @classmethod
    def from_response(cls, response, prompt=None, **kwargs):
        token_likelihoods = response.get("token_likelihoods")
        if token_likelihoods:
            token_likelihoods = [TokenLikelihood(d["token"], d.get("likelihood")) for d in token_likelihoods]
        return cls(
            text=response.get("text"),
            likelihood=response.get("likelihood"),
            token_likelihoods=token_likelihoods,
            prompt=prompt,
            id=response.get("id"),
            finish_reason=response.get("finish_reason"),
            **kwargs,
        )

    # nice jupyter output
    def visualize_token_likelihoods(self, ignore_first_n=0, midpoint=-3, value_range=8, display=True):  # very WIP
        if self.token_likelihoods is None:
            return None

        def color_token(i, t: TokenLikelihood):
            if t.likelihood is None or i < ignore_first_n:
                col = "#EDEDED"
            else:
                col = next(c for thr, c in TOKEN_COLORS if t.likelihood >= thr)  # first hit
            return f"<span style='background-color:{col}'>{t.token}</span>"

        html = "".join(color_token(i, t) for i, t in enumerate(self.token_likelihoods))
        if display:
            from IPython.display import HTML

            return HTML(html)  # show in jupyter by default, but allow to be used as helper
        return html

    def _visualize_helper(self):
        return dict(
            prompt=self.prompt,
            text=self.text,
            likelihood=self.likelihood,
            token_likelihoods=self.visualize_token_likelihoods(display=False),
        )

    def visualize(self, **kwargs) -> str:
        import pandas as pd

        with pd.option_context("display.max_colwidth", 250):
            return _df_html(pd.DataFrame([self._visualize_helper()]), **kwargs)


class TextGenerations(UserList, MaelumaObject):
    def __init__(self, generations,  meta: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(generations)
        self.meta = meta

    @classmethod
    def from_dict(cls, response: Dict[str, Any]) -> List[TextGeneration]:
        generations: List[TextGeneration] = []
        for gen in response["generations"]:
            likelihood = None
            token_likelihoods = None
            if "token_likelihoods" in gen.keys():
                token_likelihoods = []
                for likelihoods in gen["token_likelihoods"]:
                    token_likelihood = likelihoods["likelihood"] if "likelihood" in likelihoods.keys() else None
                    token_likelihoods.append(TokenLikelihood(likelihoods["token"], token_likelihood))
            generations.append(
                TextGeneration(
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
    def generations(self) -> List[TextGeneration]:  # backward compatibility
        return self.data

    # nice jupyter output
    def visualize(self, **kwargs) -> str:
        import pandas as pd

        with pd.option_context("display.max_colwidth", 250):
            return _df_html(pd.DataFrame([g._visualize_helper() for g in self]), **kwargs)

    @property
    def prompt(self) -> str:
        """Returns the prompt used as input"""
        return self[0].prompt  # should all be the same


# ("likelihood", Optional[float])]) not supported
StreamingText = NamedTuple("StreamingText", [("index", Optional[int]), ("text", str), ("is_finished", bool)])


class StreamingTextGenerations(MaelumaObject):
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
        self.generations = TextGenerations.from_dict(generation_response)
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