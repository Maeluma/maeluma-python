from typing import Any, Dict, Iterator, List, Optional

from maeluma.responses.base import MaelumaObject


class Embeddings(MaelumaObject):
    def __init__(
        self,
        embeddings: List[List[float]],
        compressed_embeddings: List[List[int]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.embeddings = embeddings
        self.compressed_embeddings = compressed_embeddings
        self.meta = meta

    def __iter__(self) -> Iterator:
        return iter(self.embeddings)

    def __len__(self) -> int:
        return len(self.embeddings)