import asyncio
import json as jsonlib
import os
import posixpath
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

try:
    from typing import Literal, TypedDict
except ImportError:
    from typing_extensions import Literal, TypedDict

import aiohttp
import backoff

import maeluma
from maeluma.client import Client
from maeluma.custom_model_dataset import CustomModelDataset
from maeluma.error import MaelumaAPIError, MaelumaConnectionError, MaelumaError
from maeluma.logging import logger
from maeluma.responses import (
    Detokenization,
    Embeddings,
    TextGenerations,
    StreamingTextGenerations,
    Tokens,
)
from maeluma.responses.chat import AsyncChat, Mode, StreamingChat
from maeluma.utils import async_wait_for_job, is_api_key_valid, np_json_dumps

JSON = Union[Dict, List]


class AsyncClient(Client):
    """AsyncClient

    This client provides an asyncio/aiohttp interface.
    Using this client is recommended when you are making highly parallel request,
    or when calling the Maeluma API from a server such as FastAPI."""

    def __init__(
        self,
        api_key: str = None,
        num_workers: int = 16,
        request_dict: dict = {},
        check_api_key: bool = True,
        client_name: Optional[str] = None,
        max_retries: int = 3,
        timeout=120,
    ) -> None:
        self.api_key = api_key or os.getenv("CO_API_KEY")
        self.api_url = os.getenv("CO_API_URL", maeluma.MAE_API_URL)
        self.batch_size = maeluma.MAELUMA_EMBED_BATCH_SIZE
        self.num_workers = num_workers
        self.request_dict = request_dict
        self.request_source = "python-sdk-" + maeluma.SDK_VERSION
        self.max_retries = max_retries
        if client_name:
            self.request_source += ":" + client_name
        self.api_version = f"v{maeluma.API_VERSION}"
        self._check_api_key_on_enter = check_api_key
        self._backend = AIOHTTPBackend(logger, num_workers, max_retries, timeout)

    async def _request(self, endpoint, json=None, method="POST", full_url=None, stream=False) -> JSON:
        headers = {
            "Authorization": f"BEARER {self.api_key}",
            "Request-Source": self.request_source,
        }
        if endpoint is None and full_url is not None:  # api key
            url = full_url
        else:
            url = posixpath.join(self.api_url, self.api_version, endpoint)

        response = await self._backend.request(url, json, method, headers, stream=stream)
        if stream:
            return response

        try:
            json_response = await response.json()
        #   `MaelumaAPIError.from_response()` will capture the http status code
        except jsonlib.decoder.JSONDecodeError:
            raise MaelumaAPIError.from_response(response, message=f"Failed to decode json body: {await response.text()}")
        except aiohttp.ClientPayloadError as e:
            raise MaelumaAPIError.from_response(
                response, message=f"An unexpected error occurred while receiving the response: {e}"
            )

        logger.debug(f"JSON response: {json_response}")
        self._check_response(json_response, response.headers, response.status)
        return json_response

    async def close(self):
        return await self._backend.close()

    async def __aenter__(self):
        if self._check_api_key_on_enter:
            await self.check_api_key()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.close()

    # API methods
    async def check_api_key(self) -> Dict[str, bool]:
        """
        check_api_key raises an exception when the key is invalid, but the return value for valid keys is kept for
        backwards compatibility.
        """
        return {"valid": is_api_key_valid(self.api_key)}

    async def batch_generate(
        self, prompts: List[str], return_exceptions=False, **kwargs
    ) -> List[Union[Exception, TextGenerations]]:
        return await asyncio.gather(
            *[self.generate(prompt, **kwargs) for prompt in prompts], return_exceptions=return_exceptions
        )

    async def generate(
        self,
        prompt: Optional[str] = None,
        prompt_vars: object = {},
        model: Optional[str] = None,
        preset: Optional[str] = None,
        num_generations: Optional[int] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        k: Optional[int] = None,
        p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        end_sequences: Optional[List[str]] = None,
        stop_sequences: Optional[List[str]] = None,
        return_likelihoods: Optional[str] = None,
        truncate: Optional[str] = None,
        logit_bias: Dict[int, float] = {},
        stream: bool = False,
    ) -> Union[TextGenerations, StreamingTextGenerations]:
        json_body = {
            "model": model,
            "prompt": prompt,
            "prompt_vars": prompt_vars,
            "preset": preset,
            "num_generations": num_generations,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "k": k,
            "p": p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "end_sequences": end_sequences,
            "stop_sequences": stop_sequences,
            "return_likelihoods": return_likelihoods,
            "truncate": truncate,
            "logit_bias": logit_bias,
            "stream": stream,
        }
        response = await self._request(maeluma.GENERATE_URL, json=json_body, stream=stream)
        if stream:
            return StreamingTextGenerations(response)
        else:
            return TextGenerations.from_dict(response=response, return_likelihoods=return_likelihoods)

    async def chat(
        self,
        message: Optional[str] = None,
        query: Optional[str] = None,
        conversation_id: Optional[str] = "",
        model: Optional[str] = None,
        return_chatlog: Optional[bool] = False,
        return_prompt: Optional[bool] = False,
        return_preamble: Optional[bool] = False,
        chat_history: Optional[List[Dict[str, str]]] = None,
        preamble_override: Optional[str] = None,
        user_name: Optional[str] = None,
        temperature: Optional[float] = 0.8,
        max_tokens: Optional[int] = None,
        stream: Optional[bool] = False,
        p: Optional[float] = None,
        k: Optional[float] = None,
        logit_bias: Optional[Dict[int, float]] = None,
        mode: Optional[Mode] = None,
        documents: Optional[List[Dict[str, str]]] = None,
    ) -> Union[AsyncChat, StreamingChat]:
        if chat_history is not None:
            should_warn = True
            for entry in chat_history:
                if "text" in entry:
                    entry["message"] = entry["text"]

                if "text" in entry and should_warn:
                    logger.warning(
                        "The 'text' parameter is deprecated and will be removed in a future version of this function. "
                        + "Use 'message' instead.",
                    )
                    should_warn = False

        if query is None and message is None:
            raise MaelumaError("Either 'query' or 'message' must be provided.")

        if query is not None:
            logger.warning(
                "The 'query' parameter is deprecated and will be removed in a future version of this function. "
                + "Use 'message' instead.",
            )
            message = query

        json_body = {
            "message": message,
            "conversation_id": conversation_id,
            "model": model,
            "return_chatlog": return_chatlog,
            "return_prompt": return_prompt,
            "return_preamble": return_preamble,
            "chat_history": chat_history,
            "preamble_override": preamble_override,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            "user_name": user_name,
            "p": p,
            "k": k,
            "logit_bias": logit_bias,
            "mode": mode,
            "documents": documents,
        }

        response = await self._request(maeluma.CHAT_URL, json=json_body, stream=stream)

        if stream:
            return StreamingChat(response)
        else:
            return AsyncChat.from_dict(response, message=message, client=self)

    async def embed(
        self,
        texts: List[str],
        model: Optional[str] = None,
        truncate: Optional[str] = None,
        compress: Optional[bool] = False,
        compression_codebook: Optional[str] = "default",
    ) -> Embeddings:
        """Returns an Embeddings object for the provided texts. Visit https://maeluma.ai/embed to learn about embeddings.

        Args:
            text (List[str]): A list of strings to embed.
            model (str): (Optional) The model ID to use for embedding the text.
            truncate (str): (Optional) One of NONE|START|END, defaults to END. How the API handles text longer than the maximum token length.
            compress (bool): (Optional) Whether to compress the embeddings. When True, the compressed_embeddings will be returned as integers in the range [0, 255].
            compression_codebook (str): (Optional) The compression codebook to use for compressed embeddings. Defaults to "default".
        """
        json_bodys = [
            dict(
                texts=texts[i : i + maeluma.MAELUMA_EMBED_BATCH_SIZE],
                model=model,
                truncate=truncate,
                compress=compress,
                compression_codebook=compression_codebook,
            )
            for i in range(0, len(texts), maeluma.MAELUMA_EMBED_BATCH_SIZE)
        ]
        responses = await asyncio.gather(*[self._request(maeluma.EMBED_URL, json) for json in json_bodys])
        meta = responses[0]["meta"] if responses else None

        return Embeddings(
            embeddings=[e for res in responses for e in res["embeddings"]],
            compressed_embeddings=[e for res in responses for e in res["compressed_embeddings"]] if compress else None,
            meta=meta,
        )


    async def batch_tokenize(
        self, texts: List[str], return_exceptions=False, **kwargs
    ) -> List[Union[Tokens, Exception]]:
        return await asyncio.gather(*[self.tokenize(t, **kwargs) for t in texts], return_exceptions=return_exceptions)

    async def tokenize(self, text: str, model: Optional[str] = None) -> Tokens:
        json_body = {"text": text, "model": model}
        res = await self._request(maeluma.TOKENIZE_URL, json_body)
        return Tokens(tokens=res["tokens"], token_strings=res["token_strings"], meta=res["meta"])

    async def batch_detokenize(
        self, list_of_tokens: List[List[int]], return_exceptions=False, **kwargs
    ) -> List[Union[Detokenization, Exception]]:
        return await asyncio.gather(
            *[self.detokenize(t, **kwargs) for t in list_of_tokens], return_exceptions=return_exceptions
        )

    async def detokenize(self, tokens: List[int], model: Optional[str] = None) -> Detokenization:
        json_body = {"tokens": tokens, "model": model}
        res = await self._request(maeluma.DETOKENIZE_URL, json_body)
        return Detokenization(text=res["text"], meta=res["meta"])

class AIOHTTPBackend:
    """HTTP backend which handles retries, concurrency limiting and logging"""

    SLEEP_AFTER_FAILURE = defaultdict(lambda: 0.25, {429: 5})

    def __init__(self, logger, max_concurrent_requests: int = 64, max_retries: int = 5, timeout: int = 120):
        self.logger = logger
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_concurrent_requests = max_concurrent_requests
        self._semaphore: asyncio.Semaphore = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._requester = None

    def build_aio_requester(self) -> Callable:  # returns a function for retryable requests
        @backoff.on_exception(
            backoff.expo,
            (aiohttp.ClientError, aiohttp.ClientResponseError),
            max_tries=self.max_retries + 1,
            max_time=self.timeout,
        )
        async def make_request_fn(session, *args, **kwargs):
            async with self._semaphore:  # this limits total concurrency by the client
                response = await session.request(*args, **kwargs)
            if response.status in maeluma.RETRY_STATUS_CODES:  # likely temporary, raise to retry
                self.logger.info(f"Received status {response.status}, retrying...")
                await asyncio.sleep(self.SLEEP_AFTER_FAILURE[response.status])
                response.raise_for_status()

            return response

        return make_request_fn

    async def request(
        self, url, json=None, method: str = "post", headers=None, session=None, stream=False, **kwargs
    ) -> JSON:
        session = session or await self.session()
        self.logger.debug(f"Making request to {url} with content {json}")

        request_start = time.time()
        try:
            response = await self._requester(session, method, url, headers=headers, json=json, **kwargs)
        except aiohttp.ClientConnectionError as e:  # ensure the SDK user does not have to deal with knowing aiohttp
            self.logger.debug(f"Fatal connection error after {time.time()-request_start:.1f}s: {e}")
            raise MaelumaConnectionError(str(e)) from e
        except aiohttp.ClientResponseError as e:  # status 500 or something remains after retries
            self.logger.debug(f"Fatal ClientResponseError error after {time.time()-request_start:.1f}s: {e}")
            raise MaelumaConnectionError(str(e)) from e
        except asyncio.TimeoutError as e:
            self.logger.debug(f"Fatal timeout error after {time.time()-request_start:.1f}s: {e}")
            raise MaelumaConnectionError("The request timed out") from e
        except Exception as e:  # Anything caught here should be added above
            self.logger.debug(f"Unexpected fatal error after {time.time()-request_start:.1f}s: {e}")
            raise MaelumaError(f"Unexpected exception ({e.__class__.__name__}): {e}") from e

        self.logger.debug(f"Received response with status {response.status} after {time.time()-request_start:.1f}s")
        return response

    async def session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession(
                json_serialize=np_json_dumps,
                timeout=aiohttp.ClientTimeout(self.timeout),
                connector=aiohttp.TCPConnector(limit=0),
            )
            self._semaphore = asyncio.Semaphore(self.max_concurrent_requests)
            self._requester = self.build_aio_requester()
        return self._session

    async def close(self):
        if self._session is not None:
            await self._session.close()
            self._session = None

    def __del__(self):
        # https://stackoverflow.com/questions/54770360/how-can-i-wait-for-an-objects-del-to-finish-before-the-async-loop-closes
        if self._session:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.close())
                else:
                    loop.run_until_complete(self.close())
            except Exception:
                pass