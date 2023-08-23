import json as jsonlib
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from functools import partial
from typing import Any, Dict, Iterable, List, Optional, Union
from io import BytesIO, BufferedReader
from pathlib import Path  
try:
    from typing import Literal, TypedDict
except ImportError:
    from typing_extensions import Literal, TypedDict

import requests
from requests.adapters import HTTPAdapter
from urllib3 import Retry

import maeluma
from maeluma.enums import VolumeEnum, TextFormat, LengthEnum
from maeluma.custom_model_dataset import CustomModelDataset
from maeluma.error import MaelumaAPIError, MaelumaConnectionError, MaelumaError
from maeluma.logging import logger
from maeluma.responses import (
    Detokenization,
    TextGenerations,
    StreamingTextGenerations,
    Tokens,
    ImageGenerations,
    UpscaledImage,
    SpeechGeneration,
    SummarizeResponse,
)
from maeluma.responses.chat import Chat, Mode, StreamingChat
#from maeluma.responses.classify import Example as ClassifyExample
#from maeluma.responses.classify import LabelPrediction
#from maeluma.responses.cluster import ClusterJobResult, CreateClusterJobResponse
#from maeluma.responses.custom_model import (
#    CUSTOM_MODEL_PRODUCT_MAPPING,
#    CUSTOM_MODEL_STATUS,
#    CUSTOM_MODEL_TYPE,
#    INTERNAL_CUSTOM_MODEL_TYPE,
#    CustomModel,
#    HyperParametersInput,
#
# )
# from maeluma.responses.detectlang import DetectLanguageResponse, Language
from maeluma.responses.embeddings import Embeddings

from maeluma.utils import is_api_key_valid, threadpool_map, wait_for_job


class Client:
    """Maeluma Client

    Args:
        api_key (str): Your API key.
        num_workers (int): Maximal number of threads for parallelized calls.
        request_dict (dict): Additional parameters for calls with the requests library. Currently ignored in AsyncClient
        check_api_key (bool): Whether to check the api key for validity on initialization.
        client_name (str): A string to identify your application for internal analytics purposes.
        max_retries (int): maximal number of retries for requests.
        timeout (int): request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str = None,
        num_workers: int = 64,
        request_dict: dict = {},
        check_api_key: bool = True,
        client_name: Optional[str] = None,
        max_retries: int = 3,
        timeout: int = 120,
    ) -> None:
        self.api_key = api_key or os.getenv("MAE_API_KEY")
        self.api_url = os.getenv("MAE_API_URL", maeluma.MAE_API_URL)
        self.batch_size = maeluma.MAELUMA_EMBED_BATCH_SIZE
        self._executor = ThreadPoolExecutor(num_workers)
        self.num_workers = num_workers
        self.request_dict = request_dict
        self.request_source = "python-sdk-" + maeluma.SDK_VERSION
        self.max_retries = max_retries
        self.timeout = timeout
        self.api_version = f"v{maeluma.API_VERSION}"
        if client_name:
            self.request_source += ":" + client_name

        if check_api_key:
            self.check_api_key()

    def check_api_key(self) -> Dict[str, bool]:
        """
        Checks the api key, which happens automatically during Client initialization, but not in AsyncClient.
        check_api_key raises an exception when the key is invalid, but the return value for valid keys is kept for
        backwards compatibility.
        """
        return {"valid": is_api_key_valid(self.api_key)}

    def batch_generate(
        self, prompts: List[str], return_exceptions=False, **kwargs
    ) -> List[Union[TextGenerations, Exception]]:
        """A batched version of generate with multiple prompts.

        Args:
            prompts: list of prompts
            return_exceptions (bool): Return exceptions as list items rather than raise them. Ensures your entire batch is not lost on one of the items failing.
            kwargs: other arguments to `generate`
        """
        return threadpool_map(
            self.generate,
            [dict(prompt=prompt, **kwargs) for prompt in prompts],
            num_workers=self.num_workers,
            return_exceptions=return_exceptions,
        )

    def generate(
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
        truncate: Optional[str] = None,
        stream: bool = False,
        logit_bias: Dict[int, float] = {},
    ) -> Union[TextGenerations, StreamingTextGenerations]:
        """Generate endpoint.
        See https://docs.maeluma.ai/reference/generate for advanced arguments

        Args:
            prompt (str): Represents the prompt or text to be completed. Trailing whitespaces will be trimmed.
            model (str): (Optional) The model ID to use for generating the next reply.
            preset (str): (Optional) The ID of a custom playground preset.
            num_generations (int): (Optional) The number of generations that will be returned, defaults to 1.
            max_tokens (int): (Optional) The number of tokens to predict per generation, defaults to 20.
            temperature (float): (Optional) The degree of randomness in generations from 0.0 to 5.0, lower is less random.
            truncate (str): (Optional) One of NONE|START|END, defaults to END. How the API handles text longer than the maximum token length.\
            stream (bool): Return streaming tokens.
        Returns:
            if stream=False: a Generations object

        Examples:
            A simple generate message:
                >>> res = ml.generate(prompt="Hey! How are you doing today?")
                >>> print(res.text)
            Streaming generate:
                >>> res = ml.generate(
                >>>     prompt="Hey! How are you doing today?",
                >>>     stream=True)
                >>> for token in res:
                >>>     print(token)
        """
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
            "truncate": truncate,
            "logit_bias": logit_bias,
            "stream": stream,
        }
        response = self._request(maeluma.GENERATE_URL, json=json_body, stream=stream)
        return TextGenerations.from_dict(response=response)


    def generate_image(
            self,
            prompt:str,
            num_generations: Optional[int] = 1,
            negative_prompt: Optional[str] = None,
            model: Optional[str] = 'general',
            width: Optional[int] = 512,
            height: Optional[int] = 512,
            samples: Optional[int] = 1,
            enhance_prompt=False,
            num_inference_steps: Optional[int] = 21,
            seed: Optional[int] = None,
            guidance_scale: Optional[float] = 7.5,
            safety_checker: Optional[bool] = False,
            multi_lingual: Optional[bool] = True,
            panorama: Optional[bool] = False,
            self_attention: Optional[bool] = None,
            upscale: Optional[bool] = None,
            embeddings_model: Optional[bool] = None,
            webhook: Optional[bool] = None,
            idempotency_key=None, 
            stream=False):
        json_body = {
          "key": idempotency_key,
          "prompt": prompt,
          'num_generations': num_generations,
          "negative_prompt": negative_prompt,
          'enhance_prompt': enhance_prompt,
          "width": width,
          "height": height,
          "samples": samples,
          'model': model, # 'general' or 'faces
          "num_inference_steps": num_inference_steps,
          "seed": seed,
          "guidance_scale": guidance_scale,
          "safety_checker": safety_checker,
          "multi_lingual": multi_lingual,
          "panorama": panorama,
          "self_attention": self_attention,
          "upscale": upscale,
          "embeddings_model": None,
          "webhook": None,
          "track_id": None
        }
        response = self._request(maeluma.GENERATE_IMAGE_URL, json=json_body, stream=stream)
        return response
    
    def generate_speech(self, text, voice) -> SpeechGeneration:
        json_body = {
            "text": text,
            "voice": voice
        }
        response = self._request(maeluma.GENERATE_SPEECH_URL, json=json_body, stream=False)
        return SpeechGeneration.from_response(response=response)
    
    def batch_generate_image(
        self, prompts: List[str], return_exceptions=False, **kwargs
    ) -> List[Union[ImageGenerations, Exception]]:
        """A batched version of generate with multiple prompts.

        Args:
            prompts: list of prompts
            return_exceptions (bool): Return exceptions as list items rather than raise them. Ensures your entire batch is not lost on one of the items failing.
            kwargs: other arguments to `generate`
        """
        return threadpool_map(
            self.generate_image,
            [dict(prompt=prompt, **kwargs) for prompt in prompts],
            num_workers=self.num_workers,
            return_exceptions=return_exceptions,
        )

    
    def chat(
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
    ) -> Union[Chat, StreamingChat]:
        """Returns a Chat object with the query reply.

        Args:
            message (str): The message to send to the chatbot.
            conversation_id (str): (Optional) The conversation id to continue the conversation.
            model (str): (Optional) The model to use for generating the next reply.
            return_chatlog (bool): (Optional) Whether to return the chatlog.
            return_prompt (bool): (Optional) Whether to return the prompt.
            return_preamble (bool): (Optional) Whether to return the preamble.
            chat_history (List[Dict[str, str]]): (Optional) A list of entries used to construct the conversation. If provided, these messages will be used to build the prompt and the conversation_id will be ignored so no data will be stored to maintain state.
            preamble_override (str): (Optional) A string to override the preamble.
            user_name (str): (Optional) A string to override the username.
            temperature (float): (Optional) The temperature to use for the next reply. The higher the temperature, the more random the reply.
            max_tokens (int): (Optional) The max tokens generated for the next reply.
            stream (bool): Return streaming tokens.
            p (float): (Optional) The nucleus sampling probability.
            k (float): (Optional) The top-k sampling probability.
            logit_bias (Dict[int, float]): (Optional) A dictionary of logit bias values to use for the next reply.
            mode Mode: (Optional) This property determines which functionality of retrieval augmented generation to use.
                                    chat mode doesn't use any retrieval augmented generation functionality.
                                    search_query_generation uses the provided query to produce search terms that you can use to search for documents.
                                    augmented_generation uses the provided documents and query to produce citations
            document Document: (Optional) The documents to use in augmented_generation mode. Shape: ("title", str), ("snippet", str), ("url", str)
        Returns:
            a Chat object if stream=False, or a StreamingChat object if stream=True

        Examples:
            A simple chat message:
                >>> res = ml.chat(message="Hey! How are you doing today?")
                >>> print(res.text)
                >>> print(res.conversation_id)
            Continuing a session using a specific model:
                >>> res = ml.chat(
                >>>     message="Hey! How are you doing today?",
                >>>     conversation_id="1234",
                >>>     model="command",
                >>>     return_chatlog=True)
                >>> print(res.text)
                >>> print(res.chatlog)
            Streaming chat:
                >>> res = ml.chat(
                >>>     message="Hey! How are you doing today?",
                >>>     stream=True)
                >>> for token in res:
                >>>     print(token)
            Stateless chat with chat history:
                >>> res = ml.chat(
                >>>     message="Tell me a joke!",
                >>>     chat_history=[
                >>>         {'user_name': 'User', message': 'Hey! How are you doing today?'},
                >>>         {'user_name': 'Bot', message': 'I am doing great! How can I help you?'},
                >>>     ],
                >>>     return_prompt=True)
                >>> print(res.text)
                >>> print(res.prompt)
            Augmented generation example:
                >>> res = ml.chat(query="What are the tallest penguins?",
                                  mode="augmented_generation",
                                  documents = [{"title":"Tall penguins", "snippet":"Emperor penguins are the tallest", "url":"http://example.com/foo"}])
                >>> print(res.text)
                >>> print(res.citations)
        """
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

        if query is not None:
            logger.warning(
                "The chat_history 'text' key is deprecated and will be removed in a future version of this function. "
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
        response = self._request(maeluma.CHAT_URL, json=json_body, stream=stream)

        if stream:
            return StreamingChat(response)
        else:
            return Chat.from_dict(response, message=message, client=self)


    def upscale_image(
            self,
            url: Optional[str] = None,
            image: Optional[Union[str, BytesIO]] = None,
            filetype: Optional[str] = "png",
    ):
      """
        Returns the url of the upscaled image.
        Args:
            url (str): The url of the image to upscale.
            image (Union[str, bytes, BytesIO]): (Optional) The image or image file path to upscale. If provided, the url will be ignored.
      """
      headers = {"Content-Disposition": "attachment; filename={url}"}
      files = {}
      if image is None and url is None:
          raise MaelumaError("You must provide either a url or an image.")
      if image is not None:
          files={"image": image}
          if isinstance(image, str):
              filename = Path(image).name
              image = self._get_image(image)
              headers = {"Content-Disposition": "attachment; filename={filename}".format(filename=filename)}
          else:
              filename = "image." + filetype
              headers = {"Content-Disposition": "attachment; filename={filename}".format(filename=filename)}
      else:
          image = url
      response = self._request(maeluma.UPSCALE_IMAGE_URL, json={"url": url}, files=files, headers=headers)
      return UpscaledImage(**response)
      

    def summarize(self, 
                  text: str,
                  length: Optional[LengthEnum] = LengthEnum.AUTO,
                  format: Optional[TextFormat] = TextFormat.AUTO,
                  model: Optional[str] = 'general',
                  extractiveness: Optional[VolumeEnum] = VolumeEnum.AUTO,
                  temperature: Optional[float] = 0.75,
                  additional_command: Optional[str] = '',
                ) -> SummarizeResponse:
        """Returns a generated summary of the specified length for the provided text.

        Args:
            text (str): Text to summarize.
            model (str): (Optional) ID of the model.
            length (str): (Optional) One of {"short", "medium", "long"}, defaults to "medium". \
                Controls the length of the summary.
            format (str): (Optional) One of {"paragraph", "bullets"}, defaults to "paragraph". \
                Controls the format of the summary.
            extractiveness (str) One of {"high", "medium", "low"}, defaults to "high". \
                Controls how close to the original text the summary is. "High" extractiveness \
                summaries will lean towards reusing sentences verbatim, while "low" extractiveness \
                summaries will tend to paraphrase more.
            temperature (float): Ranges from 0 to 5. Controls the randomness of the output. \
                Lower values tend to generate more “predictable” output, while higher values \
                tend to generate more “creative” output. The sweet spot is typically between 0 and 1.
            additional_command (str): (Optional) Modifier for the underlying prompt, must \
                complete the sentence "Generate a summary _".

        Examples:
            Summarize a text:
                >>> res = ml.summarize(text="Stock market report for today...")
                >>> print(res.summary)

            Summarize a text with a specific model and prompt:
                >>> res = ml.summarize(
                >>>     text="Stock market report for today...",
                >>>     model="summarize-xlarge",
                >>>     length="long",
                >>>     format="bullets",
                >>>     temperature=0.3,
                >>>     additional_command="focusing on the highest performing stocks")
                >>> print(res.summary)
        """
        json_body = {
            "text": text,
            "length": length,
            "format": format,
            "model": model,
            "extractiveness": extractiveness,
            "temperature": temperature,
            "additional_command": additional_command,
        }
        json_body = {k: v for k, v in json_body.items() if v is not None}
        response = self._request(maeluma.SUMMARIZE_URL, json=json_body) 
        return SummarizeResponse(id=response["id"], summary=response["summary"], meta=response["meta"])


    def embed(
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
        responses = {
            "embeddings": [],
            "compressed_embeddings": [],
        }
        json_bodys = []

        for i in range(0, len(texts), self.batch_size):
            texts_batch = texts[i : i + self.batch_size]
            json_bodys.append(
                {
                    "model": model,
                    "texts": texts_batch,
                    "truncate": truncate,
                    "compress": compress,
                    "compression_codebook": compression_codebook,
                }
            )

        meta = None
        for result in self._executor.map(lambda json_body: self._request(maeluma.EMBED_URL, json=json_body), json_bodys):
            responses["embeddings"].extend(result["embeddings"])
            responses["compressed_embeddings"].extend(result.get("compressed_embeddings", []))
            meta = result["meta"] if not meta else meta

        return Embeddings(
            embeddings=responses["embeddings"],
            compressed_embeddings=responses["compressed_embeddings"],
            meta=meta,
        )


    def batch_tokenize(self, texts: List[str], return_exceptions=False, **kwargs) -> List[Union[Tokens, Exception]]:
        """A batched version of tokenize.

        Args:
            texts: list of texts
            return_exceptions (bool): Return exceptions as list items rather than raise them. Ensures your entire batch is not lost on one of the items failing.
            kwargs: other arguments to `tokenize`
        """
        return threadpool_map(
            self.tokenize,
            [dict(text=text, **kwargs) for text in texts],
            num_workers=self.num_workers,
            return_exceptions=return_exceptions,
        )

    def tokenize(self, text: str, model: Optional[str] = None) -> Tokens:
        """Returns a Tokens object of the provided text, see https://docs.maeluma.ai/reference/tokenize for advanced usage.

        Args:
            text (str): Text to summarize.
            model (str): An optional model name that will ensure that the tokenization uses the tokenizer used by that model, which can be critical for counting tokens properly.
        """
        json_body = {"text": text, "model": model}
        res = self._request(maeluma.TOKENIZE_URL, json=json_body)
        return Tokens(tokens=res["tokens"], token_strings=res["token_strings"], meta=res.get("meta"))

    def batch_detokenize(
        self, list_of_tokens: List[List[int]], return_exceptions=False, **kwargs
    ) -> List[Union[Detokenization, Exception]]:
        """A batched version of detokenize.

        Args:
            list_of_tokens: list of list of tokens
            return_exceptions (bool): Return exceptions as list items rather than raise them. Ensures your entire batch is not lost on one of the items failing.
            kwargs: other arguments to `detokenize`
        """
        return threadpool_map(
            self.detokenize,
            [dict(tokens=tokens, **kwargs) for tokens in list_of_tokens],
            num_workers=self.num_workers,
            return_exceptions=return_exceptions,
        )

    def detokenize(self, tokens: List[int], model: Optional[str] = None) -> Detokenization:
        """Returns a Detokenization object of the provided tokens, see https://docs.maeluma.ai/reference/detokenize for advanced usage.

        Args:
            tokens (List[int]): A list of tokens to convert to strings
            model (str): An optional model name. This will ensure that the detokenization is done by the tokenizer used by that model.
        """
        json_body = {"tokens": tokens, "model": model}
        res = self._request(maeluma.DETOKENIZE_URL, json=json_body)
        return Detokenization(text=res["text"], meta=res.get("meta"))


    def _check_response(self, json_response: Dict, headers: Dict, status_code: int):
        if "X-API-Warning" in headers:
            logger.warning(headers["X-API-Warning"])
        if "message" in json_response:  # has errors
            raise MaelumaAPIError(
                message=json_response["message"],
                http_status=status_code,
                headers=headers,
            )
        if 400 <= status_code < 500:
            raise MaelumaAPIError(
                message=f"Unexpected client error (status {status_code}): {json_response}",
                http_status=status_code,
                headers=headers,
            )
        if status_code >= 500:
            raise MaelumaError(message=f"Unexpected server error (status {status_code}): {json_response}")
    

    def _get_image(self, file_path: str) -> BufferedReader:
        """Returns a file object of the image at the provided file path."""
        return open(file_path, "rb")
    

    def _request(self, endpoint, json=None, method="POST", stream=False, files: dict = {}, headers={}) -> Any:
        headers = {
            "Authorization": "BEARER {}".format(self.api_key),
            "Content-Type": "application/json",
            "Request-Source": self.request_source,
            **headers,
        }

        url = f"{self.api_url}/{self.api_version}/{endpoint}"
        with requests.Session() as session:
            retries = Retry(
                total=self.max_retries,
                backoff_factor=0.5,
                allowed_methods=["POST", "GET"],
                status_forcelist=maeluma.RETRY_STATUS_CODES,
                raise_on_status=False,
            )
            session.mount("https://", HTTPAdapter(max_retries=retries))
            session.mount("http://", HTTPAdapter(max_retries=retries))

            if stream:
                return session.request(method, url, headers=headers, json=json, files=files, **self.request_dict, stream=True)

            try:
                response = session.request(
                    method, url, headers=headers, json=json, timeout=self.timeout, files=files, **self.request_dict
                )
            except requests.exceptions.ConnectionError as e:
                raise MaelumaConnectionError(str(e)) from e
            except requests.exceptions.RequestException as e:
                raise MaelumaError(f"Unexpected exception ({e.__class__.__name__}): {e}") from e

            try:
                json_response = response.json()
            except jsonlib.decoder.JSONDecodeError:  # MaelumaAPIError will capture status
                raise MaelumaAPIError.from_response(response, message=f"Failed to decode json body: {response.text}")

            self._check_response(json_response, response.headers, response.status_code)
        return json_response