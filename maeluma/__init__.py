from importlib_metadata import version  # use package to support python 3.7

from maeluma.client import Client
from maeluma.client_async import AsyncClient
from maeluma.error import MaelumaAPIError, MaelumaConnectionError, MaelumaError

MAE_API_URL =  "http://127.0.0.1:8000"  #"https://api.maeluma.io"
RETRY_STATUS_CODES = [429, 500, 502, 503, 504]

API_VERSION = "1"
SDK_VERSION = version("maeluma")
MAELUMA_EMBED_BATCH_SIZE = 96
CHAT_URL = "chat"
CLASSIFY_URL = "classify"
CODEBOOK_URL = "embed-codebook"
DETECT_LANG_URL = "detect-language"
EMBED_URL = "embed"
GENERATE_FEEDBACK_URL = "feedback/generate"
GENERATE_PREFERENCE_FEEDBACK_URL = "feedback/generate/preference"
GENERATE_URL = "generate/text"
GENERATE_IMAGE_URL = "generate/image"
GENERATE_SPEECH_URL = "generate/speech"
SUMMARIZE_URL = "summarize"
RERANK_URL = "rerank"
UPSCALE_IMAGE_URL = "upscale-image"

CHECK_API_KEY_URL = "check-api-key"
TOKENIZE_URL = "tokenize"
DETOKENIZE_URL = "detokenize"

CLUSTER_JOBS_URL = "cluster-jobs"
BULK_EMBED_JOBS_URL = "embed-jobs"
CUSTOM_MODEL_URL = "finetune"