import os
import json
import yaml
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
API_KEY  = os.getenv("OPENROUTER_API_KEY")
TEMPERATURE = float(os.getenv("TEMPERATURE", 0))


def load_config() -> dict:
    config_path = Path(__file__).parent.parent / "config" / "models.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def load_models() -> list[dict]:
    return load_config().get("models") or []


def load_execution_config() -> dict:
    return load_config().get("execution") or {}


class UpstreamError(Exception):
    """
    OpenRouter às vezes responde 200 OK com um corpo {"error": {"message":...,
    "code": N}} quando o provedor upstream falha (ex.: sobrecarga temporária).
    response.raise_for_status() não detecta isso, pois o status HTTP é 200.
    """

    def __init__(self, message: str, code: int | None):
        super().__init__(message)
        self.code = code


def call_model(model_id: str, provider: str, messages: list[dict]) -> dict:
    """Chama um modelo no OpenRouter e retorna o objeto de resposta completo."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    execution_config = load_execution_config()

    payload = {
        "model": model_id,
        "messages": messages,
        "max_tokens": execution_config["max_tokens"],
        "temperature": TEMPERATURE,
        "tool_choice": "none",    # sem ferramentas externas
        "provider": {
            "order": [provider],
            "allow_fallbacks": execution_config["fallbacks"],
        },
    }
    response = requests.post(
    f"{BASE_URL}/chat/completions",
    headers=headers,
    json=payload,
    timeout=execution_config["request_timeout_seconds"],
)
    response.raise_for_status()
    body = response.json()

    # Erro upstream disfarçado de 200 OK (sem "choices", com "error" no corpo).
    if "choices" not in body and isinstance(body.get("error"), dict):
        err = body["error"]
        raise UpstreamError(err.get("message", "erro upstream sem mensagem"), err.get("code"))

    return body


def is_retryable_error(error: Exception) -> bool:
    if isinstance(
        error,
        (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
        ),
    ):
        return True

    if isinstance(error, requests.exceptions.HTTPError):
        response = error.response
        if response is None:
            return False

        status_code = response.status_code
        return status_code == 429 or 500 <= status_code < 600

    if isinstance(error, UpstreamError):
        code = error.code
        return code is not None and (code == 429 or 500 <= code < 600)

    return False


def extract_content(response: dict) -> str:
    """Extrai o texto da resposta, tratando modelos com 'thinking' separado."""
    message = response["choices"][0]["message"]
    content = message.get("content")
    if content is not None:
        return content
    # Modelos com thinking podem retornar content=null e resposta em outro campo
    for field in ("reasoning_content", "reasoning", "text"):
        alt = message.get(field)
        if alt:
            return alt
    # Se chegou aqui, loga o response para diagnóstico
    raise ValueError(
        f"Resposta sem conteúdo extraível. Campos disponíveis: {list(message.keys())}\n"
        f"Response completo: {response}"
    )


def parse_json_response(content: str) -> dict:
    """Extrai JSON da resposta, tolerando markdown code fences."""
    content = content.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(lines[1:-1])
    return json.loads(content)
