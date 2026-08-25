import os
import json
import yaml
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
API_KEY  = os.getenv("OPENROUTER_API_KEY")
MAX_TOKENS  = int(os.getenv("MAX_TOKENS", 4096))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0))


def load_config() -> dict:
    config_path = Path(__file__).parent.parent / "config" / "models.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def load_models() -> list[dict]:
    return load_config().get("models") or []


def load_execution_config() -> dict:
    return load_config().get("execution") or {}


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
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "tool_choice": "none",    # sem ferramentas externas
        "provider": {
            "order": [provider],
            "allow_fallbacks": execution_config["fallbacks"],
        },
    }
    response = requests.post(f"{BASE_URL}/chat/completions", headers=headers, json=payload)
    response.raise_for_status()
    return response.json()


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
