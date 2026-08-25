"""
Execução do Experimento Principal

Para cada combinação (modelo × especificação):
  - Cria dois contextos de conversa independentes (T_WH e T_WD)
  - No T_WD: obtém C_im na mesma sessão, logo após a estimativa
  - Registra respostas brutas + metadados em experiment/results/

Pré-requisitos:
  1. config/models.yaml preenchido com os 12 modelos selecionados
  2. data/specifications/ com as 88 especificações em .txt
  3. experiment/randomization.json gerado por randomize.py
  4. Instrumentos congelados após o estudo piloto
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from openrouter_client import load_models, call_model, extract_content, parse_json_response
from prompt_builder import build_estimation_prompt, build_conversion_prompt

ROOT        = Path(__file__).parent.parent
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

RAND_FILE   = Path(__file__).parent / "randomization.json"


def load_spec(spec_id: str) -> str:
    return (ROOT / "data" / "specifications" / f"{spec_id}.txt").read_text()


def load_randomization() -> dict:
    with open(RAND_FILE) as f:
        return json.load(f)


def result_path(model_id: str, spec_id: str, treatment: str) -> Path:
    safe_model = model_id.replace("/", "__")
    return RESULTS_DIR / safe_model / f"{spec_id}__{treatment}.json"

def load_result(model_id: str, spec_id: str, treatment: str) -> dict | None:
    path = result_path(model_id, spec_id, treatment)

    if not path.exists():
        return None

    with open(path) as f:
        return json.load(f)


def already_done(model_id: str, spec_id: str, treatment: str) -> bool:
    return result_path(model_id, spec_id, treatment).exists()


def save_result(model_id: str, spec_id: str, treatment: str, data: dict):
    path = result_path(model_id, spec_id, treatment)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def run_single(model_id: str, provider: str, spec_id: str, treatment: str) -> dict:
    spec_text = load_spec(spec_id)
    estimation_prompt = build_estimation_prompt(spec_text, treatment)
    messages = [{"role": "user", "content": estimation_prompt}]

    result = {
        "model_id": model_id,
        "spec_id": spec_id,
        "treatment": treatment,
        "collection": {
            "estimation": {
                "status": "pending",
                "attempts": 1,
            }
        },
    }

    if treatment == "WD":
        result["collection"]["conversion"] = {
            "status": "pending",
            "attempts": 0,
        }

    save_result(model_id, spec_id, treatment, result)

    raw = call_model(model_id, provider, messages)

    result["estimation_response"] = raw
    result["usage"] = raw.get("usage")
    result["model_used"] = raw.get("model")

    save_result(model_id, spec_id, treatment, result)

    content = extract_content(raw)

    result["collection"]["estimation"]["status"] = "received"

    result["estimation_raw"] = content
    save_result(model_id, spec_id, treatment, result)

    if treatment == "WD":
        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content": build_conversion_prompt()})

        result["collection"]["conversion"]["attempts"] = 1
        save_result(model_id, spec_id, treatment, result)

        raw_conv = call_model(model_id, provider, messages)

        result["conversion_response"] = raw_conv
        save_result(model_id, spec_id, treatment, result)

        conv_content = extract_content(raw_conv)

        result["collection"]["conversion"]["status"] = "received"

        result["conversion_raw"] = conv_content
        save_result(model_id, spec_id, treatment, result)

    estimation = parse_json_response(content)
    result["estimation"] = estimation
    save_result(model_id, spec_id, treatment, result)

    if treatment == "WD":
        conversion = parse_json_response(conv_content)
        result["conversion"] = conversion
        save_result(model_id, spec_id, treatment, result)

    return result


def run_model(model: dict, randomization: dict):
    model_id   = model["id"]
    provider = model["provider"]
    model_rand = randomization["models"].get(model_id, {})

    sequences = [
        (model_rand.get("block_WH_first", []), ["WH", "WD"]),
        (model_rand.get("block_WD_first", []), ["WD", "WH"]),
    ]

    for specs, order in sequences:
        for spec_id in specs:
            for treatment in order:
                if already_done(model_id, spec_id, treatment):
                    continue
                try:
                    result = run_single(model_id, provider, spec_id, treatment)
                    save_result(model_id, spec_id, treatment, result)
                    print(f"  OK  {spec_id} {treatment}")
                except Exception as e:
                    print(f"  ERR {spec_id} {treatment}: {e}")
                time.sleep(0.5)  # respeito ao rate limit


def main():
    if not RAND_FILE.exists():
        print("randomization.json não encontrado. Execute randomize.py primeiro.")
        return

    models = load_models()
    if not models:
        print("Nenhum modelo em config/models.yaml")
        return

    randomization = load_randomization()

    for model in models:
        print(f"\n=== {model['name']} ({model['id']}) ===")
        run_model(model, randomization)

    print("\nExperimento concluído.")


if __name__ == "__main__":
    main()
