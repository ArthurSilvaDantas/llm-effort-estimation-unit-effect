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

from openrouter_client import load_models, call_model, extract_content, parse_json_response, is_retryable_error
from prompt_builder import build_estimation_prompt, build_conversion_prompt

ROOT        = Path(__file__).parent.parent
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

RAND_FILE   = Path(__file__).parent / "randomization.json"
MAX_ATTEMPTS = 2


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


def get_processing_status(result: dict, stage: str) -> str:
    processing = result.get("processing", {})

    if stage in processing:
        return processing[stage]

    # Compatibilidade com registros criados antes do campo "processing".
    if stage in result:
        return "parsed"

    return "pending"


def execution_complete(
    model_id: str,
    spec_id: str,
    treatment: str,
) -> bool:
    result = load_result(model_id, spec_id, treatment)

    if result is None:
        return False

    collection = result.get("collection")

    # Resultados anteriores à introdução dos estados são preservados.
    if collection is None:
        return True

    estimation_status = collection["estimation"]["status"]

    if estimation_status == "failed":
        return True

    if estimation_status == "pending":
        return False

    if estimation_status != "received":
        raise ValueError(
            f"Estado de estimativa desconhecido: {estimation_status}"
        )

    estimation_processing = get_processing_status(
        result,
        "estimation",
    )

    if treatment == "WH":
        return estimation_processing in {"parsed", "failed"}

    conversion_status = collection["conversion"]["status"]

    if conversion_status == "pending":
        return False

    if conversion_status == "failed":
        return estimation_processing in {"parsed", "failed"}

    if conversion_status != "received":
        raise ValueError(
            f"Estado de conversão desconhecido: {conversion_status}"
        )

    conversion_processing = get_processing_status(
        result,
        "conversion",
    )

    return (
        estimation_processing in {"parsed", "failed"}
        and conversion_processing in {"parsed", "failed"}
    )


def save_result(model_id: str, spec_id: str, treatment: str, data: dict):
    path = result_path(model_id, spec_id, treatment)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def record_failure(
    result: dict,
    stage: str,
    attempt: int,
    error: Exception,
    retryable: bool,
    outcome: str,
):
    result.setdefault("attempt_log", []).append({
        "stage": stage,
        "attempt": attempt,
        "outcome": outcome,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "retryable": retryable,
    })


def call_with_retry(
    model_id: str,
    provider: str,
    spec_id: str,
    treatment: str,
    result: dict,
    stage: str,
    messages: list[dict],
) -> dict:
    state = result["collection"][stage]

    while state["attempts"] < MAX_ATTEMPTS:
        state["attempts"] += 1
        attempt = state["attempts"]

        save_result(model_id, spec_id, treatment, result)

        try:
            raw_response = call_model(model_id, provider, messages)

        except Exception as error:
            retryable = is_retryable_error(error)

            record_failure(
                result,
                stage,
                attempt,
                error,
                retryable,
                "technical_failure",
            )

            if not retryable or attempt >= MAX_ATTEMPTS:
                state["status"] = "failed"

            save_result(model_id, spec_id, treatment, result)

            if state["status"] == "failed":
                raise

            continue

        result[f"{stage}_response"] = raw_response
        save_result(model_id, spec_id, treatment, result)

        return raw_response

    state["status"] = "failed"
    save_result(model_id, spec_id, treatment, result)

    raise RuntimeError(
        f"Nenhuma tentativa restante para a etapa {stage}."
    )


def collect_stage(
    model_id: str,
    provider: str,
    spec_id: str,
    treatment: str,
    result: dict,
    stage: str,
    messages: list[dict],
) -> str | None:
    state = result["collection"][stage]

    response_key = f"{stage}_response"
    content_key = f"{stage}_raw"

    if state["status"] == "received":
        return result[content_key]

    if state["status"] == "failed":
        return None

    if state["status"] != "pending":
        raise ValueError(
            f"Estado de coleta desconhecido para {stage}: "
            f"{state['status']}"
        )

    if response_key in result:
        raw_response = result[response_key]
    else:
        raw_response = call_with_retry(
            model_id,
            provider,
            spec_id,
            treatment,
            result,
            stage,
            messages,
        )

    try:
        content = extract_content(raw_response)

    except Exception as error:
        state["status"] = "failed"

        record_failure(
            result,
            stage,
            state["attempts"],
            error,
            False,
            "response_processing_failure",
        )

        save_result(model_id, spec_id, treatment, result)
        raise

    state["status"] = "received"
    result[content_key] = content

    save_result(model_id, spec_id, treatment, result)

    return content


def process_json_stage(
    model_id: str,
    spec_id: str,
    treatment: str,
    result: dict,
    stage: str,
    content: str,
) -> Exception | None:
    processing = result["processing"]
    status = processing[stage]

    if status in {"parsed", "failed"}:
        return None

    if status != "pending":
        raise ValueError(
            f"Estado de processamento desconhecido para {stage}: "
            f"{status}"
        )

    try:
        parsed = parse_json_response(content)
    except Exception as error:
        processing[stage] = "failed"

        result.setdefault("processing_log", []).append({
            "stage": stage,
            "outcome": "parse_failure",
            "error_type": type(error).__name__,
            "error_message": str(error),
        })

        save_result(model_id, spec_id, treatment, result)

        return error

    result[stage] = parsed
    processing[stage] = "parsed"

    save_result(model_id, spec_id, treatment, result)

    return None


def run_single(
    model_id: str,
    provider: str,
    spec_id: str,
    treatment: str,
) -> dict:
    spec_text = load_spec(spec_id)
    estimation_prompt = build_estimation_prompt(spec_text, treatment)

    estimation_messages = [
        {"role": "user", "content": estimation_prompt}
    ]

    result = load_result(model_id, spec_id, treatment)

    if result is None:
        result = {
            "model_id": model_id,
            "spec_id": spec_id,
            "treatment": treatment,
            "collection": {
                "estimation": {
                    "status": "pending",
                    "attempts": 0,
                }
            },
            "processing": {
                "estimation": "pending",
            },
            "attempt_log": [],
            "processing_log": [],
        }

        if treatment == "WD":
            result["collection"]["conversion"] = {
                "status": "pending",
                "attempts": 0,
            }
            result["processing"]["conversion"] = "pending"

        save_result(model_id, spec_id, treatment, result)

    else:
        result.setdefault("attempt_log", [])
        result.setdefault("processing_log", [])

        processing = result.setdefault("processing", {})

        processing.setdefault(
            "estimation",
            "parsed" if "estimation" in result else "pending",
        )

        if treatment == "WD":
            processing.setdefault(
                "conversion",
                "parsed" if "conversion" in result else "pending",
            )

        save_result(model_id, spec_id, treatment, result)

    content = collect_stage(
        model_id,
        provider,
        spec_id,
        treatment,
        result,
        "estimation",
        estimation_messages,
    )

    if content is None:
        return result

    estimation_response = result.get("estimation_response")

    if estimation_response is not None:
        result["usage"] = estimation_response.get("usage")
        result["model_used"] = estimation_response.get("model")
        save_result(model_id, spec_id, treatment, result)

    conv_content = None
    collection_error = None

    if treatment == "WD":
        conversion_messages = [
            {"role": "user", "content": estimation_prompt},
            {"role": "assistant", "content": content},
            {"role": "user", "content": build_conversion_prompt()},
        ]

        try:
            conv_content = collect_stage(
                model_id,
                provider,
                spec_id,
                treatment,
                result,
                "conversion",
                conversion_messages,
            )
        except Exception as error:
            collection_error = error

    processing_errors = []

    estimation_error = process_json_stage(
        model_id,
        spec_id,
        treatment,
        result,
        "estimation",
        content,
    )

    if estimation_error is not None:
        processing_errors.append(estimation_error)

    if treatment == "WD" and conv_content is not None:
        conversion_error = process_json_stage(
            model_id,
            spec_id,
            treatment,
            result,
            "conversion",
            conv_content,
        )

        if conversion_error is not None:
            processing_errors.append(conversion_error)

    if collection_error is not None:
        raise collection_error

    if processing_errors:
        raise processing_errors[0]

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
                if execution_complete(model_id, spec_id, treatment):
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
