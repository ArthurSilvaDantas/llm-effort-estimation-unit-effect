"""
Estudo Piloto

Executa com o modelo mais utilizado do ranking (posição 1 na models.yaml).
Usa 3 especificações — uma de cada projeto de origem do corpus:
  - Spring XD
  - Hyperledger Fabric
  - Mule

Objetivos:
  - Verificar conformidade com os schemas JSON
  - Checar ocorrência de recusas ou respostas fora do escopo
  - Avaliar plausibilidade dos valores retornados
  - Validar adequação dos instrumentos de registro

Os dados do piloto NÃO entram na análise principal.
As 3 especificações PODERÃO integrar o experimento principal,
mas serão submetidas a novas execuções após o congelamento dos instrumentos.
"""

import sys
import json
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from openrouter_client import load_models, call_model, extract_content, parse_json_response
from prompt_builder import build_estimation_prompt, build_conversion_prompt

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# IDs das 3 especificações piloto (uma por projeto de origem)
# Preencha após organizar o dataset
PILOT_SPEC_IDS = [
    "spring-xd-001",       # Spring XD
    "hyperledger-001",     # Hyperledger Fabric
    "mule-001",            # Mule
]

TREATMENTS = ["WH", "WD"]


def load_spec(spec_id: str) -> str:
    specs_path = Path(__file__).parent.parent / "data" / "specifications" / f"{spec_id}.txt"
    return specs_path.read_text()


def run_single(model_id: str, spec_id: str, treatment: str) -> dict:
    spec_text = load_spec(spec_id)
    estimation_prompt = build_estimation_prompt(spec_text, treatment)

    messages = [{"role": "user", "content": estimation_prompt}]
    raw_estimation = call_model(model_id, messages)
    estimation_content = extract_content(raw_estimation)
    estimation_json = parse_json_response(estimation_content)

    result = {
        "model_id": model_id,
        "spec_id": spec_id,
        "treatment": treatment,
        "estimation_raw": estimation_content,
        "estimation": estimation_json,
        "usage": raw_estimation.get("usage"),
    }

    # Para T_WD: obter fator de conversão C_im na mesma sessão
    if treatment == "WD":
        messages.append({"role": "assistant", "content": estimation_content})
        messages.append({"role": "user", "content": build_conversion_prompt()})

        raw_conversion = call_model(model_id, messages)
        conversion_content = extract_content(raw_conversion)
        conversion_json = parse_json_response(conversion_content)

        result["conversion_raw"] = conversion_content
        result["conversion"] = conversion_json

    return result


def main():
    models = load_models()
    if not models:
        print("Nenhum modelo configurado em config/models.yaml")
        return

    # Piloto usa apenas o modelo #1 do ranking
    pilot_model = models[0]
    model_id = pilot_model["id"]
    print(f"Modelo piloto: {pilot_model['name']} ({model_id})")

    all_results = []
    for spec_id in PILOT_SPEC_IDS:
        for treatment in TREATMENTS:
            print(f"  spec={spec_id} treatment={treatment} ... ", end="", flush=True)
            try:
                result = run_single(model_id, spec_id, treatment)
                all_results.append(result)
                print("OK")
            except Exception as e:
                print(f"ERRO: {e}")

    output_path = RESULTS_DIR / "pilot_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResultados salvos em {output_path}")


if __name__ == "__main__":
    main()
