"""
Gera e persiste a randomização do experimento.

Para cada modelo:
  - 88 especificações divididas em dois blocos de 44
  - Bloco A: sequência T_WH → T_WD
  - Bloco B: sequência T_WD → T_WH
  - Atribuição aleatória e balanceada entre blocos

O resultado é salvo em experiment/randomization.json e deve ser
preservado no pacote de replicação antes do início da coleta.
"""

import json
import random
import yaml
from pathlib import Path

ROOT = Path(__file__).parent.parent


def load_spec_ids() -> list[str]:
    specs_dir = ROOT / "data" / "specifications"
    return sorted(p.stem for p in specs_dir.glob("*.txt"))


def load_models() -> list[dict]:
    with open(ROOT / "config" / "models.yaml") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("models") or []


def generate_randomization(seed: int = 42) -> dict:
    rng = random.Random(seed)
    spec_ids = load_spec_ids()
    models = load_models()

    if len(spec_ids) != 88:
        print(f"Aviso: esperado 88 especificações, encontrado {len(spec_ids)}")

    randomization = {"seed": seed, "models": {}}

    for model in models:
        model_id = model["id"]
        specs = spec_ids.copy()
        rng.shuffle(specs)

        block_a = specs[:44]   # T_WH → T_WD
        block_b = specs[44:]   # T_WD → T_WH

        randomization["models"][model_id] = {
            "block_WH_first": block_a,
            "block_WD_first": block_b,
        }

    return randomization


def main():
    out_path = Path(__file__).parent / "randomization.json"
    if out_path.exists():
        print("randomization.json já existe. Delete manualmente para regenerar.")
        return

    rand = generate_randomization()
    with open(out_path, "w") as f:
        json.dump(rand, f, indent=2)
    print(f"Randomização gerada com seed={rand['seed']} → {out_path}")


if __name__ == "__main__":
    main()
