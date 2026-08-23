"""
Extrai as 88 especificações dos arquivos brutos e salva em data/specifications/

Fonte:  data/raw/calikli_alhamed_2025_batches/
Saída:  data/specifications/<id>.txt

Cada arquivo bruto contém o prompt completo do estudo original (Çalikli & Alhamed, 2025),
que inclui a descrição do projeto + 8 user stories. Este script extrai apenas
essa parte, descartando as instruções experimentais do estudo original.

Uso:
    python scripts/extract_specifications.py           # extrai tudo
    python scripts/extract_specifications.py --force   # re-extrai mesmo se já existir
"""

import re
import sys
import argparse
from pathlib import Path

ROOT       = Path(__file__).parent.parent
SOURCE_DIR = ROOT / "data" / "raw" / "calikli_alhamed_2025_batches"
DEST_DIR   = ROOT / "data" / "specifications"

PROJECT_PREFIX = {
    "Spring XD":          "spring-xd",
    "Hyperledger Fabric": "hyperledger",
    "Mule":               "mule",
}


def extract_spec_text(content: str) -> str | None:
    """
    Extrai descrição do projeto + user stories do arquivo de batch.
    O conteúdo útil está após 'Requirements of the application ... are as follows:'.
    """
    match = re.search(
        r"'role': 'user', 'content': '(.*?)'\}]",
        content,
        re.DOTALL,
    )
    if not match:
        return None

    prompt_text = match.group(1)
    prompt_text = (
        prompt_text
        .replace("\\n", "\n")
        .replace("\\'", "'")
        .replace('\\"', '"')
    )

    spec_match = re.search(
        r"Requirements of the application .+? are as follows:\n(.+)",
        prompt_text,
        re.DOTALL,
    )
    if not spec_match:
        return None

    return spec_match.group(1).strip()


def main():
    parser = argparse.ArgumentParser(
        description="Extrai especificações dos arquivos brutos para data/specifications/"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-extrai mesmo se o arquivo de destino já existir"
    )
    args = parser.parse_args()

    if not SOURCE_DIR.exists():
        print(f"Diretório fonte não encontrado: {SOURCE_DIR}")
        print("Certifique-se de que data/raw/calikli_alhamed_2025_batches/ existe.")
        sys.exit(1)

    DEST_DIR.mkdir(parents=True, exist_ok=True)

    batch_files = sorted(SOURCE_DIR.glob("*.txt"))
    print(f"{len(batch_files)} arquivos brutos encontrados em {SOURCE_DIR.name}/")

    saved = skipped = errors = 0

    for batch_file in batch_files:
        stem = batch_file.stem  # ex: "Spring XD_batch_3"

        project = next(
            (p for p in PROJECT_PREFIX if stem.startswith(p)), None
        )
        if not project:
            print(f"  SKIP (projeto desconhecido): {stem}")
            continue

        num_match = re.search(r"_batch_(\d+)$", stem)
        if not num_match:
            print(f"  SKIP (sem número de batch): {stem}")
            continue

        batch_num = int(num_match.group(1))
        spec_id   = f"{PROJECT_PREFIX[project]}-{batch_num:03d}"
        dest_file = DEST_DIR / f"{spec_id}.txt"

        if dest_file.exists() and not args.force:
            skipped += 1
            continue

        content   = batch_file.read_text(encoding="utf-8", errors="replace")
        spec_text = extract_spec_text(content)

        if not spec_text:
            print(f"  ERR (extração falhou): {stem}")
            errors += 1
            continue

        dest_file.write_text(spec_text, encoding="utf-8")
        saved += 1
        print(f"  OK  {spec_id}")

    print()
    print(f"Resultado: {saved} extraídas, {skipped} já existiam, {errors} erros")
    print(f"Especificações em: {DEST_DIR}")


if __name__ == "__main__":
    main()
