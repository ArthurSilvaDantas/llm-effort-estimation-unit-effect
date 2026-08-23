import json
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

EFFORT_UNITS = {
    "WH": "work-hours",
    "WD": "workdays",
}


def load_template(filename: str) -> str:
    return (PROMPTS_DIR / filename).read_text()


def load_schema(filename: str) -> str:
    schema = json.loads((PROMPTS_DIR / filename).read_text())
    return json.dumps(schema, indent=2)


def build_estimation_prompt(specification_text: str, treatment: str) -> str:
    """
    treatment: "WH" (work-hours) ou "WD" (workdays)
    """
    effort_unit = EFFORT_UNITS[treatment]
    schema_str  = load_schema("estimation_schema.json")
    template    = load_template("estimation_template.txt")

    return template.replace("{{EFFORT_UNIT}}", effort_unit) \
                   .replace("{{SPECIFICATION_TEXT}}", specification_text) \
                   .replace("{{RESPONSE_SCHEMA}}", schema_str)


def build_conversion_prompt() -> str:
    schema_str = load_schema("conversion_schema.json")
    template   = load_template("conversion_template.txt")
    return template.replace("{{RESPONSE_SCHEMA}}", schema_str)
