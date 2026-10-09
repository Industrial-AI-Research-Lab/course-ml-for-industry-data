"""Prompts. Kept in one place so that replay keys stay stable: change a prompt, re-run live."""

from __future__ import annotations

from .schema import fields_description

SYSTEM_EXTRACTION = (
    "You extract structured data from maintenance log entries written by technicians at an industrial plant. "
    "Entries can be in Russian, English or a mix, with abbreviations and typos. "
    "Use only facts that are stated in the entry. If a value is not stated, use null; never guess. "
    "Equipment tags must use Latin letters, e.g. 'P-101' even if the technician wrote 'Р-101' or 'P101'."
)

USER_EXTRACTION = "Maintenance log entry:\n<<<\n{text}\n>>>"

# The baseline has no schema in the API call: the model only sees a field list in the prompt.
BASELINE_INSTRUCTIONS = (
    "Return the result as JSON with these fields:\n{fields}\n"
    "Return only JSON."
)

RETRY_MESSAGE = (
    "Your previous answer has these problems:\n{errors}\n"
    "Fix them and return the corrected JSON only."
)


# Self-hosted servers (vLLM, Ollama) use the JSON Schema only to constrain tokens: the model never
# reads the field descriptions. OpenAI-style APIs put the schema into the context. So we repeat the
# field definitions in the prompt: the schema guarantees the format, the prompt explains the meaning.
FIELD_GUIDE = "Field definitions:\n{fields}"


def extraction_messages(text: str, field_guide: bool = True) -> list[dict[str, str]]:
    system = SYSTEM_EXTRACTION
    if field_guide:
        system += "\n\n" + FIELD_GUIDE.format(fields=fields_description())
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": USER_EXTRACTION.format(text=text)},
    ]


def baseline_messages(text: str) -> list[dict[str, str]]:
    system = SYSTEM_EXTRACTION + "\n\n" + BASELINE_INSTRUCTIONS.format(fields=fields_description())
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": USER_EXTRACTION.format(text=text)},
    ]


def retry_messages(previous: list[dict[str, str]], answer: str, errors: list[str]) -> list[dict[str, str]]:
    error_text = "\n".join(f"- {e}" for e in errors)
    return previous + [
        {"role": "assistant", "content": answer},
        {"role": "user", "content": RETRY_MESSAGE.format(errors=error_text)},
    ]
