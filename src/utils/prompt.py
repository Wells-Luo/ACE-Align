from __future__ import annotations

import re
from typing import Dict, Optional


def build_survey_prompt(
    *,
    question_text: str,
    options: Dict[str, str],
    country: str,
    gender: Optional[str] = None,
    residence: Optional[str] = None,
    education: Optional[str] = None,
    marital_status: Optional[str] = None,
) -> str:
    persona_parts: list[str] = []

    if gender:
        persona_parts.append(f"a {gender.lower()}")

    if residence:
        residence_lower = residence.lower()
        if residence_lower == "urban":
            residence_text = "an urban area"
        elif residence_lower == "rural":
            residence_text = "a rural area"
        else:
            residence_text = residence_lower
        persona_parts.append(f"currently living in {residence_text} in {country}")
    else:
        persona_parts.append(f"living in {country}")

    first_sentence = f"You are {' '.join(persona_parts)}."

    extra: list[str] = []
    if education:
        if education.lower() == "college":
            extra.append("have completed a college education")
        else:
            extra.append("have not completed a college education")
    if marital_status:
        normalized = marital_status.lower().replace("_", " ")
        if "married" in normalized and "not" not in normalized:
            extra.append("are married")
        else:
            extra.append("are not married")

    if extra:
        persona = f"{first_sentence} You {', and you '.join(extra)}."
    else:
        persona = first_sentence

    def format_option(key: str, value: str) -> str:
        value = str(value).strip()
        if value.isdigit():
            return f"{key}: Point {value} on scale"
        cleaned = re.sub(r"^\d+[,.]?\s*", "", value)
        return f"{key}: {cleaned or value}"

    option_text = "\n".join(format_option(k, v) for k, v in options.items())
    return (
        f"{persona}\n"
        "Please answer the following survey question, and ensure that your choice reflects "
        "the persona described above. Do not add explanations or additional content. "
        "Respond with the number of the option you choose.\n\n"
        f"Question: {question_text}\n\n"
        f"Options (scale from 1 to {len(options)}):\n"
        f"{option_text}\n\n"
        "Answer:\n"
    )

