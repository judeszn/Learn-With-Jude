"""Extract skills from CV using Anthropic Claude API."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import anthropic
from dotenv import load_dotenv


class CVExtractorError(RuntimeError):
    """Raised when CV extraction fails."""


def load_cv_text(cv_path: Path) -> str:
    """Read CV file and extract text (supports .txt, .pdf via text conversion)."""

    if not cv_path.exists():
        raise CVExtractorError(f"CV file not found: {cv_path}")

    if cv_path.suffix.lower() == ".txt":
        return cv_path.read_text(encoding="utf-8")
    elif cv_path.suffix.lower() == ".pdf":
        try:
            import PyPDF2
            with cv_path.open("rb") as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                return text
        except ImportError:
            raise CVExtractorError(
                "PyPDF2 not installed. Install it with: pip install PyPDF2"
            )
    else:
        raise CVExtractorError(
            f"Unsupported file format: {cv_path.suffix}. Use .txt or .pdf"
        )


def extract_cv_with_llm(cv_text: str, *, api_key: str, prompt: str) -> dict[str, Any]:
    """Send CV text to Anthropic Claude API with custom prompt."""

    client = anthropic.Anthropic(api_key=api_key)

    message = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=2048,
        messages=[
            {
                "role": "user",
                "content": f"{prompt}\n\n---CV CONTENT---\n\n{cv_text}",
            },
        ],
    )

    response_text = message.content[0].text

    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        return {"raw_response": response_text}


def load_api_key() -> str:
    """Load Anthropic API key from environment."""

    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise CVExtractorError(
            "Missing ANTHROPIC_API_KEY in environment. Set it in your .env file."
        )
    return api_key


def extract_skills(cv_text: str, *, api_key: str) -> dict[str, Any]:
    """Extract skills from CV text using Claude."""

    prompt = """Extract all skills from this CV. Categorize them as follows:
- Technical Skills: programming languages, tools, frameworks, technologies
- Professional Skills: soft skills, methodologies, certifications
- Domain Skills: industry-specific expertise

Return ONLY valid JSON with this structure:
{
  "technical_skills": ["skill1", "skill2", ...],
  "professional_skills": ["skill1", "skill2", ...],
  "domain_skills": ["skill1", "skill2", ...]
}"""

    return extract_cv_with_llm(cv_text, api_key=api_key, prompt=prompt)


def main() -> None:
    cv_pdf_path = Path("bolu.pdf")  # Change this to your CV file path
    
    api_key = load_api_key()
    cv_text = load_cv_text(cv_pdf_path)
    result = extract_skills(cv_text, api_key=api_key)
    
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()