from functools import lru_cache
from pathlib import Path

_DIR = Path(__file__).parent


@lru_cache
def load_prompt(name: str) -> str:
    """The system prompt of an agent (prompts/<name>.md)."""
    return (_DIR / f"{name}.md").read_text(encoding="utf-8").strip()
