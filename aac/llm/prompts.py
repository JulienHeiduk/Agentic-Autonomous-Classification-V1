"""Prompt files: one Markdown file per agent, versioned, no inline f-strings (README 5.4).

    ---
    version: 1
    ---
    ## system
    ...text with {{placeholders}}...
    ## user
    ...

``Prompt.messages(**values)`` substitutes ``{{name}}`` placeholders and refuses to render
with any left unfilled, so a typo in a placeholder name is an error and never reaches a model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent / "prompts"
_PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")
_FRONT = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.S)
_SECTION = re.compile(r"^## (system|user)\s*$", re.M)


class PromptError(ValueError):
    pass


@dataclass(frozen=True)
class Prompt:
    name: str
    version: int
    system: str
    user: str

    @property
    def placeholders(self) -> set[str]:
        return set(_PLACEHOLDER.findall(self.system)) | set(_PLACEHOLDER.findall(self.user))

    def render(self, text: str, values: dict[str, object]) -> str:
        def sub(match: re.Match[str]) -> str:
            key = match.group(1)
            if key not in values:
                raise PromptError(
                    f"prompt {self.name!r} v{self.version}: no value for {{{{{key}}}}}"
                )
            return str(values[key])

        return _PLACEHOLDER.sub(sub, text)

    def messages(self, **values: object) -> list[dict[str, str]]:
        out = []
        if self.system.strip():
            out.append({"role": "system", "content": self.render(self.system, values)})
        out.append({"role": "user", "content": self.render(self.user, values)})
        return out


def parse_prompt(name: str, text: str) -> Prompt:
    front = _FRONT.match(text)
    if not front:
        raise PromptError(f"prompt {name!r}: missing front matter with a version")
    meta = dict(line.split(":", 1) for line in front.group(1).splitlines() if ":" in line)
    try:
        version = int(str(meta.get("version", "")).strip())
    except ValueError:
        raise PromptError(f"prompt {name!r}: front matter needs an integer version") from None
    body = text[front.end() :]
    parts = _SECTION.split(body)
    sections: dict[str, str] = {}
    for i in range(1, len(parts), 2):
        sections[parts[i]] = parts[i + 1].strip()
    if "user" not in sections:
        raise PromptError(f"prompt {name!r}: no '## user' section")
    return Prompt(
        name=name, version=version, system=sections.get("system", ""), user=sections["user"]
    )


@cache
def load_prompt(name: str, directory: Path | None = None) -> Prompt:
    path = (directory or PROMPTS_DIR) / f"{name}.md"
    if not path.is_file():
        raise PromptError(f"prompt {name!r} not found at {path}")
    return parse_prompt(name, path.read_text(encoding="utf-8"))


def available_prompts(directory: Path | None = None) -> list[str]:
    return sorted(p.stem for p in (directory or PROMPTS_DIR).glob("*.md"))
