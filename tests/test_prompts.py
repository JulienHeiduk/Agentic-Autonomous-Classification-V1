import pytest

from aac.llm.prompts import PromptError, available_prompts, load_prompt, parse_prompt

TEXT = """---
version: 3
---
## system
You are {{role}}.
## user
Do {{task}} for {{ role }}.
"""


def test_parse_and_render():
    p = parse_prompt("t", TEXT)
    assert p.version == 3 and p.placeholders == {"role", "task"}
    msgs = p.messages(role="a critic", task="analysis")
    assert msgs == [
        {"role": "system", "content": "You are a critic."},
        {"role": "user", "content": "Do analysis for a critic."},
    ]


def test_missing_placeholder_is_an_error():
    p = parse_prompt("t", TEXT)
    with pytest.raises(PromptError, match="task"):
        p.messages(role="x")


def test_user_only_prompt_has_no_system_message():
    p = parse_prompt("t", "---\nversion: 1\n---\n## user\nhi\n")
    assert p.messages() == [{"role": "user", "content": "hi"}]


@pytest.mark.parametrize(
    "text",
    [
        "## user\nno front matter\n",
        "---\nversion: x\n---\n## user\nhi\n",
        "---\nversion: 1\n---\n## system\nonly\n",
    ],
)
def test_malformed_prompts(text):
    with pytest.raises(PromptError):
        parse_prompt("bad", text)


def test_shipped_prompts_parse_and_render():
    names = available_prompts()
    assert {"repair", "ping"} <= set(names)
    for name in names:
        p = load_prompt(name)
        assert p.version >= 1 and p.user
    assert "schema" in load_prompt("repair").placeholders
    with pytest.raises(PromptError, match="not found"):
        load_prompt("does-not-exist")
