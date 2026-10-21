"""Compare offline prompt requests, results and logs with the thesis implementation."""

import copy
import json
import subprocess
import types
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpr_zero.llm import prompt_manager as current


THESIS_COMMIT = "290b7c4c8eb900a712384deea0b68e2e47009624"
ROOT = Path(__file__).resolve().parents[1]


class FrozenDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 10, 21, 9, 30, 0, tzinfo=tz)


@pytest.fixture
def thesis():
    source = subprocess.check_output(
        ["git", "show", f"{THESIS_COMMIT}:src/gpr_zero/llm/prompt_manager.py"],
        cwd=ROOT,
        text=True,
    )
    module = types.ModuleType("thesis_prompt_manager")
    exec(compile(source, "thesis_prompt_manager.py", "exec"), module.__dict__)
    return module


@pytest.mark.parametrize("provider", ["openai", "openrouter"])
@pytest.mark.parametrize("style", ["file", "ad_hoc"])
@pytest.mark.parametrize("response_kind", ["json", "invalid_json", "error", "no_key"])
def test_prompt_results_requests_and_logs_match_thesis(
    thesis, monkeypatch, tmp_path, provider, style, response_kind
):
    config = tmp_path / "config.yaml"
    config.write_text(
        f"providers:\n  default: {provider}\nmodels:\n  default: configured-model\n",
        encoding="utf-8",
    )
    prompt = tmp_path / "prompt.json"
    prompt.write_text(json.dumps({
        "prompt": {
            "input": [
                {"role": "system", "content": "Role: {role}"},
                {"role": "user", "content": "Analyze {topic}; keep {missing}."},
            ],
            "text": {"format": {"type": "json_object"}},
        },
        "api_config": {"model": "template-model", "temperature": 0.3},
    }), encoding="utf-8")

    observed = []
    for index, module in enumerate([thesis, current]):
        monkeypatch.setattr(module, "datetime", FrozenDatetime)
        for key in ("OPENAI_API_KEY", "OPENROUTER_API_KEY"):
            if response_kind == "no_key":
                monkeypatch.delenv(key, raising=False)
            else:
                monkeypatch.setenv(key, "offline-test-key")
        # Never instantiate a network client, even when credentials are present.
        monkeypatch.setattr(module.openai, "OpenAI", lambda **kwargs: object())
        manager = module.PromptManager(
            config_file_path=config, log_dir=str(tmp_path / f"logs-{index}")
        )
        requests = []

        def fake_call(params):
            requests.append(copy.deepcopy(params))
            if response_kind == "error":
                raise RuntimeError("recorded provider failure")
            content = '{"word_sets": {"risk": ["軍演"]}}' if response_kind == "json" else "invalid JSON"
            raw = {
                "id": "recorded-response", "model": "recorded-model",
                "usage": {"total_tokens": 12},
                "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            }
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                model_dump=lambda: copy.deepcopy(raw),
            )

        monkeypatch.setattr(manager, "_make_api_call", fake_call)
        if style == "file":
            result = manager.call_llm_with_prompt_file(
                prompt, [{"role": "researcher"}, {"topic": "Taiwan"}],
                model_override="override-model", temperature_override=0.0,
            )
        else:
            result = manager.call_llm(
                "Analyze {topic}; keep {missing}.", system_message="Role: {role}",
                params={"model": "override-model", "temperature": 0.0},
                template_name="recorded", template_args={"role": "researcher", "topic": "Taiwan"},
            )
        logs = {p.name: json.loads(p.read_text()) for p in manager.log_dir.glob("*.json")}
        observed.append((result, requests, logs))

    assert observed[1] == observed[0]
    assert len(observed[1][2]) == 2
    assert len(observed[1][1]) == (0 if response_kind == "no_key" else 1)
