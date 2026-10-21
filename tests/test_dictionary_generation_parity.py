"""Compare generated artifacts and entry points with pinned thesis behavior.

All responses are recorded fixtures; neither implementation opens an LLM client.
"""

# The executed oracle is code from the pinned local thesis commit.
# ruff: noqa: S102

import ast
import copy
import importlib
import json
import subprocess
import types
from datetime import datetime
from pathlib import Path

import pytest

from gpr_zero.llm import dictionary_generator as current

ROOT = Path(__file__).resolve().parents[1]
THESIS_COMMIT = "290b7c4c8eb900a712384deea0b68e2e47009624"


class FrozenDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 10, 21, 9, 30, 0, tzinfo=tz)


def load_thesis_module(path, **dependencies):
    source = subprocess.check_output(
        ["git", "show", f"{THESIS_COMMIT}:{path}"], cwd=ROOT, text=True
    )
    tree = ast.parse(source, filename=path)
    # Replace thesis imports with the recorded prompt adapter and generator.
    # Current workflow modules are imported normally through the installed package.
    tree.body = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.ImportFrom)
            and (node.module or "").startswith(("gpr_zero.", "src.gpr_zero."))
        )
    ]
    module = types.ModuleType(f"thesis_{Path(path).stem}")
    module.__dict__.update(dependencies)
    exec(compile(tree, path, "exec"), module.__dict__)
    return module


@pytest.fixture
def thesis():
    return load_thesis_module(
        "src/gpr_zero/llm/dictionary_generator.py",
        PromptManager=object,
    )


@pytest.fixture
def prompt(tmp_path):
    path = tmp_path / "prompt.json"
    path.write_text(
        json.dumps(
            {
                "metadata": {
                    "description": "Recorded prompt",
                    "target_context": "Taiwan",
                },
                "prompt": {"input": [{"role": "user", "content": "Generate"}]},
            }
        ),
        encoding="utf-8",
    )
    return path


def recorded_response(kind):
    dictionary = {
        "word_sets": {"risk": ["軍演", "制裁"]},
        "categories": {f"Category {i}": "risk" for i in range(8)},
    }
    text = json.dumps(dictionary, ensure_ascii=False)
    if kind == "fenced":
        text = f"```json\n{text}\n```"
    elif kind == "prose":
        text = f"Here is the dictionary:\n{text}\nEnd."
    elif kind == "invalid":
        text = "The recorded response has no JSON."
    elif kind == "missing_keys":
        text = '{"word_sets": {"risk": ["軍演"]}}'
    elif kind == "invalid_categories":
        dictionary["categories"] = {"one category": "risk"}
        text = json.dumps(dictionary, ensure_ascii=False)
    return {
        "text": text,
        # Thesis parses text, even when a structured result is present.
        "structured_output": {"categories": {}, "word_sets": {}},
        "metadata": {"model": "recorded-model", "request_id": "recorded-request"},
    }


def prompt_adapter(response, init_calls, calls):
    class RecordedPromptManager:
        model_name = "configured-model"
        temperature = 0.2

        def __init__(self, **kwargs):
            init_calls.append(kwargs)

        def call_llm_with_prompt_file(self, **kwargs):
            calls.append(kwargs)
            return copy.deepcopy(response)

        def call_llm(self, *args, **kwargs):
            calls.append((args, kwargs))
            return copy.deepcopy(response)

    return RecordedPromptManager


def artifacts(directory):
    return {path.name: path.read_bytes() for path in sorted(directory.glob("*.json"))}


@pytest.mark.parametrize("kind", ["json", "fenced", "prose", "invalid", "missing_keys"])
@pytest.mark.parametrize("strategy", ["prompt_file", "translation", "direct", "guided"])
def test_dictionary_artifacts_match_thesis(
    thesis, monkeypatch, tmp_path, prompt, kind, strategy
):
    observed = []
    for index, module in enumerate([thesis, current]):
        monkeypatch.setattr(module, "datetime", FrozenDatetime)
        init_calls, calls = [], []
        manager = prompt_adapter(recorded_response(kind), init_calls, calls)()
        output_dir = tmp_path / f"artifacts-{index}"
        generator = module.DictionaryGenerator(manager, str(output_dir))
        result = generator.generate_dictionary(
            strategy=strategy,
            name="recorded",
            model="requested-model",
            temperature=0.0,
            prompt_file_path=prompt,
            template_args_list=[{"role": "Researcher"}],
        )
        observed.append((result, artifacts(output_dir), calls))
    assert observed[1] == observed[0]
    saved = json.loads(observed[1][1]["recorded.json"])
    assert "filename" not in saved["metadata"]
    assert observed[1][0]["metadata"]["filename"] == "recorded.json"


@pytest.mark.parametrize("step", [1, 2])
@pytest.mark.parametrize(
    "kind", ["json", "fenced", "invalid", "missing_keys", "invalid_categories"]
)
def test_batch_artifacts_and_provider_selection_match_thesis(
    thesis, monkeypatch, tmp_path, prompt, step, kind
):
    monkeypatch.chdir(tmp_path)
    observed = []
    for index, generator_module in enumerate([thesis, current]):
        monkeypatch.setattr(generator_module, "datetime", FrozenDatetime)
        init_calls, calls = [], []
        manager_class = prompt_adapter(recorded_response(kind), init_calls, calls)
        if index == 0:
            module = load_thesis_module(
                f"scripts/batch_generate_step{step}.py",
                PromptManager=manager_class,
                DictionaryGenerator=generator_module.DictionaryGenerator,
            )
        else:
            module = importlib.import_module(
                f"gpr_zero.workflows.batch_generate_step{step}"
            )
            monkeypatch.setattr(
                module, "DictionaryGenerator", generator_module.DictionaryGenerator
            )
        monkeypatch.setattr(
            module, "PROVIDERS", {"recorded": {"model": "requested-model", "runs": 1}}
        )
        monkeypatch.setattr(
            module, "ROLE_MAP", {"researcher": "Researcher"}, raising=False
        )
        monkeypatch.setattr(module, "PROMPT_FILE", prompt)
        output_dir = tmp_path / f"batch-{index}"
        if index == 0:
            module.OUTPUT_DIR = output_dir
            module.main()
            assert init_calls == [{}]
        else:
            config = tmp_path / "llm_config.yaml"
            project = types.SimpleNamespace(
                root=tmp_path,
                llm_config=config,
                data_path=lambda *parts, output_dir=output_dir: output_dir,
                prompt_manager=lambda manager_class=manager_class, config=config: (
                    manager_class(config_file_path=config)
                ),
            )
            module.main(project)
            assert init_calls == [{"config_file_path": config}]
        observed.append((artifacts(output_dir), calls))
    assert observed[1] == observed[0]
    if step == 2 and kind in {"invalid", "missing_keys", "invalid_categories"}:
        assert len(observed[1][0]) == 1  # Thesis saves failures before validation.
        assert not any(name.endswith("_meta.json") for name in observed[1][0])
