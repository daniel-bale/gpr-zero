"""Exercise project discovery and the public command surface without production data."""

import importlib
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpr_zero import cli
from gpr_zero.project import Project, get_project, use_project

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    monkeypatch.delenv("GPR_PROJECT", raising=False)
    monkeypatch.delenv("GPR_DB_PATH", raising=False)


def test_paths_resolve_from_config_not_working_directory(tmp_path, monkeypatch):
    config = tmp_path / "gpr.toml"
    config.write_text('[paths]\ndata = "corpus"\nllm_config = "provider.yaml"\n')
    nested = tmp_path / "analysis" / "nested"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    project = Project.load()
    assert project.root == tmp_path
    assert project.database == tmp_path / "corpus/news/stage2_enhanced_full.duckdb"
    assert project.llm_config == tmp_path / "provider.yaml"
    assert project.dictionaries_dir == tmp_path / "corpus/dictionaries/all_dictionaries"
    assert (
        project.step2_dictionaries_dir == tmp_path / "corpus/dictionaries/step2_batch"
    )
    assert not project.data_dir.exists()


def test_project_selection_and_database_come_from_project_configuration(
    tmp_path, monkeypatch
):
    config = tmp_path / "research.toml"
    config.write_text('[paths]\ndatabase = "configured.duckdb"\n')
    monkeypatch.setenv("GPR_PROJECT", str(config))
    assert Project.load().database == tmp_path / "configured.duckdb"
    monkeypatch.setenv("GPR_DB_PATH", "overridden.duckdb")
    assert Project.load().database == tmp_path / "configured.duckdb"
    other = tmp_path / "other.toml"
    other.write_text('[paths]\ndata = "other"\n')
    assert Project.load(other).data_dir == tmp_path / "other"
    assert (
        Project.load(other).database
        == tmp_path / "other/news/stage2_enhanced_full.duckdb"
    )


@pytest.mark.parametrize(
    "contents",
    [
        '[path]\ndata = "typo"',
        '[paths]\ndat = "typo"',
        'paths = "not a table"',
        "[paths]\ndata = false",
        "[paths]\ndatabase = 0",
        '[paths]\nllm_config = ""',
    ],
)
def test_invalid_configuration_is_rejected(tmp_path, contents):
    config = tmp_path / "gpr.toml"
    config.write_text(contents)
    with pytest.raises(ValueError):
        Project.load(config)


def test_missing_explicit_configuration_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        Project.load(tmp_path / "missing.toml")


def test_project_context_restored_after_nested_failure(tmp_path, monkeypatch):
    config = tmp_path / "gpr.toml"
    config.write_text("")
    project = Project.load(config)
    other = Project(
        tmp_path, tmp_path / "other", tmp_path / "other.db", tmp_path / "llm.yaml"
    )
    monkeypatch.setenv("GPR_PROJECT", str(config))
    with use_project(project):
        with pytest.raises(RuntimeError), use_project(other):
            assert get_project() is other
            raise RuntimeError("workflow failed")
        assert get_project() is project
    assert get_project() == project


TASKS = [
    (group, action, module, function)
    for group, actions in cli.WORKFLOWS.items()
    for action, (module, function, _) in actions.items()
]


@pytest.mark.parametrize("group,action,module,function", TASKS)
def test_commands_dispatch_with_one_active_project(
    tmp_path, monkeypatch, group, action, module, function
):
    config = tmp_path / "gpr.toml"
    config.write_text('[paths]\ndata = "fixtures"\n')
    invoked = []

    def load(name):
        assert name == f"gpr_zero.workflows.{module}"
        return SimpleNamespace(
            **{function: lambda argv=None: invoked.append((get_project(), argv))}
        )

    monkeypatch.setattr(cli.importlib, "import_module", load)
    assert cli.main(["--project", str(config), group, action]) == 0
    assert invoked == [
        (Project.load(config), [] if group in {"ingest", "verify"} else None)
    ]
    assert not (tmp_path / "fixtures").exists()


def test_workflow_failures_reach_command_exit_status(tmp_path, monkeypatch, caplog):
    config = tmp_path / "gpr.toml"
    config.write_text("")

    def fail():
        raise ValueError("cache vocabulary mismatch")

    monkeypatch.setattr(
        cli.importlib, "import_module", lambda _: SimpleNamespace(main=fail)
    )
    assert cli.main(["--project", str(config), "index", "all"]) == 1
    assert "cache vocabulary mismatch" in caplog.text


def test_config_command_does_not_open_database(tmp_path, monkeypatch, capsys):
    config = tmp_path / "gpr.toml"
    config.write_text("")
    monkeypatch.setattr(
        Project, "connect", lambda *a, **k: pytest.fail("config opened DB")
    )
    assert cli.main(["--project", str(config), "config"]) == 0
    resolved = json.loads(capsys.readouterr().out)
    assert resolved["database"] == str(
        tmp_path / "data/news/stage2_enhanced_full.duckdb"
    )
    assert set(tmp_path.iterdir()) == {config}


@pytest.mark.parametrize("group,action,module,function", TASKS)
def test_catalog_resolves_installed_workflow_modules(group, action, module, function):
    current = importlib.import_module(f"gpr_zero.workflows.{module}")
    assert current.__name__ == f"gpr_zero.workflows.{module}"
    assert current.__package__ == "gpr_zero.workflows"
    assert Path(current.__file__).name == f"{module}.py"
    assert callable(getattr(current, function))


def test_installed_console_entry_point_uses_the_canonical_dispatcher():
    with (ROOT / "pyproject.toml").open("rb") as stream:
        package = tomllib.load(stream)
    assert package["project"]["scripts"] == {"gpr": "gpr_zero.cli:main"}


@pytest.mark.parametrize("status", [0, 1])
def test_profile_verification_receives_arguments_and_returns_its_status(
    tmp_path, monkeypatch, status
):
    config = tmp_path / "gpr.toml"
    config.write_text("")
    profile = tmp_path / "profile directory"
    calls = []

    def verify(argv):
        calls.append((get_project(), argv))
        return status

    def load(name):
        assert name == "gpr_zero.workflows.verify_profile"
        return SimpleNamespace(main=verify)

    monkeypatch.setattr(cli.importlib, "import_module", load)
    assert (
        cli.main(["--project", str(config), "verify", "thesis-profile", str(profile)])
        == status
    )
    assert calls == [(Project.load(config), [str(profile)])]


@pytest.mark.parametrize(
    "action,arguments",
    [
        (
            "warc",
            ["--input", "first.warc.gz", "second.warc.gz", "--limit-records", "5"],
        ),
        ("html", ["--input-dir", "extracted", "--years", "2020", "2021", "--truncate"]),
        ("warc", ["--help"]),
        ("html", ["--help"]),
    ],
)
def test_ingestion_arguments_are_owned_by_the_workflow_parser(
    tmp_path, monkeypatch, action, arguments
):
    config = tmp_path / "gpr.toml"
    config.write_text("")
    calls = []

    def load(name):
        assert name == f"gpr_zero.workflows.ingest_{action}"
        return SimpleNamespace(main=lambda argv: calls.append((get_project(), argv)))

    monkeypatch.setattr(cli.importlib, "import_module", load)
    assert cli.main(["--project", str(config), "ingest", action, *arguments]) == 0
    assert calls == [(Project.load(config), arguments)]


@pytest.mark.parametrize(
    "arguments", [["generate-dictionary"], ["index", "all", "--database", "other.db"]]
)
def test_removed_commands_and_workflow_path_flags_are_rejected(arguments, capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(arguments)
    assert error.value.code == 2
    assert "error:" in capsys.readouterr().err


def test_command_imports_from_outside_checkout_without_writing_data(tmp_path):
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    code = """
import importlib
from gpr_zero.cli import WORKFLOWS
for actions in WORKFLOWS.values():
    for module, function, _ in actions.values():
        assert callable(getattr(importlib.import_module("gpr_zero.workflows." + module), function))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert not list(tmp_path.iterdir())
    help_result = subprocess.run(
        [sys.executable, "-m", "gpr_zero", "--help"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert help_result.returncode == 0, help_result.stderr
    assert "--project" in help_result.stdout


def test_prompt_manager_uses_project_environment_and_paths(tmp_path, monkeypatch):
    monkeypatch.delenv("GPR_TEST_PROJECT_CREDENTIAL", raising=False)
    (tmp_path / ".env").write_text("GPR_TEST_PROJECT_CREDENTIAL=fixture-only\n")
    config = tmp_path / "gpr.toml"
    config.write_text('[paths]\nllm_config = "provider.yaml"\n')
    module = importlib.import_module("gpr_zero.llm.prompt_manager")
    calls = []

    def manager(**kwargs):
        calls.append((kwargs, os.environ.get("GPR_TEST_PROJECT_CREDENTIAL")))
        return "recorded-manager"

    monkeypatch.setattr(module, "PromptManager", manager)
    project = Project.load(config)
    assert "GPR_TEST_PROJECT_CREDENTIAL" not in os.environ
    assert project.prompt_manager() == "recorded-manager"
    assert calls == [
        (
            {
                "config_file_path": tmp_path / "provider.yaml",
                "log_dir": tmp_path / "logs/llm",
            },
            "fixture-only",
        )
    ]
