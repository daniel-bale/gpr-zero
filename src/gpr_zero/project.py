"""Project paths and database access shared by every workflow.

Paths in a project file are relative to that file, so commands work from any
directory. Workflows receive one Project instead of separate path settings.
"""

import os
import tomllib
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path

_active_project: ContextVar["Project | None"] = ContextVar("gpr_project", default=None)


@dataclass(frozen=True)
class Project:
    root: Path
    data_dir: Path
    database: Path
    llm_config: Path

    @property
    def dictionaries_dir(self) -> Path:
        return self.data_path("dictionaries", "all_dictionaries")

    @property
    def step2_dictionaries_dir(self) -> Path:
        return self.data_path("dictionaries", "step2_batch")

    def data_path(self, *parts: str) -> Path:
        return self.data_dir.joinpath(*parts)

    def connect(self, *, read_only: bool = True):
        from gpr_zero.db import get_connection

        return get_connection(self.database, read_only=read_only)

    def prompt_manager(self):
        """Load this project's credentials, provider settings, and request logs."""
        from dotenv import load_dotenv

        from gpr_zero.llm.prompt_manager import PromptManager

        load_dotenv(self.root / ".env")
        return PromptManager(
            config_file_path=self.llm_config, log_dir=self.root / "logs" / "llm"
        )

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Project":
        """Load the project's single path configuration.

        An explicit file must exist. With no explicit file, look in the current
        directory and its parents, then use the repository's portable defaults.
        """
        selected = path or os.environ.get("GPR_PROJECT")
        if selected is not None:
            config_path = Path(selected).expanduser().resolve()
            if not config_path.is_file():
                raise FileNotFoundError(
                    f"Project configuration not found: {config_path}"
                )
        else:
            bundled = Path(__file__).resolve().parents[2] / "gpr.toml"
            config_path = next(
                (
                    p / "gpr.toml"
                    for p in (Path.cwd(), *Path.cwd().parents)
                    if (p / "gpr.toml").is_file()
                ),
                bundled if bundled.is_file() else Path.cwd() / "gpr.toml",
            )
        root = config_path.parent
        settings = {}
        if config_path.is_file():
            with config_path.open("rb") as stream:
                settings = tomllib.load(stream)
        unknown_sections = set(settings) - {"paths"}
        if unknown_sections:
            raise ValueError(
                f"Unknown project sections: {', '.join(sorted(unknown_sections))}"
            )
        paths = settings.get("paths", {})
        if not isinstance(paths, dict):
            raise ValueError("Project paths must be a TOML table")
        unknown_paths = set(paths) - {"data", "database", "llm_config"}
        if unknown_paths:
            raise ValueError(
                f"Unknown project paths: {', '.join(sorted(unknown_paths))}"
            )
        if any(
            not isinstance(value, str) or not value.strip() for value in paths.values()
        ):
            raise ValueError("Project paths must be nonempty strings")

        def resolve(value):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("Project paths must be nonempty strings")
            result = Path(value).expanduser()
            return (
                (root / result).resolve()
                if not result.is_absolute()
                else result.resolve()
            )

        data = resolve(paths.get("data", "data"))
        database = paths.get("database")
        return cls(
            root=root,
            data_dir=data,
            database=resolve(database)
            if database
            else data / "news" / "stage2_enhanced_full.duckdb",
            llm_config=resolve(paths.get("llm_config", "llm_config.yaml")),
        )


def get_project() -> Project:
    """Use the command's Project, or discover one for a direct package call."""
    return _active_project.get() or Project.load()


@contextmanager
def use_project(project: Project):
    """Bind one project for a command without changing global environment or cwd."""
    token = _active_project.set(project)
    try:
        yield project
    finally:
        _active_project.reset(token)
