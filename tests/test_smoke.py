"""Cheap checks for application assets that otherwise fail only at runtime."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from app.dependencies import templates


REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_ROOT = REPO_ROOT / "app" / "templates"


def test_application_import_is_independent_of_working_directory(tmp_path: Path) -> None:
    """ASGI servers may be launched outside the checkout directory."""
    environment = os.environ.copy()
    environment.update(
        ENVIRONMENT="test",
        DATABASE_URL=f"sqlite:///{tmp_path / 'smoke.db'}",
        REDIS_URL="",
        SCHEDULER_ENABLED="false",
    )
    python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + python_path if python_path else "")

    result = subprocess.run(
        [sys.executable, "-c", "import app.main; print(len(app.main.app.routes))"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_all_templates_compile() -> None:
    """Catch broken includes, syntax, globals, and custom filters in CI."""
    for path in TEMPLATE_ROOT.rglob("*.html"):
        templates.env.get_template(path.relative_to(TEMPLATE_ROOT).as_posix())
