"""リポジトリに実際に入っているかの検査。

一度、`.gitignore` の `data/` が `src/data/` にも一致して、データ取得コード一式が
push されないまま「テストは全部通っている」と報告した事故があった。
手元で動くことと、clone した人の手元で動くことは別問題なので、機械で固定する。
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args],
                          capture_output=True, text=True, check=True).stdout


def _is_git_repo() -> bool:
    return (REPO / ".git").exists()


@pytest.mark.skipif(not _is_git_repo(), reason="not a git checkout")
def test_every_source_file_is_tracked():
    tracked = set(_git("ls-files").splitlines())
    on_disk = {
        str(p.relative_to(REPO))
        for p in list(REPO.glob("src/**/*.py")) + list(REPO.glob("tests/**/*.py"))
        if "__pycache__" not in p.parts
    }
    missing = sorted(on_disk - tracked)
    assert not missing, f"git に入っていないソースがある（clone すると動かない）: {missing}"


@pytest.mark.skipif(not _is_git_repo(), reason="not a git checkout")
def test_config_files_are_tracked():
    tracked = set(_git("ls-files").splitlines())
    for name in ("config/params.yaml", "config/symbols.yaml", "requirements.txt"):
        assert name in tracked, f"{name} が git に入っていない"


@pytest.mark.skipif(not _is_git_repo(), reason="not a git checkout")
def test_gitignore_does_not_swallow_source_directories():
    """`data/` のような無指定パターンが src/ 配下に波及していないこと。"""
    probes = ["src/data/download.py", "src/analysis/viability.py", "tests/conftest.py"]
    r = subprocess.run(["git", "-C", str(REPO), "check-ignore", "-v", *probes],
                       capture_output=True, text=True)
    assert r.returncode == 1, f"ソースが .gitignore に一致している:\n{r.stdout}"
