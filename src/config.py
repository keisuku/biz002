"""設定ローダ。閾値・窓幅はすべて config/*.yaml に置き、コード側に定数を書かない。"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARAMS = REPO_ROOT / "config" / "params.yaml"
DEFAULT_SYMBOLS = REPO_ROOT / "config" / "symbols.yaml"


class Params(dict):
    """ネストした dict に "a.b.c" でアクセスできるだけの薄いラッパ。"""

    def get_path(self, dotted: str, default: Any = "__raise__") -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                if default == "__raise__":
                    raise KeyError(f"missing config key: {dotted}")
                return default
            node = node[part]
        return node

    def with_overrides(self, overrides: dict[str, Any]) -> "Params":
        """"a.b.c" 形式のキーで上書きしたコピーを返す（探索用）。"""
        out = Params(copy.deepcopy(dict(self)))
        for dotted, value in overrides.items():
            parts = dotted.split(".")
            node: Any = out
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = value
        return out

    def fingerprint(self) -> str:
        return json.dumps(self, sort_keys=True, default=str)


def load_params(path: str | os.PathLike[str] | None = None) -> Params:
    with open(path or DEFAULT_PARAMS, "r", encoding="utf-8") as fh:
        return Params(yaml.safe_load(fh))


def load_symbols(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    with open(path or DEFAULT_SYMBOLS, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_dir(params: Params, key: str, root: Path | None = None) -> Path:
    """params.data.<key> の相対パスをリポジトリルート基準の絶対パスにする。"""
    base = Path(root) if root is not None else REPO_ROOT
    p = Path(params.get_path(f"data.{key}"))
    return p if p.is_absolute() else base / p
