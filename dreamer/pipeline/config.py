from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


STAGE_ORDER = ["tokenizer", "dynamics", "bc_rew", "policy", "eval", "report"]


@dataclass(frozen=True)
class PipelinePaths:
    run_dir: Path
    checkpoints_dir: Path
    metrics_dir: Path
    media_dir: Path


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _parse_scalar(text: str) -> Any:
    lower = text.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower in {"null", "none"}:
        return None
    if text.startswith('"') and text.endswith('"'):
        return text[1:-1]
    if text.startswith("'") and text.endswith("'"):
        return text[1:-1]
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return text


def _load_yaml(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]

    for raw in lines:
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        content = line.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise ValueError(f"Invalid indentation in {path}: {raw}")
        container = stack[-1][1]

        if content.startswith("- "):
            if not isinstance(container, list):
                raise ValueError(f"List item without list parent in {path}: {raw}")
            container.append(_parse_scalar(content[2:].strip()))
            continue

        if ":" not in content:
            raise ValueError(f"Invalid line in {path}: {raw}")
        key, value = content.split(":", 1)
        key = key.strip()
        value = value.strip()

        if value == "":
            # Look-ahead heuristic: next meaningful line with '- ' indicates list.
            node: Any = {}
            # create list for known list keys
            if key == "includes":
                node = []
            if isinstance(container, dict):
                container[key] = node
            else:
                raise ValueError(f"Expected mapping container in {path}: {raw}")
            stack.append((indent, node))
        else:
            if isinstance(container, dict):
                container[key] = _parse_scalar(value)
            else:
                raise ValueError(f"Expected mapping container in {path}: {raw}")

    if not isinstance(root, dict):
        raise ValueError(f"Expected mapping at {path}")
    return root


def _scalar_to_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if any(ch in text for ch in [":", "#", "{", "}", "[", "]"]):
        return f'"{text}"'
    return text


def dump_yaml(obj: dict[str, Any]) -> str:
    def emit(value: Any, indent: int) -> list[str]:
        prefix = " " * indent
        if isinstance(value, dict):
            lines: list[str] = []
            for k, v in value.items():
                if isinstance(v, (dict, list)):
                    lines.append(f"{prefix}{k}:")
                    lines.extend(emit(v, indent + 2))
                else:
                    lines.append(f"{prefix}{k}: {_scalar_to_text(v)}")
            return lines
        if isinstance(value, list):
            lines = []
            for item in value:
                if isinstance(item, (dict, list)):
                    lines.append(f"{prefix}-")
                    lines.extend(emit(item, indent + 2))
                else:
                    lines.append(f"{prefix}- {_scalar_to_text(item)}")
            return lines
        return [f"{prefix}{_scalar_to_text(value)}"]

    return "\n".join(emit(obj, 0)) + "\n"


def load_config(config_path: str | Path, *, _validate: bool = True) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    root = _load_yaml(config_path)

    merged: dict[str, Any] = {}
    includes = root.get("includes", [])
    if includes:
        if not isinstance(includes, list):
            raise ValueError(f"'includes' must be a list in {config_path}")
        for include in includes:
            include_path = (config_path.parent / include).resolve()
            merged = _deep_merge(merged, load_config(include_path, _validate=False))

    root = dict(root)
    root.pop("includes", None)
    merged = _deep_merge(merged, root)
    if _validate:
        validate_config(merged, source=str(config_path))
    return merged


def validate_config(cfg: dict[str, Any], *, source: str) -> None:
    for key in ("experiment_name", "env_name", "output_root", "stages"):
        if key not in cfg:
            raise ValueError(f"Missing required key '{key}' in {source}")
    if cfg["env_name"] not in {"grasping_2p5d", "bouncing_square"}:
        raise ValueError(f"Unsupported env_name={cfg['env_name']!r} in {source}")
    stages = cfg["stages"]
    if not isinstance(stages, dict):
        raise ValueError(f"'stages' must be a mapping in {source}")
    for stage in STAGE_ORDER:
        if stage not in stages:
            raise ValueError(f"Missing stage config '{stage}' in {source}")

    for stage in ("tokenizer", "dynamics", "bc_rew", "policy"):
        stage_cfg = stages[stage]
        if stage_cfg.get("max_steps", 1) <= 0:
            raise ValueError(f"{stage}.max_steps must be > 0 in {source}")
        if stage_cfg.get("B", 1) <= 0 or stage_cfg.get("T", 1) <= 0:
            raise ValueError(f"{stage}.B/T must be > 0 in {source}")


def compute_paths(run_dir: Path) -> PipelinePaths:
    return PipelinePaths(
        run_dir=run_dir,
        checkpoints_dir=run_dir / "checkpoints",
        metrics_dir=run_dir / "metrics",
        media_dir=run_dir / "media",
    )
