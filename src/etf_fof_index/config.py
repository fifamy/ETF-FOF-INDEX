from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: Path) -> Dict[str, Any]:
    path = config_path.resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError("Config must be a mapping.")

    config["_config_path"] = str(path)
    config["_root_dir"] = str(path.parent.parent)
    return config


def resolve_path(config: Dict[str, Any], relative_or_absolute: str) -> Path:
    candidate = Path(relative_or_absolute)
    if candidate.is_absolute():
        return candidate
    root = Path(config["_root_dir"])
    return (root / candidate).resolve()


def validate_config(config: Dict[str, Any]) -> None:
    bucket_order = config["bucket_order"]
    strategic_weights = config["strategic_weights"]
    bounds = config["bucket_bounds"]

    if sorted(bucket_order) != sorted(strategic_weights):
        raise ValueError("bucket_order and strategic_weights keys must match.")

    if sorted(bucket_order) != sorted(bounds):
        raise ValueError("bucket_order and bucket_bounds keys must match.")

    total = sum(float(strategic_weights[bucket]) for bucket in bucket_order)
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"Strategic weights must sum to 1.0, got {total:.6f}.")

