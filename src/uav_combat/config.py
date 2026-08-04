"""YAML 配置加载与校验。"""
from pathlib import Path
from typing import Any
import yaml
from .models import AircraftSpec


def load_config(path: str | Path) -> dict[str, Any]:
    """读取 YAML 配置并检查必要顶层字段。"""
    with Path(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    # Some scenario-specific generators, including functional 4v3 v9, own
    # their reset geometry and intentionally do not use initial_state.
    required = {"simulation", "action", "aircraft", "battlefield", "combat", "scenario"}
    if not isinstance(config, dict) or not required.issubset(config):
        raise ValueError(f"configuration must contain: {sorted(required)}")
    return config


def aircraft_spec(config: dict[str, Any]) -> AircraftSpec:
    """从配置构造不可变飞机规格。"""
    return AircraftSpec(**config["aircraft"])
