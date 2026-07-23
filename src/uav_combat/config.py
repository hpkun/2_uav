"""YAML 配置加载与校验。"""
from pathlib import Path
from typing import Any
import yaml
from .models import AircraftSpec


def load_config(path: str | Path) -> dict[str, Any]:
    """读取 YAML 配置并检查必要顶层字段。"""
    with Path(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    required = {"simulation", "action", "aircraft", "battlefield", "combat", "scenario", "initial_state"}
    if not isinstance(config, dict) or not required.issubset(config):
        raise ValueError(f"configuration must contain: {sorted(required)}")
    return config


def aircraft_spec(config: dict[str, Any]) -> AircraftSpec:
    """从配置构造不可变飞机规格。"""
    return AircraftSpec(**config["aircraft"])
