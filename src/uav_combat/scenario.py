"""同构 1v1 场景创建逻辑。"""
from typing import Any
from .config import aircraft_spec
from .models import Aircraft, AircraftState


class HomogeneousScenario:
    """用实体列表管理共享同一规格的红蓝双方。"""
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.spec = aircraft_spec(config)
        self.aircraft: list[Aircraft] = []

    def reset(self, seed: int | None = None) -> list[Aircraft]:
        """按配置确定性创建 red_0 和 blue_0。"""
        self.aircraft = []
        for team in ("red", "blue"):
            item = self.config["initial_state"][team]
            state = AircraftState(item["x"], item["y"], -item["altitude"], item["v"], item["theta"], item["psi"])
            self.aircraft.append(Aircraft(f"{team}_0", team, self.spec, state))
        return self.aircraft
