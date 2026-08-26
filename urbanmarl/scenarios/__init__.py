"""UrbanMARL Scenarios Module Registry.

Provides dynamic scenario loading and registration utilities for UrbanMARL environments.
"""

import importlib
import os
from typing import Optional

from .base import UrbanScenario

_registry = {}


def register_scenario(name: str, scenario_class: type) -> None:
    """Registers a scenario class under a specific string key identifier.

    Args:
        name (str): Unique scenario string name.
        scenario_class (type): UrbanScenario subclass.
    """
    _registry[name] = scenario_class


def load_scenario(name: str, config: Optional[dict] = None) -> UrbanScenario:
    """Loads and instantiates an UrbanScenario by name.

    Args:
        name (str): Name identifier of the scenario.
        config (Optional[dict]): Configuration parameters dictionary.

    Returns:
        UrbanScenario: An instance of the requested scenario class.

    Raises:
        ValueError: If scenario name is not found in registry or module path.
    """
    if name in _registry:
        return _registry[name](config or {})
    try:
        module = importlib.import_module(f"urbanmarl.scenarios.{name}")
        scenario_class = getattr(module, "Scenario")
        register_scenario(name, scenario_class)
        return scenario_class(config or {})
    except (ImportError, AttributeError) as e:
        raise ValueError(
            f"Scenario '{name}' not found in registry or as a module."
        ) from e


from .default import Scenario as DefaultScenario
register_scenario("default", DefaultScenario)

from .uav_navigation import Scenario as NavigationScenario
register_scenario("uav_navigation", NavigationScenario)

from .uav_ue_los import Scenario as UavUeLosScenario
register_scenario("uav_ue_los", UavUeLosScenario)

from .uavmec_offloading import Scenario as UAVMECScenario
register_scenario("uavmec_offloading", UAVMECScenario)

