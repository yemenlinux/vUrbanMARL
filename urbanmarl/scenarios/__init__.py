import importlib
import os
from .base import UrbanScenario

_registry = {}

def register_scenario(name: str, scenario_class):
    _registry[name] = scenario_class

def load_scenario(name: str, config: dict = None) -> UrbanScenario:
    """Load a scenario by name, optionally with config."""
    if name in _registry:
        return _registry[name](config or {})
    # Try to import from a file
    try:
        module = importlib.import_module(f"urbanmarl.scenarios.{name}")
        scenario_class = getattr(module, "Scenario")
        register_scenario(name, scenario_class)
        return scenario_class(config or {})
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Scenario '{name}' not found in registry or as a module.") from e

# Auto‑register default scenario
from .default import Scenario as DefaultScenario
register_scenario("default", DefaultScenario) 
from .uav_navigation import Scenario as NavigationScenario
register_scenario("uav_navigation", NavigationScenario)
from .uav_ue_los import Scenario as UavUeLosScenario
register_scenario("uav_ue_los", UavUeLosScenario)
from .uavmec_offloading import Scenario as UAVMECScenario
register_scenario("uavmec_offloading", UAVMECScenario)

