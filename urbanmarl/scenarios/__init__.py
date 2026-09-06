"""UrbanMARL Scenarios Module Registry.

Provides dynamic, automated scenario discovery, loading, and registration
utilities for UrbanMARL environments. Scenarios in this package implementing
the ``UrbanScenario`` interface are automatically discovered and registered
at runtime without requiring manual hardcoded imports.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from .base import UrbanScenario

# Global registry mapping scenario names (lowercase) to their UrbanScenario classes
_registry: Dict[str, Type[UrbanScenario]] = {}

# Legacy PascalCase aliases for backward compatibility with direct module imports
_SCENARIO_NAME_ALIASES: Dict[str, str] = {
    "DefaultScenario": "default",
    "NavigationScenario": "uav_navigation",
    "UavNavigationScenario": "uav_navigation",
    "UavUeLosScenario": "uav_ue_los",
    "UAVMECScenario": "uavmec_offloading",
    "UavMecOffloadingScenario": "uavmec_offloading",
    "UavMobileUeScenario": "uav_mobile_ue",
    "UavLidarNavigationScenario": "uav_lidar_navigation",
    "UavMecAdvancedPhysicsScenario": "uavmec_advanced_physics",
    "CoverageScenario": "coverage",
    "MecOffloadingScenario": "mec_offloading",
}


def register_scenario(
    name: str,
    scenario_class: Type[UrbanScenario],
    aliases: Optional[List[str]] = None,
) -> None:
    """Registers a scenario class under a specific string key identifier.

    Args:
        name: Unique scenario string name.
        scenario_class: Concrete subclass of UrbanScenario.
        aliases: Optional list of additional alias names for the scenario.

    Raises:
        TypeError: If scenario_class is not a subclass of UrbanScenario or is abstract.
    """
    if not (
        inspect.isclass(scenario_class) and issubclass(scenario_class, UrbanScenario)
    ):
        raise TypeError(
            f"Scenario class {scenario_class} must be a subclass of UrbanScenario."
        )
    if inspect.isabstract(scenario_class):
        raise TypeError(
            f"Cannot register abstract scenario class {scenario_class.__name__}."
        )

    _registry[name.lower().strip()] = scenario_class
    if aliases:
        for alias in aliases:
            _registry[alias.lower().strip()] = scenario_class


def get_scenario_class(name: str) -> Type[UrbanScenario]:
    """Retrieves the scenario class for a given scenario name without instantiating it.

    Args:
        name: Name identifier of the scenario (case-insensitive).

    Returns:
        Type[UrbanScenario]: The registered scenario class.

    Raises:
        ValueError: If scenario name is not found in registry.
    """
    key = name.lower().strip()
    if key in _registry:
        return _registry[key]

    # Attempt on-demand dynamic import in case the module was added post-initialization
    try:
        module = importlib.import_module(f"urbanmarl.scenarios.{key}")
        scenario_class = getattr(module, "Scenario", None)
        if (
            scenario_class is not None
            and inspect.isclass(scenario_class)
            and issubclass(scenario_class, UrbanScenario)
            and not inspect.isabstract(scenario_class)
        ):
            register_scenario(key, scenario_class)
            return scenario_class
    except (ImportError, AttributeError, TypeError):
        pass

    available = sorted(_registry.keys())
    raise ValueError(
        f"Scenario '{name}' not found in registry. Available scenarios: {available}"
    )


def load_scenario(name: str, config: Optional[Dict[str, Any]] = None) -> UrbanScenario:
    """Loads and instantiates an UrbanScenario by name.

    Args:
        name: Name identifier of the scenario (case-insensitive).
        config: Optional configuration parameters dictionary passed to constructor.

    Returns:
        UrbanScenario: An initialized instance of the requested scenario class.

    Raises:
        ValueError: If scenario name is not found in registry or cannot be loaded.
    """
    scenario_cls = get_scenario_class(name)
    return scenario_cls(config or {})


def list_scenarios() -> List[str]:
    """Returns a sorted list of all currently registered scenario names.

    Returns:
        List[str]: List of unique scenario names.
    """
    return sorted(_registry.keys())


def auto_register_scenarios(
    package_path: Optional[Path] = None,
) -> Dict[str, Type[UrbanScenario]]:
    """Dynamically scans and registers all concrete Scenario classes in the scenarios package.

    This function automatically iterates through all python modules in the package
    directory, imports them, and registers any concrete subclass of ``UrbanScenario``
    (typically defined as ``class Scenario(UrbanScenario)``).

    Args:
        package_path: Optional custom Path directory to scan. Defaults to this package's directory.

    Returns:
        Dict[str, Type[UrbanScenario]]: Dictionary of all registered scenarios.
    """
    if package_path is None:
        package_dir = Path(__file__).resolve().parent
    else:
        package_dir = Path(package_path).resolve()

    for module_info in pkgutil.iter_modules([str(package_dir)]):
        module_name = module_info.name
        # Skip private modules and the base class module
        if module_name.startswith("_") or module_name == "base":
            continue

        try:
            module = importlib.import_module(f"urbanmarl.scenarios.{module_name}")

            scenario_cls: Optional[Type[UrbanScenario]] = None
            # Standard UrbanMARL convention: module defines `class Scenario(UrbanScenario):`
            if hasattr(module, "Scenario"):
                cand = module.Scenario
                if (
                    inspect.isclass(cand)
                    and issubclass(cand, UrbanScenario)
                    and not inspect.isabstract(cand)
                ):
                    scenario_cls = cand

            # Fallback: scan for any concrete UrbanScenario subclass defined in the module
            if scenario_cls is None:
                for attr_name in dir(module):
                    cand = getattr(module, attr_name)
                    if (
                        inspect.isclass(cand)
                        and issubclass(cand, UrbanScenario)
                        and cand is not UrbanScenario
                        and not inspect.isabstract(cand)
                    ):
                        scenario_cls = cand
                        break

            if scenario_cls is not None:
                aliases = getattr(scenario_cls, "scenario_aliases", [])
                custom_name = getattr(scenario_cls, "scenario_name", None)
                register_scenario(module_name, scenario_cls, aliases=aliases)
                if custom_name and custom_name != module_name:
                    register_scenario(custom_name, scenario_cls)

        except Exception as e:
            warnings.warn(
                f"Failed to auto-register scenario module '{module_name}': {e}",
                stacklevel=2,
            )

    return _registry


# Automatically discover and register all scenarios upon package import
auto_register_scenarios()


def __getattr__(name: str) -> Any:
    """Enables backward-compatible and dynamic attribute access on the module.

    Allows importing scenario classes directly using legacy PascalCase names
    (e.g., `from urbanmarl.scenarios import NavigationScenario`) or by scenario name.
    """
    if name in _SCENARIO_NAME_ALIASES:
        key = _SCENARIO_NAME_ALIASES[name]
        if key in _registry:
            return _registry[key]
    key = name.lower().strip()
    if key in _registry:
        return _registry[key]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> List[str]:
    """Returns directory listing of module attributes including registered scenarios."""
    return sorted(
        list(globals().keys())
        + list(_registry.keys())
        + list(_SCENARIO_NAME_ALIASES.keys())
    )


__all__ = [
    "UrbanScenario",
    "register_scenario",
    "get_scenario_class",
    "load_scenario",
    "list_scenarios",
    "auto_register_scenarios",
    "_registry",
]
