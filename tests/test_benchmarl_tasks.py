"""Unit tests verifying BenchMARL task configuration loading and environment instantiation."""

import pytest
from benchmarl.environments.urbanmarl.common import UrbanEnvTask


@pytest.mark.parametrize("task", list(UrbanEnvTask))
def test_task_yaml_loading(task):
    """Verifies that all UrbanEnvTask tasks load their YAML configurations and type-check correctly."""
    task_instance = task.get_from_yaml()
    assert task_instance is not None
    assert isinstance(task_instance.config, dict)
    assert len(task_instance.config) > 0


@pytest.mark.parametrize(
    "task",
    [
        UrbanEnvTask.UAV_MOBILE_UE,
        UrbanEnvTask.UAV_LIDAR_NAVIGATION,
        UrbanEnvTask.UAVMEC_ADVANCED_PHYSICS,
        UrbanEnvTask.MEC_OFFLOADING,
    ],
)
def test_new_tasks_env_fun(task):
    """Verifies that get_env_fun creates an UrbanEnv that can be reset properly."""
    task_instance = task.get_from_yaml()
    env = task_instance.get_env_fun(
        num_envs=2, continuous_actions=True, seed=123, device="cpu"
    )()
    td = env.reset()
    assert td is not None
    assert env.batch_size == (2,)
    env.close()
