import pytest
import torch
from torchrl.envs.utils import check_env_specs
from urbanmarl.envs.base_env import UrbanEnv


def test_urbanmarl_env_specs():
    """
    Validates that UrbanEnv produces TensorDicts that strictly
    adhere to defined observation, state, action, and reward specs.
    """
    config = {
        "num_uavs": 3,
        "num_ues": 20,
        "volume_size": (500, 500, 200),
        "max_steps": 50,
        "max_horizontal_speed": 49.0,
        "max_vertical_speed": 12.0,
        "max_transmit_power": 5.0,
        "frequency_ghz": 29.0,
        "g2a_bandwidth": 10e6,
        "noise_figure_db": 7.0,
    }

    device = torch.device("cpu")

    # Initialize env with a batch size of 2 to ensure vectorization works
    env = UrbanEnv(
        num_envs=2,
        continuous_actions=True,
        seed=42,
        device=device,
        scenario="default",
        **config
    )

    # Check specs natively
    env.check_env_specs()

    # Execute a manual step check
    td = env.reset()
    group = list(env.group_map.keys())[0]
    assert group in td.keys()
    assert "observation" in td[group].keys()

    # Generate random actions based on the spec
    action = env.action_spec.rand()
    td.update(action)

    # Step the environment
    td_step = env.step(td)

    assert "done" in td_step.keys()
    group = list(env.group_map.keys())[0]
    assert ("next", group, "reward") in td_step.keys(True, True) or "reward" in td_step.get(("next", group), {}).keys()


if __name__ == "__main__":
    test_urbanmarl_env_specs()
    print("All environment spec tests passed successfully!")
