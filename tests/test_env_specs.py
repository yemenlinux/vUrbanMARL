import pytest
import torch
from torchrl.envs.utils import check_env_specs
from urbanmarl.envs.urbanmarl_env import UrbanMARLEnv

def test_urbanmarl_env_specs():
    """
    Validates that the UrbanMARLEnv produces TensorDicts that strictly 
    adhere to the defined observation, state, action, and reward specs.
    """
    config = {
        "num_uavs": 3,
        "num_ues": 20,
        "area_size": (500, 500),
        "max_time_slots": 50,
        "max_horizontal_speed": 49.0,
        "max_vertical_speed": 12.0,
        "max_transmit_power": 5.0,
        "frequency_ghz": 29.0,
        "g2a_bandwidth": 10e6,
        "noise_figure_db": 7.0
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize env with a batch size of 2 to ensure vectorization works
    env = UrbanMARLEnv(config=config, device=device, batch_size=torch.Size([2]))
    
    # Check specs natively
    check_env_specs(env)
    
    # Execute a manual step check
    td = env.reset()
    assert "agents" in td.keys()
    assert "observation" in td["agents"].keys()
    assert "state" in td.keys()
    
    # Generate random actions based on the spec
    action = env.action_spec.rand()
    td.update(action)
    
    # Step the environment
    td = env.step(td)
    
    assert "reward" in td["agents"].keys()
    assert td["agents", "reward"].shape == torch.Size([2, 3, 1])
    assert td["done"].shape == torch.Size([2, 1])

if __name__ == "__main__":
    test_urbanmarl_env_specs()
    print("All environment spec tests passed successfully!")
