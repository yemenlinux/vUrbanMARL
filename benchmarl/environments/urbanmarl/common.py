from benchmarl.environments.common import Task, TaskClass
from benchmarl.utils import DEVICE_TYPING

from typing import Callable, Optional, Dict, List

from tensordict import TensorDictBase
from torchrl.data import Composite
from torchrl.envs import EnvBase
# from urbanmarl.envs.urbanmarl_env import UrbanEnv
from urbanmarl.envs.base_env import UrbanEnv

class UrbanEnvClass(TaskClass):
    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        return lambda: UrbanEnv(
            num_envs=num_envs,
            continuous_actions=continuous_actions,
            seed=seed,
            device=device,
            scenario=self.name.lower(),
            # scenario=self.config.get("scenario", "default"),  # read from config
            **self.config,
        )

    def supports_continuous_actions(self) -> bool:
        return True

    def supports_discrete_actions(self) -> bool:
        return True

    def has_render(self, env: EnvBase) -> bool:
        return False

    def max_steps(self, env: EnvBase) -> int:
        return env.max_steps

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        return env.group_map

    def observation_spec(self, env: EnvBase) -> Composite:
        return env.full_observation_spec_unbatched

    def action_spec(self, env: EnvBase) -> Composite:
        return env.full_action_spec_unbatched

    def state_spec(self, env: EnvBase) -> Optional[Composite]:
        if "state" in env.full_observation_spec_unbatched.keys():
            return Composite({"state": env.full_observation_spec_unbatched["state"]})
        return None

    def action_mask_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    def info_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    @staticmethod
    def env_name() -> str:
        return "urbanmarl"

    def log_info(self, batch: TensorDictBase) -> Dict[str, float]:
        if "state" not in batch.keys():
            return {}
        info = {}
        for i in range(batch.batch_size[0]):
            alpha, beta, gamma, E = batch.get("state")[i, 0, :4]
            name = f"reward_{alpha.item():.2f}_{int(beta.item())}_{gamma.item():.2f}_{E.item():.4f}"
            #
            reward = 0
            for key in batch.keys(True, True):
                if isinstance(key, tuple):
                    if key[0] == "next" and key[-1] == "reward":
                        reward = batch.get(key)[i].mean()
            info[name] = reward
            if ('next', 'info', 'collisions') in batch.keys(True, True):
                col_name = f"collisions_{alpha.item():.2f}_{int(beta.item())}_{gamma.item():.2f}_{E.item():.4f}"
                info[col_name] = batch.get(('next', 'info', 'collisions'))[i].mean()
            if ('next', 'info', 'velocity') in batch.keys(True, True):
                vel_name = f"velocity_{alpha.item():.2f}_{int(beta.item())}_{gamma.item():.2f}_{E.item():.4f}"
                info[vel_name] = batch.get(('next', 'info', 'velocity'))[i].mean()
            if ('next', 'info', 'los') in batch.keys(True, True):
                los_name = f"los_{alpha.item():.2f}_{int(beta.item())}_{gamma.item():.2f}_{E.item():.4f}"
                info[los_name] = batch.get(('next', 'info', 'los'))[i].mean()
                
        return info
        

class UrbanEnvTask(Task):
    UAV_NAVIGATION = None
    UAV_UE_LOS = None
    UAVMEC_OFFLOADING = None
    COVERAGE = None

    @staticmethod
    def associated_class():
        return UrbanEnvClass


