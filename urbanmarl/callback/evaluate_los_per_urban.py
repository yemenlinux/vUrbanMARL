from typing import List

from benchmarl.algorithms import MappoConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.experiment.callback import Callback
from benchmarl.models.mlp import MlpConfig
from tensordict import TensorDict, TensorDictBase


class EvaluateLoS(Callback):
    def on_train_step(self, batch: TensorDictBase, group: str) -> TensorDictBase:
        """
        A callback called for every training step.

        Args:
           batch (TensorDictBase): tensordict with the training batch
           group (str): group name

        Returns:
            TensorDictBase: a new tensordict containing the loss values

        """
        # print("-"*60)
        # print(f"group: {group}")
        # for key in batch.keys(True, True):
        #     print(f"{key}, shape: {batch[key].shape}")
        # print("-"*60)
        pass
        
    def on_train_end(self, training_td: TensorDictBase, group: str):
        """
        A callback called at the end of training.

        Args:
            training_td (TensorDictBase): tensordict containing the loss values
            group (str): group name

        """
        # print("-"*60)
        # print(f"group: {group}")
        # for key in training_td.keys(True, True):
        #     print(f"{key}, shape: {training_td[key].shape}")
        # print("-"*60)