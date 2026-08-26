"""UrbanMARL Callback Module for Line-of-Sight Evaluation.

Provides BenchMARL experiment callbacks for tracking LoS and network metrics during training.
"""

from typing import List

from benchmarl.algorithms import MappoConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.experiment.callback import Callback
from benchmarl.models.mlp import MlpConfig
from tensordict import TensorDict, TensorDictBase


class EvaluateLoS(Callback):
    """BenchMARL Callback to evaluate LoS ratios per urban environment configuration.

    Interprets training and evaluation batches to record LoS statistics.
    """

    def on_train_step(self, batch: TensorDictBase, group: str) -> TensorDictBase:
        """Invoked on each training step.

        Args:
            batch (TensorDictBase): Tensordict containing training batch data.
            group (str): Agent group identifier name.

        Returns:
            TensorDictBase: Tensordict with optional computed metrics.
        """
        pass

    def on_train_end(self, training_td: TensorDictBase, group: str) -> None:
        """Invoked at the conclusion of experiment training.

        Args:
            training_td (TensorDictBase): Tensordict containing training history metrics.
            group (str): Agent group identifier name.
        """
        pass