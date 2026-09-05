"""UrbanMARL Callback Module for Line-of-Sight Evaluation.

Provides BenchMARL experiment callbacks for tracking LoS and network metrics during training.
"""

from benchmarl.experiment.callback import Callback
from tensordict import TensorDictBase


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
