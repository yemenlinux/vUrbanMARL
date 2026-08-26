import pytest
import torch
from tensordict import TensorDict
from urbanmarl.callback.evaluate_los_per_urban import EvaluateLoS


def test_evaluate_los_callback():
    """
    Verifies that EvaluateLoS callback methods (on_train_step, on_train_end)
    can be invoked with mocked TensorDict inputs without throwing errors.
    """
    callback = EvaluateLoS()

    mock_batch = TensorDict(
        {"loss": torch.tensor([0.5, 0.4]), "reward": torch.tensor([1.2, 1.5])},
        batch_size=torch.Size([2]),
    )

    # Invoke callback hooks
    res_step = callback.on_train_step(batch=mock_batch, group="agents")
    # on_train_step can return None or TensorDictBase
    assert res_step is None or isinstance(res_step, TensorDict)

    callback.on_train_end(training_td=mock_batch, group="agents")
