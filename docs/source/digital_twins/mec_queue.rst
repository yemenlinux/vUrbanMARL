MEC Queue Digital Twin
======================

The **MEC Queue Digital Twin** (`VectorizedMECQueue`) models multi-server **M/M/c queuing theory** dynamics for edge server computing clusters deployed on UAVs or Base Stations.

M/M/c Queuing Dynamics
----------------------

Each Mobile Edge Computing (MEC) node operates :math:`c` parallel CPU/GPU processing cores with individual service rate :math:`\mu = f_{\text{cpu}} / s_{\text{task}}` (tasks/sec).

For a aggregate task arrival rate :math:`\lambda`, server traffic intensity :math:`\rho` is defined as:

.. math::

   \rho = \frac{\lambda}{c \cdot \mu}

For queue stability, :math:`\rho < 1`.

Erlang-C Queue Waiting Probability
----------------------------------

The probability :math:`P_q` that an offloaded task must wait in the queue before processing follows the **Erlang-C formula**:

.. math::

   P_q = \frac{\frac{(c\rho)^c}{c!} \frac{1}{1-\rho}}{\sum_{k=0}^{c-1} \frac{(c\rho)^k}{k!} + \frac{(c\rho)^c}{c!} \frac{1}{1-\rho}}

Average Waiting & Execution Delay
---------------------------------

The average queuing wait time :math:`W_q` and total task response delay :math:`T_{\text{total}}` are:

.. math::

   W_q = \frac{P_q}{c\mu - \lambda}

.. math::

   T_{\text{total}} = T_{\text{comm}} + W_q + \frac{1}{\mu}

where :math:`T_{\text{comm}} = D_{\text{task}} / R_{ij}` is the radio transmission delay over the mmWave link.

Vectorized PyTorch Implementation
---------------------------------

`VectorizedMECQueue` calculates queuing metrics across thousands of MEC servers simultaneously:

.. code-block:: python

   import torch
   from urbanmarl.models.mec_queue import VectorizedMECQueue

   queue_model = VectorizedMECQueue(
       num_servers=4,
       service_rate=100.0,  # tasks per second
       device="cuda"
   )

   # Compute queuing delays for batched arrival rates (B, N_mec)
   delays, waiting_times, queue_lengths = queue_model.compute_delays(
       arrival_rates=lambda_matrix,
       data_rates=transmission_rates,
       task_sizes=task_sizes,
   )
