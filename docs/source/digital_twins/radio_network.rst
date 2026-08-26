Radio Network Digital Twin
==========================

The **Radio Network Digital Twin** (`VectorizedChannelModel`) simulates millimeter-wave (mmWave) 29 GHz propagation dynamics across 3D urban topographies.

Propagation Path Loss Model
---------------------------

Path loss :math:`\text{PL}(d)` between transmitter :math:`i` and receiver :math:`j` at 3D distance :math:`d = \|\mathbf{p}_i - \mathbf{p}_j\|` is modeled as:

.. math::

   \text{PL}(d) \; [\text{dB}] = \text{PL}_0 + 10 \cdot \eta \cdot \log_{10}\left(\frac{d}{d_0}\right) + X_\sigma

where:

- :math:`\text{PL}_0`: Reference path loss at distance :math:`d_0 = 1\text{m}` for carrier frequency :math:`f_c = 28\text{ GHz}`.
- :math:`\eta`: Path loss exponent, taking distinct values based on geospatial LoS state:

.. math::

   \eta = \begin{cases} 
   \eta_{\text{LoS}} \approx 2.0, & \text{if Line-of-Sight} \\
   \eta_{\text{NLoS}} \approx 3.8, & \text{if Non-Line-of-Sight}
   \end{cases}

- :math:`X_\sigma \sim \mathcal{N}(0, \sigma^2)`: Log-normal shadow fading.

SINR & Shannon Data Rates
-------------------------

Signal-to-Interference-plus-Noise Ratio (SINR) for link :math:`(i, j)` with transmit power :math:`P_{\text{tx}}`:

.. math::

   \text{SINR}_{ij} = \frac{P_{\text{tx}} \cdot g_{ij} \cdot h_{ij}^{\text{PL}}}{\sigma_n^2 + \sum_{k \neq i} P_{\text{tx}} \cdot g_{kj} \cdot h_{kj}^{\text{PL}}}

Achievable data transmission rate :math:`R_{ij}` follows Shannon channel capacity:

.. math::

   R_{ij} = B \cdot \log_2\left(1 + \text{SINR}_{ij}\right)

Vectorized Channel Computation
------------------------------

`VectorizedChannelModel` computes data rates for all links across thousands of environments simultaneously:

.. code-block:: python

   import torch
   from urbanmarl.models.channel import VectorizedChannelModel

   channel = VectorizedChannelModel(
       carrier_frequency=28e9,
       bandwidth=20e6,
       tx_power_dbm=30.0,
       noise_figure_db=9.0,
       device="cuda"
   )

   # Calculate data rates (B, N_tx, M_rx)
   data_rates = channel.compute_data_rates(
       tx_pos=tx_positions,
       rx_pos=rx_positions,
       los_matrix=los_states,
   )
