==================
Installation Guide
==================

UrbanMARL is a vectorized multi-agent reinforcement learning simulation platform for 6G network digital twins built natively on PyTorch and TorchRL.

Prerequisites
=============

Before installing UrbanMARL, ensure your system meets the following requirements:

* **Python**: :math:`\ge 3.10` (tested thoroughly on Python 3.12).
* **PyTorch**: :math:`\ge 2.2.0` (CUDA-enabled GPU recommended for large vectorized batches).
* **Operating System**: Linux (Ubuntu 22.04 or later; 26.04 recommended), Windows (WSL2), or macOS.

-------------------------------------------------
Method 1: Installation using ``uv`` (Recommended)
-------------------------------------------------

`uv <https://github.com/astral-sh/uv>`_ is an extremely fast Python package installer and resolver. UrbanMARL's ``pyproject.toml`` is configured with ``torch-backend = "auto"``, enabling ``uv`` to automatically detect your host hardware (CPU or NVIDIA CUDA) and resolve the appropriate PyTorch dependencies without requiring manual index flags.

Step 1: Install ``uv``
~~~~~~~~~~~~~~~~~~~~~~

First, ensure ``uv`` is installed on your system:

.. code-block:: bash

    curl -LsSf https://astral.sh/uv/install.sh | sh

Step 2: Clone the Repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Clone the UrbanMARL repository and navigate into the project directory:

.. code-block:: bash

    git clone https://github.com/yemenlinux/vUrbanMARL.git
    cd vUrbanMARL

Step 3: Create and Activate Virtual Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a dedicated virtual environment with Python 3.12 and activate it:

.. code-block:: bash

    uv venv --python 3.12
    source .venv/bin/activate

Step 4: Install UrbanMARL
~~~~~~~~~~~~~~~~~~~~~~~~~

**Option 1: Install from PyPI**

To install the latest release directly from PyPI:

.. code-block:: bash

    uv pip install urbanmarl

**Option 2: Install for Development (From Source)**

If you plan to modify scenarios, contribute code, or run benchmarks, install the package in editable mode with test and documentation dependencies:

.. code-block:: bash

    uv pip install -e .[test,docs]

.. tip::
   **BenchMARL Dependency Note:** If you encounter a dependency conflict with upstream BenchMARL, install the compatible fork directly:

   .. code-block:: bash

       uv pip install git+https://github.com/yemenlinux/BenchMARL.git

-------------------------------------------
Method 2: Installation using Standard `pip`
-------------------------------------------

If you prefer using standard Python ``venv`` and ``pip``:

Step 1: Create Virtual Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip

Step 2: Install Package
~~~~~~~~~~~~~~~~~~~~~~~

**From PyPI:**

.. code-block:: bash

    pip install urbanmarl

**From Source (Editable / Development):**

.. code-block:: bash

    git clone https://github.com/yemenlinux/vUrbanMARL.git
    cd vUrbanMARL
    pip install -e .[test,docs]

.. note::
   If ``pip`` defaults to CPU-only PyTorch on an NVIDIA GPU system, you can explicitly specify the CUDA extra index URL matching your driver:

   .. code-block:: bash

       pip install -e .[test,docs] --extra-index-url https://download.pytorch.org/whl/cu126

    For other later CUDA versions, replace ``cu126`` with the appropriate version (e.g., ``cu130`` for CUDA 13.0 or ``cu132`` for CUDA 13.2).

-----------------------
Verifying Installation
-----------------------

Verify that UrbanMARL and TorchRL are correctly installed and hardware acceleration is functioning:

.. code-block:: bash

    python -c "import torch, urbanmarl; print(f'UrbanMARL version: {urbanmarl.__version__}, CUDA available: {torch.cuda.is_available()}')"

To run the unit test suite and verify that all registered scenarios load properly:

.. code-block:: bash

    pytest tests/
