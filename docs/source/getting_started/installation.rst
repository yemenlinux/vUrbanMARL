==================
Installation Guide
==================

UrbanMARL requires Python 3.10 or newer. Due to the heavy reliance on PyTorch and hardware-accelerated tensor operations, it is crucial to install the correct version of the package that matches your system's hardware capabilities.

We provide two primary methods for installation: utilizing the modern ``uv`` package manager (recommended for faster dependency resolution) or utilizing the standard ``pip`` installer.

---------------------------------
Method 1: Installation using `uv` (Recommended)
---------------------------------

`uv` is an extremely fast Python package installer. UrbanMARL's configuration is optimized for `uv`, allowing it to automatically route PyTorch dependencies to the correct hardware-specific index without requiring manual URL flags.

First, ensure `uv` is installed on your system:

.. code-block:: bash

    curl -LsSf https://astral.sh/uv/install.sh | sh

Once installed, you can install UrbanMARL directly by specifying your target hardware as an extra:

**CPU Only (No NVIDIA GPU):**

.. code-block:: bash

    uv pip install urbanmarl[cpu]

**CUDA 12.6:**

.. code-block:: bash

    uv pip install urbanmarl[cuda126]

**CUDA 13.0:**

.. code-block:: bash

    uv pip install urbanmarl[cuda130]

**CUDA 13.2:**

.. code-block:: bash

    uv pip install urbanmarl[cuda132]

---------------------------------------
Method 2: Installation using standard `pip`
---------------------------------------

If you prefer to use the standard Python package installer (``pip``), you must explicitly provide the extra index URL. This ensures ``pip`` fetches the correct hardware-specific PyTorch binaries instead of defaulting to the standard PyPI release.

Ensure your ``pip`` is up to date before proceeding:

.. code-block:: bash

    python -m pip install --upgrade pip

**CPU Only:**

.. code-block:: bash

    pip install urbanmarl[cpu] --extra-index-url https://download.pytorch.org/whl/cpu

**CUDA 12.6:**

.. code-block:: bash

    pip install urbanmarl[cuda126] --extra-index-url https://download.pytorch.org/whl/cu126

**CUDA 13.0:**

.. code-block:: bash

    pip install urbanmarl[cuda130] --extra-index-url https://download.pytorch.org/whl/cu130

**CUDA 13.2:**

.. code-block:: bash

    pip install urbanmarl[cuda132] --extra-index-url https://download.pytorch.org/whl/cu132

.. note::
    Failure to include the ``--extra-index-url`` flag when using ``pip`` may result in downloading incompatible PyTorch wheels, which can cause the simulation environment to crash or fail to recognize your GPU hardware.

-----------------------
Development Setup
-----------------------

If you intend to modify the UrbanMARL source code, contribute to the repository, or run the test suites, install the package in editable mode. 

Clone the repository and install it along with the testing and documentation dependencies:

.. code-block:: bash

    git clone https://github.com/yemenlinux/vUrbanMARL.git
    cd vUrbanMARL

Using ``uv`` (for CUDA 12.6):

.. code-block:: bash

    uv pip install -e .[cuda126,test,docs]

Using ``pip`` (for CUDA 12.6):

.. code-block:: bash

    pip install -e .[cuda126,test,docs] --extra-index-url https://download.pytorch.org/whl/cu126

