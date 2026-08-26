Installation Guide
==================

This guide covers installing **UrbanMARL** using standard `pip` or the modern fast package manager `uv`, with hardware-specific PyTorch configurations (CPU, CUDA 12.6, CUDA 13.0, and CUDA 13.2).


Installing from PyPI
--------------------

Standard installation from PyPI with default dependencies:

.. code-block:: bash

   # Using standard pip
   pip install urbanmarl

   # Using uv
   uv add urbanmarl

Hardware Extras Installation (CPU vs CUDA)
-------------------------------------------

Depending on your target compute hardware (CPU or specific CUDA toolkit versions), install `urbanmarl` with optional hardware extras.

1. CPU Installation
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Using pip
   pip install urbanmarl[cpu] --extra-index-url https://download.pytorch.org/whl/cpu

   # Using uv
   uv pip install urbanmarl[cpu]

2. CUDA 12.6 Installation
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Using pip
   pip install urbanmarl[cuda126] --extra-index-url https://download.pytorch.org/whl/cu126

   # Using uv
   uv pip install urbanmarl[cuda126]

3. CUDA 13.0 Installation
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Using pip
   pip install urbanmarl[cuda130] --extra-index-url https://download.pytorch.org/whl/cu130

   # Using uv
   uv pip install urbanmarl[cuda130]

4. CUDA 13.2 Installation
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Using pip
   pip install urbanmarl[cuda132] --extra-index-url https://download.pytorch.org/whl/cu132

   # Using uv
   uv pip install urbanmarl[cuda132]

Installing from Source (Development)
------------------------------------

To edit or contribute to `UrbanMARL`, clone the repository and install in editable mode:


.. code-block:: bash

   git clone https://github.com/yemenlinux/vUrbanMARL.git
   cd vUrbanMARL

   # Install editable package with testing and documentation tools
   pip install -e .[test,docs]

   # Or using uv
   uv pip install -e .[test,docs]

Using Requirements Files
------------------------

Alternatively, install using preconfigured requirements files under `requirements/`:

.. code-block:: bash

   # For CPU environments
   pip install -r requirements/cpu.txt

   # For CUDA 12.6 environments
   pip install -r requirements/cuda126.txt

   # For CUDA 13.0 environments
   pip install -r requirements/cuda130.txt

   # For CUDA 13.2 environments
   pip install -r requirements/cuda132.txt

Verifying Installation
----------------------

Run unit tests with `pytest` to confirm your setup:

.. code-block:: bash

   pytest

Troubleshooting Locale Errors in Sphinx
----------------------------------------

On minimal or headless Linux environments, running `python -m sphinx` may fail with `locale.Error: unsupported locale setting`.

To resolve this issue, set the `LC_ALL` environment variable in your terminal before building documentation:

.. code-block:: bash

   # Set LC_ALL in shell
   export LC_ALL=C.UTF-8
   export LANG=C.UTF-8

   # Build documentation
   python -m sphinx -b html docs/source docs/_build/html

   # Or using make
   make -C docs html

