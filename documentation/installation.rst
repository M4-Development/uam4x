Installation
============

Prerequisites
-------------

uam4x requires Python 3.11 or higher. The following packages are required
(as defined in ``setup.py``):

* ``numpy>=2.2.6``
* ``pytest>=8.4.2``
* ``openmdao>=3.39.0``
* ``cpylog``
* ``pytest-cov``
* ``pytest``
* ``pytest-html``
* ``pygit2``
* ``matplotlib>=3.10.0``
* ``openturns==1.26.0``
* ``scipy>=1.16.1``
* ``tzdata==2025.2``
* ``sympy==1.14.0``
* ``pandas==2.3.1``

Installation Procedure
-----------------------

1. Clone the repository:

   .. code-block:: bash

      # Using https:
      git clone https://github.com/M4-Development/uam4x.git
      cd uam4x

      # Using SSH:
      git clone git@github.com:M4-Development/uam4x.git
      cd uam4x


2. Install Anaconda (if needed) and create a conda environment (recommended):

   Download Anaconda: https://www.anaconda.com/download

   .. code-block:: bash

      conda create -n uam4x python=3.11
      conda activate uam4x

3. Install uam4x in development mode (this installs all required packages from ``setup.py``):

   .. code-block:: bash

      pip install -e .

Verification
------------

To verify the installation, run:

.. code-block:: python

   import uam4x
   print("uam4x version:", uam4x.__version__)

Optional Dependencies
---------------------

For documentation generation:

.. code-block:: bash

   pip install -r requirements-docs.txt

For the GUI:

The uam4x GUI requires Python 3.11.

The GUI depends on the following packages:

* ``wxpython``
* ``numpy``
* ``uam4x``
* ``openvsp``
* ``rcotools``
* ``uam4x_openvsp``

Install those dependencies in your active environment before installing the
GUI wheel.

To install the GUI itself, use the wheel file provided for your platform:

.. code-block:: bash

   pip install path/to/uam4x_gui-<version>-<platform>.whl

The GUI wheel is platform dependent, so install the wheel that matches your
operating system and Python environment.

For plugin packages (from ``uam4x_plugins``):

Some plugins require additional external software and licenses. Install only the
plugin package(s) you need.

.. code-block:: bash

   git clone https://github.com/M4-Development/uam4x_plugins.git
   cd uam4x_plugins
   pip install -r requirements.txt

To install only the OpenVSP plugin package:

.. code-block:: bash

   cd uam4x_plugins/openvsp
   pip install -e .

Troubleshooting
---------------

Common installation issues:

1. **wxPython installation fails**: Use conda instead of pip:

   .. code-block:: bash

      conda install wxpython

2. **wxPython installation help Linux**: Ensure you have the necessary system dependencies:

   Follow instructions at: https://wxpython.org/blog/2017-08-17-builds-for-linux-with-pip/index.html

3. **OpenMDAO conflicts**: Ensure you're using a compatible version:

   .. code-block:: bash

      pip install openmdao>=3.39.0

4. **Import errors**: Make sure the uam4x package is in your Python path.
