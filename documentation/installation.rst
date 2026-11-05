Installation
============

UAM4X requires Python 3.11 or newer. From a checkout of the ``uam4x``
project, create and activate an environment, then run:

.. code-block:: bash

   python -m pip install -e .

The project installation includes the framework and the public
``uam4x_general``, ``uam4x_ndarc``, ``uam4x_openvsp``, and ``uam4x_ixgen``
plugin packages in ``plugins/``. Their code is in this repository; a separate
public plugin checkout is unnecessary. CAMRAD II and M4SS are private packages
and must be installed separately when needed. See :doc:`plugins/index`.

Check the core and general plugin imports in the active environment:

.. code-block:: bash

   python -c "import uam4x, uam4x_general; print('UAM4X imports OK')"

External analysis software
--------------------------

The Python installation does not install NDARC, IXGEN, or OpenVSP.
Examples that use them also need the corresponding executable or Python
binding, local input files, and executable paths configured in their scripts.
The OpenVSP plugin imports its Python bindings and Pillow at module import time,
so these are also needed when an example imports the plugin while creating a
study. NDARC and IXGEN
examples require ``rcotools``. See the individual :doc:`examples` pages
before running these.

Build the documentation
-----------------------

From the ``uam4x`` project directory:

.. code-block:: bash

   python -m pip install -r documentation/requirements.txt
   sphinx-build -M html documentation documentation/_build

Open ``documentation/_build/html/index.html``. API pages mock optional solver
bindings during the documentation build; building pages does not run solvers.
