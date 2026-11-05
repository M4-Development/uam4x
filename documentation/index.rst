UAM4X documentation
====================

UAM4X provides aircraft models, solution graphs, trade studies, and analysis
plugins. This is the single documentation source for the framework, the public
plugins installed with it, and the examples in this repository.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   basic_overview
   installation
   user_guide
   api_reference
   plugins/index
   examples
   contributing

Quick start
-----------

From the ``uam4x`` project directory in an active Python environment:

.. code-block:: bash

   python -m pip install -e .
   python examples/aleatory_cantilever/branching_script.py
   python examples/aleatory_cantilever/run_analysis.py

See :doc:`examples` for all eight examples and their additional requirements.

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
