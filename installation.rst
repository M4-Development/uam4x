Installation
============

Prerequisites
-------------

UAM4X requires Python 3.8 or higher. The following packages are required:

* NumPy
* SciPy
* Matplotlib
* OpenMDAO (for optimization)
* wxPython (for GUI)

Installation
------------

1. Clone the repository:

   .. code-block:: bash

      git clone <repository-url>
      cd uam4x

2. Create a virtual environment (recommended):

   .. code-block:: bash

      python -m venv uam4x_env
      source uam4x_env/bin/activate  # On Windows: uam4x_env\Scripts\activate

3. Install dependencies:

   .. code-block:: bash

      pip install -r requirements.txt

4. Install UAM4X in development mode:

   .. code-block:: bash

      pip install -e .

Verification
------------

To verify the installation, run:

.. code-block:: python

   import uam4x
   print("UAM4X version:", uam4x.__version__)

Optional Dependencies
---------------------

For documentation generation:

.. code-block:: bash

   pip install -r requirements-docs.txt

For GUI development:

.. code-block:: bash

   pip install wxpython

Troubleshooting
---------------

Common installation issues:

1. **wxPython installation fails**: Use conda instead of pip:
   .. code-block:: bash
      conda install wxpython

2. **OpenMDAO conflicts**: Ensure you're using a compatible version:
   .. code-block:: bash
      pip install openmdao>=3.0.0

3. **Import errors**: Make sure the uam4x package is in your Python path.
