UAM4X Documentation
===================

Welcome to UAM4X, the Urban Air Mobility 4X Framework for aircraft design and optimization.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   user_guide
   api_reference
   examples
   contributing

Overview
--------

UAM4X is a comprehensive framework for Urban Air Mobility (UAM) aircraft design, analysis, and optimization. It provides tools for:

* Aircraft component modeling and analysis
* Multi-objective optimization
* Study management and variant handling
* Plugin system for extensibility

Key Features
------------

* **Subsystem System**: Modular aircraft subsystems (wings, rotors, aircraft, etc.)
* **Plugin Architecture**: Extensible plugin system for analysis tools
* **Study Management**: Comprehensive study and variant management
* **GUI Interface**: User-friendly graphical interface for workflow management

Quick Start
-----------

.. code-block:: python

   from uam4x import Study, Vehicle, Wing

   # Create a new study
   study = Study()

   # Create a vehicle
   vehicle = Vehicle('my_aircraft')

   # Add a wing component
   wing = Wing('main_wing')
   wing.set_parameter('span', 10.0)
   vehicle.add_component(wing)

   # Add vehicle to study
   study.add_vehicle(vehicle)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
