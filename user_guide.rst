User Guide
==========

This guide provides step-by-step instructions for using UAM4X.

Getting Started
---------------

Creating Your First Study
~~~~~~~~~~~~~~~~~~~~~~~~~

A study in UAM4X represents a complete aircraft analysis or optimization problem. Here's how to create one:

.. code-block:: python

   from uam4x import Study, Vehicle
   from uam4x.components import Wing, Rotor

   # Create a new study
   study = Study("My First Study")

   # Create a vehicle
   vehicle = Vehicle("my_aircraft")

   # Add components
   wing = Wing("main_wing")
   wing.set_parameter("span", 10.0)
   wing.set_parameter("chord", 1.0)
   vehicle.add_component(wing)

   # Add vehicle to study
   study.add_vehicle(vehicle)

Working with Components
~~~~~~~~~~~~~~~~~~~~~~~

Components are the building blocks of aircraft in UAM4X:

.. code-block:: python

   from uam4x.components import Wing, Rotor, Aircraft

   # Create a wing
   wing = Wing("main_wing")
   wing.set_parameter("span", 15.0)
   wing.set_parameter("chord", 1.2)
   wing.set_parameter("sweep", 5.0)

   # Create a rotor
   rotor = Rotor("main_rotor")
   rotor.set_parameter("diameter", 8.0)
   rotor.set_parameter("blades", 4)

   # Create an aircraft and add components
   aircraft = Aircraft("my_aircraft")
   aircraft.add_component(wing)
   aircraft.add_component(rotor)


Working with Variants
~~~~~~~~~~~~~~~~~~~~~

Variants allow you to create different configurations of the same study:

.. code-block:: python

   from uam4x.study import Study, Variants

   study = Study("Multi-Config Study")

   # Create variants
   variants = Variants()

   # Short wing variant
   short_wing = study.create_variant("short_wing")
   short_wing.set_parameter("wing.span", 8.0)

   # Long wing variant
   long_wing = study.create_variant("long_wing")
   long_wing.set_parameter("wing.span", 12.0)

Using Plugins
~~~~~~~~~~~~~

Plugins extend UAM4X's capabilities:

.. code-block:: python

   from uam4x.solution import register_plugin

   @register_plugin
   def my_analysis_plugin(study):
       """Custom analysis plugin."""
       # Perform analysis
       results = analyze_aircraft(study)
       return results

GUI Usage
---------

The UAM4X GUI provides a graphical interface for managing studies and workflows:

1. **Launch the GUI**: Run the GUI application
2. **Create Studies**: Use the study management interface
3. **Add Components**: Drag and drop components onto vehicles
4. **Configure Parameters**: Set component parameters through the GUI
5. **Run Analysis**: Execute studies and view results

Best Practices
--------------

1. **Use Descriptive Names**: Give meaningful names to studies, vehicles, and components
2. **Document Your Work**: Add docstrings to custom functions and plugins
3. **Version Control**: Use version control for your study configurations
4. **Modular Design**: Break complex studies into smaller, reusable components
5. **Parameter Management**: Use consistent parameter naming conventions

Troubleshooting
---------------

Common issues and solutions:

1. **Import Errors**: Ensure all required packages are installed
2. **Parameter Errors**: Check that parameter names match component definitions
3. **Study Execution**: Verify that all required components are properly configured
4. **GUI Issues**: Restart the GUI if it becomes unresponsive
