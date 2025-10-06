Examples
========

This section provides practical examples of using UAM4X.

Basic Aircraft Study
--------------------

This example shows how to create a simple aircraft study:

.. code-block:: python

   from uam4x import Study, Vehicle
   from uam4x.components import Wing, Rotor

   def create_basic_aircraft():
       """Create a basic aircraft study."""
       # Create study
       study = Study("Basic Aircraft Study")

       # Create vehicle
       vehicle = Vehicle("basic_aircraft")

       # Add wing
       wing = Wing("main_wing")
       wing.set_parameter("span", 10.0)
       wing.set_parameter("chord", 1.0)
       wing.set_parameter("sweep", 0.0)
       vehicle.add_component(wing)

       # Add rotor
       rotor = Rotor("main_rotor")
       rotor.set_parameter("diameter", 8.0)
       rotor.set_parameter("blades", 4)
       vehicle.add_component(rotor)

       # Add vehicle to study
       study.add_vehicle(vehicle)

       return study

Multi-Objective Optimization
----------------------------

This example demonstrates multi-objective optimization:

.. code-block:: python

   from uam4x import Study, Vehicle
   from uam4x.components import Wing
   from uam4x.control import Optimization

   def multi_objective_optimization():
       """Set up a multi-objective optimization study."""
       study = Study("Multi-Objective Optimization")

       # Create vehicle
       vehicle = Vehicle("optimized_aircraft")

       # Add wing with design variables
       wing = Wing("main_wing")
       wing.set_parameter("span", 10.0)  # Initial value
       wing.set_parameter("chord", 1.0)  # Initial value
       vehicle.add_component(wing)

       # Set up optimization
       optimization = Optimization()
       optimization.add_design_variable("wing.span", lower=8.0, upper=15.0)
       optimization.add_design_variable("wing.chord", lower=0.8, upper=1.5)
       optimization.add_objective("weight", minimize=True)
       optimization.add_objective("drag", minimize=True)

       study.set_control(optimization)
       study.add_vehicle(vehicle)

       return study


Plugin Development
------------------

This example shows how to create a custom plugin:

.. code-block:: python

   from uam4x.solution import register_plugin
   from uam4x.study import Study

   @register_plugin
   def weight_analysis(study: Study):
       """Calculate aircraft weight."""
       total_weight = 0.0

       for vehicle in study.vehicles:
           for component in vehicle.components:
               # Simple weight calculation
               if hasattr(component, 'get_parameter'):
                   span = component.get_parameter('span', 0.0)
                   chord = component.get_parameter('chord', 0.0)
                   weight = span * chord * 10.0  # Simplified weight model
                   total_weight += weight

       study.set_result("total_weight", total_weight)
       return total_weight

Parameter Sweep
---------------

This example demonstrates parameter sweep analysis:

.. code-block:: python

   from uam4x import Study, Vehicle
   from uam4x.components import Wing
   from uam4x.control import ParameterSweep

   def parameter_sweep_study():
       """Set up a parameter sweep study."""
       study = Study("Parameter Sweep")

       # Create vehicle
       vehicle = Vehicle("sweep_aircraft")

       # Add wing
       wing = Wing("main_wing")
       wing.set_parameter("span", 10.0)
       wing.set_parameter("chord", 1.0)
       vehicle.add_component(wing)

       # Set up parameter sweep
       sweep = ParameterSweep()
       sweep.add_parameter("wing.span", values=[8.0, 10.0, 12.0, 15.0])
       sweep.add_parameter("wing.chord", values=[0.8, 1.0, 1.2, 1.5])

       study.set_control(sweep)
       study.add_vehicle(vehicle)

       return study

Running the Examples
--------------------

To run these examples:

1. Save the code to Python files
2. Run them in your Python environment
3. Modify parameters to explore different configurations
4. Use the GUI to visualize and interact with the results

For more complex examples, see the `examples/` directory in the UAM4X repository.
