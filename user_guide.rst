User Guide
==========

This guide provides step-by-step instructions for using UAM4X.

Getting Started
---------------

Creating Your First Study
~~~~~~~~~~~~~~~~~~~~~~~~~

A study in UAM4X represents a complete aircraft analysis or optimization problem. Here's how to create one:

.. code-block:: python

   from uam4x.study import Study
   from uam4x.vehicle import Vehicle
   from uam4x.subsystems import Wing

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

Working with Subsystems
~~~~~~~~~~~~~~~~~~~~~~~

 Subsystems are the building blocks of aircraft in UAM4X:

.. code-block:: python

   from uam4x.subsystems import Wing, Rotor
   from uam4x.vehicles import Vehicle

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
   aircraft = Vehicle("my_aircraft")
   aircraft.add_component(wing)
   aircraft.add_component(rotor)


Working with Variants, an example problem:
~~~~~~~~~~~~~~~~~~~~~

multi_objective
^^^^^^^^^^^^^^^

The ``multi_objective`` example is a complete end-to-end pattern for:

1. Declaring custom subsystems and plugins
2. Building a model and trade studies
3. Generating a variant JSON
4. Running the analysis and reading results

First, check out the example repository:

.. code-block:: bash

   git clone git@github.com:M4-Development/uam4x_multiobjective.git
   cd uam4x_multiobjective

You can also use HTTPS:

.. code-block:: bash

   git clone https://github.com/M4-Development/uam4x_multiobjective.git
   cd uam4x_multiobjective

Required Plugin: OpenVSP
""""""""""""""""""""""""

The ``multi_objective`` example uses ``OpenvspPlugin``:

.. code-block:: python

   from uam4x_openvsp.plugin import OpenvspPlugin

Before running the example, install both:

1. OpenVSP itself (includes the Python API used by ``import openvsp``)
2. The UAM4X OpenVSP plugin package (``uam4x_openvsp``)

OpenVSP install:

.. code-block:: text

   Download and install OpenVSP from https://openvsp.org/

Install the plugin package from the plugins repo:

.. code-block:: bash

   git clone https://github.com/M4-Development/uam4x_plugins.git
   cd uam4x_plugins/openvsp
   pip install -e .

Verify the OpenVSP Python module is available in your active environment:

.. code-block:: bash

   python -c "import openvsp; print('openvsp import OK')"

Step 1: ``custom.py`` (custom subsystem + plugins)
""""""""""""""""""""""""""""""""""""""""""""""""""

This file defines the reusable building blocks used by the solution graph.

The subsystem is registered with ``@register_subsystem`` so it can be added to a vehicle:

.. code-block:: python

   @register_subsystem
   class WeightingFactor(Subsystem):
       def setup(self):
           self.add_parameter(
               "weighting_factor",
               0.0,
               float,
               unit_type=UnitType.LENGTH,
               description="Wing span",
           )

The first plugin computes area and exposes exact derivatives:

.. note::

   Plugin inputs must declare a concrete Python type in ``setup_plugin`` (for example ``float`` or ``int``).
   Use ``self.declare_input("name", input_type, default_value)`` for every input so UAM4X can validate links and
   correctly construct optimization variables/derivatives.

.. code-block:: python

   @register_plugin
   class CalcArea(Plugin):
       def setup_plugin(self, plugin_options):
           self.declare_input("aspect_ratio", float, 1.0)
           self.declare_input("span", float, 1.0)
           self.declare_output("area", float, 1.0)
           self.declare_partials_plugin(["area"], ["span", "aspect_ratio"], method="exact")

       def run_plugin(self, links, work_dir):
           area = (links["span"] ** 2) / links["aspect_ratio"]
           return {"area": area}

       def compute_partials_plugin(self, links):
           return {
               ("area", "span"): 2 * links["span"] / links["aspect_ratio"],
               ("area", "aspect_ratio"): -links["span"] ** 2 / links["aspect_ratio"] ** 2,
           }

The second plugin computes chord, drag, wing weight, and objective:

.. code-block:: python

   @register_plugin
   class CalcObjective(Plugin):
       def setup_plugin(self, plugin_options):
           self.declare_input("aspect_ratio", float, 1.0)
           self.declare_input("span", float, 1.0)
           self.declare_input("area", float, 1.0)
           self.declare_input("weighting_factor", float, 1.0)
           self.declare_output("chord", float, 1.0)
           self.declare_output("CD", float, 1.0)
           self.declare_output("wing_weight", float, 1.0)
           self.declare_output("objective", float, 1.0)
           self.declare_partials_plugin("*", "*", method="exact")

       def run_plugin(self, links, work_dir):
           if links["weighting_factor"] < 0:
               raise PluginException("plugin failed")
           # ... compute CD, wing_weight, objective
           return {
               "chord": chord,
               "CD": self.CD,
               "wing_weight": W_box,
               "objective": objective,
           }

       def compute_partials_plugin(self, links):
           # ... return analytical derivative dictionary
           return partial_to_val


Step 2: ``branching_script.py`` (model setup + variant generation)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

This file wires the study in three setup functions.

Model setup:

.. code-block:: python

   def setup_model(study: Study):
       veh1 = Vehicle("multi_objective_vehicle")
       wing1 = Wing("wing1")
       wing1.set_parameter("span", 100.0)
       factor = WeightingFactor("weighting_factor")
       veh1.add_component(wing1)
       veh1.add_component(factor)
       study.add_vehicle(veh1)

Solution setup:

.. code-block:: python

   def setup_solution(study: Study):
       study.solution.add_step(CalcArea, "calc_area")
       study.solution.add_link(
           "vehicles.multi_objective_vehicle.wing1.aspect_ratio",
           "calc_area.aspect_ratio",
       )
       study.solution.add_link(
           "vehicles.multi_objective_vehicle.wing1.span",
           "calc_area.span",
       )
       study.solution.add_step(CalcObjective, "calc_objective")
       study.solution.add_link("calc_area.area", "calc_objective.area")
       study.solution.add_tracked_output("calc_objective.objective")

Trade-study setup:

.. code-block:: python

   def setup_tradestudy(study: Study):
       optimization = Optimization()
       optimization.add_design_var(
           "vehicles.multi_objective_vehicle.wing1.aspect_ratio",
           lower=1.0,
           upper=10.0,
       )
       optimization.add_design_var(
           "vehicles.multi_objective_vehicle.wing1.span",
           lower=100.0,
           upper=1000.0,
       )
       optimization.add_constraint("calc_area.area", lower=49999.0, upper=50001.0)
       optimization.add_objective("calc_objective.objective")

       parameter_sweep = ParameterSweep(
           {"vehicles.multi_objective_vehicle.weighting_factor.weighting_factor": [-10, 0, 5000, 10000]},
           optimization,
       )
       study.control.add_tradestudy("sweep", parameter_sweep)
       study.control.add_tradestudy("optimization_only", optimization)

Variant generation in ``__main__``:

.. code-block:: python

   if __name__ == "__main__":
       demo = [setup_model, setup_solution, setup_tradestudy]
       framework = Variants(THIS_DIR)
       framework.add_variant("demo", demo)
       study = framework.generate_variants()[0]

Step 3: ``run_analysis.py`` (execute and inspect results)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

This script loads the generated variant and runs both trade studies.

.. code-block:: python

   if __name__ == "__main__":
       file = os.path.join(THIS_DIR, "variants", "demo", "data.json")
       study = Study.from_json(file)
       engine = Engine(study, os.path.join(THIS_DIR, "variants", "demo", "work"))

       engine.run_tradestudy("sweep", num_processors=10)
       engine.run_solution()
       engine.run_tradestudy("optimization_only")

       study.save_results_to_json(file)

Then read outputs from each trade study:

.. code-block:: python

   sweep_ts = study.results.get_tradestudy("sweep")
   opt_ts = study.results.get_tradestudy("optimization_only")
   sp_ts = study.results.get_tradestudy("Default State")

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
2. **Parameter Errors**: Check that parameter names match subsystem definitions
3. **Study Execution**: Verify that all required components are properly configured
4. **GUI Issues**: Restart the GUI if it becomes unresponsive
