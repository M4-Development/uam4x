Examples
========

This section provides practical examples of using UAM4X.

Basic Aircraft Study
--------------------

This example shows how to create a simple aircraft study:

.. code-block:: python

   from uam4x.study import Study
   from uam4x.vehicle import Vehicle
   from uam4x.subsystems import Wing, Rotor

   def create_basic_aircraft():
       """Create a basic aircraft study."""
       study = Study()

       vehicle = Vehicle("basic_aircraft")

       wing = Wing("main_wing")
       wing.set_parameter("span", 10.0)
       wing.set_parameter("chord", 1.0)
       wing.set_parameter("sweep", 0.0)
       vehicle.add_component(wing)

       rotor = Rotor("main_rotor")
       rotor.set_parameter("diameter", 8.0)
       rotor.set_parameter("blades", 4)
       vehicle.add_component(rotor)

       study.add_vehicle(vehicle)
       return study

Multiobjective Example
----------------------

What this demonstrates:
An OpenMDAO optimization nested inside a parameter sweep, plus OpenVSP geometry updates through ``OpenvspPlugin``.

Get the example:

.. code-block:: bash

   git clone https://github.com/M4-Development/uam4x_multiobjective.git
   cd uam4x_multiobjective

Required plugins/tools:

1. Install OpenVSP (https://openvsp.org/)
2. Install UAM4X OpenVSP plugin

.. code-block:: bash

   git clone https://github.com/M4-Development/uam4x_plugins.git
   cd uam4x_plugins/openvsp
   pip install -e .

Run:

.. code-block:: bash

   python branching_script.py
   python run_analysis.py

Core code walkthrough:

``custom.py`` defines domain plugins and analytical derivatives:

.. code-block:: python

   @register_plugin
   class CalcArea(Plugin):
       def setup_plugin(self, plugin_options):
           self.declare_input("aspect_ratio", float, 1.0)
           self.declare_input("span", float, 1.0)
           self.declare_output("area", float, 1.0)
           self.declare_partials_plugin(["area"], ["span", "aspect_ratio"], method="exact")

``branching_script.py`` builds model, solution, and trade studies:

.. code-block:: python

   study.solution.add_step(CalcArea, "calc_area")
   study.solution.add_step(CalcObjective, "calc_objective")
   study.solution.add_step(OpenvspPlugin, "openvsp", {...})
   study.solution.add_link("calc_area.area", "calc_objective.area")

   optimization = Optimization()
   optimization.add_design_var("vehicles.multi_objective_vehicle.wing1.aspect_ratio", lower=1., upper=10.)
   parameter_sweep = ParameterSweep(
       {"vehicles.multi_objective_vehicle.weighting_factor.weighting_factor": [...]},
       optimization
   )
   study.control.add_tradestudy("sweep", parameter_sweep)

``run_analysis.py`` executes and processes results:

.. code-block:: python

   engine.run_tradestudy("sweep", num_processors=10)
   engine.run_solution()
   engine.run_tradestudy("optimization_only")

   sweep_ts = study.results.get_tradestudy("sweep")
   obj = sweep_ts.get_variable("calc_objective.objective")

Insight:
This example shows composition of analyses: parameter sweep -> optimization child -> plugin chain.

Tilt Rotor Example
------------------

What this demonstrates:
A coupled rotorcraft workflow linking NDARC sizing outputs with M4SS/OpenVSP-based structural/geometry calculations.

Get the example:

.. code-block:: bash

   git clone https://github.com/M4-Development/uam4x_tiltrotor.git
   cd uam4x_tiltrotor

Required plugins/tools:

1. ``uam4x_ndarc`` plugin
2. ``uam4x_m4ss`` plugin
3. OpenVSP install (https://openvsp.org/)
4. NDARC executable
5. M4SS executable

.. code-block:: bash

   git clone https://github.com/M4-Development/uam4x_plugins.git
   cd uam4x_plugins/ndarc && pip install -e .
   cd ../m4ss && pip install -e .

Run:

.. code-block:: bash

   python branching_script.py
   python run_analysis.py

Core code walkthrough:

``custom.py`` provides coupled helper plugins/subsystems (including M4SS-style weight processing):

.. code-block:: python

   @register_plugin
   class FakeFastOpenvspM4ssPlugin(Plugin):
       def setup_plugin(self, plugin_options):
           self.declare_input("openvsp.parm_id.HFZNSRXUBZN", float, 0.0)
           self.declare_output("m4ss.weight.total", float, 0.0)

``branching_script.py`` sets up NDARC + OpenVSP/M4SS coupling and sweep:

.. code-block:: python

   study.solution.add_step(NdarcPlugin, "ndarc", {...})
   study.solution.add_step(FakeFastOpenvspM4ssPlugin, "openvsp_m4ss_vertical", {...})
   study.solution.add_link("vehicles.mtr.wing.span", "ndarc.Wing(1).span")
   study.solution.add_link("ndarc....", "openvsp_m4ss_vertical.m4ss.MODIFY....")
   study.control.add_tradestudy("ndarc_m4ss_couple", parameter_sweep)

``run_analysis.py`` executes the coupled trade and persists results:

.. code-block:: python

   engine.run_tradestudy("ndarc_m4ss_couple")
   study.save_results_to_json(variant_path)

Insight:
This is a coupled multidisciplinary pattern where tool outputs become structural/weight inputs for downstream steps.

Quad Rotor Example
------------------

What this demonstrates:
A multi-variant quad-rotor workflow sweeping rotor tip speed and linking NDARC rotor sizing outputs into OpenVSP.

Get the example:

.. code-block:: bash

   git clone https://github.com/M4-Development/uam4x_quad.git
   cd uam4x_quad

Required plugins/tools:

1. ``uam4x_ndarc`` plugin
2. ``uam4x_openvsp`` plugin
3. OpenVSP install (https://openvsp.org/)
4. NDARC executable

.. code-block:: bash

   git clone https://github.com/M4-Development/uam4x_plugins.git
   cd uam4x_plugins/ndarc && pip install -e .
   cd ../openvsp && pip install -e .

Run:

.. code-block:: bash

   python branching_script.py
   python quad_rotor_analysis.py

Core code walkthrough:

``branching_script.py`` links NDARC rotor sizing to OpenVSP geometry:

.. code-block:: python

   study.solution.add_step(NdarcPlugin, "ndarc", {...})
   study.solution.add_step(OpenvspPlugin, "openvsp", {...})
   study.solution.add_link("vehicles.mtr.rotor.vtip", "ndarc.ROTOR(1).Vtip_ref")
   study.solution.add_link("ndarc.soln.Rotor(1).RADIUS", "openvsp.UserParm.Default_Group.rotor_radius")
   study.control.add_tradestudy("vtip_trade", ParameterSweep({"vehicles.mtr.rotor.vtip": [...]}))

``quad_rotor_analysis.py`` runs each generated variant and writes results:

.. code-block:: python

   for active_variant in ACTIVE_VARIANTS:
       study = Study.from_json(variant_path)
       engine = Engine(study, work_dir)
       engine.run_tradestudy("vtip_trade")
       study.save_results_to_json(variant_path)

Insight:
This example shows multi-variant orchestration with the same trade definition reused across configurations.

Sellar Example
--------------

What this demonstrates:
The classical Sellar multidisciplinary optimization problem with two coupled disciplines solved in a nonlinear cycle.

Get the example:

.. code-block:: bash

   git clone https://github.com/M4-Development/uam4x_sellar_example.git
   cd uam4x_sellar_example

Required plugins/tools:
No external UAM4X plugin package is required.

Run:

.. code-block:: bash

   python branching_script.py
   python run_analysis.py

Core code walkthrough:

``custom.py`` implements Sellar disciplines/objective plugins.

``branching_script.py`` builds a cyclic group and enables a cycle solver:

.. code-block:: python

   sellar_cycle = Group()
   sellar_cycle.add_step(SellarDis1, "sellardis1")
   sellar_cycle.add_step(SellarDis2, "sellardis2")
   sellar_cycle.declare_cycle_step_order(["sellardis1", "sellardis2"])
   sellar_cycle.set_cycle_solver("NonlinearBlockGS")

   optimization = Optimization()
   optimization.add_design_var("vehicles.sellar_vehicle.sellar_base.x", lower=0.0, upper=10.)
   optimization.add_objective("sellar_objective.obj")

``run_analysis.py`` executes and inspects objective/constraint/design outputs:

.. code-block:: python

   engine.run_tradestudy("sellar_problem")
   trade = study.results.get_tradestudy("sellar_problem")
   obj = trade.get_variable("sellar_objective.obj")
   con1 = trade.get_variable("sellar_objective.con1")

Insight:
This demonstrates strongly-coupled MDO with explicit cycle solver configuration in UAM4X ``Group``.

Borehole Example
----------------

What this demonstrates:
Uncertainty quantification using Polynomial Chaos Expansion (regression PCE) on the borehole flow-rate model.

Get the example:

.. code-block:: bash

   git clone https://github.com/M4-Development/uam4x_borehole_example.git
   cd uam4x_borehole_example

Required plugins/tools:
No external UAM4X plugin package is required.

Run:

.. code-block:: bash

   python branching_script.py
   python run_analysis.py

Core code walkthrough:

``custom.py`` defines the borehole response plugin:

.. code-block:: python

   @register_plugin
   class BoreholeAnalysis(Plugin):
       def run_plugin(self, links, work_dir):
           flow_rate = numerator / denominator
           return {"flow_rate": flow_rate}

``branching_script.py`` configures a UQ workflow using regression PCE:

.. code-block:: python

   workflow = RegressionPCE(num_samples=3*comb(5+8, 5), SamplingClass=LatinHypercubeSampler, poly_order=5)
   uq = UQAnalysis(seed=42, UQ_workflow=workflow)
   uq.add_aleatory_variable("vehicles.borehole_vehicle.borehole_base.rw", NormalDistribution(...))
   study.control.add_tradestudy("borehole_problem", uq)

``run_analysis.py`` performs post-processing on UQ outputs:

.. code-block:: python

   uq = study.results.get_tradestudy("borehole_problem")
   surrogate_model = uq.post.generate_surrogate("borehole_calc.flow_rate")
   analytical_mean = float(surrogate_model.analytical_mean)
   new_samples, predicted_outputs = surrogate_model.sample(num_samples=5000, method="LHS")

Insight:
This shows end-to-end polynomial chaos workflow: model evaluation, surrogate construction, and moment/error validation.

Ishigami Example
----------------

What this demonstrates:
Sobol sensitivity analysis on the Ishigami function, including first-order and total-order index comparison against analytical values.

Get the example:

.. code-block:: bash

   git clone https://github.com/M4-Development/uam4x_ishigami.git
   cd uam4x_ishigami

Required plugins/tools:
No external UAM4X plugin package is required.

Run:

.. code-block:: bash

   python branching_script.py
   python run_analysis.py

Core code walkthrough:

``custom.py`` defines the Ishigami response plugin:

.. code-block:: python

   @register_plugin
   class IshigamiAnalysis(Plugin):
       def run_plugin(self, links, work_dir):
           g = np.sin(X1) + a*np.sin(X2)**2 + b*X3**4*np.sin(X1)
           return {"g": g}

``branching_script.py`` configures Sobol sensitivity UQ:

.. code-block:: python

   workflow = SobolSensitivity(5000, SamplingClass=SobolSampler)
   uq = UQAnalysis(seed=42, UQ_workflow=workflow)
   uq.add_aleatory_variable("vehicles.ishigami_vehicle.ishigami_base.X1", UniformDistribution(low=-np.pi, high=np.pi))
   study.control.add_tradestudy("ishigami_problem", uq)

``run_analysis.py`` extracts first/total Sobol indices and compares to analytical values:

.. code-block:: python

   uq = study.results.get_tradestudy("ishigami_problem")
   first_order_terms = uq.post.get_first_order_indices("ishigami_calc.g")
   total_order_terms = uq.post.get_total_order_indices("ishigami_calc.g")
   # compare S1/S2/S3 and ST1/ST2/ST3 against analytical expressions

Insight:
This demonstrates global sensitivity analysis with direct computed-vs-analytical Sobol index checks.
