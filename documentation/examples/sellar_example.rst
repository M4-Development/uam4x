Sellar Example
==============

First, check out the example repository:

.. code-block:: bash

    git clone git@github.com:M4-Development/uam4x_sellar_example.git
    cd uam4x_sellar_example

You can also use HTTPS:

.. code-block:: bash

    git clone https://github.com/M4-Development/uam4x_sellar_example.git
    cd uam4x_sellar_example

The ``sellar_example`` implements the classical Sellar multidisciplinary
optimization problem. It demonstrates a cyclic group, nonlinear cycle
solver selection, vector design variables, constraints, and an
optimization.

Run
---

.. code-block:: bash

    cd uam4x_examples/sellar_example
    python branching_script.py
    python run_analysis.py
    python run_post.py

``custom.py``
-------------

The base subsystem stores the design variables.

.. code-block:: python

    @register_subsystem
    class SellarBase(Subsystem):
        def setup(self):
            self.add_parameter("x", 1.0, float)
            self.add_parameter("z", np.array([5.0, 2.0]), np.ndarray)

The two discipline plugins exchange ``y1`` and ``y2`` through a cycle.
Note, the full scripts are available in the example repository, and the snippets here are just highlights
demonstrating key components. Partials are declared here between all inputs and outputs
for each plugin via the finite difference method.

.. code-block:: python

    @register_plugin
    class SellarDis1(Plugin):
        def setup_plugin(self, plugin_options):
            self.declare_input("x", float, 1.0)
            self.declare_input("z", np.ndarray, np.array([5.0, 2.0]))
            self.declare_input("y2", float, 1.0)
            self.declare_output("y1", float, 1.0)

            self.declare_partials('*','*',method='fd')


        def run_plugin(self, links, work_dir):
            z1 = links["z"][0]
            z2 = links["z"][1]
            y1 = z1 ** 2 + z2 + links["x"] - 0.2 * links["y2"]
            return {"y1": y1}

    @register_plugin
    class SellarDis2(Plugin):
        def setup_plugin(self, plugin_options):
            self.declare_input("z", np.ndarray, np.array([5.0, 2.0]))
            self.declare_input("y1", float, 1.0)
            self.declare_output("y2", float, 1.0)

            self.declare_partials('*','*',method='fd')


        def run_plugin(self, links, work_dir):
            z1 = links["z"][0]
            z2 = links["z"][1]
            y2 = links["y1"] ** 0.5 + z1 + z2
            return {"y2": y2}

The objective plugin computes the objective and constraint values.

.. code-block:: python

    @register_plugin
    class SellarObjective(Plugin):
        def setup_plugin(self, plugin_options):
            self.declare_input("x", float, 1.0)
            self.declare_input("z", np.ndarray, np.array([5.0, 2.0]))
            self.declare_input("y1", float, 1.0)
            self.declare_input("y2", float, 1.0)
            self.declare_output("obj", float, 1.0)
            self.declare_output("con1", float, 1.0)
            self.declare_output("con2", float, 1.0)

            self.declare_partials('*','*',method='fd')


        def run_plugin(self, links, work_dir):
            obj = links["x"] ** 2 + links["z"][1] + links["y1"] + np.exp(-links["y2"])
            con1 = 3.16 - links["y1"]
            con2 = links["y2"] - 24.0
            return {"obj": obj, "con1": con1, "con2": con2}

``branching_script.py``
-----------------------

The model contains a single Sellar base component.

.. code-block:: python

    def setup_model(study: Study):
        vehicle = Vehicle("sellar_vehicle")
        sb = SellarBase("sellar_base")
        sb.set_parameter("x", 1.0)
        sb.set_parameter("z", np.array([5.0, 2.0]))
        vehicle.add_component(sb)
        study.add_vehicle(vehicle)

The solution creates the cycle group and connects both disciplines.

.. code-block:: python

    from uam4x.solution import Group, Solution

    def setup_solution(study: Study):
        sellar_basic = Solution("sellar_basic")
        sellar_cycle = Group()
        sellar_cycle.add_step(SellarDis1, "sellardis1")
        sellar_cycle.add_step(SellarDis2, "sellardis2")
        sellar_cycle.declare_cycle_step_order(["sellardis1", "sellardis2"])
        sellar_cycle.set_cycle_solver("NonlinearBlockGS")

        sellar_basic.add_step(sellar_cycle, "sellar_cycle")
        sellar_cycle.add_link("sellardis1.y1", "sellardis2.y1")
        sellar_cycle.add_link("sellardis2.y2", "sellardis1.y2")

        sellar_basic.add_link(
            "vehicles.sellar_vehicle.sellar_base.x",
            "sellar_cycle.sellardis1.x",
        )
        sellar_basic.add_link(
            "vehicles.sellar_vehicle.sellar_base.z",
            "sellar_cycle.sellardis1.z",
        )
        sellar_basic.add_link(
            "vehicles.sellar_vehicle.sellar_base.z",
            "sellar_cycle.sellardis2.z",
        )

        sellar_basic.add_step(SellarObjective, "sellar_objective")
        sellar_basic.add_link("sellar_cycle.sellardis1.y1", "sellar_objective.y1")
        sellar_basic.add_link("sellar_cycle.sellardis2.y2", "sellar_objective.y2")
        sellar_basic.add_tracked_output("sellar_objective.obj")
        study.add_solution(sellar_basic)

The optimizer uses ``x`` and ``z`` as design variables.

.. code-block:: python

    def setup_tradestudy(study: Study):
        optimization = Optimization()
        optimization.add_optimizer_option("ACC", 1e-8)
        optimization.set_driver("pyOptSparseDriver")
        optimization.add_design_var(
            "vehicles.sellar_vehicle.sellar_base.x",
            lower=0.0,
            upper=10.0,
        )
        optimization.add_design_var(
            "vehicles.sellar_vehicle.sellar_base.z",
            lower=np.array([0.0, 0.0]),
            upper=np.array([10.0, 10.0]),
        )
        optimization.add_constraint("sellar_objective.con1", upper=0.0)
        optimization.add_constraint("sellar_objective.con2", upper=0.0)
        optimization.add_objective("sellar_objective.obj")
        study.control.add_tradestudy("sellar_problem", optimization)

``run_analysis.py``
-------------------

.. code-block:: python

    if __name__ == "__main__":
        study_dir = os.path.join(THIS_DIR, "variants", "sellar")
        study = Study.from_json(study_dir)
        engine = Engine(study, os.path.join(study_dir, "work"))
        engine.run_tradestudy("sellar_problem", solution_name="sellar_basic")
        study.save_to_json(study_dir)

``run_post.py``
---------------
The various variables of interest can be extracted. The results API allows
for index based requests for vector variables or the full array if desired.

.. code-block:: python

    solution_results = study.results.get_solution_result("sellar_basic")
    trade = solution_results.get_tradestudy("sellar_problem")
    out_z = trade.get_variable("vehicles.sellar_vehicle.sellar_base.z")
    out_z0 = trade.get_variable('vehicles.sellar_vehicle.sellar_base.z[0]')
    out_x = trade.get_variable("vehicles.sellar_vehicle.sellar_base.x")
    obj = trade.get_variable("sellar_objective.obj")
    con1 = trade.get_variable("sellar_objective.con1")
    con2 = trade.get_variable("sellar_objective.con2")
