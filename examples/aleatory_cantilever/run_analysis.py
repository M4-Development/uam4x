from uam4x.study import Study
from uam4x.execution_engine import Engine
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if __name__ == "__main__":
    save_path = os.path.join(THIS_DIR, "variants", "nipc_uq")

    #--------------------------------------Run NIPC UQ analysis
    study = Study.load_study_from_directory(save_path)
    engine = Engine(study, os.path.join(save_path, "work"))
    engine.run_tradestudy("lhs_pce_analysis", solution_name="basic", num_processors=1,
                          create_step_work_dirs=False)
    engine.run_tradestudy("mc_pce_analysis", solution_name="basic", num_processors=1,
                          create_step_work_dirs=False)
    engine.run_tradestudy("quad_pce_analysis", solution_name="basic", num_processors=1,
                          create_step_work_dirs=False)
    study.save_results_to_json(save_path)
