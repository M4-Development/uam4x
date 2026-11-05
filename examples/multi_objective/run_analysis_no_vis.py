import os

from uam4x.study import Study
from uam4x.execution_engine import Engine

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


if __name__ == "__main__":

    study_dir=os.path.join(THIS_DIR, "variants", "demo")
    study = Study.load_study_from_directory(study_dir)
    engine = Engine(study, os.path.join(study_dir,"work"))
    engine.run_tradestudy("sweep", solution_name="no_vis",num_processors=10)
    engine.run_solution("no_vis")
    engine.run_tradestudy("optimization_only", solution_name="no_vis")

    study.save_results_to_json(study_dir)

