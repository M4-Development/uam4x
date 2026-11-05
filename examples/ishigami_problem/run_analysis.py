import os
from uam4x.study import Study
from uam4x.execution_engine import Engine

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__=="__main__":
    study_dir=os.path.join(THIS_DIR, "variants", "ishigami")
    study = Study.load_study_from_directory(study_dir)
    engine = Engine(study, os.path.join(study_dir,"work"))
    engine.run_tradestudy('pce_ishigami_problem', solution_name="basic",create_step_work_dirs=False)
    engine.run_tradestudy('mc_ishigami_problem', solution_name="basic",create_step_work_dirs=False)
    study.save_results_to_json(study_dir)

