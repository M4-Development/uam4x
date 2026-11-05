import os
from uam4x.study import Study
from uam4x.execution_engine import Engine
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
STACKED_COMPARISON_PATH = os.path.join(THIS_DIR, "predicted_vs_mc_stacked.csv")

def _relative_error(estimate, reference):
    if reference == 0.0:
        return np.nan
    return 100.0 * (estimate - reference) / reference

if __name__=="__main__":
    study_dir=os.path.join(THIS_DIR, "variants", "borehole")
    study = Study.load_study_from_directory(study_dir)
    engine = Engine(study, os.path.join(study_dir,"work"))
    engine.run_tradestudy('borehole_problem', solution_name="basic",create_step_work_dirs=False)
    study.save_results_to_json(study_dir)
