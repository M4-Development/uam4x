import os
from uam4x.study import Study
from uam4x.execution_engine import Engine


THIS_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__=="__main__":
    study_dir=os.path.join(THIS_DIR, "variants", "sellar")
    study = Study.load_study_from_directory(study_dir)
    engine = Engine(study, os.path.join(THIS_DIR, "variants", "sellar","work"))
    engine.run_tradestudy('sellar_problem', solution_name="sellar_basic")
    study.save_to_json(study_dir)

