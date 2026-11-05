
import os
from uam4x.study import Study
from uam4x.execution_engine import Engine

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
VARIANTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "variants")


if __name__ == "__main__":

    variant_path = os.path.join(VARIANTS_DIR, "quad")
    study = Study.load_study_from_directory(variant_path)
    engine = Engine(study, os.path.join(VARIANTS_DIR, "quad", "work"))
    engine.run_tradestudy("vel_trade", solution_name="basic")

    # Save the updated study back to JSON
    study.save_results_to_json(variant_path)

