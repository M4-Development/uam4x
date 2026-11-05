
import os
from uam4x.study import Study
from uam4x.execution_engine import Engine
from conf import (TURBO_ELECTRIC,TURBO_ELECTRIC_AT_SEA,ELECTRIC,ELECTRIC_AT_SEA)
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
VARIANTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "variants")
ACTIVE_VARIANTS = [TURBO_ELECTRIC, ELECTRIC, TURBO_ELECTRIC_AT_SEA, ELECTRIC_AT_SEA]


if __name__ == "__main__":

    # vsp.StartGUI()
    for active_variant in ACTIVE_VARIANTS:
        work_dir = os.path.join(VARIANTS_DIR, active_variant, "work")
        variant_path = os.path.join(VARIANTS_DIR, active_variant)
        study = Study.load_study_from_directory(variant_path)
        engine = Engine(study, work_dir)
        engine.run_tradestudy(trade_name="vtip_trade",solution_name="basic",num_processors=6)

        # Save the updated study back to JSON
        study.save_results_to_json(variant_path)