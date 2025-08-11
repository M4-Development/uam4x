
import os
from nasa_uam.study import Study
from nasa_uam.git_controller import GitController
from nasa_uam.execution_engine import Engine
class Framework():
    def __init__(self, directory) -> None:
        self.dir = directory
        self.variants = {}

        self.has_git = os.path.isdir(os.path.join(self.dir, '.git'))
        if self.has_git:
            self.git_controller = GitController(self.dir)


    def add_variant(self, name, func_list:list):
        self.variants[name] = func_list

    def generate_variants(self, commit = False):
        if commit and not self.has_git:
            raise RuntimeError("Cannot commit, cannot find git repository")
        for name, func_list in self.variants.items():
            variants_dir = os.path.join(self.dir, "variants")
            if not os.path.exists(variants_dir):
                os.mkdir(variants_dir)

            this_variant_dir = os.path.join(variants_dir, name)
            if not os.path.exists(this_variant_dir):
                os.mkdir(this_variant_dir)

            study = Study()
            for func in func_list:
                func(study)
            json_path = os.path.join(this_variant_dir, 'data.json')
            study.save_as(json_path)
            if commit:
                self.git_controller.commit_data_json(json_path, f"AutoCommit: Created data for variant {name}")


    def run_tradestudy(self, trade_study_name, variant_names, commit=True):
        variant_dir = os.path.join(self.dir, "variants")
        for variant in variant_names:
            if variant in os.listdir(variant_dir):
                this_variant_dir = os.path.join(variant_dir, variant)
                study = Study()
                json_path = os.path.join(this_variant_dir, "data.json")
                if os.path.exists(json_path):
                    study.load(json_path)
                else:
                    raise RuntimeError(f"Cannot find variant {variant}")
                engine = Engine(study)
                engine.run_tradestudy(trade_study_name, commit=True)
            else:
                raise RuntimeError(f"Cannot find variant {variant}")
            if commit:
                self.git_controller.commit_data_json(json_path, f"AutoCommit: Writing results for variant {variant}")
    def run_singlepoint(self, variant_names, commit=True):
        variant_dir = os.path.join(self.dir, "variants")
        for variant in variant_names:
            if variant in os.listdir(variant_dir):
                variant_dir = os.path.join(variant_dir, variant)
                study = Study()
                json_path = os.path.join(variant_dir, "data.json")
                if os.path.exists(json_path):
                    study.load(json_path)
                else:
                    raise RuntimeError(f"Cannot find variant {variant}")
                engine = Engine(study)
                engine.run_solution({'optimization': False},None)
            else:
                raise RuntimeError(f"Cannot find variant {variant}")
            if commit:
                self.git_controller.commit_data_json(json_path, f"AutoCommit: Writing results for variant {variant}")