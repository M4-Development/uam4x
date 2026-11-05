import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uam4x.study import Study, Variants
from uam4x.execution_engine import Engine
from branching_script import setup_model,setup_solution,setup_tradestudy
import os
import math
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
VSP_FILE_PATH = os.path.join(THIS_DIR, "NASA_Altered_Multi-Tiltrotor.vsp3")

def test_multi_objective():
    #run branching:
    # define variants
    demo = [setup_model, setup_solution, setup_tradestudy]

    # setup framework
    framework = Variants(THIS_DIR)
    framework.add_variant('demo', demo)
    study = framework.generate_variants()[0]
    save_path = os.path.join(THIS_DIR, "variants","demo")


    study = Study.load_study_from_directory(save_path)
    engine = Engine(study, os.path.join(save_path,"work"))
    engine.run_tradestudy("sweep", solution_name="no_vis")


    solution_results = study.results.get_solution_result("no_vis")
    trade_results = solution_results.get_tradestudy('sweep')
    calc_area_results = trade_results.get_variable("calc_area.area")
    cd_results = trade_results.get_variable("calc_objective.CD")
    wing_weight_results = trade_results.get_variable("calc_objective.wing_weight")
    objective_results = trade_results.get_variable("calc_objective.objective")
    aspect_ratio_results = trade_results.get_variable("vehicles.multi_objective_vehicle.wing1.aspect_ratio")
    span_results = trade_results.get_variable("vehicles.multi_objective_vehicle.wing1.span")

    #Note first three points (-10,-20,-30 fail, so they do not contain results)
    weighting_factor_sweep_values = [-10, -20, -30, 0, 5000, 7500, 10000, 15000, 20000,
                                     25000, 30000, 40000, 50000, 100000, 150000, 200000,
                                     250000, 300000, 350000, 400000, 450000, 500000]

    calc_areas = [49998.999999999774, 50001.00000027564, 50001.00000016379, 50001.00000003263,
                       50001.00000000095, 50001.00000023285, 50001.000000706605, 50001.000000428205,
                       50001.00000030467, 50001.000000283864, 50001.00000000009, 50001.000000067535,
                       50001.00000011257, 50001.0000009542, 50001.00000063622, 50001.00000029626,
                       50001.000000427826, 50001.00000093053, 50001.00000002121]
    CD_list = [0.04276707143266288, 0.04048597525316287, 0.036809884609613694, 0.034613030130147784,
               0.03200019194954755, 0.0304380753994824, 0.029369540775481727, 0.028579243630762745,
               0.027467309454638857, 0.026706490163191224, 0.024808833148918207, 0.023962086343584824,
               0.023454826166661093, 0.023107307853671284, 0.02284995081638699, 0.022649375319604988,
               0.02248729523406657, 0.02235274958325178, 0.02227652501397989]

    wing_weight_list = [88.92178070843681, 99.1555424011207, 121.66902759460294,
                        140.69532595585028, 172.69946878547762, 199.7576893582335,
                        223.65166780768496, 245.29538789554178, 283.81684240474726,
                        317.84342075751897, 452.0928429399247, 555.817172824079,
                        643.686188033256, 721.3986581478139, 791.8819545256465,
                        856.8780050549608, 917.5244424011403, 974.6085615813698, 1010.1284630542882]

    objective_list = [88.92178070843681, 301.5854186669351, 397.74316216670564,
                      486.82562725732816, 652.7023480286908, 808.5191973478815,
                      957.8901871947281, 1102.6726968184241, 1382.5092205903015,
                      1653.1679289170802, 2932.9761578317452, 4150.130124361803,
                      5334.651421365475, 6498.225621565634, 7646.867199441744,
                      8784.159366916707, 9912.442536027767, 11033.34587404467, 12148.390970044233]

    aspect_ratio_list = [1.0, 1.1112602577236215, 1.354277597279386, 1.5578733457070177,
                         1.8970738331113086, 2.1809815764233074, 2.429708209257099,
                         2.653526478462604, 3.0486549777819643, 3.3945103303796356,
                         4.734048663119386, 5.745773354131758, 6.589405382934045,
                         7.326358124456346, 7.98794491761851, 8.592685970564924,
                         9.152612776968658, 9.676019199412167, 9.999999999992227]

    span_list = [223.60456167082052, 235.7204364215057, 260.2215097600281,
                 279.09716078589446, 307.98634503724406, 330.2291019932217,
                 348.551058200633, 364.25125593571386, 390.43027232145414,
                 411.98168773657375, 486.52560796389, 535.9985200355782,
                 574.000747867829, 605.2480752451283, 631.9851531728615,
                 655.4715029783998, 676.4907918553848, 695.5649761156836, 707.1138522188795]

    area_tol = 1e-3
    cd_tol = 1e-3
    weight_tol = 1e-3
    span_tol = 1e-3
    ar_tol = 1e-3

    for i, wf in enumerate(weighting_factor_sweep_values):
        # First three cases fail → no outputs
        if i < 3:
            assert np.isnan(calc_area_results[i])
            assert np.isnan(objective_results[i])
            continue

        j = i - 3  # offset into reference lists

        assert math.isclose(
            calc_area_results[i],
            calc_areas[j],
            abs_tol=area_tol
        )

        assert math.isclose(
            cd_results[i],
            CD_list[j],
            abs_tol=cd_tol
        )

        assert math.isclose(
            wing_weight_results[i],
            wing_weight_list[j],
            abs_tol=weight_tol
        )

        assert math.isclose(
            objective_results[i],
            objective_list[j],
            abs_tol=weight_tol
        )

        assert math.isclose(
            aspect_ratio_results[i],
            aspect_ratio_list[j],
            abs_tol=ar_tol
        )

        assert math.isclose(
            span_results[i],
            span_list[j],
            abs_tol=span_tol
        )

if __name__=="__main__":
    test_multi_objective()
