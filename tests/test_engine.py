from uam4x.execution_engine import Engine
from uam4x.study import Study
import unittest
import os, shutil
import json

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_WORK_DIR = os.path.join(THIS_DIR, "work")

class Test_Engine(unittest.TestCase):
    """
    Tests for Engine.

    """
    @classmethod
    def setUpClass(cls):
        """
        This function is called once before testing starts
        """

    def setUp(self):
        """
        This function is called immediatly before calling every test
        """
        pass
    def tearDown(self):
        """
        This function is called immediatly after a test method has been called and the result recorded
        """

        # Remove any file that may have been created in the work dir after the test is finished
        for filename in os.listdir(TEST_WORK_DIR):
            file_path = os.path.join(TEST_WORK_DIR, filename)
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
            else:
                os.remove(file_path)

    def initialize_engine(self):
        test_study = Study(TEST_WORK_DIR)
        v1 = test_study.vehicles.new_vehicle("Vehicle_1")
        wing = v1.new_component("LiftingSurface", "wing")
        wing.set_attribute("span", 30)
        wing.set_attribute("root_chord", 7)
        fuse = v1.new_component("Fuselage", "fuselage")
        fuse.set_attribute("length", 35)
        fuse.set_attribute("width", 3)
        fuse.set_attribute("height", 3)
        hstab = v1.new_component("LiftingSurface", "hstab")
        hstab.set_attribute("span", 10)
        hstab.set_attribute("root_chord", 2)
        sol = test_study.solution
        ind = "Vehicles.Vehicle_1.wing.baseline.span"
        dep = "step1.wing.WING.WingGeom.TotalSpan"
        sol.add_link(ind, dep)
        ind = "Vehicles.Vehicle_1.wing.baseline.root_chord"
        dep = "step1.wing.WING.WingGeom.TotalChord"
        sol.add_link(ind, dep)
        ind = "Vehicles.Vehicle_1.fuselage.baseline.length"
        dep = "step1.fuselage.FUSELAGE.Design.Length"
        sol.add_link(ind, dep)
        ind = "Vehicles.Vehicle_1.fuselage.baseline.width"
        dep = "step1.fuselage.FUSELAGE.XSec.Ellipse_Width"
        sol.add_link(ind, dep)
        ind = "Vehicles.Vehicle_1.fuselage.baseline.height"
        dep = "step1.fuselage.FUSELAGE.XSec.Ellipse_Height"
        sol.add_link(ind, dep)
        ind = "Vehicles.Vehicle_1.hstab.baseline.span"
        dep = "step1.hstab.WING.WingGeom.TotalSpan"
        sol.add_link(ind, dep)
        ind = "Vehicles.Vehicle_1.hstab.baseline.root_chord"
        dep = "step1.hstab.WING.WingGeom.TotalChord"
        sol.add_link(ind, dep)

        sol.add_step(plugin_name="OpenvspPlugin", step_name="step1", plugin_options={"analysis": "CompGeom"}, tracked_outputs=["Total_Wet_Area", "Total_Wet_Vol"])
        test_study.save("test/data.json")

        return test_study, sol

    def test_init(self):

        # check inputs
        test_study, sol  = self.initialize_engine()

        # should raise type error
        # input needs to be a study not a string
        with self.assertRaises(AttributeError):
            eng = Engine("This should be a Study, not a String")

        # there should only be one input
        with self.assertRaises(TypeError):
            eng = Engine()

        # there should only be one input
        with self.assertRaises(TypeError):
            eng = Engine(test_study, "one arg too many")

        # should not raise any error
        eng = Engine(test_study)
        eng.run()

        # check outputs
        self.assertEqual(eng.study_results,{'standard_analysis': {'step1.CompGeom.Total_Wet_Area': 653.6958359311172, 'step1.CompGeom.Total_Wet_Vol': 295.3398259787079}})



    def test_export_results(self):
       # intialize tests
        test_study, sol = self.initialize_engine()
        eng = Engine(test_study)
        filename = 'test.json'
        filepath = os.path.join(TEST_WORK_DIR,filename)
        eng.run()

        # should raise type error

        # should have only one input that is a string
        with self.assertRaises(TypeError):
            eng.export_results()

        with self.assertRaises(TypeError):
            eng.export_results(filename,'extra input')

        # with self.assertRaises(TypeError):
        #     eng.export_results(float(1.27)) ## Ask Thomas (permission error)

        # should not raise any error
        eng.export_results(filename)

        f = open(filepath, "r")
        data = json.load(f)
        f.close()

        self.assertEqual(eng.study_results,data)

        print('.')

    def test_run(self):

        # initalize test
        study, sol = self.initialize_engine()
        eng = Engine(study)

        # should raise error
        # needs zero or one input that is a sting
        with self.assertRaises(TypeError):
            eng.run('standard_analysis','extra string')

        with self.assertRaises(AttributeError): # ask thomas if okay
            eng.run(1.2)


        # check and run other active study that does not exist
        eng.run('false_analysis')
        self.assertEqual(eng.study_results,{})

        # check and run standard analysis
        eng.run('standard_analysis')
        self.assertEqual(eng.study_results,{'standard_analysis': {'step1.CompGeom.Total_Wet_Area': 653.6958359311172, 'step1.CompGeom.Total_Wet_Vol': 295.3398259787079}})

        eng.run()
        self.assertEqual(eng.study_results,{'standard_analysis': {'step1.CompGeom.Total_Wet_Area': 653.6958359311172, 'step1.CompGeom.Total_Wet_Vol': 295.3398259787079}})

        # check and run trade study
        trade_study = self.initialize_engine()
        trade_study.control.create_trade_study("trade_study1", ["Vehicles.Vehicle_1.wing.baseline.span", "Vehicles.Vehicle_1.wing.baseline.root_chord"], [[21, 24, 27, 30, 33], [5, 6, 7]])
        eng = Engine(trade_study)
        eng.run("trade_study1")

        # ask joe

        check = {'Vehicles.Vehicle_1.wing.baseline.span': [21, 24, 27],
                 'Vehicles.Vehicle_1.wing.baseline.root_chord': [5, 6, 7],
                 'step1.CompGeom.Total_Wet_Area': [465.2329275138641, 532.6799906495105, 611.3023384175077],
                 'step1.CompGeom.Total_Wet_Vol': [234.6127946317888, 255.46944857292036, 284.4033286306588]
                 }
        eng.study_results['trade_study1']['Vehicles.Vehicle_1.wing.baseline.span'].sort()
        eng.study_results['trade_study1']['Vehicles.Vehicle_1.wing.baseline.root_chord'].sort()
        eng.study_results['trade_study1']['step1.CompGeom.Total_Wet_Area'].sort()
        eng.study_results['trade_study1']['step1.CompGeom.Total_Wet_Vol'].sort()
        self.assertEqual(check['Vehicles.Vehicle_1.wing.baseline.span'],eng.study_results['trade_study1']['Vehicles.Vehicle_1.wing.baseline.span'])
        self.assertEqual(check['Vehicles.Vehicle_1.wing.baseline.root_chord'],eng.study_results['trade_study1']['Vehicles.Vehicle_1.wing.baseline.root_chord'])
        self.assertEqual(check['step1.CompGeom.Total_Wet_Area'],eng.study_results['trade_study1']['step1.CompGeom.Total_Wet_Area'])
        self.assertEqual(check['step1.CompGeom.Total_Wet_Vol'],eng.study_results['trade_study1']['step1.CompGeom.Total_Wet_Vol'])





    def test_standard_analysis(self): # returns values
        # initalize test
        test_out, sol = self.initialize_engine()
        eng = Engine(test_out)
        step_info = {"plugin_options":{'analysis': 'CompGeom'},
                     "plugin_type": 'OpenvspPlugin',
                     "tracked_outputs":['Total_Wet_Area', 'Total_Wet_Vol']
                     }
        step_name = 'step1'
        out_dict={}

        # should raise error
        # should have inputs (self, step_name:str, step_info:dict, outputs:dict)
        with self.assertRaises(TypeError):
            stand_out = eng.standard_analysis()

        with self.assertRaises(TypeError):
            stand_out = eng.standard_analysis(step_name,step_info,out_dict,'Extra Input')

        # with self.assertRaises(TypeError):
        #     stand_out = eng.standard_analysis(step_name,step_info,1.2)

        stand_out = eng.standard_analysis('step1', step_info, out_dict) # Permission error?
        self.assertEqual(stand_out,{'step1.CompGeom.Total_Wet_Area': 653.6958359311172, 'step1.CompGeom.Total_Wet_Vol': 295.3398259787079})



    def test_trade_study(self): # returns values
        study, sol  = self.initialize_engine()
        study.control.create_trade_study("trade_study1", ["Vehicles.Vehicle_1.wing.baseline.span", "Vehicles.Vehicle_1.wing.baseline.root_chord"], [[21, 24, 27, 30, 33], [5, 6, 7]])
        eng = Engine(study)
        trade_study = {'parameter_sweep': {'Vehicles.Vehicle_1.wing.baseline.span': [21, 24, 27, 30, 33], 'Vehicles.Vehicle_1.wing.baseline.root_chord': [5, 6, 7]}}
        step_name = 'step1'
        step_info = {'plugin_type': 'OpenvspPlugin',
                     'plugin_options': {'analysis': 'CompGeom'},
                     'tracked_outputs': ['Total_Wet_Area', 'Total_Wet_Vol']}
        outputs = {}
        step_num = 1

        # should raise errors
        # there should be the following inputs: (self, trade_study:dict, step_name:str, step_info:dict, outputs:dict, step_num:int)
        with self.assertRaises(TypeError):
            out = eng.trade_study()

        with self.assertRaises(TypeError):
            out = eng.trade_study(trade_study,step_name)

        with self.assertRaises(TypeError):
            out = eng.trade_study(trade_study, step_name, step_info, outputs, step_num, 'Extra String')

        with self.assertRaises(AttributeError):
            out = eng.trade_study('String but should be dict',step_name, step_info, outputs, step_num)

        # this should not raise errors
        out = eng.trade_study(trade_study, step_name, step_info, outputs, step_num)
        check = {'Vehicles.Vehicle_1.wing.baseline.span': [21, 24, 27],
                 'Vehicles.Vehicle_1.wing.baseline.root_chord': [5, 6, 7],
                 'step1.CompGeom.Total_Wet_Area': [465.2329275138641, 532.6799906495105, 611.3023384175077],
                 'step1.CompGeom.Total_Wet_Vol': [234.6127946317888, 255.46944857292036, 284.4033286306588]
                 }


        out['Vehicles.Vehicle_1.wing.baseline.span'].sort()
        out['Vehicles.Vehicle_1.wing.baseline.root_chord'].sort()
        out['step1.CompGeom.Total_Wet_Area'].sort()
        out['step1.CompGeom.Total_Wet_Vol'].sort()
        self.assertEqual(check['Vehicles.Vehicle_1.wing.baseline.span'],out['Vehicles.Vehicle_1.wing.baseline.span'])
        self.assertEqual(check['Vehicles.Vehicle_1.wing.baseline.root_chord'],out['Vehicles.Vehicle_1.wing.baseline.root_chord'])
        self.assertEqual(check['step1.CompGeom.Total_Wet_Area'],out['step1.CompGeom.Total_Wet_Area'])
        self.assertEqual(check['step1.CompGeom.Total_Wet_Vol'],out['step1.CompGeom.Total_Wet_Vol'])



    def test_parm_sweep(self): # return
        study, sol  = self.initialize_engine()
        study.control.create_trade_study("trade_study1", ["Vehicles.Vehicle_1.wing.baseline.span", "Vehicles.Vehicle_1.wing.baseline.root_chord"], [[21, 24, 27, 30, 33], [5, 6, 7]])
        eng = Engine(study)
        trade_study = {'parameter_sweep': {'Vehicles.Vehicle_1.wing.baseline.span': [21, 24, 27, 30, 33], 'Vehicles.Vehicle_1.wing.baseline.root_chord': [5, 6, 7]}}
        value = trade_study['parameter_sweep']
        step_name = 'step1'
        step_info = {'plugin_type': 'OpenvspPlugin',
                     'plugin_options': {'analysis': 'CompGeom'},
                     'tracked_outputs': ['Total_Wet_Area', 'Total_Wet_Vol']}
        outputs = {}
        step_num = 1
        original_values = {}

        with self.assertRaises(TypeError):
            out = eng.parm_sweep()

        with self.assertRaises(TypeError):
            out = eng.parm_sweep(value, step_name, step_info, outputs, step_num, original_values,'Extra String')

        with self.assertRaises(AttributeError):
            out = eng.parm_sweep('This should be a dictionary', step_name, step_info, outputs, step_num, original_values)

        out = eng.parm_sweep(value, step_name, step_info, outputs, step_num, original_values)
        check = {'Vehicles.Vehicle_1.wing.baseline.span': [21, 24, 27],
                 'Vehicles.Vehicle_1.wing.baseline.root_chord': [5, 6, 7],
                 'step1.CompGeom.Total_Wet_Area': [465.2329275138641, 532.6799906495105, 611.3023384175077],
                 'step1.CompGeom.Total_Wet_Vol': [234.6127946317888, 255.46944857292036, 284.4033286306588]
                 }

        print('.')
        out['Vehicles.Vehicle_1.wing.baseline.span'].sort()
        out['Vehicles.Vehicle_1.wing.baseline.root_chord'].sort()
        out['step1.CompGeom.Total_Wet_Area'].sort()
        out['step1.CompGeom.Total_Wet_Vol'].sort()
        self.assertEqual(check['Vehicles.Vehicle_1.wing.baseline.span'],out['Vehicles.Vehicle_1.wing.baseline.span'])
        self.assertEqual(check['Vehicles.Vehicle_1.wing.baseline.root_chord'],out['Vehicles.Vehicle_1.wing.baseline.root_chord'])
        self.assertEqual(check['step1.CompGeom.Total_Wet_Area'],out['step1.CompGeom.Total_Wet_Area'])
        self.assertEqual(check['step1.CompGeom.Total_Wet_Vol'],out['step1.CompGeom.Total_Wet_Vol'])



    def test_run_sweep_combo(self):
        pass

    def test_multi_parm_sweep(self): # return
        study, sol  = self.initialize_engine()
        eng = Engine(study)
        sol.add_step(plugin_name="OpenvspPlugin", step_name="step2", plugin_options={"analysis": "CompGeom"}, tracked_outputs=["Total_Wet_Area", "Total_Wet_Vol"])


        pass


    def test_sort_links(self): # return values
        study, sol  = self.initialize_engine()
        study.control.create_trade_study("trade_study1", ["Vehicles.Vehicle_1.wing.baseline.span", "Vehicles.Vehicle_1.wing.baseline.root_chord"], [[21, 24, 27, 30, 33], [5, 6, 7]])
        eng = Engine(study)

        trade_study = {'parameter_sweep': {'Vehicles.Vehicle_1.wing.baseline.span': [21, 24, 27, 30, 33], 'Vehicles.Vehicle_1.wing.baseline.root_chord': [5, 6, 7]}}
        step_name = 'step1'
        outputs = {}


        # should raise errors
        # inputs should look like the following: sort_links(self, step_name:str, study:Study, outputs:dict, sweep_num:int=None)

        with self.assertRaises(TypeError):
            out = eng.sort_links()

        with self.assertRaises(TypeError):
            out = eng.sort_links(step_name)

        with self.assertRaises(TypeError):
            out = eng.sort_links(step_name,outputs, None, 'Extra String')

        # with self.assertRaises(TypeError):
        #     sortlink_out = eng.sort_links(step_name,'Should be a dictionary', None)

        # should not cause an error
        out = eng.sort_links(step_name,outputs, None)
        check  = [['wing.WING.WingGeom.TotalSpan', 30], ['wing.WING.WingGeom.TotalChord', 7], ['fuselage.FUSELAGE.Design.Length', 35], ['fuselage.FUSELAGE.XSec.Ellipse_Width', 3], ['fuselage.FUSELAGE.XSec.Ellipse_Height', 3], ['hstab.WING.WingGeom.TotalSpan', 10], ['hstab.WING.WingGeom.TotalChord', 2]]
        self.assertEqual(out,check)

        outputs  = {'step1.CompGeom.Total_Wet_Area': 653.6958359311172, 'step1.CompGeom.Total_Wet_Vol': 295.3398259787079}

        out = eng.sort_links(step_name,outputs, None)


    def test_solution_sequence(self): # return
        study, sol  = self.initialize_engine()
        eng = Engine(study)
        step_name = 'step1'
        step_info = {'plugin_type': 'OpenvspPlugin',
                     'plugin_options': {'analysis': 'CompGeom'},
                     'tracked_outputs': ['Total_Wet_Area', 'Total_Wet_Vol']}
        sorted_links = [['wing.WING.WingGeom.TotalSpan', 30], ['wing.WING.WingGeom.TotalChord', 7], ['fuselage.FUSELAGE.Design.Length', 35], ['fuselage.FUSELAGE.XSec.Ellipse_Width', 3], ['fuselage.FUSELAGE.XSec.Ellipse_Height', 3], ['hstab.WING.WingGeom.TotalSpan', 10], ['hstab.WING.WingGeom.TotalChord', 2]]

        # should raise errors
        # solution_sequence(self, sorted_links:list, step_name:str, step_info:dict, study:Study, sweep_num:int=None)

        with self.assertRaises(TypeError):
            out = eng.solution_sequence()

        with self.assertRaises(TypeError):
            out = eng.solution_sequence(sorted_links,step_name,step_info,'Extra String')

        # with self.assertRaises(TypeError):
        #     solseq_out = eng.solution_sequence(sorted_links,step_name,step_info)

        # no error
        out = eng.solution_sequence(sorted_links,step_name,step_info)
        check = {'step1.CompGeom.Total_Wet_Area': 653.6958359311172, 'step1.CompGeom.Total_Wet_Vol': 295.3398259787079}
        self.assertEqual(check,out)

    def test_parse_results(self): # return
        study, sol  = self.initialize_engine()
        eng = Engine(study)
        results  = {'step1.CompGeom.Total_Wet_Area': 10, 'step1.CompGeom.Total_Wet_Vol': 5}
        link = ['step1.CompGeom.Total_Wet_Area', 'step1.wing.WING.WingGeom.Total_Wet_Area']
        # should raise errors
        # inputs should look like the following: (self, results:dict, link:list, sweep_num:int=None)

        with self.assertRaises(TypeError):
            out = eng.parse_results()

        with self.assertRaises(TypeError):
            out = eng.parse_results(results)

        with self.assertRaises(TypeError):
            out = eng.parse_results(results,link,None,'extra string')

        out = eng.parse_results(results,link)
        self.assertEqual(['wing.WING.WingGeom.Total_Wet_Area', 10],out)


        results  = {'step1.CompGeom.Total_Wet_Area': {1:10},
                        'step1.CompGeom.Total_Wet_Vol': {1:5}}
        out = eng.parse_results(results,link,1)
        self.assertEqual(['wing.WING.WingGeom.Total_Wet_Area', 10],out)

        results = {}
        out = eng.parse_results(results,link)
        self.assertEqual([],out)









    # def test_run_single_sweep_combo(self): # return
    #     pass

    # def test_single_var_parm_sweep(self): # return
    #     pass

    # def test_setup_sweep(self):
    #     pass

    # def test_execute_sweep(self): # return
    #     pass

    # def change_parm(self):
    #     pass





if __name__ == '__main__':
    unittest.main()