from nasa_uam.plugins.ndarc_plugin import NdarcPlugin
from nasa_uam.study import Study

import unittest
import os, shutil




THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_WORK_DIR = os.path.join(THIS_DIR, "work")

class Test_NdarcPlugin(unittest.TestCase):
    """
    Tests for ndarc_plugin.

    """
    @classmethod
    def setUpClass(cls):
        """
        This function is called once before testing starts
        """
        pass
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

    def test_init(self):

        test_study  = Study(TEST_WORK_DIR)

        # test inputs

        # input needs to be a study - not raising type error
        with self.assertRaises(TypeError):
            eng_out = NdarcPlugin("This should be a Study, not a String")

        # there should only be one input
        with self.assertRaises(TypeError):
            nc_out = NdarcPlugin(test_study,"There should only be one input")

        # there should be one input
        with self.assertRaises(TypeError):
            nc_out = NdarcPlugin()

        # should not raise any error
        nc_out = NdarcPlugin(test_study)

    def test_setup_plugin(self):
        test_study  = Study(TEST_WORK_DIR)
        nc_test = NdarcPlugin(test_study)
        plugin_options = {'filename':'test_info',
                          'new_filename': 'test_info'}
        tracked_outputs = ["test list"]

        # test inputs
        # inputs: (self, plugin_options:dict=None, tracked_outputs:list=None)

        # there should be more than 
        with self.assertRaises(TypeError):
            setup_out = nc_test.setup_plugin(plugin_options,tracked_outputs,"extra input")

        # first input should be dict, second input should be list
        with self.assertRaises(AttributeError):
            setup_out = nc_test.setup_plugin(tracked_outputs,plugin_options)

        # should not raise an error
        setup_out = nc_test.setup_plugin(plugin_options,tracked_outputs)
        setup_out = nc_test.setup_plugin(plugin_options)
        setup_out = nc_test.setup_plugin()

        # check outputs 

        # check plugin_options output
        check_filename = 'test_info'
        check_new_filename = "_".join("test_info") + "_"
        self.assertEqual(nc_test._filename,check_filename)
        self.assertEqual(nc_test._new_filename,check_new_filename)

        # check tracked_outputs outplut
        self.assertEqual(nc_test._tracked_outputs, tracked_outputs) 

    def test_run_plugin(self):

        # intialize test case
        test_study  = Study(TEST_WORK_DIR)
        nc_test = NdarcPlugin(test_study)
        links = [['ROTOR(1).vtip_ref', 450], ['ROTOR(2).vtip_ref', 450], ['ROTOR(3).vtip_ref', 450], ['ROTOR(4).vtip_ref', 450]]
        nc_test._tracked_outputs = ['Rotor(1).RADIUS', 'Rotor(2).RADIUS', 'Rotor(3).RADIUS', 'Rotor(4).RADIUS']
        nc_test._currentdir = os.getcwd()
        nc_test._filename = os.path.join(nc_test._currentdir,"examples","uam_quad", "UAM-N-quad-5-turbo-xshaft-coll-flap-550.njob")
        self._analysis_name = 'design'

        # check inputs 
        # inputs: (self, links:list)

        # needs one input tha is a list
        with self.assertRaises(TypeError):
            run_out = nc_test.run_plugin()

        with self.assertRaises(TypeError):
            run_out = nc_test.run_plugin(links,"extra input")

        with self.assertRaises(TypeError):
            run_out = nc_test.run_plugin(1)

        # should not cause an error
        run_out = nc_test.run_plugin(links)

        # test outputs
        check_results = {'.design.Rotor(1).RADIUS': 8.533422, '.design.Rotor(2).RADIUS': 8.533422, '.design.Rotor(3).RADIUS': 8.533422, '.design.Rotor(4).RADIUS': 8.533422}
        self.assertEqual(check_results,run_out)

    def test_use_rcotools(self):

        # intialize test case
        test_study  = Study(TEST_WORK_DIR)
        nc_test = NdarcPlugin(test_study)
        nc_test._currentdir = os.getcwd()
        nc_test._filename = os.path.join(nc_test._currentdir,"examples","uam_quad", "UAM-N-quad-5-turbo-xshaft-coll-flap-550.njob")
        ['Rotor(1).RADIUS', 'Rotor(2).RADIUS', 'Rotor(3).RADIUS', 'Rotor(4).RADIUS']
        self._analysis_name = 'design'
        update_dict = {'ROTOR(1)%vtip_ref': 450, 'ROTOR(2)%vtip_ref': 450, 'ROTOR(3)%vtip_ref': 450, 'ROTOR(4)%vtip_ref': 450}

        # test inputs
        # inputs: (self,update_dict,write_input = False)

        # needs at least one input
        with self.assertRaises(TypeError):
            rco_out = nc_test.use_rcotools()

        with self.assertRaises(TypeError):
            rco_out = nc_test.use_rcotools(['Should be dict not list'])

        with self.assertRaises(TypeError):
            rco_out = nc_test.use_rcotools(update_dict,False,'extra_input')

        # should not raise error
        rco_out = nc_test.use_rcotools(update_dict,False)

        # check output type
        self.assertTrue(type(rco_out),'nasa_uam.plugins.rcotools_ndarc.rcotools.ndarc.ndarc_core.NdarcSolution')

        
















if __name__ == '__main__':
    unittest.main()

