from uam4x.openvsp_plugin import OpenvspPlugin
from uam4x.study import Study

import unittest
import os, shutil


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_WORK_DIR = os.path.join(THIS_DIR, "work")

class Test_OpenvspPlugin(unittest.TestCase):
    """
    Tests for openvsp plugin.

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
        # input needs to be a study
        with self.assertRaises(TypeError):
            eng_out = OpenvspPlugin("this should be a study not a string")

        # there should only be one input
        with self.assertRaises(TypeError):
            eng_out = OpenvspPlugin(test_study,"There should only be one input")

        # there should be one input
        with self.assertRaises(TypeError):
            eng_out = OpenvspPlugin()

    def test_setup_plugin(self):
        test_study  = Study(TEST_WORK_DIR)
        vsp_test = OpenvspPlugin(test_study)
        plugin_options = {'filename':'test_info',
                          'new_filename': 'test_info'}
        tracked_outputs = ["test list"]

        # test inputs
        # inputs: (self, plugin_options:dict=None, tracked_outputs:list=None)

        # there should be more than
        with self.assertRaises(TypeError):
            setup_out = vsp_test.setup_plugin(plugin_options,tracked_outputs,"extra input")

        # first input should be dict, second input should be list
        with self.assertRaises(AttributeError):
            setup_out = vsp_test.setup_plugin(tracked_outputs,plugin_options)

        # should not raise an error
        setup_out = vsp_test.setup_plugin(plugin_options,tracked_outputs)
        setup_out = vsp_test.setup_plugin(plugin_options)
        setup_out = vsp_test.setup_plugin()

        # check outputs

        # check plugin_options output
        check_filename = 'test_info'
        check_new_filename = "_".join("test_info") + "_"
        self.assertEqual(vsp_test._filename,check_filename)
        self.assertEqual(vsp_test._new_filename,check_new_filename)

        # check tracked_outputs outplut
        self.assertEqual(vsp_test._tracked_outputs, tracked_outputs)

    def test_run_plugin(self):

        # initalize parameters
        test_study  = Study(TEST_WORK_DIR)
        vsp_test = OpenvspPlugin(test_study)
        links = [['wing.WING.WingGeom.TotalSpan', 30], ['wing.WING.WingGeom.TotalChord', 7], ['fuselage.FUSELAGE.Design.Length', 35], ['fuselage.FUSELAGE.XSec.Ellipse_Width', 3], ['fuselage.FUSELAGE.XSec.Ellipse_Height', 3], ['hstab.WING.WingGeom.TotalSpan', 10], ['hstab.WING.WingGeom.TotalChord', 2]]
        plugin_options = {'filename':'test_info',
                          'new_filename': 'test_info'}
        vsp_test._tracked_outputs =  ['Total_Wet_Area', 'Total_Wet_Vol']
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

        # check inputs
        # inputs: (self, links:list)

        with self.assertRaises(TypeError):
            run_out = vsp_test.run_plugin()

        with self.assertRaises(TypeError):
            run_out = vsp_test.run_plugin(links,'extra input')

        with self.assertRaises(TypeError):
            run_out = vsp_test.run_plugin(1)

        # should not raise any errors
        run_out = vsp_test.run_plugin(links)

        # test results
        check_results = {'.CompGeom.Total_Wet_Area': 653.6958359311172, '.CompGeom.Total_Wet_Vol': 295.3398259787079}
        self.assertEqual(check_results,run_out)



if __name__ == '__main__':
    unittest.main()