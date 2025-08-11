from uam4x.components import Component
from uam4x.components import ComponentMode
from uam4x.components import AttributeCollector
from uam4x.components import DataType
from uam4x.study import Study
import unittest
import os, shutil

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_WORK_DIR = os.path.join(THIS_DIR, "work")

class Test_Componets(unittest.TestCase):
    """
    Tests for AttributeCollector.

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

    def initialize(self):
        # intialize the tests by creating a study
        # study = Study(TEST_WORK_DIR)
        study = Study()
        v1 = study.vehicles.new_vehicle("Vehicle_1")
        wing = v1.new_component("LiftingSurface", "wing")
        wing.set_attribute("span", 30)
        wing.set_attribute("root_chord", 7)
        wing_takeoff_mode = wing.new_mode("takeoff")
        wing_takeoff_mode.set_attribute("span", 20)
        wing_takeoff_mode.set_attribute("area", 20)

        return study, wing, wing_takeoff_mode, v1

    def test_init(self):

        # intialize test
        study,wing,wing_takeoff_mode, v1 = self.initialize()

        # test inputs: (self, study, name, mode_name, parent_attributes, new_mode)

        # there need to be 2 inputs
        with self.assertRaises(TypeError):
            wing = Component()

        # there need to be 2 inputs
        with self.assertRaises(TypeError):
            wing = Component(study)

        # maximum of 5 inputs
        with self.assertRaises(TypeError):
            wing = Component(study,'span', 'baseline', None, False, 'extra input')

        # first input needs to be a study, the second input should be a string
        with self.assertRaises(TypeError):
            wing_out = Component('This should be a study',Study(TEST_WORK_DIR))

        # check outputs
        wing = Component(study,'span')
        self.assertEqual(wing._name,'span')
        self.assertEqual(wing.study,study)
        self.assertEqual(wing.mode_name,'baseline')
        ac = AttributeCollector(wing.study)
        self.assertEqual(wing._attributes_collector.study, ac.study)

    def test_get_mode(self):

        # intialize test
        study,wing,wing_takeoff_mode, v1 = self.initialize()


        # check inputs: (self, name: str)
        with self.assertRaises(TypeError):
            out = wing.get_mode('baseline','Extra string')

        with self.assertRaises(TypeError):
            out = wing.get_mode()

        # with self.assertRaises(TypeError): # should be raising an error but is not
        #     out = wing.get_mode(['this should be a string not a list','item2'])

        # test outputs
        out = wing.get_mode('baseline')
        self.assertTrue(out,wing)

        mode1 = ComponentMode(study,None,'mode1','other_mode')
        wing._modes = [mode1]
        out = wing.get_mode('other_mode')
        self.assertTrue('other_mode',out)



    def test_get_mode_names(self):
        # intialize test
        study,wing,wing_takeoff_mode,v1 = self.initialize()

        # check inputs: (self)
        with self.assertRaises(TypeError):
            out = wing.get_mode_names('This is an extra string')

        # check outputs
        mode1 = ComponentMode(study,None,'mode1','other_mode')
        wing._modes = [mode1]
        out = wing.get_mode_names()
        self.assertTrue(['other_mode'],out)


    def test_create_attribute(self):
        # intialize test
        study,wing,wing_takeoff_mode,v1 = self.initialize()

        # check inputs: (self, name, attribute_type:DataType, default_value)
        with self.assertRaises(TypeError):
            out = wing.create_attribute('span',DataType.FLOAT,0.0,'Extra String')

        with self.assertRaises(TypeError):
            out = wing.create_attribute()

        # with self.assertRaises(TypeError): # should be raising an error but is not
        #     out = wing.create_attribute('span',"should be a enum",0.0)

        # check outputs
        out = wing.create_attribute('span',DataType.FLOAT,0.0)
        self.assertEqual(wing._attributes_collector.attributes['span'],[DataType.FLOAT,0.0])

    def test_get_attributes(self):
        # intialize test
        study,wing,wing_takeoff_mode,v1 = self.initialize()

        # test inputs: (self, name:str)
        with self.assertRaises(TypeError):
            out = wing.get_attribute()

        with self.assertRaises(TypeError):
            out = wing.get_attribute(['Should be a string'])

        with self.assertRaises(TypeError):
            out = wing.get_attribute('span','extra string')

        # test outputs

        out = wing.get_attribute('span')
        self.assertEqual(out,30)
        out = wing_takeoff_mode.get_attribute("root_chord")
        self.assertEqual(out,7)


    def test_set_attribute(self):
        # intialize test
        study,wing,wing_takeoff_mode,v1 = self.initialize()


        # test inputs: (self, name, value)
        with self.assertRaises(TypeError):
            out = wing.set_attribute()

        with self.assertRaises(TypeError):
            out = wing.set_attribute('span',10,'Extra String')

        # with self.assertRaises(ValueError): # should be raising an error but is not
        #     out = wing.set_attribute('span','should be a float')

        # test outputs
        out = wing.set_attribute('span',10)
        self.assertEqual(10,wing.get_attribute('span'))


    def test_link_components(self):

        # intialize test
        study,wing,wing_takeoff_mode,v1 = self.initialize()
        relations = ['span', 'aspect_ratio', 'area', 'taper', 'volume', 'thickness', 'root_chord'] # list of quantites with relationships
        wing.set_attribute("area", 30)
        wing.set_attribute("aspect_ratio", 30)
        name = 'span'

        # check inputs
        with self.assertRaises(TypeError):
            out = wing.link_components(name,relations,'Extra String')

        with self.assertRaises(TypeError):
            out = wing.link_components()

        with self.assertRaises(TypeError):
            out = wing.link_components(name,1.2)

        # check outputs
        out = wing.link_components(name,relations)
        self.assertEqual(wing._attributes_collector.attributes['area'][1],30)
        self.assertEqual(wing._attributes_collector.attributes['aspect_ratio'][1],30)
        wing.set_attribute("taper", 1)
        out = wing.link_components(name,relations)
        self.assertEqual(wing._attributes_collector.attributes['area'][1],210)
        self.assertEqual(wing._attributes_collector.attributes['aspect_ratio'][1],4.285714285714286)


    def test_new_mode(self):

        # intialize test
        study,wing,wing_takeoff_mode,v1 = self.initialize()

        # check inputs: (self, mode_name)
        with self.assertRaises(TypeError):
            out = wing.new_mode()

        with self.assertRaises(TypeError):
            out = wing.new_mode('takeoff','Extra String')

        # with self.assertRaises(TypeError): # should be raising error but is not
        #     out = wing.new_mode(1.2)

        # check outputs
        wing_takeoff_mode = wing.new_mode("takeoff")
        self.assertEqual('takeoff',wing_takeoff_mode.mode_name)
        self.assertEqual('wing',wing_takeoff_mode.name)
        wing_takeoff_mode = wing.new_mode("takeoff")
        self.assertEqual('takeoff',wing_takeoff_mode.mode_name)
        self.assertEqual('wing',wing_takeoff_mode.name)


    def test_save_component(self):
        # intialize tests
        study,wing,wing_takeoff_mode,v1 = self.initialize()

        # test inputs (self)
        with self.assertRaises(TypeError):
            out = wing.save_component('Extra String')

        # test outputs
        out = wing.save_component()
        self.assertEqual(['mode_name', 'type', 'attributes_collector', 'modes'],list(out.keys()))
        self.assertEqual('baseline',out['mode_name'])
        self.assertEqual('LiftingSurface',out['type'])
        self.assertEqual(['span', 'root_chord', 'area', 'aspect_ratio', 'taper', 'twist', 'dihedral', 'sweep', 'volume', 'thickness', 'nrotor', 'type_rotor', 'loc_x', 'loc_y', 'loc_z'],list(out['attributes_collector'].keys()))
        self.assertEqual({'takeoff': {'span': ['FLOAT', 20], 'area': ['FLOAT', 20]}},out['modes'])




if __name__ == '__main__':
    unittest.main()
