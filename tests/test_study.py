from uam4x.components import AttributeCollector
from uam4x.study import Study
from uam4x.study import Solution
from uam4x.components import DataType
from uam4x.study import Vehicles
from uam4x.study import Control

import unittest
import os, shutil
import json


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_WORK_DIR = os.path.join(THIS_DIR, "work")

class Test_Study(unittest.TestCase):
    """
    Tests for Study.

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
        study = Study(TEST_WORK_DIR)
        v1 = study.vehicles.new_vehicle("Vehicle_1")
        wing = v1.new_component("LiftingSurface", "wing")
        wing.set_attribute("span", 30)
        wing.set_attribute("root_chord", 7)

        return study, wing


    def test_init(self):
        with self.assertRaises(TypeError):
            study = Study()

        with self.assertRaises(TypeError):
            study = Study(TEST_WORK_DIR,'Extra Input')

        # with self.assertRaises(TypeError):
        #     study = Study('This should be a directory')

        study = Study(TEST_WORK_DIR)
        self.assertEqual(study._study_dir,TEST_WORK_DIR)
        # add tests for the vehicle and controls

    def test_get_value(self):
        study, wing = self.initialize()
        info = 'Vehicle_1.wing.baseline.span'

        with self.assertRaises(TypeError):
            out = study.get_value()

        with self.assertRaises(TypeError):
            out = study.get_value(info,'Extra String')

        out = study.get_value(info)
        self.assertEqual(30,out)

        out = study.get_value('Vehicle_2.baseline.wing.span')
        self.assertEqual('This value does not exist',out)

        out = study.get_value('Vehicle_2.takeoff.wing.span')
        self.assertEqual('This value does not exist',out)


    def test_save(self):

        # intialize test
        study, wing = self.initialize()
        filename = os.path.join(TEST_WORK_DIR,'test.json')

        # test inputs: (self, filename)
        with self.assertRaises(TypeError):
            out = study.save()

        with self.assertRaises(TypeError):
            out = study.save(filename,'Extra String')

        with self.assertRaises(TypeError):
            out = study.save(1.3)

        # test output
        out = study.save(filename)
        f = open(filename, "r")
        data = json.load(f)
        attributes_dict = {'span': ['FLOAT', 30], 'root_chord': ['FLOAT', 7], 'area': ['FLOAT', 0], 'aspect_ratio': ['FLOAT', 0], 'taper': ['FLOAT', 0], 'twist': ['FLOAT', 0], 'dihedral': ['FLOAT', 0], 'sweep': ['FLOAT', 0], 'volume': ['FLOAT', 0], 'thickness': ['FLOAT', 0], 'nrotor': ['FLOAT', 0], 'type_rotor': ['INT', 0], 'loc_x': ['FLOAT', 0], 'loc_y': ['FLOAT', 0], 'loc_z': ['FLOAT', 0]}
        self.assertEqual(data['vehicles']['Vehicle_1']['components']['wing']['mode_name'],'baseline')
        self.assertEqual(data['vehicles']['Vehicle_1']['components']['wing']['type'],'LiftingSurface')
        self.assertEqual(data['vehicles']['Vehicle_1']['components']['wing']['modes'],{})
        self.assertEqual(data['vehicles']['Vehicle_1']['components']['wing']['attributes_collector'],attributes_dict)
        f.close()

        wing.set_attribute("root_chord", 1)
        out = study.save(filename)
        f = open(filename, "r")
        data = json.load(f)
        attributes_dict = {'span': ['FLOAT', 30], 'root_chord': ['FLOAT', 1], 'area': ['FLOAT', 0], 'aspect_ratio': ['FLOAT', 0], 'taper': ['FLOAT', 0], 'twist': ['FLOAT', 0], 'dihedral': ['FLOAT', 0], 'sweep': ['FLOAT', 0], 'volume': ['FLOAT', 0], 'thickness': ['FLOAT', 0], 'nrotor': ['FLOAT', 0], 'type_rotor': ['INT', 0], 'loc_x': ['FLOAT', 0], 'loc_y': ['FLOAT', 0], 'loc_z': ['FLOAT', 0]}
        self.assertEqual(data['vehicles']['Vehicle_1']['components']['wing']['mode_name'],'baseline')
        self.assertEqual(data['vehicles']['Vehicle_1']['components']['wing']['type'],'LiftingSurface')
        self.assertEqual(data['vehicles']['Vehicle_1']['components']['wing']['modes'],{})
        self.assertEqual(data['vehicles']['Vehicle_1']['components']['wing']['attributes_collector'],attributes_dict)
        f.close()


    def test_load(self):
        # intialize test
        study, wing = self.initialize()
        filename = os.path.join(TEST_WORK_DIR,'test.json')
        study.save(filename)

        # check inputs: (self, filename)
        with self.assertRaises(TypeError):
            out = study.save()

        with self.assertRaises(TypeError):
            out = study.save(filename,'Extra String')

        with self.assertRaises(TypeError):
            out = study.save(1.3)

        # test output
        out = study.load(filename)
        self.assertEqual(study._control._trade_studies,{})
        self.assertEqual(study._solution.steps,{})
        self.assertEqual(study._solution.links,[])
        self.assertEqual(study.get_value("Vehicles.Vehicle_1.wing.baseline.root_chord"),7)
        self.assertEqual(study.get_value("Vehicles.Vehicle_1.wing.baseline.span"),30)



if __name__ == '__main__':
    unittest.main()
