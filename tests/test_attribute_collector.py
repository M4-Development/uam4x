from uam4x.components import AttributeCollector
from uam4x.study import Study
from uam4x.components import DataType

import unittest
import os, shutil


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_WORK_DIR = os.path.join(THIS_DIR, "work")

class Test_AttributeCollector(unittest.TestCase):
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
        # test_study = Study(TEST_WORK_DIR)
        test_study = Study()

        return test_study


    def test_init(self):
        # intitalize test
        study = self.initialize()

        # should raise type error
        # with self.assertRaises(TypeError): # currently not raising a type error but it should
        #     ac = AttributeCollector('This should be a study not a string')

        with self.assertRaises(TypeError):
            ac = AttributeCollector()

        with self.assertRaises(TypeError):
            ac = AttributeCollector(study, "one arg too many")

        # should not raise any error
        ac = AttributeCollector(study)

        # check outputs
        self.assertTrue(ac.study,study)



    def test_check_data_type(self):

        # intitalize test
        study = self.initialize()
        ac = AttributeCollector(study)
        ac._attributes = {'span':[DataType.FLOAT, 30]}

        # check inputs (name, value, parent)
        with self.assertRaises(TypeError):
           check = ac.check_data_type('span')

        with self.assertRaises(TypeError):
           check = ac.check_data_type('span', 30, None, 'Extra input')

        with self.assertRaises(TypeError):
           check = ac.check_data_type(['should be string, not list'], 30, None)

        # check outputs
        check = ac.check_data_type('span',30,None)
        self.assertTrue(check)

        check = ac.check_data_type('span',None,'span')
        self.assertTrue(check)

        ac._attributes = {'span':[DataType.INT, 30]}
        check = ac.check_data_type('span',30,None)
        self.assertTrue(check)

    def test_load_attributes_collector(self):
        # intialize test
        # intitalize test
        study = self.initialize()
        ac = AttributeCollector(study)
        attribute_dict = {'span':[DataType.FLOAT, 30]}

        # test inputs
        # inputs: (attributes_dict)
        with self.assertRaises(TypeError):
            ac.load_attributes_collector()

        with self.assertRaises(AttributeError):
            ac.load_attributes_collector(['this should be a dictionary'])

        with self.assertRaises(TypeError):
            ac.load_attributes_collector(attribute_dict,'Extra String')

        # test outputs
        ac.load_attributes_collector(attribute_dict)
        self.assertEqual(ac._attributes,attribute_dict)


    def test_save_attributes_collector(self):
        # initialize test
        study = self.initialize()
        ac = AttributeCollector(study)
        ac._attributes = {'span':[DataType.FLOAT, 30]}

        # test inputs
        # inputs: ()
        with self.assertRaises(TypeError):
            out = ac.save_attributes_collector('Extra inputs')

        # test outputs
        out = ac.save_attributes_collector()
        self.assertEqual(out,{'span':['FLOAT',30]})

if __name__ == '__main__':
    unittest.main()
