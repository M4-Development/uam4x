import pytest
from uam4x.study import Study
import re
from uam4x.solution import Plugin, register_plugin
import numpy as np

@pytest.fixture
def study():
    return Study()

#-----------Happy checks---------------

def test_plugin_shape():
    @register_plugin
    class ShapeTester(Plugin):
        def setup_plugin(self):
            self.declare_input('test_var', var_type=np.ndarray, shape=5)
            
    ShapeTest = ShapeTester()
    ShapeTest.setup_plugin()
    assert ShapeTest.plugin_input_names['test_var']['shape'] == (5, )
    assert ShapeTest.plugin_input_names['test_var']['type'] == np.ndarray
    

#---------------- Negative checks----------------
#def test_dependent_slice(study):
    
    #dependent = 'calc_area.test_array[1:]'
    
        ## Define the exact expected message once, so both raise() and match use it
    #msg = (
        #f"Only independent variables can have indices specified. "
        #f"The independent variable {dependent}\n"
        #f"has slices indices specified."
    #)

    #with pytest.raises(ValueError, match=re.escape(msg)):  
        ## Simulate what your code should do
        #study.solution.add_link(
            #"vehicles.multi_objective_vehicle.weighting_factor.test_array",
            #dependent,
        #)
        

@pytest.mark.parametrize("name", [
    ('will~mess_up'), 
    ('will[mess_up'),
    ('will]mess_up'),
    ('~wi[ll]mess_up')
])
def test_invalid_plugin_var_names(name):
    msg = (f'Tildes (~), square brackets "[","]" are not allowed in variable names\n'
           f'The following name is not allowed: {name}')

    @register_plugin
    class TildePlugin(Plugin):
        def setup_plugin(self):
            pass
        
    # If Plugin.__init__ does NOT call setup_plugin():
    with pytest.raises(ValueError, match=re.escape(msg)):	
        TildePlugin().declare_input(name, float)
        
        # If Plugin.__init__ does NOT call setup_plugin():
    with pytest.raises(ValueError, match=re.escape(msg)):
        TildePlugin().declare_output(name, float) 
        
        
@pytest.mark.parametrize("var_type", [
    (float),
    (str),
    (int),
    (list),
    (np.float64)
])        
def test_bad_type_to_shape(var_type):
    name = 'test_var'
    msg = (f'{name} has declared type: {var_type} but a shape argument was provided\n'
           f'only np.ndarray types are allowed to have shapes')
    
    @register_plugin
    class BadTypePlugin(Plugin):
        def setup_plugin(self):
            pass
        
    with pytest.raises(ValueError, match=msg):
        BadTypePlugin().declare_input(name, var_type, shape=(1, 3))
        
            
    with pytest.raises(ValueError, match=msg):
        BadTypePlugin().declare_output(name, var_type, shape=(1, 3))
        


@pytest.mark.parametrize("var_type", [
    (56.),
    (56),
    ([1, 2, 3, 4]), 
    (np.zeros(5)),
    ('hello')
])             
def test_float_type(var_type):
    name = 'test_var'
    msg = (f'Declared type must be a type. {name} has declared type {var_type} which is not a type.')
    
    @register_plugin
    class BadTypePlugin(Plugin):
        def setup_plugin(self):
            pass
        
    with pytest.raises(TypeError, match=re.escape(msg)):
        BadTypePlugin().declare_input(name, var_type)
    
    with pytest.raises(TypeError, match=re.escape(msg)):
        BadTypePlugin().declare_output(name, var_type)    
        
## Numpy valid shape checking should be a few different examples
#@pytest.mark.parametrize("shape,value", [
    #((5,),        np.zeros((6,))),      # 1D length off by one
    #((6,),        np.zeros((5,))),      # 1D shorter than declared
    #((3,),        np.zeros((3, 1))),    # 1D declared vs 2D actual
    #((2, 2),      np.zeros((2, 3))),    # 2D inner dim mismatch
    #((4, 1),      np.zeros((4,))),      # 2D declared vs 1D actual
    #((1, 4),      np.zeros((4, 1))),    # Row vs column mismatch
    #((3, 4),      np.zeros((3, 4, 1))), # Extra trailing axis
    #((2, 3, 4),   np.zeros((2, 3, 5))), # 3D last-dim mismatch
    #((2, 3, 4),   np.zeros((2, 4, 3))), # 3D permuted dims
    #((7, ),       np.arange(5)),  
#])
#def test_valid_shapes():
    #name = 'test_var'
    #msg = (f'If the value of an array is not declared as a scalar, then the shape must match\n'
        #f'the declared shape. The variable {name} has declared shape {shape} and actual shape {value.shape}')
    
    #@register_plugin
    #class BadShapePlugin(Plugin):
        #pass

    #with pytest.raises(ValueError, match=re.escape(msg)):
        #BadTypePlugin().declare_input(name, np.ndarray)    