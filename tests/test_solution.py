import pytest
from uam4x.study import Study
import re
import uam4x.solution as sol
from uam4x.solution import Plugin, register_plugin, Solution, _plugin_registry, log, get_plugin_class, get_plugin_names
import numpy as np
from uam4x.utils import check_and_get_config
import os
import logging
import json
import importlib
import inspect
import hashlib
#----------------------------Test Solution
# Positive checks
def test_solution_initialization():
    S = Solution()
    assert S.tracked_outputs == []
    assert S.links == []
    assert S.steps == {}

def test_add_link_good():
    S = Solution()
    S.add_link('vehicles.test_vehicle.vtail.span', 'VSP_step.vtail_span')
    S.add_link('vehicles.test_vehicle.vtail.chord', 'VSP_step.vtail_chord')
    S.add_link('vehicles.test_vehicle.vtail.chord', 'VSP_step.vtail_chord')
    S.add_link('vehicles.test_vehicle.vtail.aspect_ratio', 'VSP_step.vtail_aspect_ratio')

    assert len(S.links) == 3
    assert S.links[0][0] == 'vehicles.test_vehicle.vtail.span'
    assert S.links[0][1] == 'VSP_step.vtail_span'
    assert len(S.links[0]) == 2

def test_remove_link_good():
    S = Solution()
    S.add_link('vehicles.test_vehicle.vtail.span', 'VSP_step.vtail_span')
    S.add_link('vehicles.test_vehicle.vtail.chord', 'VSP_step.vtail_chord')
    S.add_link('vehicles.test_vehicle.vtail.aspect_ratio', 'VSP_step.vtail_aspect_ratio')
    S.remove_link('vehicles.test_vehicle.vtail.chord', 'VSP_step.vtail_chord')


    assert len(S.links) == 2
    assert S.links[1] == ['vehicles.test_vehicle.vtail.aspect_ratio', 'VSP_step.vtail_aspect_ratio']

def test_add_step_good():

    class TestPlugin(Plugin):
        def setup_plugin(self, plugin_options):


            self.declare_input('vtail_aspect_ratio', float)
            self.base_path = plugin_options['file_path']

            for name,data in  plugin_options['output_request_dict'].items():
                self.declare_output(name, data['type'])

    S = Solution()
    S.add_step(TestPlugin, step_name=None, plugin_options={'file_path': 'some\\generic\\filepath',
                                                                 'output_request_dict': {'out_var_1': {'type': float},
                                                                                         'out_var_2': {'type': str}}})
    S.add_step(TestPlugin, step_name='some_step')

    src = inspect.getsource(TestPlugin)
    h = hashlib.sha256()
    h.update(src.encode("utf-8"))

    assert S.steps['step1'] == {'plugin': TestPlugin,'plugin_options': {'file_path': 'some\\generic\\filepath',
                                                                 'output_request_dict': {'out_var_1': {'type': float},
                                                                                         'out_var_2': {'type': str}}},
                                                     'plugin_hash': h.hexdigest()
                                                     }
    assert 'some_step' in S.steps
    P = TestPlugin()
    P.setup_plugin(S.steps['step1']['plugin_options'])
    assert P.plugin_input_names['vtail_aspect_ratio']['type'] == float
    assert P.plugin_output_names['out_var_1']['type'] == float
    assert P.plugin_output_names['out_var_2']['type'] == str
    assert P.base_path == 'some\\generic\\filepath'


def test_get_step_names_good():
    class Plugin1(Plugin):
        pass

    class Plugin2(Plugin):
        pass

    S = Solution()
    S.add_step(Plugin1, step_name='sample_step_one')
    S.add_step(Plugin2, step_name='sample_step_two')

    assert  S.get_step_names() == ['sample_step_one', 'sample_step_two']

def test_remove_step_good():
    class Plugin1(Plugin):
        pass

    class Plugin2(Plugin):
        pass

    S = Solution()
    S.add_step(Plugin1, step_name='sample_step_one')
    S.add_step(Plugin2, step_name='sample_step_two')
    src = inspect.getsource(Plugin2)
    h = hashlib.sha256()
    h.update(src.encode("utf-8"))


    S.remove_step('sample_step_one')

    assert S.steps == {'sample_step_two': {'plugin': Plugin2,'plugin_options': {}, 'plugin_hash': h.hexdigest(),}}

    S.remove_step('sample_step_fifty')

    assert S.steps == {'sample_step_two': {'plugin': Plugin2,'plugin_options': {}, 'plugin_hash': h.hexdigest()}}

def test_edit_step_good():
    class TestPlugin(Plugin):
        def setup_plugin(self, plugin_options):


            self.declare_input('vtail_aspect_ratio', float)
            self.base_path = plugin_options['file_path']

            for name,data in  plugin_options['output_request_dict'].items():
                self.declare_output(name, data['type'])

    S = Solution()
    S.add_step(TestPlugin, step_name=None, plugin_options={'file_path': 'some\\generic\\filepath',
                                                                 'output_request_dict': {'out_var_1': {'type': float},
                                                                                         'out_var_2': {'type': str}}})

    S.add_step(TestPlugin, step_name='random_step', plugin_options={'file_path': 'random_path',
                                                                 'output_request_dict': {'some_var_4': {'type': np.ndarray},
                                                                                         'some_var_67': {'type': int}}})

    S.edit_step('step1', plugin_options={'file_path': 'new\\path',
                                                                 'output_request_dict': {'out_var_3': {'type': int},
                                                                                         'out_var_2': {'type': np.ndarray}}})
    src = inspect.getsource(TestPlugin)
    h = hashlib.sha256()
    h.update(src.encode("utf-8"))

    # dictionary is modified as desired
    assert S.steps['step1']['plugin_options']['file_path'] == 'new\\path'
    assert S.steps['step1'] == {'plugin': TestPlugin,'plugin_options': {'file_path': 'new\\path',
                                                                 'output_request_dict': {'out_var_3': {'type': int},
                                                                                         'out_var_2': {'type': np.ndarray}},
                                                                 },
                                'plugin_hash': h.hexdigest()}

    #step doesn't exist, so nothing should happen
    S.edit_step('step3', plugin_options={'file_path': 'new\\path',
                                                                 'output_request_dict': {'out_var_3': {'type': int},
                                                                                         'out_var_2': {'type': np.ndarray}}})
    assert 'step3' not in S.steps

    S.edit_step('step1')

    #when no plugin dictionary passed in, dictionary is empty
    assert S.steps['step1']['plugin_options'] == {}

    # steps not wanted to be edited are unedited.
    assert S.steps['random_step'] == {'plugin': TestPlugin,'plugin_options': {'file_path': 'random_path',
                                                                 'output_request_dict': {'some_var_4': {'type': np.ndarray},
                                                                                         'some_var_67': {'type': int}}},
                                                           'plugin_hash': h.hexdigest(),}


def test_tracked_output_good():
    S = Solution()
    S.add_tracked_output('happy_gillmore')
    S.add_tracked_output('dark_knight_rises')
    S.add_tracked_output('batman_begins')

    assert S.tracked_outputs == ['happy_gillmore', 'dark_knight_rises', 'batman_begins']

def test_remove_tracked_output_good():
    S = Solution()
    S.add_tracked_output('happy_gillmore')
    S.add_tracked_output('dark_knight_rises')
    S.add_tracked_output('batman_begins')

    S.remove_tracked_output('dark_knight_rises')
    assert S.tracked_outputs == ['happy_gillmore', 'batman_begins']

    S.remove_tracked_output('start_wars_III')
    assert S.tracked_outputs == ['happy_gillmore', 'batman_begins']


def test_add_plugin_option_good(capsys):
    class TestPlugin(Plugin):
        def setup_plugin(self, plugin_options):


            self.declare_input('vtail_aspect_ratio', float)
            self.base_path = plugin_options['file_path']

            for name,data in  plugin_options['output_request_dict'].items():
                self.declare_output(name, data['type'])


    src = inspect.getsource(TestPlugin)
    h = hashlib.sha256()
    h.update(src.encode("utf-8"))
    S = Solution()
    S.add_step(TestPlugin, step_name=None, plugin_options={'file_path': 'some\\generic\\filepath',
                                                                 'output_request_dict': {'out_var_1': {'type': float},
                                                                                         'out_var_2': {'type': str}}})

    S.add_plugin_option(step_name='step1', option_name='misc_option', option_value=1738)

    assert S.steps['step1'] == {'plugin': TestPlugin,'plugin_options': {'file_path': 'some\\generic\\filepath',
                                                                 'output_request_dict': {'out_var_1': {'type': float},
                                                                                         'out_var_2': {'type': str}},
                                                                 'misc_option': 1738},
                                                     'plugin_hash': h.hexdigest(),
                                                     }
    S.add_plugin_option(step_name='step3', option_name='misc_option', option_value=1738)
    assert 'step3' not  in S.steps
    assert len(list(S.steps.keys()))
    assert S.steps['step1'] == {'plugin': TestPlugin,'plugin_options': {'file_path': 'some\\generic\\filepath',
                                                                 'output_request_dict': {'out_var_1': {'type': float},
                                                                                         'out_var_2': {'type': str}},
                                                                 'misc_option': 1738},
                                                     'plugin_hash': h.hexdigest()
                                                     }

    S.add_plugin_option(step_name='step1', option_name='misc_option', option_value=69)
    captured = capsys.readouterr()
    out = captured.out
    assert f'misc_option already a plugin option. Overwriting value: {1738}' in  out
    assert S.steps['step1'] == {'plugin': TestPlugin,'plugin_options': {'file_path': 'some\\generic\\filepath',
                                                             'output_request_dict': {'out_var_1': {'type': float},
                                                                                     'out_var_2': {'type': str}},
                                                             'misc_option': 69},
                                                     'plugin_hash': h.hexdigest()}

def test_remove_plugin_option_good():
    class TestPlugin(Plugin):
        def setup_plugin(self, plugin_options):


            self.declare_input('vtail_aspect_ratio', float)
            self.base_path = plugin_options['file_path']

            for name,data in  plugin_options['output_request_dict'].items():
                self.declare_output(name, data['type'])

    S = Solution()
    S.add_step(TestPlugin, step_name=None, plugin_options={'file_path': 'some\\generic\\filepath',
                                                                 'output_request_dict': {'out_var_1': {'type': float},
                                                                                         'out_var_2': {'type': str}}})
    assert 'file_path' in S.steps['step1']['plugin_options']

    S.remove_plugin_option(step_name='step1', option_name='file_path')

    assert 'file_path' not in S.steps['step1']['plugin_options']

    S.remove_plugin_option('step1738', 'file_path')

    assert 'step1738' not  in S.steps
    assert S.steps['step1']['plugin_options'] ==  {'output_request_dict': {'out_var_1': {'type': float},
                                                                                         'out_var_2': {'type': str}}}

# Negative checks
@pytest.mark.parametrize('indep,dep', [
    ('vehicles.test_vehicle.vtail.span', 'VSP_step.vtail.span[1]'),
])
def test_add_link_bad(indep, dep):
    S = Solution()
    with pytest.raises(ValueError):
        S.add_link(indep, dep)


def test_add_step_no_callable():
    S = Solution()
    with  pytest.raises(TypeError):
        S.add_step('not_a_callable')

@pytest.mark.parametrize('name', [
    ('some_ste[p'),
    ('some_]step')
])
def test_add_step_name_bad_characters(name):
    class TestPlugin(Plugin):
        def setup_plugin(self):
            self.declare_input('vtail_aspect_ratio', float)

    S = Solution()
    with  pytest.raises(ValueError):

        S.add_step(TestPlugin, step_name=name)

def test_tracked_output_repeated_outputs():
    S = Solution()
    S.add_tracked_output('happy_gillmore')
    S.add_tracked_output('dark_knight_rises')
    S.add_tracked_output('batman_begins')

    with pytest.raises(RuntimeError):
        S.add_tracked_output('dark_knight_rises')

#----------------------------Test Plugin------------------------------

#-----------Happy checks---------------

def test_declaring_array_in():
    @register_plugin
    class ShapeTester(Plugin):
        def setup_plugin(self):
            self.declare_input('test_var_in', var_type=np.ndarray, shape=5)
            self.declare_output('test_var_out1', var_type=np.ndarray, shape=(6, 7), value=np.full((6, 7), 67))
            self.declare_output('test_var_out2', var_type=np.ndarray, value=np.array([1., 33., 66., 420.]))
    ShapeTest = ShapeTester()
    ShapeTest.setup_plugin()
    #inputs
    assert ShapeTest.plugin_input_names['test_var_in']['shape'] == (5, )
    assert ShapeTest.plugin_input_names['test_var_in']['type'] == np.ndarray
    assert ShapeTest.plugin_input_names['test_var_in']['value'] is None
    #outputs
    assert ShapeTest.plugin_output_names['test_var_out1']['shape'] == (6, 7)
    assert ShapeTest.plugin_output_names['test_var_out1']['type'] == np.ndarray
    np.testing.assert_array_equal(ShapeTest.plugin_output_names['test_var_out1']['value'] , np.full((6, 7), 67))

    assert ShapeTest.plugin_output_names['test_var_out2']['shape'] is  None
    assert ShapeTest.plugin_output_names['test_var_out1']['type'] == np.ndarray
    np.testing.assert_array_equal(ShapeTest.plugin_output_names['test_var_out2']['value'] ,np.array([1., 33., 66., 420.]))


def test_array_may_be_scalar_with_shape_plugin():
    @register_plugin
    class FloatValueOkay(Plugin):
        def setup_plugin(self):
            self.declare_input('test_var_in', var_type=np.ndarray, value=5., shape=5)
            self.declare_output('test_var_out', var_type=np.ndarray, value=10., shape=(4, 3))

    FVO = FloatValueOkay()
    FVO.setup_plugin()
    #inputs
    assert FVO.plugin_input_names['test_var_in']['shape'] == (5, )
    assert FVO.plugin_input_names['test_var_in']['type'] == np.ndarray
    assert FVO.plugin_input_names['test_var_in']['value'] == 5.

    assert FVO.plugin_output_names['test_var_out']['shape'] == (4,3 )
    assert FVO.plugin_output_names['test_var_out']['type'] == np.ndarray
    assert FVO.plugin_output_names['test_var_out']['value'] == 10.

@pytest.mark.parametrize("var_type,value", [
    (float, 67.),
    (str, "six-seven"),
    (int, 67),
    (list, [6, 7]),
    (tuple, (6, 7))
])
def test_declaring_inputs_outputs(var_type, value):
    @register_plugin
    class RandomPlugin(Plugin):
        def setup_plugin(self):
            self.declare_input('var_in', var_type, value)
            self.declare_output('var_out', var_type, value)

    RP = RandomPlugin()
    RP.setup_plugin()
    assert RP.plugin_input_names['var_in']['type'] == var_type
    assert RP.plugin_input_names['var_in']['value'] == value
    assert RP.plugin_input_names['var_in']['shape'] is None

    assert RP.plugin_output_names['var_out']['type'] == var_type
    assert RP.plugin_output_names['var_out']['value'] == value
    assert RP.plugin_output_names['var_out']['shape'] is None


#Partial derivatives:
def test_fd_none_step_and_valid_forms():
    @register_plugin
    class P(Plugin):
        def setup_plugim(self):
            self.declare_partials_plugin('y')
    P().declare_partials_plugin('y','x',method='fd',step=None,form='forward')
    P().declare_partials_plugin('y','x',method='fd',step=1e-4,form='central')


def test_of_wrt_wrapping_and_passthrough():
    @register_plugin
    class P(Plugin): pass
    p = P()
    p.declare_partials_plugin('y',['x1','x2'])
    assert p.initial_partials[-1]['of'] == ['y']
    assert p.initial_partials[-1]['wrt'] == ['x1','x2']

def test_build_partials_wildcards():
    @register_plugin
    class P(Plugin): pass
    p = P()
    # Simulate declared names as your code expects them (no 'vehicles.' prefix after strip)
    ins  = ['c.x','c.z']; outs = ['c.y','c.w']
    p.initial_partials = [{'of':['*'],'wrt':['*'],'method':'fd','step':1e-3,'form':'forward',
                           'rows':None,'cols':None}]
    d = p.build_partials(input_names=ins, output_names=outs)
    assert set(d.keys()) == {(o,i) for o in {'y','w'} for i in {'x','z'}}


def test_redeclare_overwrites():
    @register_plugin
    class P(Plugin): pass
    p = P()
    p.initial_partials = [
        {'of':['y'],'wrt':['x'],'method':'fd','form':'forward','step':1e-3,'rows':None,'cols':None},
        {'of':['y'],'wrt':['x'],'method':'exact','form':None,'step':None,'rows':None,'cols':None},
    ]
    d = p.build_partials(['c.x'],['c.y'])
    assert d[('y','x')]['method'] == 'exact'
    assert d[('y', 'x')]['form'] is None
    assert d[('y', 'x')]['step'] is  None

def test_categorize_partials_and_flag():
    @register_plugin
    class P(Plugin): pass
    p = P()
    p.comprehensive_partials_dict = {
        ('y','x'): {'method':'fd','form':'forward','step':1e-3,'rows':None,'cols':None},
        ('y','z'): {'method':'exact','form':None,'step':None,'rows':None,'cols':None},
    }
    m = p.categorize_partials()
    assert set(m['fd']) == {('y','x')}
    assert set(m['exact']) == {('y','z')}
    assert p._declare_partials is True

def test_list_of_str_wrt():

    class SamplePlugin(Plugin):
        def  setup_plugin(self):
            self.declare_partials_plugin(of=['c.x', 'c.y'], wrt='*')

    p = SamplePlugin()
    p.setup_plugin()
    assert p.initial_partials[0]['wrt'] == ['*']


#-------------------------------Negative checks------------

#----PLUGIN INPUT/OUTPUT DECLARATION
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
def test_type_must_be_array_if_shape(var_type):
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
def test_declared_type_must_be_type(var_type):
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



@pytest.mark.parametrize("shape,value", [
    ((5,),        np.zeros((6,))),      # 1D length off by one
    ((6,),        np.zeros((5,))),      # 1D shorter than declared
    ((3,),        np.zeros((3, 1))),    # 1D declared vs 2D actual
    ((2, 2),      np.zeros((2, 3))),    # 2D inner dim mismatch
    ((4, 1),      np.zeros((4,))),      # 2D declared vs 1D actual
    ((1, 4),      np.zeros((4, 1))),    # Row vs column mismatch
    ((3, 4),      np.zeros((3, 4, 1))), # Extra trailing axis
    ((2, 3, 4),   np.zeros((2, 3, 5))), # 3D last-dim mismatch
    ((2, 3, 4),   np.zeros((2, 4, 3))), # 3D permuted dims
    ((7, ),       np.arange(5)),
])
def test_shape_must_match_value_shape(shape, value):
    name = 'test_var'
    msg = (f'If the value of an array is not declared as a scalar, then the shape must match\n'
        f'the declared shape. The variable {name} has declared shape {shape} and actual shape {value.shape}')

    @register_plugin
    class BadShapePlugin(Plugin):
        pass

    with pytest.raises(ValueError, match=re.escape(msg)):
        BadShapePlugin().declare_input(name, np.ndarray,value,shape )

    with pytest.raises(ValueError, match=re.escape(msg)):
        BadShapePlugin().declare_output(name, np.ndarray,value,shape )


def test_if_array_value_scalar_must_shape():
    name = 'test_var'
    value = 67.
    var_type = np.ndarray
    msg = (f'If array value is declared with scalar, shape must be declared for broadcasting.\n'
                                 f'The variable {name} has value: {value} and type: {var_type} but no shape declared.')

    @register_plugin
    class FloatValueNoShape(Plugin):
        pass

    with pytest.raises(ValueError, match=re.escape(msg)):
        FloatValueNoShape().declare_input(name, var_type, value)

    with  pytest.raises(ValueError, match=re.escape(msg)):
        FloatValueNoShape().declare_output(name, var_type, value)


def test_array_requires_shape_or_value():
    @register_plugin
    class P(Plugin): pass
    with pytest.raises(ValueError, match=r'no shape or value'):
        P().declare_input('a', np.ndarray)  # neither shape nor value

@pytest.mark.parametrize("shape", [
    True,                 # bool not allowed
    -1,                   # negative int
    2.5,                  # float not allowed
    "3",                  # string
    [3, 4],               # list (must be tuple or int)
    (3, -1),              # tuple with negative dim
    (3, 2.5),             # tuple with non-integer dim
    (3, True),            # tuple with bool dim
    ("3", 4),             # tuple with string dim
])
def test_invalid_array_shape_rejected(shape):
    name = "test_var"
    msg = ('The shape must be declared as a tuple of integers or as a\n'
           f'single integer or as NoneType. The shape of {name} is {shape} which is invalid')

    @register_plugin
    class P(Plugin):
        pass

    with pytest.raises(ValueError, match=re.escape(msg)):
        P().declare_output(name, np.ndarray, shape=shape)

#----PARTIALS DECLARATION
@pytest.mark.parametrize("method,msg,form,step",[
    ('something_random',(f'Cannot use methods other than exact or fd, user used method: something_random'),None,None),
    ('fd', (f'Step size must be a real number. The step size invalid_step is not valid'), None, 'invalid_step'),
    ('fd', (f'Step size must be greater than zero. The step size {-5.} is not valid'), None, -5.),
    ('fd', ('Only allowed finite difference forms are: forward,\nbackward, and central. The form: invalid_form is not valid'), 'invalid_form', None),
    ('exact',(f'User specified exact method, which does not take in a form.\nInstead, leave step blank or set it equal to None'), 'forward', None),
    ('exact',(f'User specified exact method, which does not take in a step.\nInstead, leave step blank or set it equal to None'), None, 10.**-3)
])
def test_partials_declaration_negative(method, msg, form, step):

    @register_plugin
    class BadPartialDeclare(Plugin):
        pass

    with  pytest.raises(ValueError, match=re.escape(msg)):
        BPD = BadPartialDeclare()
        BPD.declare_partials_plugin(of='hello', wrt='goodbye', method=method, form=form, step=step)


def test_build_partials_name_subset_checks_out():
    @register_plugin
    class P(Plugin): pass
    p = P()
    p.initial_partials = [{'of':['bad_out'],'wrt':['x'],'method':'fd','form':'forward','step':1e-3,
                           'rows':None,'cols':None}]
    with pytest.raises(ValueError, match=r'Partial outputs'):
        p.build_partials(['c.x'], ['c.y'])

def test_build_partials_name_subset_checks_in():
    class P(Plugin):
        def setup_plugin(self):
            self.declare_partials_plugin(of='idk', wrt='bad', method='fd')

    Plug = P()
    Plug.setup_plugin()

    with pytest.raises(ValueError):

        Plug.build_partials(['will_raise'], ['idk'])


#--------------Registry stuff

def test_register_plugin_good():


    file_path = os.path.abspath(__file__)
    test_dir = os.path.dirname(file_path)
    proj_dir = os.path.dirname(test_dir)


    config_path = os.path.join(proj_dir,'uam4x', 'conf.json')

    if os.path.exists(config_path):
        os.remove(config_path)

    config = check_and_get_config(config_path)
    assert config == {"PLUGINS": {},"SUBSYSTEMS": {}}

    @register_plugin
    class P(Plugin):
        pass

    config = check_and_get_config(config_path)
    assert config == {"PLUGINS": {P.__name__: file_path},"SUBSYSTEMS": {}}
    assert _plugin_registry[P.__name__] == P


def test_get_plugin_class_good():
    file_path = os.path.abspath(__file__)
    test_dir = os.path.dirname(file_path)
    proj_dir = os.path.dirname(test_dir)


    config_path = os.path.join(proj_dir,'uam4x', 'conf.json')

    if os.path.exists(config_path):
        os.remove(config_path)

    @register_plugin
    class P(Plugin):
        pass

    assert get_plugin_class(P.__name__) == P
    os.remove(config_path)


#----Warnings and negatives
def test_register_plugin_warning(capsys):
    # Locate conf.json in the project
    file_path = os.path.abspath(__file__)
    test_dir = os.path.dirname(file_path)
    proj_dir = os.path.dirname(test_dir)
    config_path = os.path.join(proj_dir, 'uam4x', 'conf.json')

    # Start clean
    if os.path.exists(config_path):
        os.remove(config_path)

    config = check_and_get_config(config_path)
    assert config == {"PLUGINS": {}, "SUBSYSTEMS": {}}

    # Seed a conflicting plugin entry for name "P"
    config["PLUGINS"]["P"] = "/some/other/module/path.py"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    # Register a plugin with the same name "P"
    @register_plugin
    class P(Plugin):
        pass

    # Capture printed output from FileLogger
    captured = capsys.readouterr()
    combined = captured.out + captured.err

    # Sanity check (optional debug)
    # print("\n--- CAPTURED ---\n", combined)

    assert "Plugin 'P' already registered" in combined
    assert "Overwritting existing data" in combined

    # Confirm that conf.json has been updated to the actual module origin
    new_config = check_and_get_config(config_path)
    actual_origin = importlib.util.find_spec(P.__module__).origin
    assert new_config["PLUGINS"]["P"] == actual_origin

    # Cleanup
    if os.path.exists(config_path):
        os.remove(config_path)




#-----------------Note, last test because modification of _plugin_registry which should ordinarily not be done
@pytest.fixture
def clean_registry():
    # Clear before the test
    sol._plugin_registry.clear()
    yield sol._plugin_registry
    # Optionally clear after the test as well
    sol._plugin_registry.clear()

def test_get_plugin_class_non_existent_plugin(clean_registry):

    with  pytest.raises(ImportError):
        get_plugin_class('DefNotInPluginDict')

def test_get_plugin_names(clean_registry):

    assert get_plugin_names() == []

    @register_plugin
    class P1(Plugin):
        pass
    @register_plugin
    class P2(Plugin):
        pass

    assert get_plugin_names() == ['P1', 'P2']


if __name__ == "__main__":
    test_edit_step_good()
