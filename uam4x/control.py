from typing import Any, Dict, List, Union
import copy
from uam4x.log import get_log
from uam4x.utils import log
import numpy as np

class Optimization:
    def __init__(self) -> None:
        self._optimization_dict={'type': 'Optimization',
                                         'des_vars': {},
                                         'constraints': {},
                                         'objective_name': None}

    def _to_dict(self):
        #if user accidentally calls, won't modify the actual optimization dict
        return copy.deepcopy(self._optimization_dict)

    @classmethod
    def _from_dict(cls, data: dict) -> 'Optimization':
        obj = cls()
        # if upper/lower for DV or constraints are lists, must be
        #desired to be arrays, so convert. 
        for constraint, constraint_vals in data['constraints'].items():
            if isinstance(constraint_vals['lower'], list):
                constraint_vals['lower'] = np.array(constraint_vals['lower'])
            if isinstance(constraint_vals['upper'], list):
                constraint_vals['upper'] = np.array(constraint_vals['upper'])
        for des_var, des_var_vals in data['des_vars'].items():
            if isinstance(des_var_vals['lower'], list):
                des_var_vals['lower'] = np.array(des_var_vals['lower'])
            if isinstance(des_var_vals['upper'], list):
                des_var_vals['upper'] = np.array(des_var_vals['upper'])
                
        obj._optimization_dict = {"type": "Optimization", 
                                   "des_vars": data['des_vars'],
                                   "constraints": data['constraints'],
                                   "objective_name": data['objective_name']
                                   }
        return obj
    
    def add_design_var(self, name: str, lower: np.ndarray|list|None=None, upper:np.ndarray|list|None=None, indices: list|None=None):
        self._optimization_dict['des_vars'][name] = {'lower': lower,
                                               'upper': upper,
                                               'indices': indices}

    def add_constraint(self, name: str, lower: list|None=None, upper: list|None=None, indices=None):
        self._optimization_dict['constraints'][name] = {'lower': lower,
                                                        'upper': upper,
                                                        'indices': indices}
    def add_objective(self, name: str):
        self._optimization_dict['objective_name'] = name
class ParameterSweep:
    log = get_log(log, object_name='parameter sweep')
    def __init__(self, param_name_to_value_list: Dict[str, Any], child:Union['ParameterSweep', Optimization]=None) -> None:
        self.child = child
        self.length: int = None
        self.param_name_to_value_list = param_name_to_value_list
        self.log = ParameterSweep.log
        self.log.debug('initialize')
        for name, value_list in self.param_name_to_value_list.items():
            if not isinstance(value_list, list):
                self.log.error(f"Values for parameter '{name}' must be a list, got {type(value_list).__name__}")
                raise TypeError(f"Values for parameter '{name}' must be a list, got {type(value_list).__name__}")
            if self.length is None:
                self.length = len(value_list)
            elif self.length != len(value_list):
                self.log.error(f"All parameter value lists must have the same length, but '{name}' has length {len(value_list)} while others have length {self.length}")
                raise ValueError(f"All parameter value lists must have the same length, but '{name}' has length {len(value_list)} while others have length {self.length}")

    def set_child(self, child: Union['ParameterSweep', Optimization]) -> None:
        self.log.debug('set child')
        if not isinstance(child, (Optimization, ParameterSweep)):
            self.log.error("Child must be ParameterSweep or Optimization")
            raise TypeError("Child must be ParameterSweep or Optimization")
        self.child = child

    def get_states(self) -> List[Dict[str, Any]]:
        states = []
        self.log.debug('get states')
        for i in range(self.length):
            state = {}
            for name, values in self.param_name_to_value_list.items():
                state[name] = values[i]
            if isinstance(self.child, Optimization):
                state['optimization'] = True
                state.update(self.child._to_dict())
                states.append(state)
            elif isinstance(self.child, ParameterSweep):
                states.extend(self.child.get_states())
            else:
                state['optimization'] = False
                states.append(state)
        return states
    def _to_dict(self):
        self.log.debug('write to dict')
        return {
            "type": "ParameterSweep",
            "param_name_to_value_list": self.param_name_to_value_list,
            "child": self.child._to_dict() if self.child else None
        }
    @classmethod
    def _from_dict(cls, data):
        cls.log.debug('parameter sweep from dict')
        child_dict = data["child"]
        if child_dict is None:
            child = None
        elif child_dict['type'] == "ParameterSweep":
            child = ParameterSweep._from_dict(child_dict)
        elif child_dict['type'] == "Optimization":
            child = Optimization._from_dict(child_dict)
        return ParameterSweep(data["param_name_to_value_list"], child=child)

class Control:
    log = get_log(log, object_name='control')
    def __init__(self, title: str = "Default Title", author: str = "Default Author", description: str = "Default Description") -> None:
        self.title = title
        self.author = author
        self.description = description
        self.trade_studies = {}
        self.log = Control.log

    def add_trade(self, trade_name:str, trade:Union[ParameterSweep, Optimization]) -> None:
        self.log.debug(f'add trade: {trade_name}')
        self.trade_studies[trade_name] = trade
    def _to_dict(self) -> dict:
        self.log.debug('write to dict')
        return {
            "title": self.title,
            "author": self.author,
            "description": self.description,
            "trade_studies": {k: v._to_dict() for k, v in self.trade_studies.items()}
        }
    @classmethod
    def _from_dict(cls, data: dict) -> 'Control':
        cls.log.debug('control from dict')
        control = cls(
            title=data["title"],
            author=data["author"],
            description=data["description"],
        )
        for k, v in data["trade_studies"].items():
            if v['type'] == "ParameterSweep":
                trade = ParameterSweep._from_dict(v)
            elif v['type'] == "Optimization":
                trade = Optimization._from_dict(v)
            else:
                raise ValueError(f"Unknown trade study type: {v['type']}")
            control.add_trade(k, trade)

        return control

if __name__ == "__main__":
    #TODO: Update this to use the new Optimization class
    opt = Optimization(
        des_var_names=["wing_span", "wing_area"],
        des_var_lower=[10.0, 20.0],
        des_var_upper=[15.0, 30.0],
        constraint_names=["drag"],
        constraint_lower=[0.1],
        constraint_upper=[0.5],
        objective_name="weight"
    )
    param_sweep = ParameterSweep(
        param_name_to_value_list={
            "aspect_ratio": [10.0, 12.5, 15.0],
        },
        child=opt
    )
    control = Control(title="Vehicle Control", author="Tom", description="Control for vehicle design")
    control.add_trade("aspect_ratio_parameter_sweep", param_sweep)

    # Print all trade studies and their details
    print("\nTrade studies in control:")
    for name, trade in control.trade_studies.items():
        print(f"Trade name: {name}")
        print(f"Type: {type(trade).__name__}")
        if isinstance(trade, ParameterSweep):
            print("Parameter sweep states:")
            for state in trade.get_states():
                print(state)
        print()

    # Add and print a direct optimization trade
    control.add_trade("direct_optimization", opt)
    print("Direct optimization trade as dict:")
    print(vars(opt))
