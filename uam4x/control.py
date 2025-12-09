from typing import Any, Dict, List, Union
from uam4x.log import get_log
from uam4x.utils import log

class Optimization:
    def __init__(
        self,
        des_var_names: list[str],
        des_var_lower: list[float],
        des_var_upper: list[float],
        constraint_names: list[str],
        constraint_lower: list[float],
        constraint_upper: list[float],
        objective_name: str
    ) -> None:
        self.des_var_names = des_var_names
        self.des_var_lower = des_var_lower
        self.des_var_upper = des_var_upper
        self.constraint_names = constraint_names
        self.constraint_lower = constraint_lower
        self.constraint_upper = constraint_upper
        self.objective_name = objective_name
        self.log = get_log(log, object_name='optimization')

        if len(self.des_var_names) != len(self.des_var_lower):
            self.log.error("des_var_lower must match length of des_var_names")
            raise ValueError("des_var_lower must match length of des_var_names")
        if len(self.des_var_names) != len(self.des_var_upper):
            self.log.error("des_var_upper must match length of des_var_names")
            raise ValueError("des_var_upper must match length of des_var_names")
        if len(self.constraint_names) != len(self.constraint_lower):
            self.log.error("constraint_lower must match length of constraint_names")
            raise ValueError("constraint_lower must match length of constraint_names")
        if len(self.constraint_names) != len(self.constraint_upper):
            self.log.error("constraint_upper must match length of constraint_names")
            raise ValueError("constraint_upper must match length of constraint_names")
    def _to_dict(self):
        self.log.debug('write to dict')
        return {
            "type": "Optimization",
            "des_var_names": self.des_var_names,
            "des_var_lower": self.des_var_lower,
            "des_var_upper": self.des_var_upper,
            "constraint_names": self.constraint_names,
            "constraint_lower": self.constraint_lower,
            "constraint_upper": self.constraint_upper,
            "objective_name": self.objective_name
        }
    @classmethod
    def _from_dict(cls, data: dict) -> 'Optimization':
        log.debug('optimization from dict')
        return cls(
            data["des_var_names"],
            data["des_var_lower"],
            data["des_var_upper"],
            data["constraint_names"],
            data["constraint_lower"],
            data["constraint_upper"],
            data["objective_name"]
        )
class ParameterSweep:
    def __init__(self, param_name_to_value_list: Dict[str, Any], child:Union['ParameterSweep', Optimization]=None) -> None:
        self.child = child
        self.length: int = None
        self.param_name_to_value_list = param_name_to_value_list
        self.log = get_log(log, object_name='parameter sweep')
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
        log.debug('parameter sweep from dict')
        child_dict = data["child"]
        if child_dict is None:
            child = None
        elif child_dict['type'] == "ParameterSweep":
            child = ParameterSweep._from_dict(child_dict)
        elif child_dict['type'] == "Optimization":
            child = Optimization._from_dict(child_dict)
        return ParameterSweep(data["param_name_to_value_list"], child=child)
class Control:
    def __init__(self, title: str = "Default Title", author: str = "Default Author", description: str = "Default Description") -> None:
        self.title = title
        self.author = author
        self.description = description
        self.trade_studies = {}
        self.log = get_log(log, object_name='control')

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
        log.debug('control from dict')
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
