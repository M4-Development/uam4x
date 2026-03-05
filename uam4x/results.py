import copy
from uam4x.utils import hash_state, convert_numpy_to_python
import matplotlib.pyplot as plt
from uam4x.UQ import post_workflow_name_to_class_registry
import numpy as np

class Results:
    def __init__(self,study=None):
        self.tradestudies={}
        self.study=study

    def get_tradestudy(self,name):
        if name in self.tradestudies:
            return self.tradestudies[name]
        else:
            return None

    def add_tradestudy(self,name,Trade):
        Trade._results_owner=self
        self.tradestudies[name]=Trade
        if name in self.study.control.trade_studies:
            Trade.control_data = self.study.control.trade_studies[name]
            Trade.type = self.study.control.trade_studies[name].type
            Trade.tracked_outputs = self.study.solution.tracked_outputs

    def get_all_trade_studies(self):
        tradestudies = list(self.tradestudies.keys())
        return tradestudies

    def get_swept_parameters(self, trade_name):
        return list(self.study.control.trade_studies[trade_name].param_name_to_value_list.keys())

    def _to_dict(self):
        tradename_to_tradestudy={}
        for name,tradestudy in self.tradestudies.items():
            tradename_to_tradestudy[name]=tradestudy._to_dict()
        return tradename_to_tradestudy

    @classmethod
    def _from_dict(cls,tradename_to_results,study):
        res=cls(study)
        for trade_name,results in tradename_to_results.items():
            ts=Tradestudy._from_dict(trade_name,results)
            res.add_tradestudy(trade_name,ts)
            if trade_name in study.control.trade_studies:
                workflow=res.study.control.trade_studies[trade_name]
                if hasattr(workflow,'post_spec'):
                    post_spec_class=workflow.post_spec
                    ts.set_post(post_spec_class)
        return res

class Tradestudy:

    def __init__(self,trade_name):
        self._results = []
        self._hash_to_result = {}
        self.trade_name=trade_name
        self._results_owner=None
        self._results_owner = None
        self.control_data = None
        self.tracked_outputs = None
        self.type = None

        self.post=None

    def set_post(self,Post):
        self.post=Post(self)

    @property
    def study(self):
        if self._results_owner is None:
            return None
        else:
            return self._results_owner.study

    def scatter(self,var1,var2,xlabel=None,ylabel=None,title='default_title',save_dir=None):

        fig, ax=plt.subplots(figsize=(10, 6))
        x_values=self.get_variable(var1)
        y_values=self.get_variable(var2)
        ax.scatter(x_values, y_values, color='blue', alpha=0.7)

        ax.set_title(title, fontsize=16,fontweight='bold')
        if xlabel is None:
            xlabel=var1
        if ylabel is None:
            ylabel=var2
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        if save_dir is not None:
            plt.savefig(f"{save_dir}/{title.replace(' ','_')}.png", dpi=300, bbox_inches='tight')
        return ax

    def get_result_by_state(self, state):
        pass

    def get_changing_varaibles(self):

        if self.type == 'Optimization':
            changing_variables = self.tracked_outputs + self.control_data.get_desvar_names()
        elif self.type == 'ParameterSweep':
            changing_variables = self.tracked_outputs + list(self.control_data.get_all_param_names())
            if hasattr(self.control_data, "child") and self.control_data.child.type== 'Optimization' :
                changing_variables = changing_variables + self.control_data.get_desvar_names()

        return changing_variables

    def set_study_reference(self,study):
        self.study=study

    def get_variable(self,variable_name):
        if not self._results_owner:
            raise ValueError('Tradestudy has no parent Results. Tradestudy must be attached to a Results object')
        if not self._results:
            raise ValueError('No results available in Results object.')
        if self.study is None:
            raise ValueError('Study reference is not set in Results object.')


        values=[]
        #handle tradestudy desvars more robustly
        if self.trade_name in self.study.control.trade_studies:
            trade = self.study.control.trade_studies[self.trade_name]
        elif self.trade_name == "Default State":
            trade=None
        else:
            raise ValueError(f'Trade study "{self.trade_name}" not found in study control.')

        # get Design variable names if applicable
        get_desvar_names=getattr(trade,"get_desvar_names",None)
        des_vars=get_desvar_names() if callable(get_desvar_names) else []

        # get parameter variables if applicable
        get_param_names=getattr(trade,"get_all_param_names",None)
        param_names=get_param_names() if callable(get_param_names) else []

        # get UQ variable names if applicable
        aleatory_vars_to_pdf = getattr(trade, 'aleatory_var_to_dist', None) or {}

        # get sensitivity variable names if applicable
        sensitivity_vars_to_values = getattr(trade, 'variable_to_sensitivity_data', None) or {}
        if not self._results:
            print(f'No results for tradestudy {self.trade_name}')
            return None
        # first check if variable name is in tracked outputs
        if variable_name in self.study.solution.tracked_outputs:
            for result in self._results:
                if variable_name in result.outputs:
                    values.append(result.outputs[variable_name])
                else:
                    values.append(np.nan)
                    #log as missing.
            return values
        #check if the variable is a design variable
        elif des_vars and variable_name in des_vars:
            for result in self._results:
                if variable_name in result.outputs:
                    values.append(result.outputs[variable_name])
                else:
                    values.append(np.nan)
                    #log as missing.
            return values
        #check if the variable is a value in a parameter sweep
        elif variable_name in param_names:
            for result in self._results:
                values.append(result.state[variable_name])
                    #log as missing.
            return values
        # check if the variable was part of the UQ analysis:
        elif variable_name in aleatory_vars_to_pdf:
            for result in self._results:
                values.append(result.state[variable_name])
            return values
        # check if the variable was part of the sensitivity analysis:
        elif variable_name in sensitivity_vars_to_values:
            for result in self._results:
                values.append(result.state[variable_name])
            return values
        #then check if the variable name is in the vehicle parameters
        else:
            try:
                value= self.study.get_value(variable_name)
                values.append(value)
                return values
            except:
                raise ValueError(f'Variable name {variable_name} not found in tracked outputs, trade study sweep variables, or vehicle parameters.')


    def _to_dict(self):
        if self.post is not None:
            return {
            'points':[result._to_dict() for result in self._results],
            'post':self.post.__class__.__name__
            }
        else:
            return {
            'points':[result._to_dict() for result in self._results],
            }


    @classmethod
    def _from_dict(cls, trade_name,results_dict):
        results = cls(trade_name=trade_name)
        for result in results_dict['points']:
            r = Result._from_dict(result)
            results._results.append(r)
            results._hash_to_result[r.state_hash] = r
        #if 'post' in results_dict:
        #    post=post_workflow_name_to_class_registry[results_dict['post']]
        #    results.set_post(post)
        return results


class Result:

    def __init__(self, state):

        self.state = convert_numpy_to_python(copy.deepcopy(state))
        self.state_hash = hash_state(self.state)
        self.template_hashes = {}
        self.outputs = {}
        self.convergence_status = None

    def _add_convergence_status(self, convergence_status):
        self.convergence_status = convergence_status

    def _add_result(self,result_name, result_value):
        self.outputs[result_name] = convert_numpy_to_python(result_value)

    def _add_template_checksum(self, file_path, checksum):
        self.template_hashes[file_path] = checksum

    def equal_states(self, result_object):
        return self.state_hash == result_object.state_hash

    def _to_dict(self):
        d = {
            'input_state': self.state,
            'outputs': self.outputs,
        }
        if self.template_hashes:
            d['template_checksums'] = self.template_hashes

        if self.convergence_status is  not None:
            d['converged'] = self.convergence_status

        return d

    @classmethod
    def _from_dict(cls, results_dict):
        result = cls(results_dict['input_state'])
        result.outputs = copy.deepcopy(results_dict.get("outputs", {}))
        result.template_hashes = copy.deepcopy(results_dict.get("template_checksums", {}))

        if 'converged' in  results_dict:
            result._add_convergence_status(results_dict['converged'])

        return result





