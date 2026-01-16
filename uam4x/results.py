import copy
import numpy as np
from uam4x.utils import canonicalize_state, hash_state, convert_numpy_to_python
from collections import defaultdict
import matplotlib.pyplot as plt


class Results:
    
    def __init__(self,study=None):
        self._results = defaultdict(list)
        self._hash_to_result = {}
        self.study=study
        self.active_trade=None

    def scatter(self,var1,var2,x_axis_title=None,y_axis_title=None,plot_title='default_title',trade_name=None,save_dir=None):
        
        if trade_name is None:
            trade_name=self.active_trade
        if trade_name is None:
            raise ValueError('No active trade study set. Please provide trade_name or set an active trade study.')  
        
        fig, ax=plt.subplots(figsize=(10, 6))
        x_values=self.get_variable(var1,trade_name=trade_name)
        y_values=self.get_variable(var2,trade_name=trade_name)
        ax.scatter(x_values, y_values, color='blue', alpha=0.7)

        ax.set_title(plot_title, fontsize=16,fontweight='bold')
        if x_axis_title is None:
            x_axis_title=var1
        if y_axis_title is None:
            y_axis_title=var2      
        ax.set_xlabel(x_axis_title, fontsize=14)
        ax.set_ylabel(y_axis_title, fontsize=14)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        if save_dir is not None:
            plt.savefig(f"{save_dir}/{plot_title.replace(' ','_')}.png", dpi=300, bbox_inches='tight')   
        return ax     
    def get_result_by_state(self, state):
        pass

    def set_active_trade(self,trade_name):
        self.active_trade=trade_name

    def set_study_reference(self,study):
        self.study=study

    def get_variable(self,variable_name,trade_name=None):
        if not self._results:
            raise ValueError('No results available in Results object.')
        if self.study is None:
            raise ValueError('Study reference is not set in Results object.')
        if trade_name is None:  
            trade_name=self.active_trade
        if trade_name is None:
            raise ValueError('No active trade study set. Please provide trade_name or set an active trade study.')

        values=[]
        num_states=len(self._results[trade_name])
        # first check if variable name is in tracked outputs
        if variable_name in self.study.solution.tracked_outputs:
            for result in self._results[trade_name]:
                if variable_name in result.outputs:
                    values.append(result.outputs[variable_name])
                else:
                    values.append(None)
                    #log as missing. 
            return values
        #then check if variable name is in the parameter sweep variables
        #for a given tradestudy, a particular state will have all the same variable names
        elif self._results[trade_name][0].state['optimization'] and variable_name in self._results[trade_name][0].state['des_vars']:
            for result in self._results[trade_name]:
                if variable_name in result.outputs:
                    values.append(result.outputs[variable_name])
                else:
                    values.append(None)
                    #log as missing. 
            return values
        elif variable_name in self.study.control.trade_studies[trade_name].param_name_to_value_list:
            for result in self._results[trade_name]:
                values.append(result.state[variable_name])
                    #log as missing. 
            return values
        #then check if the variable name is in the vehicle parameters
        else:
            value= self.study.get_value(variable_name)
            values=[value]*num_states
            return values
        
            
        #raise ValueError(f'Variable name {variable_name} not found in tracked outputs, trade study sweep variables, or vehicle parameters.')   
        

    def _to_dict(self):
        result_dict = {}
        for trade_name,result_list in  self._results.items():
            result_dict[trade_name] = [result._to_dict() for result in result_list]

        return result_dict
        
    @classmethod
    def _from_dict(cls, results_dict,study):
        results = cls(study=study)
        for trade_name,result_list in results_dict.items():
            for result in result_list:
                r = Result._from_dict(result)
                results._results[trade_name].append(r)
                results._hash_to_result[r.state_hash] = r
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
        


        
        
