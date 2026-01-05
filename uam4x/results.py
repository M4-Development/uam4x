import copy
import numpy as np
from uam4x.utils import canonicalize_state, hash_state

class Results:
    
    def __init__(self):
        self.results = []
        self._hash_to_result = {}
    

    def _add_result(self, state, result_name, result_value):

        incoming_hash = hash_state(state)
        if incoming_hash in  self._hash_to_result:
            self._hash_to_result[incoming_hash]._add_result(result_name, result_value)
        else:
            result = Result(state)
            result._add_result(result_name, result_value)
            self._hash_to_result[incoming_hash] = result
            self.results.append(result)
            
    def _add_convergence_status(self, state, status):
        incoming_hash = hash_state(state)
        if incoming_hash in  self._hash_to_result:
            self._hash_to_result[incoming_hash]._add_convergence_status(status)
        else:
            result = Result(state)
            result._add_convergence_status(status)
            self._hash_to_result[incoming_hash] = result
            self.results.append(result)
                
                
    def _to_dict(self):
        result_list = []
        for result in  self.results:
            result_list.append(result._to_dict())
        
        return result_list
        
    @classmethod
    def _from_dict(cls, results_list):
        results = cls()
        for result in results_list:
            r = Result._from_dict(result)
            results.results.append(r)
            results._hash_to_result[r.state_hash] = r
        return results
        
    
class Result:
    
    def __init__(self, state):
        
        self.state = copy.deepcopy(state)
        self.state_hash = hash_state(self.state)        
        self.template_hashes = {}
        self.outputs = {}
        self.convergence_status = None
        
    def _add_convergence_status(self, convergence_status):
        self.convergence_status = convergence_status
    
    def _add_result(self,result_name, result_value):
        self.outputs[result_name] = result_value
        
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
        
        
    
    
    
        
        