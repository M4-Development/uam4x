from __future__ import annotations
import copy
import numpy as np
from uam4x.results import Tradestudy

post_workflow_name_to_class_registry={}


def register_post_workflow(PostWorkflow):
    """Register post workflow.

    Parameters
    ----------
    PostWorkflow : Any
        Input value for `PostWorkflow`.

    Returns
    -------
    Any
        Return value produced by this operation.

    """
    post_workflow_name_to_class_registry[PostWorkflow.__name__]=PostWorkflow
    return PostWorkflow


class LocalSensitivity:
    """Represent the localsensitivity component in the UAM4X workflow.

    Notes
    -----
    This class encapsulates related state and behavior used by UAM4X workflows.

    """
    def __init__(self,step_size=1e-6):
        """Initialize object state and default configuration.

        Parameters
        ----------
        step_size : Any, optional
            Input value for `step_size`.

        """
        self.step_size=step_size
        self.variable_to_sensitivity_data={}
        self.post_spec=post_workflow_name_to_class_registry['LocalSensitivityPost']
            
    def add_perturbation_var(self,name,evaluation_point,step_size=None):
        """Add perturbation var.

        Parameters
        ----------
        name : Any
            Identifier used by this operation.
        evaluation_point : Any
            Input value for `evaluation_point`.
        step_size : Any, optional
            Input value for `step_size`.

        """
        data={
            'evaluation_point':evaluation_point,
            'step_size': step_size if step_size is not None else self.step_size,
            }
        self.variable_to_sensitivity_data[name]=data

    def get_states(self):


        """Return states.

        Returns
        -------
        Any
            Requested value.

        """
        baseline_state={name:data['evaluation_point'] for name,data in self.variable_to_sensitivity_data.items()}
        baseline_state['optimization']=False
        states=[baseline_state]
        b_states=[]
        f_states=[]
        
        for name,sensitivity_data in self.variable_to_sensitivity_data.items():
            local_b_state=copy.deepcopy(baseline_state)
            local_f_state=copy.deepcopy(baseline_state)
            local_b_state[name]=sensitivity_data['evaluation_point']-sensitivity_data['step_size']
            local_f_state[name]=sensitivity_data['evaluation_point']+sensitivity_data['step_size']

            b_states.append(local_b_state)
            f_states.append(local_f_state)  

        states+=b_states+f_states  

        return states

    def _to_dict(self):
        """Execute to dict.

        Returns
        -------
        Any
            JSON-serializable representation.

        """
        return {
                'type':self.__class__.__name__,
                'global_step_size':self.step_size,
                'sensitivity_parameters':{name:data for name,data in self.variable_to_sensitivity_data.items()},
                'post_spec':self.post_spec.__name__ if self.post_spec else None
                }

    @classmethod
    def _from_dict(cls,sensitivity_dict):
        """Execute from dict.

        Parameters
        ----------
        sensitivity_dict : Any
            Input value for `sensitivity_dict`.

        Returns
        -------
        Any
            Reconstructed object instance.

        """
        step_size=sensitivity_dict['global_step_size']
        variable_to_sensitivity_data=sensitivity_dict['sensitivity_parameters']
        sensitivity_obj=cls(step_size)

        for name, data in variable_to_sensitivity_data.items():
            sensitivity_obj.variable_to_sensitivity_data[name]=data

        if sensitivity_dict['post_spec']:
            post_spec_name=sensitivity_dict['post_spec']
            sensitivity_obj.post_spec=post_workflow_name_to_class_registry[post_spec_name]

        return sensitivity_obj

    

@register_post_workflow
class LocalSensitivityPost:

    """Represent the localsensitivitypost component in the UAM4X workflow.

    Notes
    -----
    This class encapsulates related state and behavior used by UAM4X workflows.

    """
    def __init__(self,tradestudy:Tradestudy):
        """Initialize object state and default configuration.

        Parameters
        ----------
        tradestudy : Tradestudy
            Input value for `tradestudy`.

        """
        self.tradestudy=tradestudy

    def compute_sensitivity(self,output_name,method='forward'):
        """Compute sensitivity.

        Parameters
        ----------
        output_name : Any
            Identifier used by this operation.
        method : Any, optional
            Input value for `method`.

        Returns
        -------
        Any
            Return value produced by this operation.

        """
        trade_name=self.tradestudy.trade_name
        ts=self.tradestudy.study.control.trade_studies[trade_name]
        perturbation_vars=list(ts.variable_to_sensitivity_data.keys())
        #use np.max and np.min to locate
        outs=np.array(self.tradestudy.get_variable(output_name))
        var_to_sens={output_name:{}}
        max_sense=0.0
        max_sense_var=None
        for var_name in perturbation_vars:
            vals=self.tradestudy.get_variable(var_name)
            base=vals[0]
            pert_up=np.max(vals)
            pert_down=np.min(vals)
            out_base=outs[0]
            out_up=outs[np.argmax(vals)]
            out_down=outs[np.argmin(vals)]
            if method=='forward':
                sensitivity=(out_up-out_base)/(pert_up-base)
            elif method=='backward':
                sensitivity=(out_base-out_down)/(base-pert_down)
            elif method=='central':
                sensitivity=(out_up-out_down)/(pert_up-pert_down)
            else:
                raise ValueError(f"Method {method} not recognized. Use 'forward', 'backward', or 'central'.")
            
            if abs(sensitivity)>max_sense:
                max_sense=abs(sensitivity)
                max_sense_var=var_name

            var_to_sens[output_name][var_name]=sensitivity

        max_dict={
            'variable':max_sense_var,
            'sensitivity':max_sense
        }

        
        
        return var_to_sens,max_dict
    