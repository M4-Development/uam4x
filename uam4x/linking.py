from uam4x.utils import split_dotted_path_and_indices, parse_variable_reference
import numpy as np

#we can define a variable class which has the name and resolved indices?


class uam4xVariable:
    def __init__(self,full_name):
        self.full_name=full_name

        self.cleaned_name,self.inds=parse_variable_reference(self.full_name)

    @property
    def has_inds(self):
        if self.inds is not None:
            return True
        return False


class uam4xLinkHandler:
    def __init__(self,independent,dependent):
        self.independent=independent
        self.dependent=dependent

    def get_source_value(self,state,study):
        if self.independent.cleaned_name in state:
            source_value=state[self.independent.cleaned_name]
            if self.independent.has_inds:
                source_value=source_value[self.independent.inds]
            return source_value
        else:
            return study.get_value(self.independent.full_name)

    def apply_to_target(self,target_value,source_value):
        if not self.dependent.has_inds:
            return source_value
        target_value[self.dependent.inds]=source_value

        return target_value








