from typing import TYPE_CHECKING, Any, Mapping

from uam4x.utils import split_dotted_path_and_indices, parse_variable_reference
import numpy as np

if TYPE_CHECKING:
    from uam4x.study import Study

#we can define a variable class which has the name and resolved indices?


class uam4xVariable:
    """Represent a variable path with an optional index or slice.

    ``full_name`` retains the original path. ``cleaned_name`` omits any
    trailing index expression, and ``inds`` holds the parsed index or
    ``None``.
    """

    def __init__(self, full_name: str) -> None:
        """Parse and store a full variable path and its optional index."""
        self.full_name=full_name

        self.cleaned_name,self.inds=parse_variable_reference(self.full_name)

    @property
    def has_inds(self) -> bool:
        """Whether the variable path includes an index expression."""
        if self.inds is not None:
            return True
        return False


class uam4xLinkHandler:
    """Read a link's source value and assign it to its target."""

    def __init__(
        self, independent: uam4xVariable, dependent: uam4xVariable
    ) -> None:
        """Store source and target variable references for a link."""
        self.independent=independent
        self.dependent=dependent

    def get_source_value(
        self, state: Mapping[str, Any], study: "Study"
    ) -> Any:
        """Read the source from ``state``, or fall back to the study.

        When the source is in ``state`` and has an index expression, return
        only the indexed value.
        """
        if self.independent.cleaned_name in state:
            source_value=state[self.independent.cleaned_name]
            if self.independent.has_inds:
                source_value=source_value[self.independent.inds]
            return source_value
        else:
            return study.get_value(self.independent.full_name)

    def apply_to_target(
        self, target_value: Any, source_value: Any
    ) -> Any:
        """Apply the source value to the target.

        Return ``source_value`` for an unindexed target. For an indexed
        target, update ``target_value`` in place and return it.
        """
        if not self.dependent.has_inds:
            return source_value
        target_value[self.dependent.inds]=source_value

        return target_value








