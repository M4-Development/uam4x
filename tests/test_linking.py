import numpy as np

from uam4x.linking import uam4xLinkHandler, uam4xVariable


def test_uam4x_variable_parses_optional_indices():
    scalar = uam4xVariable("step.scalar")
    indexed = uam4xVariable("step.array[1:3]")

    assert scalar.cleaned_name == "step.scalar"
    assert scalar.inds is None
    assert scalar.has_inds is False
    assert indexed.cleaned_name == "step.array"
    assert indexed.inds == (slice(1, 3, None),)
    assert indexed.has_inds is True


def test_link_handler_applies_multiple_values_to_existing_target():
    target = np.array([10.0, 20.0, 30.0])
    first = uam4xLinkHandler(
        uam4xVariable("source.a"),
        uam4xVariable("sink.values[0]"),
    )
    second = uam4xLinkHandler(
        uam4xVariable("source.b"),
        uam4xVariable("sink.values[2]"),
    )

    first.apply_to_target(target, 2.0)
    second.apply_to_target(target, 4.0)

    np.testing.assert_array_equal(target, np.array([2.0, 20.0, 4.0]))


def test_link_handler_broadcasts_scalar_to_target_slice():
    target = np.array([10.0, 20.0, 30.0])
    handler = uam4xLinkHandler(
        uam4xVariable("source.value"),
        uam4xVariable("sink.values[1:3]"),
    )

    result = handler.apply_to_target(target, 4.0)

    np.testing.assert_array_equal(result, np.array([10.0, 4.0, 4.0]))


def test_link_handler_returns_source_for_unindexed_target():
    handler = uam4xLinkHandler(
        uam4xVariable("source.value"),
        uam4xVariable("sink.value"),
    )

    assert handler.apply_to_target(None, 3.0) == 3.0


def test_link_handler_gets_indexed_source_from_state():
    handler = uam4xLinkHandler(
        uam4xVariable("source.values[1]"),
        uam4xVariable("sink.value"),
    )

    result = handler.get_source_value(
        state={"source.values": np.array([2.0, 5.0])},
        study=None,
    )

    assert result == 5.0


def test_link_handler_gets_vehicle_source_from_study():
    class StudyStub:
        def __init__(self):
            self.requested_name = None

        def get_value(self, name):
            self.requested_name = name
            return 7.0

    study = StudyStub()
    handler = uam4xLinkHandler(
        uam4xVariable("vehicles.vehicle.component.values[1]"),
        uam4xVariable("sink.value"),
    )

    result = handler.get_source_value(state={}, study=study)

    assert result == 7.0
    assert study.requested_name == "vehicles.vehicle.component.values[1]"
