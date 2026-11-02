import pytest

from uam4x.log import FileLogger, get_log


def test_layer_keeps_actor_context_for_one_message(tmp_path):
    path = tmp_path / "events.log"
    root = FileLogger(filename_log=str(path), write_to_stdout=False)
    try:
        layer = get_log(root, object_name="motor")
        layer.info("started")
        root.info("continued")
        assert get_log(layer).object_name == "motor"

        text = path.read_text()
        assert "motor - started" in text
        assert "continued" in text
        assert "motor - continued" not in text
        assert "FileLogger(" in repr(root)
    finally:
        root.close_file()


def test_capture_level_and_lifecycle_do_not_require_file():
    logger = FileLogger(write_to_stdout=False)
    logger.begin_log_capture(capture_level_min="WARNING")
    logger.info("ignored")
    logger.warning("captured")

    assert logger.get_capture_log() == [("WARNING", "captured")]
    logger.begin_log_capture()
    assert any("without ending" in message
               for level, message in logger.get_capture_log() if level == "ERROR")
    captured = logger.end_log_capture()
    logger.error("after capture")
    assert logger.get_capture_log() == captured


def test_file_and_console_thresholds_validate_levels(tmp_path):
    logger = FileLogger(write_to_stdout=False)
    path = tmp_path / "threshold.log"
    try:
        logger.set_filename_log(str(path))
        logger.set_min_filelog_level("ERROR")
        logger.info("not written")
        logger.error("written")
        logger.set_min_console_level("CRITICAL")

        assert "not written" not in path.read_text()
        assert "written" in path.read_text()
        with pytest.raises(AssertionError):
            logger.set_min_filelog_level("UNKNOWN")
        with pytest.raises(AssertionError):
            logger.set_min_console_level("UNKNOWN")
    finally:
        logger.close_file()
