"""The run logger."""

from __future__ import annotations

import logging

import pytest

from tests import adapter as a

NAME = "mario_rl_test"


@pytest.fixture
def clean_logger():
    """Drop the handlers a test attached, so loggers do not leak between tests."""
    yield
    logger = logging.getLogger(NAME)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def test_setup_logger_writes_debug_to_a_file_and_info_to_stdout(
    tmp_path, capsys, clean_logger
) -> None:
    """The file keeps everything; the console shows INFO and above."""
    logger = a.setup_logger(tmp_path / "logs", name=NAME)
    logger.debug("only in the file")
    logger.info("on both")
    for handler in logger.handlers:
        handler.flush()
    (log_file,) = (tmp_path / "logs").glob("training_*.log")
    text = log_file.read_text()
    assert "only in the file" in text and "on both" in text
    out = capsys.readouterr().out
    assert "INFO - on both" in out
    assert "only in the file" not in out


def test_setup_logger_creates_the_log_directory(tmp_path, clean_logger) -> None:
    """Nested directories are created on demand."""
    a.setup_logger(tmp_path / "deep" / "logs", name=NAME)
    assert (tmp_path / "deep" / "logs").is_dir()
