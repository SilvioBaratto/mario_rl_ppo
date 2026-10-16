"""Checks against the trained reference checkpoint and the command line, run on this machine."""

from __future__ import annotations

import pytest

from tests import golden_io, probes

pytestmark = [pytest.mark.local_golden, pytest.mark.emulator]


def test_checkpoint_layout_loads_and_predicts_as_recorded() -> None:
    """State-dict layout, how the loader treats it, and logits for a fixed input."""
    golden_io.assert_matches("checkpoint", probes.checkpoint_contract())


@pytest.mark.slow
def test_greedy_policy_rollouts_match_the_golden() -> None:
    """The reference policy's actions and furthest x per level and NOOP offset."""
    golden_io.assert_matches("policy", probes.policy_behaviour())


@pytest.mark.slow
@pytest.mark.integration
def test_training_smoke_runs_and_resumes_as_recorded() -> None:
    """Two-worker runs (plain and curriculum) and a resume that carries the step counter."""
    golden_io.assert_matches("train_smoke", probes.train_smoke())


@pytest.mark.slow
@pytest.mark.integration
def test_evaluation_cli_reports_as_recorded() -> None:
    """The evaluate command's JSON report for the reference checkpoint on 1-1."""
    golden_io.assert_matches("evaluate_cli", probes.evaluate_cli())
