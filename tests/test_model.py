"""The feature extractor, the shared policy kwargs and checkpoint path resolution."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces

from tests import adapter as a


def test_mario_cnn_encodes_channels_first_frames_to_the_feature_width() -> None:
    """SB3 hands the extractor (batch, 4, 84, 84) floats; it returns (batch, 512)."""
    extractor = a.MarioCNN(spaces.Box(0, 255, (4, 84, 84), np.uint8))
    assert extractor.is_channels_first
    assert extractor(torch.zeros(2, 4, 84, 84)).shape == (2, 512)


def test_mario_cnn_accepts_channels_last_frames() -> None:
    """Channels-last spaces are permuted inside forward()."""
    extractor = a.MarioCNN(spaces.Box(0, 255, (84, 84, 4), np.uint8))
    assert not extractor.is_channels_first
    assert extractor(torch.zeros(2, 84, 84, 4)).shape == (2, 512)


def test_mario_cnn_has_the_layer_names_checkpoints_were_saved_with() -> None:
    """Renaming a layer would make every existing checkpoint fail to load."""
    extractor = a.MarioCNN(spaces.Box(0, 255, (4, 84, 84), np.uint8))
    assert sorted(extractor.state_dict()) == [
        "cnn.0.bias",
        "cnn.0.weight",
        "cnn.2.bias",
        "cnn.2.weight",
        "cnn.4.bias",
        "cnn.4.weight",
        "cnn.6.bias",
        "cnn.6.weight",
        "linear.0.bias",
        "linear.0.weight",
    ]


def test_policy_kwargs_are_the_single_source_for_training_and_loading() -> None:
    """Uint8 observations are normalised on the training device by SB3."""
    kwargs = a.mario_policy_kwargs()
    assert kwargs["features_extractor_class"] is a.MarioCNN
    assert kwargs["features_extractor_kwargs"] == {"features_dim": 512}
    assert kwargs["normalize_images"] is True


def test_resolve_model_path_accepts_a_path_with_or_without_the_zip_suffix(tmp_path) -> None:
    """Checkpoint paths are given both ways on the command line."""
    model = tmp_path / "best_model.zip"
    model.write_bytes(b"")
    assert a.resolve_model_path(tmp_path / "best_model") == model
    assert a.resolve_model_path(model) == model


def test_resolve_model_path_names_both_candidates_when_nothing_exists(tmp_path) -> None:
    """The error shows what was tried, so a typo is easy to spot."""
    with pytest.raises(FileNotFoundError, match="also tried"):
        a.resolve_model_path(tmp_path / "missing")
