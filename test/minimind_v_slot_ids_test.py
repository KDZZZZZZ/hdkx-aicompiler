"""Host placement contract of the MiniMind-V slot prefill."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "tools"))
import minimind_v_slots  # noqa: E402
from minimind_v_slots import (IMAGE_TOKENS, LAYOUTS, MARKER, SEQUENCE_MAX, SLOT_IMAGES,  # noqa: E402
                              VOCAB, layout_ids, slot_ids)


def image(start: int, length: int = IMAGE_TOKENS, total: int = 70) -> np.ndarray:
    ids = np.full((1, total), 100, dtype=np.int64)
    ids[0, start:start + length] = MARKER
    return ids


def test_each_layout_maps_runs_to_consecutive_slots():
    rng = np.random.default_rng(0)
    for layout in LAYOUTS:
        ids = layout_ids(rng, layout)
        remapped, images = slot_ids(ids)
        assert images == len(layout) - 1
        markers = ids[0] == MARKER
        np.testing.assert_array_equal(remapped[0, ~markers], ids[0, ~markers])
        np.testing.assert_array_equal(remapped[0, markers], VOCAB + np.arange(images * IMAGE_TOKENS))
        assert ids.shape[1] <= SEQUENCE_MAX


def test_placement_is_positional_not_fixed():
    early, _ = slot_ids(image(0))
    late, _ = slot_ids(image(6))
    assert early[0, 0] == VOCAB and late[0, 6] == VOCAB and late[0, 0] == 100


@pytest.mark.parametrize("ids, message", [
    (image(3, IMAGE_TOKENS - 1), "contiguous run"),
    (image(3, IMAGE_TOKENS + 1, 80), "contiguous run"),
    (np.full((1, 3), VOCAB, dtype=np.int64), "vocabulary"),
    (np.full((1, 3), -1, dtype=np.int64), "vocabulary"),
    (np.full((1, SEQUENCE_MAX + 1), 100, dtype=np.int64), "1 <= S"),
    (np.zeros((1, 0), dtype=np.int64), "1 <= S"),
    (np.full((2, 4), 100, dtype=np.int64), "int64 \\[1,S\\]"),
    (np.full((1, 4), 100, dtype=np.int32), "int64 \\[1,S\\]"),
])
def test_invalid_placement_is_rejected(ids, message):
    with pytest.raises(ValueError, match=message):
        slot_ids(ids)


def test_more_images_than_slots_is_rejected(monkeypatch):
    ids = np.full((1, (SLOT_IMAGES + 1) * (IMAGE_TOKENS + 1) + 1), 100, dtype=np.int64)
    for index in range(SLOT_IMAGES + 1):
        start = 1 + index * (IMAGE_TOKENS + 1)
        ids[0, start:start + IMAGE_TOKENS] = MARKER
    # Four images exceed the S bound first; a widened bound reaches the slot check.
    with pytest.raises(ValueError, match="1 <= S"):
        slot_ids(ids)
    monkeypatch.setattr(minimind_v_slots, "SEQUENCE_MAX", ids.shape[1])
    with pytest.raises(ValueError, match="at most"):
        slot_ids(ids)
