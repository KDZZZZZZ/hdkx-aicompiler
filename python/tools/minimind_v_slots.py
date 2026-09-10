"""Host placement contract for the MiniMind-V slot prefill.

Each image is one contiguous run of 64 marker tokens. The host scan maps the
k-th run to rows ``6400 + 64 * k + [0, 64)`` of the table
``Concat(embed_tokens.weight, visual_slots)``. The compiled Gather zero-fills
out-of-range indices, so this check must run before any launch.
"""
from __future__ import annotations

import numpy as np

VOCAB = 6400
MARKER = 12
IMAGE_TOKENS = 64
SLOT_IMAGES = 3
SLOT_ROWS = SLOT_IMAGES * IMAGE_TOKENS
SEQUENCE_MAX = 224
CAPACITY = 240
SENTINEL = 7.0
# Each layout alternates text runs with images; an image is one 64-marker run.
LAYOUTS = [[9], [2, 2], [5, 3, 7], [1, 1, 4, 9]]


def slot_ids(ids: np.ndarray) -> tuple[np.ndarray, int]:
    """Replace each 64-marker run with its slot rows; reject any other layout."""
    if ids.dtype != np.int64 or ids.ndim != 2 or ids.shape[0] != 1:
        raise ValueError("slot prefill requires int64 [1,S] token ids")
    if not 1 <= ids.shape[1] <= SEQUENCE_MAX:
        raise ValueError(f"slot prefill requires 1 <= S <= {SEQUENCE_MAX}")
    if (ids < 0).any() or (ids >= VOCAB).any():
        raise ValueError("token ids must lie in the text vocabulary")
    row, result, images, index = ids[0], ids.copy(), 0, 0
    while index < row.size:
        if row[index] != MARKER:
            index += 1
            continue
        start = index
        while index < row.size and row[index] == MARKER:
            index += 1
        if index - start != IMAGE_TOKENS:
            raise ValueError("every image must be one contiguous run of 64 markers")
        if images == SLOT_IMAGES:
            raise ValueError(f"at most {SLOT_IMAGES} images fit the visual slots")
        result[0, start:index] = VOCAB + images * IMAGE_TOKENS + np.arange(IMAGE_TOKENS)
        images += 1
    return result, images


def layout_ids(rng: np.random.Generator, layout: list[int]) -> np.ndarray:
    pieces = []
    for index, count in enumerate(layout):
        if index:
            pieces.append(np.full(IMAGE_TOKENS, MARKER, dtype=np.int64))
        pieces.append(rng.integers(13, VOCAB, count, dtype=np.int64))
    return np.concatenate(pieces).reshape(1, -1)
