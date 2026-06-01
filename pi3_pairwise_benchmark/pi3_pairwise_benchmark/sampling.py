from __future__ import annotations

import random

from .metrics import pairwise_indices


def _stable_rng(seed: int, sequence_key: str, policy: str) -> random.Random:
    return random.Random(f"{int(seed)}:{sequence_key}:{policy}")


def _filter_by_gap(
    pairs: list[tuple[int, int]],
    gap_min: int,
    gap_max: int | None,
) -> list[tuple[int, int]]:
    out = []
    for i, j in pairs:
        gap = j - i
        if gap < gap_min:
            continue
        if gap_max is not None and gap > gap_max:
            continue
        out.append((i, j))
    return out


def _sample_pairs(
    pairs: list[tuple[int, int]],
    max_pairs: int | None,
    rng: random.Random,
) -> list[tuple[int, int]]:
    if max_pairs is None or max_pairs >= len(pairs):
        return pairs
    if max_pairs <= 0:
        raise ValueError(f"max_pairs must be positive when set, got {max_pairs}")
    return sorted(rng.sample(pairs, max_pairs))


def select_pairs(
    num_frames: int,
    policy: str = "all",
    max_pairs: int | None = None,
    seed: int = 42,
    sequence_key: str = "",
    short_gap_min: int = 1,
    short_gap_max: int = 5,
    medium_gap_min: int = 10,
    medium_gap_max: int = 20,
    long_gap_min: int = 30,
    long_gap_max: int | None = None,
) -> list[tuple[int, int]]:
    """Select unordered frame pairs for pairwise 2-view inference/evaluation.

    The `all` policy exactly matches Pi3's multi-view metric convention:
    unordered combinations with i < j.
    """
    all_pairs = pairwise_indices(num_frames)
    policy = policy.lower()
    rng = _stable_rng(seed=seed, sequence_key=sequence_key, policy=policy)

    if policy == "all":
        return _sample_pairs(all_pairs, max_pairs=max_pairs, rng=rng)
    if policy == "short":
        candidates = _filter_by_gap(all_pairs, short_gap_min, short_gap_max)
    elif policy == "medium":
        candidates = _filter_by_gap(all_pairs, medium_gap_min, medium_gap_max)
    elif policy == "long":
        candidates = _filter_by_gap(all_pairs, long_gap_min, long_gap_max)
    elif policy == "mixed":
        buckets = [
            _filter_by_gap(all_pairs, short_gap_min, short_gap_max),
            _filter_by_gap(all_pairs, medium_gap_min, medium_gap_max),
            _filter_by_gap(all_pairs, long_gap_min, long_gap_max),
        ]
        if any(not bucket for bucket in buckets):
            raise ValueError(f"No pairs available for one or more mixed policy buckets with {num_frames} frames")
        if max_pairs is None:
            candidates = sorted({pair for bucket in buckets for pair in bucket})
        else:
            per_bucket = max(1, max_pairs // len(buckets))
            remainder = max_pairs - per_bucket * len(buckets)
            chosen: list[tuple[int, int]] = []
            for idx, bucket in enumerate(buckets):
                take = per_bucket + (1 if idx < remainder else 0)
                chosen.extend(_sample_pairs(bucket, min(take, len(bucket)), rng))
            candidates = sorted(set(chosen))
    else:
        raise ValueError(f"Unknown pair policy: {policy}")

    if not candidates:
        raise ValueError(f"No pairs available for policy={policy!r} with {num_frames} frames")
    return _sample_pairs(candidates, max_pairs=max_pairs, rng=rng)
