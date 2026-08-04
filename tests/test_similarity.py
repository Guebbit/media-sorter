"""Hamming distance and clustering — pure logic, no I/O."""

from __future__ import annotations

from app.domain.similarity import cluster, hamming, similarity


def test_hamming_of_identical_hashes_is_zero():
    assert hamming(0x1234, 0x1234) == 0


def test_hamming_counts_differing_bits():
    assert hamming(0b0000, 0b1111) == 4
    assert hamming(0b1010, 0b1000) == 1


def test_cluster_groups_transitively_even_when_the_ends_are_far_apart():
    # 1 -> 2 (distance 1), 2 -> 4 (distance 1), but 1 -> 4 is distance 2:
    # still one group, because closeness chains through 2.
    hashes = {1: 0b00, 2: 0b01, 3: 0xFFFFFFFFFFFFFFFF, 4: 0b11}
    groups = cluster(hashes, max_distance=1)
    assert groups == [[1, 2, 4]]


def test_cluster_excludes_singletons():
    hashes = {1: 0, 2: 0xFFFFFFFFFFFFFFFF}
    assert cluster(hashes, max_distance=10) == []


def test_cluster_respects_a_tighter_distance():
    hashes = {1: 0, 2: 0b11}
    assert cluster(hashes, max_distance=1) == []
    assert cluster(hashes, max_distance=2) == [[1, 2]]


def test_dismissed_pair_does_not_merge_on_its_own():
    hashes = {1: 0, 2: 1, 4: 0b11}
    groups = cluster(hashes, max_distance=1, excluded_pairs=frozenset({(1, 2)}))
    # 1-2 is dismissed and not linked any other way, but 2-4 still is.
    assert groups == [[2, 4]]


def test_dismissed_pair_can_still_be_bridged_by_a_third_image():
    # A dismissal is a verdict on one pair, not on the whole cluster: a third
    # image close to both 1 and 2 still pulls all three together.
    hashes = {1: 0, 2: 0b1, 3: 0b1}
    groups = cluster(hashes, max_distance=1, excluded_pairs=frozenset({(1, 2)}))
    assert groups == [[1, 2, 3]]


def test_cluster_needs_at_least_two_hashes():
    assert cluster({}, max_distance=10) == []
    assert cluster({1: 0}, max_distance=10) == []


# ------------------------------------------------------------- similarity %


def test_identical_hashes_are_100_percent():
    assert similarity(0xDEADBEEF, 0xDEADBEEF) == 100.0


def test_opposite_hashes_are_0_percent():
    assert similarity(0, (1 << 64) - 1) == 0.0


def test_one_differing_bit_is_just_under_100():
    # 63 of 64 bits agree.
    assert similarity(0b0, 0b1) == round(63 * 100 / 64, 1)


def test_the_percentage_falls_as_the_distance_grows():
    scores = [similarity(0, (1 << n) - 1) for n in range(0, 65, 8)]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == 100.0
