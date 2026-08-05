"""Grouping images by how alike they look, not by their bytes.

A perceptual hash (`imaging.inspect`) turns "do these look the same" into "how
many bits differ between two 64-bit integers" — the Hamming distance. This
module only does that comparison and the union-find that turns pairwise
distances into groups; it has no opinion about what a photo is a duplicate
of, what a good threshold is, or what should happen to one once grouped.
"""

from __future__ import annotations

from itertools import combinations


#: A dHash is 64 bits (`imaging._PHASH_WIDTH/_HEIGHT`), so that is the most bits
#: two of them can differ by — the denominator that turns a distance into a
#: percentage anyone can read.
HASH_BITS = 64


def hamming(a: int, b: int) -> int:
    """How many bits differ between two hashes — 0 is identical, 64 is a
    hash's bit-for-bit opposite."""
    return (a ^ b).bit_count()


def similarity(a: int, b: int) -> float:
    """How alike two hashes look, as a percentage: 100.0 is identical.

    The reading a person can act on. A distance of 6 bits out of 64 means
    nothing on its own; "91% alike" is a judgement someone can make. Note it
    measures the *hashes*, so two photos at 100% look the same to this
    comparison — they are not necessarily the same file.
    """
    return round((HASH_BITS - hamming(a, b)) * 100.0 / HASH_BITS, 1)


def cluster(hashes: dict[int, int], max_distance: int,
            excluded_pairs: frozenset[tuple[int, int]] = frozenset()) -> list[list[int]]:
    """Group image ids whose phashes are within `max_distance` bits of each
    other, transitively — if A is close to B and B is close to C, all three
    land in one group even if A and C are not directly close.

    `excluded_pairs` (each a `(smaller_id, larger_id)` tuple) are pairs a user
    has already said are not duplicates. They only stop that one pair from
    merging two images on its own; a third image close to both can still pull
    all three into one group, since a dismissal is a verdict on a pair, not on
    a whole cluster.

    Every pair is compared once — O(n^2), a few seconds for a personal
    library's few thousand photos. A library in the hundreds of thousands
    would want a cheaper pre-filter (bucket by a prefix of the hash) ahead of
    this; not built here yet, since nothing has needed it.
    """
    ids = list(hashes)
    parent = {image_id: image_id for image_id in ids}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        root_x, root_y = find(x), find(y)
        if root_x != root_y:
            parent[root_x] = root_y

    for a, b in combinations(ids, 2):
        if hamming(hashes[a], hashes[b]) > max_distance:
            continue
        pair = (a, b) if a < b else (b, a)
        if pair in excluded_pairs:
            continue
        union(a, b)

    groups: dict[int, list[int]] = {}
    for image_id in ids:
        groups.setdefault(find(image_id), []).append(image_id)
    return [group for group in groups.values() if len(group) > 1]
