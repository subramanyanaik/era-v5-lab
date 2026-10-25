"""Theorem 1: two tokens share a code exactly when their cut byte strings are equal (proof.md).

The corollary: every pair of tokens that share a code has a named cause.
The vocabulary here is synthetic and built to contain every cause. The same check runs on the real
tokenizer in build_tables.py once the vocabulary is available.
"""

import random
from collections import defaultdict
from itertools import combinations

import torch
from forward import CAUSES, code_digest, codes, decode_style_bytes, pair_cause, truncate

PS = (32, 64)
POOL = list("aZ5-_ ") + ["é", "ñ", "ಕ", "ನ", "್", "ಡ", "ಾ", "ಿ", "ि", "र", "ा", "日", "😀"]


def random_word(rng):
    return "".join(rng.choice(POOL) for _ in range(rng.randrange(1, 26))).encode("utf-8")


def synthetic_vocab(rng):
    """List of (raw, fed). fed is what the table gives the codec."""
    vocab = []

    def add(raw, fed=None):
        vocab.append((raw, raw if fed is None else fed))

    for _ in range(1500):
        add(random_word(rng))                                       # ordinary tokens
    for _ in range(30):
        base = ("ಕನ್ನಡ" * rng.randrange(10, 14)).encode("utf-8")      # long, same start: truncation collisions
        for _ in range(4):
            add(base + random_word(rng))
    for n in range(33, 75):
        add(b"-" * n)                                               # runs of one character
    for _ in range(20):
        w = random_word(rng)
        add(w)
        add(w)                                                      # exact duplicates
    add(b"<s>")
    add(b"<s>")                                                     # special token and ordinary text, same bytes
    for frag in (b"\xe0\xb2", b"\x95", b"\xa8", b"\xb2\x95", b"\xe0"):
        add(frag, decode_style_bytes(frag))                         # partial letters: decode gives U+FFFD
    add(b"")
    add(b"")                                                        # empty tokens
    return vocab


def test_shared_codes_are_exactly_equal_cut_strings_and_every_pair_has_a_cause():
    vocab = synthetic_vocab(random.Random(7))
    seen = set()
    for P in PS:
        cuts = [truncate(fed, P) for _, fed in vocab]
        for dtype in (torch.float32, torch.bfloat16):
            digests = []
            for i in range(0, len(cuts), 500):
                digests += [code_digest(z) for z in codes(cuts[i:i + 500], P, dtype)]
            by_code, by_cut = defaultdict(set), defaultdict(set)
            for i, (d, c) in enumerate(zip(digests, cuts)):
                by_code[d].add(i)
                by_cut[c].add(i)
            # Theorem 1: the two partitions of the vocabulary are the same
            assert {frozenset(g) for g in by_code.values()} == {frozenset(g) for g in by_cut.values()}, (P, dtype)
            assert len(by_code) == len(by_cut)
            if dtype is torch.float32:
                for g in by_code.values():
                    for a, b in combinations(sorted(g), 2):
                        cause = pair_cause(vocab[a][0], vocab[b][0], vocab[a][1], vocab[b][1], P)
                        assert cause in CAUSES, (P, a, b)
                        seen.add(cause)
    assert seen == set(CAUSES)  # the synthetic vocabulary really contains every cause


def test_pair_cause_is_none_when_codes_differ():
    assert pair_cause(b"ab", b"ac", b"ab", b"ac", 32) is None
    assert pair_cause(b"a" * 40, b"a" * 40 + b"b", b"a" * 40, b"a" * 40 + b"b", 32) == "truncation"
    assert pair_cause(b"<s>", b"<s>", b"<s>", b"<s>", 32) == "same_raw"
    assert pair_cause(b"\xe0\xb2", b"\x95", "�".encode(), "�".encode(), 32) == "bad_bytes_in"
    assert pair_cause(b"", b"", b"", b"", 32) == "empty"
