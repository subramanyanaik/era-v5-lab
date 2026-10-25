"""Lemma 2: the cut to P bytes is a prefix, is bounded, and keeps whole UTF-8 letters (docs/proof.md)."""

import random

from forward import truncate
from kronecker_embeddings.tokenizer_utils import utf8_safe_truncate

PS = (32, 64)
CHARS = list("aZ5-_ ") + ["é", "ñ", "ß", "ಕ", "ನ", "್", "ಡ", "ಾ", "ಿ", "ि", "र", "ा", "日", "本", "😀", "—", "‍"]


def is_valid(b: bytes) -> bool:
    try:
        b.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def random_valid(rng, max_chars=40):
    return "".join(rng.choice(CHARS) for _ in range(rng.randrange(1, max_chars))).encode("utf-8")


def random_raw(rng):
    """Arbitrary bytes, with many continuation bytes (0x80-0xBF) to stress the rule."""
    n = rng.randrange(0, 200)
    return bytes(rng.randrange(0x80, 0xC0) if rng.random() < 0.5 else rng.randrange(256) for _ in range(n))


def test_prefix_identity_length_bounds_and_idempotence_on_raw_bytes():
    rng = random.Random(0)
    for P in PS:
        for _ in range(5000):
            raw = random_raw(rng)
            t = truncate(raw, P)
            assert raw.startswith(t)                          # (a) prefix
            if len(raw) <= P:
                assert t == raw                               # (b) unchanged if it fits
            else:
                assert P - 3 <= len(t) <= P                   # (c) loses at most 3 bytes
            assert truncate(t, P) == t                        # idempotent


def test_valid_utf8_stays_valid_and_matches_the_repository_rule():
    rng = random.Random(1)
    for P in PS:
        long_seen = 0
        for _ in range(5000):
            raw = random_valid(rng)
            t = truncate(raw, P)
            assert is_valid(t)                                # (d) no letter split
            assert t == utf8_safe_truncate(raw, P)            # same as the repository on valid input
            long_seen += len(raw) > P
        assert long_seen > 500  # the test really exercises the cut


def test_a_non_empty_token_never_becomes_empty():
    rng = random.Random(2)
    for P in PS:
        for _ in range(5000):
            raw = random_raw(rng)
            if raw:
                assert truncate(raw, P)


def test_repository_rule_empties_a_partial_start_token_but_ours_does_not():
    # A token that starts with the last byte of a letter. No prefix of it is valid UTF-8,
    # so the repository rule returns b"" and the token would get the all-zero code.
    # If the first assertion fails one day, upstream fixed the rule: update docs/proof.md.
    raw = b"\x95" + "ನ್ನಡ".encode() * 6
    assert len(raw) == 73
    assert utf8_safe_truncate(raw, 32) == b""
    t = truncate(raw, 32)
    assert t and raw.startswith(t) and 29 <= len(t) <= 32
