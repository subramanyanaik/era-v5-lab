"""Theorem 1 on a real tokenizer.json (byte-level BPE, e.g. BrahmicTokenizer-131K). See proof.md section 4.

usage: python real_tokenizer_check.py <dir with tokenizer.json and special_tokens_map.json> [out.json]

Builds two tables and runs the codec on both, for P = 32 and 64 in fp32 and bf16:
  raw     the real vocabulary bytes (GPT-2 byte-to-unicode map inverted)
  decode  what the repository's token_id_to_bytes feeds (decode([id]), real `tokenizers` decoder)
For each: distinct codes, whether the groups by code equal the groups by cut string (Theorem 1),
unreachable tokens, and the cause of every colliding pair. Needs torch, tokenizers and kronecker-embeddings.
The tokenizer files are never part of this repository.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import torch
from forward import code_digest, codes, pair_cause, truncate
from tokenizers import Tokenizer


def bytes_to_unicode() -> dict[int, str]:
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def load_tables(tok_dir: Path) -> tuple[list[bytes], list[bytes], dict]:
    """Return (raw, fed, info). fed comes from the repository's token_id_to_bytes."""
    from kronecker_embeddings.tokenizer_utils import token_id_to_bytes

    tk = Tokenizer.from_file(str(tok_dir / "tokenizer.json"))
    vocab = tk.get_vocab(with_added_tokens=True)
    n = max(vocab.values()) + 1
    inv = {c: b for b, c in bytes_to_unicode().items()}
    raw: list[bytes | None] = [None] * n
    for s, i in vocab.items():
        raw[i] = bytes(inv[c] for c in s) if all(c in inv for c in s) else s.encode("utf-8")
    assert all(r is not None for r in raw), "gaps in the id range"

    smap = json.loads((tok_dir / "special_tokens_map.json").read_text(encoding="utf-8"))

    class Adapter:  # the members of a transformers tokenizer that token_id_to_bytes uses
        all_special_ids = [vocab[t] for t in smap.values() if isinstance(t, str)]

        def convert_ids_to_tokens(self, i):
            return tk.id_to_token(i)

        def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
            return tk.decode(ids, skip_special_tokens=skip_special_tokens)

    ad = Adapter()
    fed = [token_id_to_bytes(ad, i) for i in range(n)]
    h = hashlib.sha256()
    for i, r in enumerate(raw):
        h.update(i.to_bytes(4, "big") + len(r).to_bytes(2, "big") + r)
    info = {
        "vocab": n,
        "tokenizer_json_sha256": hashlib.sha256((tok_dir / "tokenizer.json").read_bytes()).hexdigest(),
        "raw_table_sha256": h.hexdigest(),
        "max_bytes": max(len(r) for r in raw),
    }
    return raw, fed, info


def check_table(raw: list[bytes], fed: list[bytes], P: int) -> dict:
    cuts = [truncate(f, P) for f in fed]
    out: dict = {}
    for dtype in (torch.float32, torch.bfloat16):
        by_code = defaultdict(list)
        for s in range(0, len(cuts), 1000):
            for k, z in enumerate(codes(cuts[s : s + 1000], P, dtype)):
                by_code[code_digest(z)].append(s + k)
        by_cut = defaultdict(list)
        for i, c in enumerate(cuts):
            by_cut[c].append(i)
        out[str(dtype).split(".")[-1]] = {
            "distinct_codes": len(by_code),
            "distinct_cut_strings": len(by_cut),
            "partitions_equal": {frozenset(g) for g in by_code.values()} == {frozenset(g) for g in by_cut.values()},
        }
        if dtype is torch.float32:
            groups = [g for g in by_code.values() if len(g) > 1]
    causes: Counter = Counter()
    for g in groups:
        for a, b in combinations(g, 2):
            causes[pair_cause(raw[a], raw[b], fed[a], fed[b], P)] += 1
    out.update(
        groups=len(groups),
        unreachable=sum(len(g) - 1 for g in groups),
        largest=sorted((len(g) for g in groups), reverse=True)[:5],
        pair_causes=dict(causes),
    )
    return out


def run(tok_dir: Path) -> dict:
    raw, fed, info = load_tables(tok_dir)
    info["fed_equals_raw"] = sum(1 for r, f in zip(raw, fed) if r == f)
    for P in (32, 64):
        info[f"P{P}_raw"] = check_table(raw, raw, P)
        info[f"P{P}_decode"] = check_table(raw, fed, P)
    return info


if __name__ == "__main__":
    res = run(Path(sys.argv[1]))
    text = json.dumps(res, indent=2)
    print(text)
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(text, encoding="utf-8")
