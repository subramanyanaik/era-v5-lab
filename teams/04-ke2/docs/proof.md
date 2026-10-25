# KE2 week 1: forward encode proof

Lemma 1, Lemma 2 and Theorem 1 for the Kronecker codec, with numerical checks.
Tests: `pytest teams/04-ke2/tests -q` (needs the kronecker-embeddings package, see the end).

**Provenance.** The codec is the one in https://github.com/theschoolofai/kronecker-embeddings
(`kronecker_embeddings/codec.py`, V1 paper arXiv:2605.29459, section 3). The statements, proofs and the cut rule in
Lemma 2 are written for this lab. The paper states no lemmas. The statements are proposals and are open to change.

## 0. Setup and notation

- P is 32 or 64 (`pos_dim`). The alphabet has 256 byte values, so a code has D = 256P entries.
- A token has raw bytes r (its real vocabulary bytes). The codec sees the cut string s = tau_P(r), with |s| = L <= P.
  |s| is the number of bytes of s.
- Code of s = s_0 ... s_{L-1}, as in `kronecker_codec`:
  1. Make a vector x of length D with x[s_p * P + p] = 1/sqrt(L) for p = 0 .. L-1, and 0 elsewhere.
  2. z-normalise: z = (x - mean(x)) / (std(x) + eps), with the unbiased std and eps = 1e-6.
- Cell i of the vector is byte i div P at position i mod P.
- For the empty string (L = 0) x = 0, so z = 0.

## 1. Lemma 1: the codec is injective

**Statement.** Let s, s' be byte strings with |s| <= P and |s'| <= P. If code(s) = code(s'), then s = s'.
Also, the code of a non-empty string is never the all-zero code.

**Proof.** Take a non-empty s with L bytes, 1 <= L <= P. Then L < D.

1. The L lit cells are distinct, because cell s_p * P + p has remainder p when divided by P, so no two share a position.
   Exactly L entries of x are 1/sqrt(L) and D - L > 0 entries are 0.
2. mean(x) = m = sqrt(L)/D. The sum of squares of x is 1, so the unbiased variance is v = (1 - L/D) / (D - 1) > 0.
   Let sigma = sqrt(v) + eps > 0.
3. So z has exactly two values:
   high h = (1/sqrt(L) - m) / sigma on the lit cells, low u = -m / sigma on all other cells.
   Their difference is h - u = 1 / (sqrt(L) * sigma) > 0, so h > u. In particular z is not the zero vector.
4. The lit cells are the cells with value h, found as the cells above (h + u)/2. In each position p = 0 .. L-1 there is
   exactly one lit cell i, and the byte at p is i div P. The number of lit cells is L.
5. So s can be computed from z alone. If code(s) = code(s'), both strings are computed from the same vector, so s = s'.
6. The empty string gives z = 0 (section 0), and every non-empty string gives a non-zero z (step 3), so they never agree.
   Two non-empty strings of different length differ in the number of lit cells, so they are covered by step 5. QED

**Remark (floating point).** Rounding to bf16 never swaps h and u, because rounding is monotone. It could merge them only
if h - u were smaller than the bf16 spacing. Since sigma <= 1/sqrt(D - 1) + eps, h - u is at least 0.999 * sqrt((D - 1)/L),
which is at least 15.9 for both P = 32 and P = 64 (the eps term changes it by less than 0.02%). The spacing of bf16 at
values of that size is at most 0.5.

**Numerical checks** (`tests/test_lemma1.py`):

| Check | Size | Result |
|---|---|---|
| exactly 2 distinct values, fp32 and bf16 | P = 32, 64; every L from 1 to P; 3 random strings each | pass |
| the two values equal the closed form h, u of step 3 (relative error 1e-3) | P = 32, 64; every L | pass |
| gap h - u after rounding to bf16 | P = 32, 64; every L | > 8 in all cases (the smallest seen earlier was 16.0) |
| bytes read back from the code equal the input, fp32 and bf16 | edge strings (`00`, `ff`, length P, all bytes) and 3,000 random strings per P | pass |
| empty string gives the zero code, no non-empty string does | 200 random strings per P | pass |
| distinct strings give distinct codes, fp32 and bf16 | at least 20,000 distinct strings per P, including near neighbours (one byte changed, two swapped, one added) | pass |

## 2. Lemma 2: the cut to P is safe and predictable

**The rule tau_P(r).**
1. If |r| <= P, return r.
2. Otherwise try c = P, P-1, P-2, P-3 in that order. Return r[0:c] for the first c where the byte r[c] is not a
   continuation byte (a byte in 0x80 .. 0xBF).
3. If none of the four works, return r[0:P].

**Statement.** For every byte string r and P >= 4:
(a) tau_P(r) is a prefix of r.
(b) If |r| <= P then tau_P(r) = r.
(c) If |r| > P then P - 3 <= |tau_P(r)| <= P. In particular a non-empty token is never cut to nothing.
(d) If r is valid UTF-8, then tau_P(r) is valid UTF-8, so no letter is split.
(e) tau_P(tau_P(r)) = tau_P(r).

**Proof.** (a) and (b): we only return r[0:c] with c <= |r|, or all of r. (c): c is one of P, P-1, P-2, P-3. (e): the
result has length <= P, so rule 1 returns it unchanged.
(d): let r be valid UTF-8 with |r| > P. A letter has at most 4 bytes: a lead byte, then at most 3 continuation bytes.
If r[P] is not a continuation byte, then c = P is chosen, and a cut just before a lead byte or an ASCII byte falls
between letters. If r[P] is a continuation byte, then the letter that contains it starts at some q with P - 3 <= q < P,
and the bytes r[q+1] .. r[P] are all continuation bytes. Going down from P, the first position that is not a continuation
byte is q, so c = q, and the cut again falls between letters. QED

**Why not the repository rule.** `utf8_safe_truncate` keeps the first of the four prefixes that decodes as valid
UTF-8, else returns `b""`. On valid UTF-8 it returns the same result as tau_P (a prefix is valid exactly when it ends
at a letter boundary). On a token that is not valid UTF-8, such as one that starts with the last byte of a letter,
no prefix is valid, so it returns `b""` and the token gets the all-zero code.

**Note on the real vocabulary.** BrahmicTokenizer-131K has no ordinary token longer than 32 bytes (longest: 32; special tokens: at most 20),
so for P = 32 and P = 64 the cut rule never fires on it and tau_P(r) = r for every token (section 4). The lemma is still needed for
smaller P, for other tokenizers, and for the provisional tokenizer from G3 (due 28 Oct), whose longest token is not yet known.

**Numerical checks** (`tests/test_lemma2.py`):

| Check | Size | Result |
|---|---|---|
| (a), (b), (c), (e) on raw bytes, half of them continuation bytes | P = 32, 64; 5,000 strings each, lengths 0 to 199 | pass |
| (d) and equal to the repository rule on valid UTF-8 (Latin, accents, Kannada, Devanagari, CJK, emoji, ZWJ) | P = 32, 64; 5,000 strings each (more than 500 longer than P) | pass |
| a non-empty token never becomes empty | P = 32, 64; 5,000 strings each | pass |
| repository rule returns empty for a 73-byte token that starts with `95`; this rule keeps 29 to 32 bytes | one token, P = 32 | pass |

## 3. Theorem 1: the codec never causes a collision

**Statement.** Let r, r' be the byte strings that a table feeds to the codec for two tokens (the raw vocabulary bytes in the
raw byte table). Then code(tau_P(r)) = code(tau_P(r')) if and only if tau_P(r) = tau_P(r').

**Proof.** (if) The code is a function of the string. (only if) By Lemma 2 (c), both cut strings have at most P bytes,
so Lemma 1 applies. QED

**Scope.** The theorem is about the strings that a table feeds to the codec. It does not say what a given way of building the table
feeds. If the table is built with `decode([id])`, partial-letter tokens are fed as U+FFFD and collide (section 4). That is a
property of the table, not of the codec, and it is why the raw byte table uses the real vocabulary bytes.

**Corollary (causes).** Two different tokens share a code only if the strings fed to the codec become equal after the cut.
For a pair of tokens, with raw bytes r, r' and fed bytes f, f' (equal to r, r' in the raw byte table), the reason is
exactly one of:

| Cause | Condition |
|---|---|
| `empty` | the cut string is empty (an empty token; the cut never empties a non-empty token) |
| `same_raw` | r = r' (duplicate surface forms, a special token with the text of an ordinary token, an agreed marker policy) |
| `bad_bytes_in` | r != r' but f = f' (for example U+FFFD from `decode([id])`; not possible in the raw byte table) |
| `truncation` | f != f' but tau_P(f) = tau_P(f') (both longer than P with the same start) |

A group of more than two tokens that share a code can mix these causes, one per pair.

**Numerical checks** (`tests/test_theorem1.py`), on a synthetic vocabulary of about 1,800 tokens that contains every cause
(families that share a long start, runs of `-` of length 33 to 74, exact duplicates, `<s>` twice, partial letters with
U+FFFD bytes, empty tokens):

| Check | Result |
|---|---|
| the grouping of tokens by code equals the grouping by cut string, P = 32 and 64, fp32 and bf16 | pass |
| every pair of tokens that share a code gets one of the four causes, and all four causes occur | pass |
| `pair_cause` on small hand-made cases | pass |

All 12 tests: `12 passed in 23 s` (CPU, torch).

## 4. The real tokenizer, and what this does not cover

**BrahmicTokenizer-131K** (`tokenizer.json` sha256 `01d968a7970b029c071b29f060abf6104b66168dedf2ff3b1a0290a545739fff`, 131,072 ids,
356 special tokens, normalizer null, no byte fallback). Raw bytes come from the vocabulary strings by inverting the GPT-2
byte-to-unicode map. The code is `src/real_tokenizer_check.py` (uses `src/forward.py`) and `tests/test_real_tokenizer.py`; run them as in section 5. The tokenizer files are not in this repository.

| Table fed to the codec | P | distinct codes (of 131,072) | tokens that cannot be produced | pair causes |
|---|---|---|---|---|
| raw vocabulary bytes | 32 and 64, fp32 and bf16 | 131,072 | 0 | none |
| `decode([id])` bytes from the repo's `token_id_to_bytes` and the real `tokenizers` decoder | 32 and 64 | 129,931 | 1,141 in 10 groups (largest 658, 396, 82) | all `bad_bytes_in` |

- Groups by code equal groups by cut string in all four cases, as Theorem 1 says. Every pair in the 10 groups has the cause `bad_bytes_in`;
  `empty`, `same_raw` and `truncation` do not occur.
- The longest ordinary token is 32 bytes (none over 32), so P = 32 and P = 64 give identical results and no token is cut.
- 1,306 tokens (1.0%) hold only part of a letter (not valid UTF-8). 197 of them start with a stray continuation byte. Only 16 touch the
  Kannada lead bytes `E0 B2` / `E0 B3`. The real U+FFFD token is id 99666.
- Hash of the raw table (for each id: the id as 4 bytes, the length as 2 bytes, then the bytes): `18c7f6a382fd6668ec73f56e589fbfbc13fec7c6ad5403a2d79ba6198ecc0d3c`.

**Not covered.**
- **The decoder is `tokenizers` 0.23.1,** the Rust engine behind HF fast tokenizers, called through a thin adapter (only the four named specials take the literal-text path). `transformers` itself was not run. The bytes fed equal a plain model (invalid bytes replaced by U+FFFD) for all 131,072 ids. The repo's `utf8_safe_truncate` gives the same groups as tau_P here.
- **Lemma 2's cut branch** is checked only on random and synthetic strings, since no real token is longer than 32 bytes.
- **Special tokens** are included here as their literal text. The final policy (the G10 list, due 31 Oct) can change what is fed to the
  codec, but not the theorem. Whether to keep `<s>` text and the special token apart is a policy decision.
- **Normalisation.** NFC or NFKC changes the bytes, so it changes the codes. The tokenizer has no normalizer, so this happens in the data pipeline. The proof starts from the bytes.
- **Exact arithmetic.** The proof is exact. The checks cover fp32 and bf16 and show that the lit and unlit values stay apart.
- **Decoding from a predicted vector** (noisy input, margins, tau) is week 2 and is not part of this proof.
## 5. Run it

```
git clone https://github.com/theschoolofai/kronecker-embeddings
pip install torch pytest
# either pip install -e kronecker-embeddings, or point the tests at its src folder:
set KRONECKER_SRC=<path to kronecker-embeddings>/src
pytest teams/04-ke2/tests -q
```

`tests/conftest.py` also finds a folder named `kronecker-embeddings` next to the lab repository.

On the real tokenizer (needs `pip install tokenizers`; download `tokenizer.json`, `tokenizer_config.json` and
`special_tokens_map.json` from https://huggingface.co/theschoolofai/BrahmicTokenizer-131K into a folder outside this repository):

```
set BRAHMIC_TOKENIZER_DIR=<that folder>
pytest teams/04-ke2/tests/test_real_tokenizer.py -q          # skipped when the variable is not set
set PYTHONPATH=<path to kronecker-embeddings>/src
python teams/04-ke2/src/real_tokenizer_check.py %BRAHMIC_TOKENIZER_DIR% results.json
```

Section 4 gives the expected numbers. All 16 tests pass together with the tokenizer (`16 passed in 144 s`); without it, 12 run and 4 are skipped.
