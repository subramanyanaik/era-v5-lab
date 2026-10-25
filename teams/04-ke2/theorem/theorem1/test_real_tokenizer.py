"""Theorem 1 on the real tokenizer (proof.md section 4).

Skipped unless BRAHMIC_TOKENIZER_DIR points to a folder with tokenizer.json and special_tokens_map.json.
"""

import os
from pathlib import Path

import pytest

tok_dir = os.environ.get("BRAHMIC_TOKENIZER_DIR")
pytestmark = pytest.mark.skipif(
    not (tok_dir and (Path(tok_dir) / "tokenizer.json").is_file()),
    reason="set BRAHMIC_TOKENIZER_DIR to run on the real tokenizer",
)


@pytest.fixture(scope="module")
def result():
    pytest.importorskip("tokenizers")
    from real_tokenizer_check import run

    return run(Path(tok_dir))


@pytest.mark.parametrize("P", [32, 64])
def test_raw_table_has_no_collision(result, P):
    for dtype in ("float32", "bfloat16"):
        r = result[f"P{P}_raw"][dtype]
        assert r["distinct_codes"] == result["vocab"]
        assert r["partitions_equal"]
    assert result[f"P{P}_raw"]["groups"] == 0


@pytest.mark.parametrize("P", [32, 64])
def test_theorem1_on_decode_table(result, P):
    for dtype in ("float32", "bfloat16"):
        assert result[f"P{P}_decode"][dtype]["partitions_equal"]
    causes = result[f"P{P}_decode"]["pair_causes"]
    assert None not in causes  # every colliding pair has a cause
    assert set(causes) <= {"empty", "same_raw", "bad_bytes_in", "truncation"}
