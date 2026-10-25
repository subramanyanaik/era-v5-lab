"""Make the kronecker-embeddings package and teams/04-ke2/theorem/theorem1 importable for the tests.

The package is found in this order: already installed, the KRONECKER_SRC environment
variable (path to the repository's src folder), a sibling folder named kronecker-embeddings.
"""

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

try:
    import kronecker_embeddings  # noqa: F401
except ImportError:
    candidates = [os.environ.get("KRONECKER_SRC"), HERE.parents[3].parent / "kronecker-embeddings" / "src"]
    for c in candidates:
        if c and (Path(c) / "kronecker_embeddings").is_dir():
            sys.path.insert(0, str(c))
            break
    else:
        raise ImportError(
            "kronecker_embeddings not found. Clone https://github.com/theschoolofai/kronecker-embeddings "
            "and either pip install it or set KRONECKER_SRC to its src folder."
        )
