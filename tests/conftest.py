"""Shared pytest setup for the whole suite.

Every engine folder (whisper/, gemini/, chirp/, wav2vec2/) defines its own
top-level `lib` package, meant to be imported as `lib.config` etc. with only
that ONE folder on sys.path (see e.g. gemini/scripts/run_baseline.py's
`sys.path.insert`) - each engine is always run as its own process, so this
never collides in practice. A test suite that needs to exercise more than one
engine in the same interpreter session can't use that same trick: adding all
four engine folders to sys.path at once would make `import lib.config`
ambiguous (all four define a top-level `lib`), silently picking whichever
one happened to load first.

Instead, this conftest puts only the REPO ROOT on sys.path, and tests import
each engine via its qualified name instead - `gemini.lib.metrics`,
`chirp.lib.stt_client`, `data.lib.dataset_utils`, and so on. Python's
implicit namespace packages (PEP 420) make this resolve correctly with no
code changes to any engine folder: none of whisper/, gemini/, chirp/,
wav2vec2/, or data/ need their own `__init__.py` for `import gemini.lib.x` to
work, as long as `gemini/lib/__init__.py` exists (it does, for every engine).
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
