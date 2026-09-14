"""Tests for wav2vec2's MMS adapter-selection logic and the --model/config.yaml
mismatch guard.

Two different techniques are used deliberately:

1. `load_model`/`load_processor` (wav2vec2/lib/model_utils.py) are tested by
   importing `wav2vec2.lib.model_utils` directly and mocking
   AutoModelForCTC.from_pretrained/AutoProcessor.from_pretrained - safe to
   import alongside other engines' `lib` modules in the same test session
   (see tests/conftest.py's docstring on why qualified imports avoid
   collisions).

2. The --model/config.yaml mismatch guard lives in
   wav2vec2/scripts/run_baseline.py's main(), which (like every engine's own
   scripts) does its own `sys.path.insert` + bare `import lib.config` at
   import time. Importing that module directly in a test process would
   claim the top-level `lib` name for whichever engine's scripts/ module
   gets imported first in the whole test session - fine in isolation, but a
   latent collision risk if a similar test were ever added for another
   engine's scripts/*.py in the same session. Running it as a real
   subprocess instead sidesteps that entirely, and is also just a more
   faithful test of what a user actually experiences running the CLI.
"""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import wav2vec2.lib.model_utils as model_utils

REPO_ROOT = Path(__file__).resolve().parent.parent


def _fake_cfg(model_type: str, mms_lang_code=None):
    return SimpleNamespace(
        model_name="some/checkpoint",
        device="cpu",
        fp16=False,
        model_type=model_type,
        mms_lang_code=mms_lang_code,
    )


def test_ctc_checkpoint_never_touches_adapter_methods():
    cfg = _fake_cfg(model_type="ctc")
    fake_model = MagicMock()
    fake_processor = MagicMock()

    with patch.object(model_utils.AutoModelForCTC, "from_pretrained", return_value=fake_model), patch.object(
        model_utils.AutoProcessor, "from_pretrained", return_value=fake_processor
    ):
        model_utils.load_model(cfg)
        model_utils.load_processor(cfg)

    fake_model.load_adapter.assert_not_called()
    fake_processor.tokenizer.set_target_lang.assert_not_called()


def test_mms_checkpoint_selects_the_adapter():
    cfg = _fake_cfg(model_type="mms", mms_lang_code="swh")
    fake_model = MagicMock()
    fake_processor = MagicMock()

    with patch.object(model_utils.AutoModelForCTC, "from_pretrained", return_value=fake_model), patch.object(
        model_utils.AutoProcessor, "from_pretrained", return_value=fake_processor
    ):
        model_utils.load_model(cfg)
        model_utils.load_processor(cfg)

    fake_model.load_adapter.assert_called_once_with("swh")
    fake_processor.tokenizer.set_target_lang.assert_called_once_with("swh")


def test_mms_type_without_lang_code_raises():
    cfg = _fake_cfg(model_type="mms", mms_lang_code=None)
    fake_model = MagicMock()
    fake_processor = MagicMock()
    with patch.object(model_utils.AutoModelForCTC, "from_pretrained", return_value=fake_model), patch.object(
        model_utils.AutoProcessor, "from_pretrained", return_value=fake_processor
    ):
        try:
            model_utils.load_processor(cfg)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError when model.type is 'mms' but lang_code is unset")

        try:
            model_utils.load_model(cfg)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError when model.type is 'mms' but lang_code is unset")


def test_cli_fails_closed_on_mms_checkpoint_with_plain_config():
    # facebook/mms-1b-all via --model, but --config is left at the default
    # wav2vec2/kiswahili/config.yaml (model.type: ctc, no adapter step) -
    # should refuse to run rather than silently loading MMS with no adapter
    # selected.
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "wav2vec2" / "scripts" / "run_baseline.py"),
            "--language",
            "kiswahili",
            "--model",
            "facebook/mms-1b-all",
            "--limit",
            "1",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert "config_mms.yaml" in result.stderr


def test_cli_rejects_limit_below_one():
    for bad_limit in ["0", "-5"]:
        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "wav2vec2" / "scripts" / "run_baseline.py"),
                "--language",
                "kiswahili",
                "--limit",
                bad_limit,
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode != 0
        assert "--limit must be >= 1" in result.stderr
