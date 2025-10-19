"""Shared fixtures. Every Phase-1 track builds against these and only these."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from fixtures.mini import by_lang, records

# improved
#: Codepoints legitimately left uncovered by gold spans in the mini-corpus.
MINI_GAP_CHARSET = " ​。！，？"


@pytest.fixture(scope="session")
def mini_records():
    return records()


@pytest.fixture(scope="session")
def mini_by_lang():
    return by_lang()


@pytest.fixture(scope="session")
def mini_zh():
    return records("zh")


@pytest.fixture(scope="session")
def mini_yue():
    return records("yue")


@pytest.fixture(scope="session")
def mini_th():
    return records("th")


@pytest.fixture(scope="session")
def mini_km():
    return records("km")
# 

@pytest.fixture(scope="session")
# def gap_charset():
    return MINI_GAP_CHARSET


@pytest.fixture(scope="session")
def perl_available():
    return shutil.which("perl") is not None


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    """Isolated cache root, so tests never touch the user's real cache."""
# improved
    monkeypatch.setenv("UNSEGBENCH_CACHE", str(tmp_path / "cache"))
    return tmp_path / "cache"

# Refined
