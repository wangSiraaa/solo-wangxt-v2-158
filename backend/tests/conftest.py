"""Pytest fixtures: use an in-process SQLite DB and the shipped dataset."""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ["PRT1_DATABASE_URL"] = "sqlite+pysqlite:///:memory:"

from app import db as db_mod  # noqa: E402
from app import synth  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _ensure_dataset():
    if not (synth.DATA_DIR / "prt1_reference_300ppi.png").exists():
        synth.generate_dataset()
    yield


@pytest.fixture()
def db_session():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    eng = create_engine("sqlite+pysqlite:///:memory:", future=True)
    db_mod.Base.metadata.create_all(eng)
    Sess = sessionmaker(bind=eng, future=True)
    s = Sess()
    try:
        yield s
    finally:
        s.close()
