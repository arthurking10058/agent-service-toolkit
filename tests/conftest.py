import os
from unittest.mock import patch

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-docker", action="store_true", default=False, help="run docker integration tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "docker: mark test as requiring docker containers")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-docker"):
        skip_docker = pytest.mark.skip(reason="need --run-docker option to run")
        for item in items:
            if "docker" in item.keywords:
                item.add_marker(skip_docker)


@pytest.fixture
def mock_env():
    """Fixture to ensure environment is clean for each test."""
    preserved_env = {
        key: os.environ[key]
        for key in ("HOME", "USERPROFILE", "HOMEDRIVE", "HOMEPATH", "TMP", "TEMP")
        if key in os.environ
    }
    with patch.dict(os.environ, preserved_env, clear=True):
        yield


@pytest.fixture(autouse=True)
def stable_home_dir(monkeypatch, tmp_path):
    """Provide HOME-style variables required by Streamlit AppTest worker threads."""
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.setenv("USERPROFILE", str(home_dir))
    monkeypatch.setenv("HOMEDRIVE", home_dir.drive or "C:")
    monkeypatch.setenv("HOMEPATH", home_dir.root if home_dir.root else "\\")
