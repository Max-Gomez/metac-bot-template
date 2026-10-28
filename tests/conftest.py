"""
Shared fixtures. Every test runs with network access blocked at the socket
level and with litellm's completion entry points stubbed out, so nothing here
can ever reach the Anthropic or Metaculus APIs.
"""
from __future__ import annotations

import os
import socket
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Fake credentials so the code paths that check for their presence run; they
# are never sent anywhere because the network is blocked below.
os.environ["METACULUS_TOKEN"] = "test-token-not-real"
os.environ["ANTHROPIC_API_KEY"] = "test-key-not-real"
os.environ.pop("GITHUB_STEP_SUMMARY", None)
os.environ.pop("GITHUB_EVENT_NAME", None)
for key in list(os.environ):
    if key.startswith("BOT_"):
        os.environ.pop(key)

import litellm  # noqa: E402
import tenacity  # noqa: E402
from forecasting_tools.ai_models import general_llm as general_llm_module  # noqa: E402
from forecasting_tools.ai_models.model_interfaces import (  # noqa: E402
    retryable_model as retryable_model_module,
)


class NetworkBlocked(RuntimeError):
    pass


def _blocked(*args, **kwargs):
    raise NetworkBlocked("Test attempted network access")


async def _blocked_async(*args, **kwargs):
    raise NetworkBlocked("Test attempted an LLM API call")


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    monkeypatch.setattr(litellm, "acompletion", _blocked_async)
    monkeypatch.setattr(litellm, "completion", _blocked)
    monkeypatch.setattr(general_llm_module, "acompletion", _blocked_async)
    monkeypatch.setattr(general_llm_module, "aresponses", _blocked_async)
    import requests

    monkeypatch.setattr(requests.Session, "request", _blocked)
    monkeypatch.setattr(requests, "request", _blocked)
    monkeypatch.setattr(requests, "get", _blocked)
    monkeypatch.setattr(requests, "post", _blocked)
    yield


@pytest.fixture(autouse=True)
def instant_retries(monkeypatch):
    """forecasting-tools waits 5-60s between retries; make that 0 in tests."""
    monkeypatch.setattr(
        retryable_model_module,
        "wait_random_exponential",
        lambda **kwargs: tenacity.wait_none(),
    )
    yield


@pytest.fixture(autouse=True)
def isolated_ledger(monkeypatch, tmp_path):
    monkeypatch.setenv("BOT_SPEND_LEDGER_PATH", str(tmp_path / "ledger.jsonl"))
    yield


@pytest.fixture
def clean_bot_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("BOT_") and key != "BOT_SPEND_LEDGER_PATH":
            monkeypatch.delenv(key, raising=False)
    yield
