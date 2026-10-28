import json

import litellm
import pytest
from litellm.llms.anthropic.chat.transformation import AnthropicConfig
from litellm.utils import get_optional_params

from bot_config import BotSettings
from claude_llm import ClaudeLlm, make_llms
from spend_guard import BudgetExceededError, SpendGuard
from tests.fakes import make_fake_llm_call


def _guard(run_cap=10.0, day_cap=100.0) -> SpendGuard:
    return SpendGuard(max_per_run_usd=run_cap, max_per_day_usd=day_cap)


def _anthropic_body(llm: ClaudeLlm) -> dict:
    """Exactly what litellm 1.80 would POST to the Anthropic Messages API."""
    kwargs = {k: v for k, v in llm.litellm_kwargs.items() if k not in ("model", "timeout")}
    model = llm.litellm_kwargs["model"].split("/")[-1]
    optional = get_optional_params(model=model, custom_llm_provider="anthropic", **kwargs)
    return AnthropicConfig().transform_request(
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        optional_params=optional,
        litellm_params={},
        headers={},
    )


def test_make_llms_builds_expected_roles(clean_bot_env):
    settings = BotSettings.from_env({})
    llms = make_llms(settings, _guard())
    assert set(llms) == {"default", "researcher", "parser", "summarizer"}
    default = llms["default"]
    assert isinstance(default, ClaudeLlm)
    assert default.model == settings.main_model
    assert default.fallback is not None and default.fallback.model == settings.fallback_model
    assert default.fallback.fallback is None
    researcher = llms["researcher"]
    assert isinstance(researcher, ClaudeLlm) and researcher.web_search_enabled
    assert researcher.litellm_kwargs["tools"][0]["type"] == settings.web_search_tool_type
    assert llms["parser"].model == settings.parser_model
    assert llms["parser"].fallback is None


def test_research_mode_none_uses_no_research(clean_bot_env):
    settings = BotSettings.from_env({"BOT_RESEARCH_MODE": "none"})
    assert make_llms(settings, _guard())["researcher"] == "no_research"


def test_request_bodies_contain_no_rejected_parameters(clean_bot_env):
    """Claude 5.x rejects temperature and budget-based thinking; make sure neither is sent."""
    settings = BotSettings.from_env({})
    llms = make_llms(settings, _guard())
    for name in ("default", "researcher", "parser", "summarizer"):
        llm = llms[name]
        body = _anthropic_body(llm)
        assert "temperature" not in body, name
        assert "thinking" not in body, name
        assert "top_p" not in body and "top_k" not in body, name
        assert body["max_tokens"] >= 256
        assert body["output_config"]["effort"] in ("low", "medium", "high"), name
    research_body = _anthropic_body(llms["researcher"])
    assert research_body["tools"] == [
        {"type": settings.web_search_tool_type, "name": "web_search", "max_uses": settings.web_search_max_uses}
    ]
    assert "tools" not in _anthropic_body(llms["default"])


def test_to_dict_has_metadata_but_no_secrets(clean_bot_env):
    llm = make_llms(BotSettings.from_env({}), _guard())["default"]
    info = llm.to_dict()
    assert info["fallback_model"] == "anthropic/claude-haiku-5-5"
    assert info["effort"] == "medium"
    blob = json.dumps(info)
    assert "api_key" not in blob and "test-key-not-real" not in blob


async def test_cost_is_recorded_from_response(monkeypatch, clean_bot_env):
    guard = _guard()
    llm = make_llms(BotSettings.from_env({}), guard)["default"]
    monkeypatch.setattr(ClaudeLlm, "_mockable_direct_call_to_model", make_fake_llm_call(cost_per_call=0.0123))
    out = await llm.invoke("hello")
    assert "Probability" in out
    assert guard.run_spent_usd == pytest.approx(0.0123)
    assert guard.calls[0].role == "forecaster"


async def test_cost_is_estimated_when_litellm_reports_zero(monkeypatch, clean_bot_env):
    guard = _guard()
    llm = make_llms(BotSettings.from_env({}), guard)["default"]
    monkeypatch.setattr(ClaudeLlm, "_mockable_direct_call_to_model", make_fake_llm_call(cost_per_call=0.0))
    await llm.invoke("hello")
    # 1000 prompt tokens * $2/M + 300 completion tokens * $10/M
    assert guard.run_spent_usd == pytest.approx(0.002 + 0.003)


async def test_retry_then_fallback_on_primary_failure(monkeypatch, clean_bot_env):
    guard = _guard()
    settings = BotSettings.from_env({"BOT_LLM_ALLOWED_TRIES": "2"})
    llm = make_llms(settings, guard)["default"]
    calls: list = []
    monkeypatch.setattr(
        ClaudeLlm,
        "_mockable_direct_call_to_model",
        make_fake_llm_call(
            calls=calls,
            fail_roles={"forecaster": lambda: litellm.Timeout("slow", model="x", llm_provider="anthropic")},
        ),
    )
    out = await llm.invoke("hello")
    assert "Probability" in out
    roles = [r for r, _ in calls]
    assert roles == ["forecaster", "forecaster", "forecaster-fallback"]  # 2 tries, then fallback
    assert guard.fallback_events == ["forecaster: anthropic/claude-sonnet-5-5 -> anthropic/claude-haiku-5-5"]
    assert guard.calls[-1].model == "anthropic/claude-haiku-5-5"


async def test_transient_failure_recovers_without_fallback(monkeypatch, clean_bot_env):
    guard = _guard()
    llm = make_llms(BotSettings.from_env({"BOT_LLM_ALLOWED_TRIES": "2"}), guard)["default"]
    attempts = {"n": 0}
    good = make_fake_llm_call()

    async def flaky(self, prompt):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("blip")
        return await good(self, prompt)

    monkeypatch.setattr(ClaudeLlm, "_mockable_direct_call_to_model", flaky)
    await llm.invoke("hello")
    assert attempts["n"] == 2
    assert guard.fallback_events == []


async def test_both_models_failing_raises_combined_error(monkeypatch, clean_bot_env):
    llm = make_llms(BotSettings.from_env({"BOT_LLM_ALLOWED_TRIES": "1"}), _guard())["default"]
    monkeypatch.setattr(
        ClaudeLlm,
        "_mockable_direct_call_to_model",
        make_fake_llm_call(fail_roles={"forecaster": lambda: RuntimeError("down"), "forecaster-fallback": lambda: RuntimeError("also down")}),
    )
    with pytest.raises(RuntimeError, match="both .* and fallback"):
        await llm.invoke("hello")


async def test_budget_error_is_not_retried_and_not_fallen_back(monkeypatch, clean_bot_env):
    guard = _guard(run_cap=0.05)
    guard.record(cost_usd=0.05, model="m", role="r")  # already at the cap
    llm = make_llms(BotSettings.from_env({}), guard)["default"]
    calls: list = []
    monkeypatch.setattr(ClaudeLlm, "_mockable_direct_call_to_model", make_fake_llm_call(calls=calls))
    with pytest.raises(BudgetExceededError):
        await llm.invoke("hello")
    assert calls == []  # no API call was even attempted
    assert guard.fallback_events == []


async def test_web_search_tool_is_dropped_if_api_rejects_it(monkeypatch, clean_bot_env):
    guard = _guard()
    llm = make_llms(BotSettings.from_env({"BOT_LLM_ALLOWED_TRIES": "1"}), guard)["researcher"]
    good = make_fake_llm_call()

    async def reject_tools(self, prompt):
        if self.litellm_kwargs.get("tools"):
            raise litellm.BadRequestError("tools.0: unsupported tool type", model="x", llm_provider="anthropic")
        return await good(self, prompt)

    monkeypatch.setattr(ClaudeLlm, "_mockable_direct_call_to_model", reject_tools)
    assert llm.web_search_enabled
    out = await llm.invoke("research this")
    assert "Research rundown" in out
    assert not llm.web_search_enabled
    assert "tools" not in llm.litellm_kwargs
    assert guard.fallback_events == []  # handled without switching model


async def test_concurrency_is_throttled(monkeypatch, clean_bot_env):
    import asyncio

    settings = BotSettings.from_env({"BOT_MAX_CONCURRENT_LLM_CALLS": "2"})
    llm = make_llms(settings, _guard())["default"]
    in_flight = {"now": 0, "max": 0}
    good = make_fake_llm_call()

    async def slow(self, prompt):
        in_flight["now"] += 1
        in_flight["max"] = max(in_flight["max"], in_flight["now"])
        await asyncio.sleep(0.01)
        in_flight["now"] -= 1
        return await good(self, prompt)

    monkeypatch.setattr(ClaudeLlm, "_mockable_direct_call_to_model", slow)
    await asyncio.gather(*[llm.invoke("x") for _ in range(6)])
    assert in_flight["max"] == 2
