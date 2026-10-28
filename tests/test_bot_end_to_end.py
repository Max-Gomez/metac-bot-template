"""
End-to-end runs of the real FallTemplateBot2026 / ForecastBot pipeline with the
LLM and Metaculus boundaries faked. Nothing here touches the network.
"""
import json

import pytest

import main
from bot_config import BotSettings
from claude_llm import ClaudeLlm
from spend_guard import SpendGuard, SpendLedger
from tests.fakes import (
    LONG_RATIONALE,
    RESEARCH_TEXT,
    SHORT_RATIONALE,
    FakeMetaculusClient,
    binary_question,
    date_question,
    default_parser_answer,
    make_fake_llm_call,
    multiple_choice_question,
    numeric_question,
)

FAST = {"BOT_LLM_ALLOWED_TRIES": "1", "BOT_PREDICTIONS_PER_QUESTION": "2"}


def _bot_and_client(env=None, run_cap=10.0, day_cap=100.0):
    settings = BotSettings.from_env({**FAST, **(env or {})})
    guard = SpendGuard(max_per_run_usd=run_cap, max_per_day_usd=day_cap)
    bot = main.build_bot(settings, guard)
    client = FakeMetaculusClient()
    bot.metaculus_client = client
    return bot, client, guard


async def test_binary_forecast_is_posted_with_reasoning(monkeypatch, clean_bot_env):
    bot, client, guard = _bot_and_client()
    monkeypatch.setattr(ClaudeLlm, "_mockable_direct_call_to_model", make_fake_llm_call())
    reports = await bot.forecast_questions([binary_question()], return_exceptions=True)
    assert not any(isinstance(r, BaseException) for r in reports), reports
    assert client.posted_binary == [(101, pytest.approx(0.72))]
    assert len(client.comments) == 1
    post_id, comment = client.comments[0]
    assert post_id == 101
    assert comment.lstrip().startswith("#")
    assert "# FORECASTS" in comment
    assert "status quo" in comment  # the bot's actual rationale text
    assert RESEARCH_TEXT.split(":")[0] in comment
    assert "anthropic/claude-sonnet-5-5" in comment  # model metadata
    assert "test-key-not-real" not in comment
    assert guard.run_spent_usd > 0


async def test_binary_probability_is_clipped_to_99_percent(monkeypatch, clean_bot_env):
    bot, client, _ = _bot_and_client()

    def parser(prompt):
        if "BinaryPrediction" in prompt:
            return json.dumps({"prediction_in_decimal": 1.0})
        return default_parser_answer(prompt)

    monkeypatch.setattr(ClaudeLlm, "_mockable_direct_call_to_model", make_fake_llm_call(parser=parser))
    reports = await bot.forecast_questions([binary_question()], return_exceptions=True)
    assert not any(isinstance(r, BaseException) for r in reports), reports
    assert client.posted_binary[0][1] == pytest.approx(0.99)


async def test_multiple_choice_posts_exact_option_names(monkeypatch, clean_bot_env):
    bot, client, _ = _bot_and_client()
    monkeypatch.setattr(ClaudeLlm, "_mockable_direct_call_to_model", make_fake_llm_call())
    reports = await bot.forecast_questions([multiple_choice_question()], return_exceptions=True)
    assert not any(isinstance(r, BaseException) for r in reports), reports
    (question_id, probs), = client.posted_mc
    assert question_id == 202
    assert list(probs) == ["Red", "Blue", "Green"]  # parser said "Option Red", "blue"
    assert sum(probs.values()) == pytest.approx(1.0)
    assert min(probs.values()) >= 0.01
    assert len(client.comments) == 1


async def test_numeric_and_date_forecasts_post_valid_cdfs(monkeypatch, clean_bot_env):
    bot, client, _ = _bot_and_client()
    monkeypatch.setattr(ClaudeLlm, "_mockable_direct_call_to_model", make_fake_llm_call())
    reports = await bot.forecast_questions([numeric_question(), date_question()], return_exceptions=True)
    assert not any(isinstance(r, BaseException) for r in reports), reports
    assert {qid for qid, _ in client.posted_numeric} == {303, 404}
    for _, cdf in client.posted_numeric:
        assert len(cdf) == 201
        assert all(0 <= v <= 1 for v in cdf)
        assert all(a <= b for a, b in zip(cdf, cdf[1:]))
    assert len(client.comments) == 2


async def test_forecast_without_real_reasoning_is_never_posted(monkeypatch, clean_bot_env):
    bot, client, _ = _bot_and_client()
    monkeypatch.setattr(ClaudeLlm, "_mockable_direct_call_to_model", make_fake_llm_call(rationale=SHORT_RATIONALE))
    reports = await bot.forecast_questions([binary_question()], return_exceptions=True)
    assert len(reports) == 1 and isinstance(reports[0], BaseException)
    assert "Refusing to publish" in str(reports[0])
    assert client.posted_binary == [] and client.comments == []


async def test_malformed_option_names_fail_safely(monkeypatch, clean_bot_env):
    bot, client, _ = _bot_and_client()

    def parser(prompt):
        if "PredictedOptionList" in prompt:
            return json.dumps({"predicted_options": [{"option_name": "Purple", "probability": 0.6}, {"option_name": "Blue", "probability": 0.4}]})
        return default_parser_answer(prompt)

    monkeypatch.setattr(ClaudeLlm, "_mockable_direct_call_to_model", make_fake_llm_call(parser=parser))
    reports = await bot.forecast_questions([multiple_choice_question()], return_exceptions=True)
    assert isinstance(reports[0], BaseException)
    assert client.posted_mc == [] and client.comments == []


async def test_spend_cap_stops_run_before_posting(monkeypatch, clean_bot_env):
    # research $0.05 + summary $0.05 = $0.10 -> cap reached before any forecast call
    bot, client, guard = _bot_and_client(run_cap=0.10)
    calls: list = []
    monkeypatch.setattr(ClaudeLlm, "_mockable_direct_call_to_model", make_fake_llm_call(cost_per_call=0.05, calls=calls))
    reports = await bot.forecast_questions([binary_question(), binary_question(102)], return_exceptions=True)
    assert all(isinstance(r, BaseException) for r in reports)
    assert client.posted_binary == [] and client.comments == []
    assert guard.tripped and "per-run cap" in guard.tripped_reason
    assert guard.run_spent_usd <= 0.10 + 1e-9
    assert all(role in ("researcher", "summarizer") for role, _ in calls)


async def test_fallback_model_still_produces_a_posted_forecast(monkeypatch, clean_bot_env):
    import litellm

    bot, client, guard = _bot_and_client()
    monkeypatch.setattr(
        ClaudeLlm,
        "_mockable_direct_call_to_model",
        make_fake_llm_call(fail_roles={"forecaster": lambda: litellm.Timeout("slow", model="x", llm_provider="anthropic")}),
    )
    reports = await bot.forecast_questions([binary_question()], return_exceptions=True)
    assert not any(isinstance(r, BaseException) for r in reports), reports
    assert client.posted_binary == [(101, pytest.approx(0.72))]
    assert guard.fallback_events  # noted for the job summary


async def test_dry_run_posts_nothing(monkeypatch, clean_bot_env):
    bot, client, _ = _bot_and_client({"BOT_PUBLISH": "false"})
    monkeypatch.setattr(ClaudeLlm, "_mockable_direct_call_to_model", make_fake_llm_call())
    reports = await bot.forecast_questions([binary_question()], return_exceptions=True)
    assert not any(isinstance(r, BaseException) for r in reports), reports
    assert client.posted_binary == [] and client.comments == []


def test_run_writes_job_summary_and_exit_code(monkeypatch, tmp_path, clean_bot_env):
    """Full main.run() in tournament mode with the fake client wired in."""
    FakeMetaculusClient.questions = [binary_question()]
    created: list[FakeMetaculusClient] = []

    class RecordingClient(FakeMetaculusClient):
        def __init__(self, *a, **k):
            super().__init__()
            created.append(self)

    import forecasting_tools.forecast_bots.forecast_bot as fb

    monkeypatch.setattr(fb, "MetaculusClient", RecordingClient)
    monkeypatch.setattr(main, "MetaculusClient", RecordingClient)
    monkeypatch.setattr(ClaudeLlm, "_mockable_direct_call_to_model", make_fake_llm_call(cost_per_call=0.02))
    summary_file = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_file))
    monkeypatch.setenv("GITHUB_RUN_ID", "4242")
    for k, v in FAST.items():
        monkeypatch.setenv(k, v)

    exit_code = main.run("tournament")

    assert exit_code == 0
    bot_client = created[0]
    # main + minibench tournaments were both queried, each posted the fake question
    assert len(bot_client.requested_tournaments) == 2
    assert len(bot_client.posted_binary) == 2 and len(bot_client.comments) == 2
    text = summary_file.read_text()
    assert "Forecast run" in text and "✅ 2 forecast(s) posted" in text
    assert "This run:" in text and "cap" in text
    ledger = SpendLedger(tmp_path / "ledger.jsonl")
    assert ledger.entries()[0]["run_id"] == "4242"
    assert ledger.entries()[0]["cost_usd"] > 0


def test_run_reports_failures_with_nonzero_exit(monkeypatch, tmp_path, clean_bot_env):
    FakeMetaculusClient.questions = [binary_question()]
    import forecasting_tools.forecast_bots.forecast_bot as fb

    monkeypatch.setattr(fb, "MetaculusClient", FakeMetaculusClient)
    monkeypatch.setattr(main, "MetaculusClient", FakeMetaculusClient)
    monkeypatch.setattr(ClaudeLlm, "_mockable_direct_call_to_model", make_fake_llm_call(rationale=SHORT_RATIONALE))
    summary_file = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_file))
    for k, v in FAST.items():
        monkeypatch.setenv(k, v)

    assert main.run("test_questions") == 1
    text = summary_file.read_text()
    assert "failed" in text and "Refusing to publish" in text


def test_run_survives_metaculus_outage(monkeypatch, tmp_path, clean_bot_env):
    class DownClient(FakeMetaculusClient):
        def get_all_open_questions_from_tournament(self, *a, **k):
            raise ConnectionError("metaculus is down")

    import forecasting_tools.forecast_bots.forecast_bot as fb

    monkeypatch.setattr(fb, "MetaculusClient", DownClient)
    monkeypatch.setattr(main, "MetaculusClient", DownClient)
    summary_file = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_file))

    assert main.run("tournament") == 1
    assert "crashed" in summary_file.read_text()
    assert "metaculus is down" in summary_file.read_text()
