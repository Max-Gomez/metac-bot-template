import pytest

import bot_helpers
from bot_config import BotSettings
from bot_helpers import build_run_summary_markdown, check_environment, missing_required_secrets
from spend_guard import SpendGuard


def test_missing_secrets_detects_placeholders(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "REPLACE_ME")
    assert missing_required_secrets() == ["ANTHROPIC_API_KEY"]
    monkeypatch.delenv("METACULUS_TOKEN")
    assert missing_required_secrets() == ["METACULUS_TOKEN", "ANTHROPIC_API_KEY"]


def test_scheduled_run_without_secrets_exits_zero_with_summary(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "schedule")
    summary = tmp_path / "s.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    with pytest.raises(SystemExit) as exc:
        check_environment(strict=True)
    assert exc.value.code == 0
    assert "not configured" in summary.read_text()
    assert "ANTHROPIC_API_KEY" in summary.read_text()


def test_manual_run_without_secrets_exits_one(monkeypatch):
    monkeypatch.delenv("METACULUS_TOKEN")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    with pytest.raises(SystemExit) as exc:
        check_environment(strict=True)
    assert exc.value.code == 1


def test_check_environment_passes_with_secrets():
    assert check_environment(strict=True) == []


def test_summary_markdown_covers_each_outcome(clean_bot_env):
    settings = BotSettings.from_env({})
    guard = SpendGuard(max_per_run_usd=4, max_per_day_usd=12)
    guard.record(cost_usd=0.25, model="anthropic/claude-sonnet-5-5", role="forecaster")

    empty = build_run_summary_markdown(run_mode="tournament", will_publish=True, forecast_reports=[], guard=guard, settings=settings)
    assert "No new questions" in empty and "$0.2500" in empty

    failed = build_run_summary_markdown(
        run_mode="tournament", will_publish=True, forecast_reports=[RuntimeError("boom")], guard=guard, settings=settings
    )
    assert "All 1 question(s) failed" in failed and "boom" in failed

    crashed = build_run_summary_markdown(
        run_mode="tournament", will_publish=True, forecast_reports=[], guard=guard, settings=settings, fatal_error=ValueError("bad")
    )
    assert "crashed" in crashed and "bad" in crashed


def test_write_job_summary_appends(monkeypatch, tmp_path, capsys):
    path = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(path))
    bot_helpers.write_job_summary("## one")
    bot_helpers.write_job_summary("## two")
    assert path.read_text() == "## one\n\n## two\n\n"
    assert "## two" in capsys.readouterr().out
