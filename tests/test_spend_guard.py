from datetime import datetime, timedelta, timezone

import pytest

from spend_guard import BudgetExceededError, SpendGuard, SpendLedger


def test_run_cap_blocks_next_call_and_marks_tripped():
    guard = SpendGuard(max_per_run_usd=0.10, max_per_day_usd=1.0)
    guard.check()
    guard.record(cost_usd=0.06, model="m", role="r")
    guard.check()  # 0.06 < 0.10, still fine
    guard.record(cost_usd=0.05, model="m", role="r")
    assert guard.run_spent_usd == pytest.approx(0.11)
    with pytest.raises(BudgetExceededError, match="per-run cap"):
        guard.check()
    assert guard.tripped
    assert "Spend cap hit" in guard.summary_markdown()


def test_day_cap_includes_previous_runs_from_ledger(tmp_path):
    now = datetime(2026, 10, 27, 12, 0, tzinfo=timezone.utc)
    ledger = SpendLedger(tmp_path / "ledger.jsonl")
    first = SpendGuard(max_per_run_usd=5, max_per_day_usd=1.0, ledger=ledger, run_id="run1", clock=lambda: now)
    first.record(cost_usd=0.7, model="m", role="r")
    first.finish()

    second = SpendGuard(max_per_run_usd=5, max_per_day_usd=1.0, ledger=ledger, run_id="run2", clock=lambda: now + timedelta(hours=1))
    assert second.day_spent_before_run_usd == pytest.approx(0.7)
    second.check()
    second.record(cost_usd=0.35, model="m", role="r")
    with pytest.raises(BudgetExceededError, match="per-day cap"):
        second.check()
    second.finish()

    # next UTC day starts fresh
    third = SpendGuard(max_per_run_usd=5, max_per_day_usd=1.0, ledger=ledger, run_id="run3", clock=lambda: now + timedelta(days=1))
    assert third.day_spent_before_run_usd == 0.0
    third.check()


def test_finish_is_idempotent_and_takes_larger_independent_measurement(tmp_path):
    ledger = SpendLedger(tmp_path / "ledger.jsonl")
    guard = SpendGuard(max_per_run_usd=5, max_per_day_usd=10, ledger=ledger, run_id="x")
    guard.record(cost_usd=0.2, model="m", role="r")
    assert guard.finish(observed_total_usd=0.3) == pytest.approx(0.3)
    guard.finish(observed_total_usd=0.1)
    entries = ledger.entries()
    assert len(entries) == 1
    assert entries[0]["cost_usd"] == pytest.approx(0.3)
    assert entries[0]["run_id"] == "x"


def test_ledger_prunes_old_entries_and_ignores_garbage(tmp_path):
    path = tmp_path / "ledger.jsonl"
    path.write_text('not json\n{"date": "2020-01-01", "cost_usd": 99}\n')
    ledger = SpendLedger(path, keep_days=14)
    now = datetime(2026, 10, 27, tzinfo=timezone.utc)
    assert ledger.spent_on("2020-01-01") == 99
    ledger.append(run_id="r", cost_usd=0.5, now=now)
    assert [e["date"] for e in ledger.entries()] == ["2026-10-27"]


def test_caps_must_be_positive():
    with pytest.raises(ValueError):
        SpendGuard(max_per_run_usd=0, max_per_day_usd=1)


def test_broken_ledger_does_not_stop_a_run(tmp_path):
    class ExplodingLedger(SpendLedger):
        def spent_on(self, day):
            raise OSError("disk on fire")

    guard = SpendGuard(max_per_run_usd=1, max_per_day_usd=2, ledger=ExplodingLedger(tmp_path / "l"))
    assert guard.day_spent_before_run_usd == 0.0
    guard.check()
