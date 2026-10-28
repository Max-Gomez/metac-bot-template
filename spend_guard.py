"""
Hard spend caps for LLM calls: per run and per UTC day.

- The per-run cap protects against a bug or a burst of questions inside one
  GitHub Actions run.
- The per-day cap protects against the schedule itself (the tournament
  workflow fires every 20 minutes). Spend is persisted to a tiny JSONL ledger
  that the workflows carry between runs with the GitHub Actions cache. If the
  ledger is lost, the day starts at $0 again, so the per-day cap is "best
  effort" while the per-run cap is always enforced.

Both caps are checked *before* every LLM call. A call that would start while
the cap is reached raises BudgetExceededError, which is deliberately not
retried and not routed to the fallback model. Calls already in flight may
finish, so the overshoot is bounded by (concurrent calls x one call's cost).
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


class BudgetExceededError(RuntimeError):
    """Raised when a spend cap has been reached. Never retried, never falls back."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SpendLedger:
    """Append-only JSONL file: one line per run with the USD spent."""

    def __init__(self, path: str | os.PathLike[str], keep_days: int = 14) -> None:
        self.path = Path(path)
        self.keep_days = keep_days

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        entries: list[dict] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"Skipping unreadable spend ledger line: {line[:80]!r}")
                continue
            if isinstance(entry, dict):
                entries.append(entry)
        return entries

    def spent_on(self, day: str) -> float:
        total = 0.0
        for entry in self.entries():
            if entry.get("date") == day:
                try:
                    total += float(entry.get("cost_usd", 0) or 0)
                except (TypeError, ValueError):
                    continue
        return total

    def append(self, *, run_id: str, cost_usd: float, now: datetime, note: str = "") -> None:
        cutoff = (now - timedelta(days=self.keep_days)).date().isoformat()
        kept = [e for e in self.entries() if str(e.get("date", "")) >= cutoff]
        kept.append(
            {
                "date": now.date().isoformat(),
                "ts": now.isoformat(timespec="seconds"),
                "run_id": run_id,
                "cost_usd": round(float(cost_usd), 6),
                "note": note,
            }
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(
            "".join(json.dumps(e, sort_keys=True) + "\n" for e in kept), encoding="utf-8"
        )
        tmp.replace(self.path)


@dataclass
class CallRecord:
    role: str
    model: str
    cost_usd: float
    prompt_tokens: int
    completion_tokens: int


@dataclass
class SpendGuard:
    max_per_run_usd: float
    max_per_day_usd: float
    ledger: SpendLedger | None = None
    run_id: str = "local"
    clock: Callable[[], datetime] = _utc_now

    run_spent_usd: float = 0.0
    day_spent_before_run_usd: float = 0.0
    calls: list[CallRecord] = field(default_factory=list)
    fallback_events: list[str] = field(default_factory=list)
    tripped_reason: str | None = None
    _finished: bool = False

    def __post_init__(self) -> None:
        if self.max_per_run_usd <= 0 or self.max_per_day_usd <= 0:
            raise ValueError("Spend caps must be positive numbers")
        if self.ledger is not None:
            try:
                self.day_spent_before_run_usd = self.ledger.spent_on(self.today)
            except Exception as e:  # a broken ledger must never stop a run
                logger.warning(f"Could not read spend ledger: {e}")
                self.day_spent_before_run_usd = 0.0

    @property
    def today(self) -> str:
        return self.clock().date().isoformat()

    @property
    def day_total_usd(self) -> float:
        return self.day_spent_before_run_usd + self.run_spent_usd

    @property
    def tripped(self) -> bool:
        return self.tripped_reason is not None

    def check(self) -> None:
        """Raise BudgetExceededError if another LLM call must not start."""
        if self.run_spent_usd >= self.max_per_run_usd:
            self.tripped_reason = (
                f"per-run cap reached: ${self.run_spent_usd:.4f} spent of "
                f"${self.max_per_run_usd:.2f} allowed"
            )
        elif self.day_total_usd >= self.max_per_day_usd:
            self.tripped_reason = (
                f"per-day cap reached: ${self.day_total_usd:.4f} spent today of "
                f"${self.max_per_day_usd:.2f} allowed (${self.day_spent_before_run_usd:.4f} "
                "before this run)"
            )
        if self.tripped_reason:
            raise BudgetExceededError(
                f"LLM call blocked, {self.tripped_reason}. Raise BOT_MAX_COST_PER_RUN_USD / "
                "BOT_MAX_COST_PER_DAY_USD if this is intended."
            )

    def record(
        self,
        *,
        cost_usd: float,
        model: str,
        role: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        cost_usd = max(float(cost_usd or 0.0), 0.0)
        self.run_spent_usd += cost_usd
        self.calls.append(
            CallRecord(
                role=role,
                model=model,
                cost_usd=cost_usd,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        )
        logger.info(
            f"LLM call [{role}] {model}: ${cost_usd:.4f} "
            f"(run total ${self.run_spent_usd:.4f} / ${self.max_per_run_usd:.2f}, "
            f"day total ${self.day_total_usd:.4f} / ${self.max_per_day_usd:.2f})"
        )

    def note_fallback(self, role: str, from_model: str, to_model: str) -> None:
        self.fallback_events.append(f"{role}: {from_model} -> {to_model}")

    def finish(self, observed_total_usd: float | None = None) -> float:
        """
        Persist this run's spend to the ledger (once). `observed_total_usd` lets
        the caller pass an independent measurement (e.g. litellm's callback
        tracker); the larger of the two is recorded.
        """
        if observed_total_usd is not None and observed_total_usd > self.run_spent_usd:
            logger.info(
                f"Independent cost tracker saw ${observed_total_usd:.4f} vs guard "
                f"${self.run_spent_usd:.4f}; recording the larger number."
            )
            self.run_spent_usd = float(observed_total_usd)
        if self._finished:
            return self.run_spent_usd
        self._finished = True
        if self.ledger is not None:
            try:
                self.ledger.append(
                    run_id=self.run_id,
                    cost_usd=self.run_spent_usd,
                    now=self.clock(),
                    note=self.tripped_reason or "",
                )
            except Exception as e:  # pragma: no cover - disk errors
                logger.warning(f"Could not write spend ledger: {e}")
        return self.run_spent_usd

    def cost_by_model(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        for call in self.calls:
            totals[call.model] = totals.get(call.model, 0.0) + call.cost_usd
        return totals

    def summary_markdown(self) -> str:
        lines = [
            f"- **This run:** ${self.run_spent_usd:.4f} of ${self.max_per_run_usd:.2f} cap "
            f"across {len(self.calls)} LLM call(s)",
            f"- **Today (UTC):** ${self.day_total_usd:.4f} of ${self.max_per_day_usd:.2f} cap",
        ]
        by_model = self.cost_by_model()
        if by_model:
            parts = ", ".join(f"{m}: ${c:.4f}" for m, c in sorted(by_model.items()))
            lines.append(f"- **By model:** {parts}")
        if self.fallback_events:
            lines.append(
                f"- **Fallback model used** {len(self.fallback_events)} time(s): "
                + "; ".join(self.fallback_events[:5])
            )
        if self.tripped:
            lines.append(
                f"- ⚠️ **Spend cap hit:** {self.tripped_reason}. Remaining questions were "
                "skipped and will be retried on the next run."
            )
        return "\n".join(lines)
