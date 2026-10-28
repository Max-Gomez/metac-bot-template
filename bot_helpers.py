"""
Runtime helpers for the template bot: environment validation, startup/result
banners, GitHub job summaries, and suppression of noisy upstream warnings.

Kept separate from main.py so that file can focus on the bot's forecasting
logic. main_with_no_framework.py keeps its own inline copies on purpose --
it's meant to be a single-file reference implementation.
"""
from __future__ import annotations

import logging
import os
import sys
import warnings
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:  # pragma: no cover
    from bot_config import BotSettings
    from spend_guard import SpendGuard


# Placeholder values shipped in .env.template. If a real env var still equals
# one of these the user forgot to replace it; we'd rather fail loudly here than
# inside the SDK three layers down.
_PLACEHOLDER_ENV_VALUES = {
    "1234567890",
    "REPLACE_ME",
    "your-token-here",
    "your-api-key-here",
}

REQUIRED_SECRETS = ("METACULUS_TOKEN", "ANTHROPIC_API_KEY")
SETUP_DOC = "SETUP-FOR-MAX.md"


def _is_real_env(name: str) -> bool:
    val = os.getenv(name)
    return bool(val and val.strip() and val.strip() not in _PLACEHOLDER_ENV_VALUES)


def silence_noisy_dependencies() -> None:
    """
    Quiet warnings from transitive deps that fire on import and confuse new
    users. Must be called *before* importing forecasting_tools.
    """
    warnings.filterwarnings(
        "ignore", message=r".*does not support cost tracking.*"
    )
    logging.getLogger("forecasting_tools.ai_models.model_tracker").setLevel(
        logging.ERROR
    )
    # Streamlit installs its own logger hierarchy; suppress via its own API.
    try:
        from streamlit.logger import set_log_level

        set_log_level("error")
    except ImportError:
        pass
    # LiteLLM is verbose at INFO; its WARNING level is enough for us.
    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False


def missing_required_secrets() -> list[str]:
    """Names of required secrets that are unset or still placeholders."""
    return [name for name in REQUIRED_SECRETS if not _is_real_env(name)]


def github_event_name() -> str:
    return os.getenv("GITHUB_EVENT_NAME", "")


def is_scheduled_run() -> bool:
    return github_event_name() == "schedule"


def check_environment(strict: bool = True) -> list[str]:
    """
    Verify METACULUS_TOKEN and ANTHROPIC_API_KEY are set.

    Returns the list of missing secret names. When something is missing and
    strict=True the process exits: with status 0 on a *scheduled* GitHub
    Actions run (nothing is configured yet, so there is nothing to fail) and
    status 1 everywhere else (a human asked for a run and should see red).
    """
    missing = missing_required_secrets()
    if not missing:
        return []

    names = ", ".join(missing)
    message = (
        f"Bot is not configured: missing repository secret(s) {names}. "
        f"Add them under Settings -> Secrets and variables -> Actions, "
        f"see {SETUP_DOC}. No forecasts were made and nothing was spent."
    )
    print(f"❌  {message}")
    write_job_summary(
        "## ⏭️ Bot not configured yet\n\n"
        f"Missing repository secret(s): **{names}**.\n\n"
        f"Follow `{SETUP_DOC}` to add them. No forecasts were made and nothing was spent."
    )
    if strict:
        sys.exit(0 if is_scheduled_run() else 1)
    return missing


def write_job_summary(markdown: str) -> None:
    """
    Append markdown to the GitHub Actions job summary (shown on the run page)
    when running in Actions; always echo it to stdout too.
    """
    path = os.getenv("GITHUB_STEP_SUMMARY")
    if path:
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(markdown.rstrip() + "\n\n")
        except OSError as e:  # pragma: no cover
            print(f"(could not write job summary: {e})")
    print()
    print(markdown)
    print()


def print_startup_banner(run_mode: str, will_publish: bool, details: str = "") -> None:
    publish = "publish=yes" if will_publish else "publish=no (dry run)"
    print(f"🤖  Running mode={run_mode}, {publish}")
    if details:
        print(f"    {details}")
    print()


def _short_error(exc: BaseException, limit: int = 300) -> str:
    msg = f"{type(exc).__name__}: {exc}"
    return msg if len(msg) <= limit else msg[:limit] + "..."


def build_run_summary_markdown(
    *,
    run_mode: str,
    will_publish: bool,
    forecast_reports: Sequence[Any],
    guard: "SpendGuard | None",
    settings: "BotSettings | None",
    fatal_error: BaseException | None = None,
    tournament_url: str | None = None,
) -> str:
    """Short, phone-readable markdown for the GitHub job summary."""
    from forecasting_tools import ForecastReport

    valid = [r for r in forecast_reports if isinstance(r, ForecastReport)]
    failed = [r for r in forecast_reports if isinstance(r, BaseException)]
    publish_word = "posted to Metaculus" if will_publish else "produced (dry run, not posted)"

    if fatal_error is not None:
        headline = f"❌ Run crashed before finishing: `{_short_error(fatal_error)}`"
    elif not forecast_reports:
        headline = "ℹ️ No new questions to forecast on this run."
    elif failed and not valid:
        headline = f"❌ All {len(failed)} question(s) failed. Nothing was posted."
    elif failed:
        headline = (
            f"⚠️ {len(valid)} forecast(s) {publish_word}, {len(failed)} question(s) failed."
        )
    else:
        headline = f"✅ {len(valid)} forecast(s) {publish_word}."

    lines = [f"## 🤖 Forecast run · mode `{run_mode}`", "", f"**{headline}**", ""]
    if guard is not None:
        lines += ["### Spend", guard.summary_markdown(), ""]
    if valid:
        lines.append("### Forecasts")
        for r in valid:
            note = f" ({len(r.errors)} minor error(s))" if r.errors else ""
            lines.append(f"- ✅ {r.question.page_url}{note}")
        lines.append("")
    if failed:
        lines.append("### Failed questions")
        for exc in failed:
            lines.append(f"- ❌ `{_short_error(exc)}`")
        lines.append("")
    if settings is not None:
        lines.append(
            f"<sub>{settings.describe()}</sub>"
        )
        if settings.warnings:
            lines.append("")
            for w in settings.warnings:
                lines.append(f"- ⚠️ {w}")
    if tournament_url:
        lines += ["", f"Tournament: {tournament_url}"]
    return "\n".join(lines).rstrip() + "\n"


def print_run_summary_banner(
    forecast_reports: Sequence[Any],
    will_publish: bool,
    tournament_url: str | None = None,
) -> None:
    """
    End-of-run summary printed via print() (not logger) so it survives log
    filtering. Shows count, per-question URLs, and any failure tracebacks.
    If tournament_url is given, it's included as a footer link.
    """
    # Lazy import so this module is usable in contexts where forecasting_tools
    # isn't installed (e.g. unit tests of the banner format).
    from forecasting_tools import ForecastReport

    valid = [r for r in forecast_reports if isinstance(r, ForecastReport)]
    exceptions = [r for r in forecast_reports if isinstance(r, BaseException)]
    banner = "=" * 80

    print()
    print(banner)

    if not forecast_reports:
        print("ℹ️   No new questions to forecast on this run.")
        print(banner)
        print()
        return

    if valid and not exceptions:
        verb = "submitted" if will_publish else "produced (dry run)"
        print(f"🎉  Bot {verb} {len(valid)} forecast(s).")
    elif valid and exceptions:
        print(
            f"⚠️   Partial — {len(valid)} succeeded, {len(exceptions)} failed."
        )
    else:
        print(f"❌  All {len(exceptions)} attempt(s) failed.")

    if valid:
        print()
        for r in valid:
            note = f"  (with {len(r.errors)} minor error(s))" if r.errors else ""
            print(f"  ✅ {r.question.page_url}{note}")
        if will_publish and tournament_url:
            print(f"\n  Tournament: {tournament_url}")

    if exceptions:
        print()
        for exc in exceptions:
            print(f"  ❌ {_short_error(exc, 200)}")

    print(banner)
    print()
