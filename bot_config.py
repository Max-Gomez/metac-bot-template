"""
Runtime settings for the bot, all read from environment variables.

Every knob has a conservative default so the bot is safe to run with only
METACULUS_TOKEN and ANTHROPIC_API_KEY set. In GitHub Actions the BOT_* values
can be overridden with repository *variables* (Settings -> Secrets and
variables -> Actions -> Variables), no code change needed.

Nothing in this module reads or prints secret values.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Mapping

logger = logging.getLogger(__name__)

# Anthropic first-party list prices, USD per 1M tokens (input, output).
# Used only as a fallback estimate when litellm cannot price a response.
# Keys are matched as substrings of the model name, longest match wins.
FALLBACK_PRICES_PER_MILLION: dict[str, tuple[float, float]] = {
    "claude-fable-5-1": (10.0, 50.0),
    "claude-fable-5": (10.0, 50.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-haiku-4-5": (1.0, 5.0),
}
# If we know nothing about a model, assume the most expensive tier so that the
# spend caps trip early rather than late.
UNKNOWN_MODEL_PRICE_PER_MILLION: tuple[float, float] = (10.0, 50.0)

VALID_EFFORTS = ("low", "medium", "high")
VALID_RESEARCH_MODES = ("web_search", "none")


def _env_str(env: Mapping[str, str], key: str, default: str) -> str:
    value = env.get(key)
    if value is None or not value.strip():
        return default
    return value.strip()


def _env_float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = _env_str(env, key, str(default))
    try:
        return float(raw)
    except ValueError as e:
        raise ValueError(f"{key} must be a number, got {raw!r}") from e


def _env_int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = _env_str(env, key, str(default))
    try:
        return int(raw)
    except ValueError as e:
        raise ValueError(f"{key} must be an integer, got {raw!r}") from e


def _env_bool(env: Mapping[str, str], key: str, default: bool) -> bool:
    raw = _env_str(env, key, "true" if default else "false").lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{key} must be true/false, got {raw!r}")


@dataclass(frozen=True)
class BotSettings:
    # --- models (litellm names; "anthropic/" prefix routes to the Anthropic API) ---
    main_model: str = "anthropic/claude-sonnet-5-5"
    fallback_model: str = "anthropic/claude-haiku-5-5"
    research_model: str = "anthropic/claude-sonnet-5-5"
    parser_model: str = "anthropic/claude-haiku-5-5"
    summarizer_model: str = "anthropic/claude-haiku-5-5"
    # Claude 4.6+ models think adaptively by default; effort controls how much.
    # litellm 1.80 only accepts low/medium/high here.
    effort: str = "medium"
    cheap_effort: str = "low"

    # --- research ---
    research_mode: str = "web_search"  # or "none"
    web_search_tool_type: str = "web_search_20260209"
    web_search_max_uses: int = 3
    # Added to the cost estimate of a research call when litellm reports no
    # cost. Anthropic bills $10 per 1,000 searches.
    web_search_cost_per_search_usd: float = 0.01

    # --- hard spend caps (USD) ---
    max_cost_per_run_usd: float = 4.0
    max_cost_per_day_usd: float = 12.0
    spend_ledger_path: str = ".spend/ledger.jsonl"

    # --- forecasting shape ---
    research_reports_per_question: int = 1
    predictions_per_question: int = 3

    # --- plumbing ---
    max_concurrent_llm_calls: int = 3
    llm_timeout_seconds: float = 240.0
    cheap_llm_timeout_seconds: float = 90.0
    main_max_tokens: int = 12000
    cheap_max_tokens: int = 4000
    allowed_tries: int = 2
    publish: bool = True

    warnings: tuple[str, ...] = field(default=(), compare=False)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "BotSettings":
        env = os.environ if env is None else env
        defaults = cls()
        main_model = _env_str(env, "BOT_MAIN_MODEL", defaults.main_model)
        fallback_model = _env_str(env, "BOT_FALLBACK_MODEL", defaults.fallback_model)
        settings = cls(
            main_model=main_model,
            fallback_model=fallback_model,
            research_model=_env_str(env, "BOT_RESEARCH_MODEL", main_model),
            parser_model=_env_str(env, "BOT_PARSER_MODEL", fallback_model),
            summarizer_model=_env_str(env, "BOT_SUMMARIZER_MODEL", fallback_model),
            effort=_env_str(env, "BOT_EFFORT", defaults.effort).lower(),
            cheap_effort=_env_str(env, "BOT_CHEAP_EFFORT", defaults.cheap_effort).lower(),
            research_mode=_env_str(env, "BOT_RESEARCH_MODE", defaults.research_mode).lower(),
            web_search_tool_type=_env_str(
                env, "BOT_WEB_SEARCH_TOOL_TYPE", defaults.web_search_tool_type
            ),
            web_search_max_uses=_env_int(
                env, "BOT_WEB_SEARCH_MAX_USES", defaults.web_search_max_uses
            ),
            web_search_cost_per_search_usd=_env_float(
                env,
                "BOT_WEB_SEARCH_COST_PER_SEARCH_USD",
                defaults.web_search_cost_per_search_usd,
            ),
            max_cost_per_run_usd=_env_float(
                env, "BOT_MAX_COST_PER_RUN_USD", defaults.max_cost_per_run_usd
            ),
            max_cost_per_day_usd=_env_float(
                env, "BOT_MAX_COST_PER_DAY_USD", defaults.max_cost_per_day_usd
            ),
            spend_ledger_path=_env_str(
                env, "BOT_SPEND_LEDGER_PATH", defaults.spend_ledger_path
            ),
            research_reports_per_question=_env_int(
                env,
                "BOT_RESEARCH_REPORTS_PER_QUESTION",
                defaults.research_reports_per_question,
            ),
            predictions_per_question=_env_int(
                env, "BOT_PREDICTIONS_PER_QUESTION", defaults.predictions_per_question
            ),
            max_concurrent_llm_calls=_env_int(
                env, "BOT_MAX_CONCURRENT_LLM_CALLS", defaults.max_concurrent_llm_calls
            ),
            llm_timeout_seconds=_env_float(
                env, "BOT_LLM_TIMEOUT_SECONDS", defaults.llm_timeout_seconds
            ),
            cheap_llm_timeout_seconds=_env_float(
                env, "BOT_CHEAP_LLM_TIMEOUT_SECONDS", defaults.cheap_llm_timeout_seconds
            ),
            main_max_tokens=_env_int(env, "BOT_MAIN_MAX_TOKENS", defaults.main_max_tokens),
            cheap_max_tokens=_env_int(
                env, "BOT_CHEAP_MAX_TOKENS", defaults.cheap_max_tokens
            ),
            allowed_tries=_env_int(env, "BOT_LLM_ALLOWED_TRIES", defaults.allowed_tries),
            publish=_env_bool(env, "BOT_PUBLISH", defaults.publish),
        )
        settings.validate()
        object.__setattr__(settings, "warnings", tuple(settings._collect_warnings()))
        return settings

    def validate(self) -> None:
        problems: list[str] = []
        if self.effort not in VALID_EFFORTS:
            problems.append(f"BOT_EFFORT must be one of {VALID_EFFORTS}, got {self.effort!r}")
        if self.cheap_effort not in VALID_EFFORTS:
            problems.append(
                f"BOT_CHEAP_EFFORT must be one of {VALID_EFFORTS}, got {self.cheap_effort!r}"
            )
        if self.research_mode not in VALID_RESEARCH_MODES:
            problems.append(
                f"BOT_RESEARCH_MODE must be one of {VALID_RESEARCH_MODES}, got {self.research_mode!r}"
            )
        if self.max_cost_per_run_usd <= 0:
            problems.append("BOT_MAX_COST_PER_RUN_USD must be > 0 (the cap is mandatory)")
        if self.max_cost_per_day_usd <= 0:
            problems.append("BOT_MAX_COST_PER_DAY_USD must be > 0 (the cap is mandatory)")
        if self.max_cost_per_day_usd < self.max_cost_per_run_usd:
            problems.append("BOT_MAX_COST_PER_DAY_USD must be >= BOT_MAX_COST_PER_RUN_USD")
        if self.predictions_per_question < 1:
            problems.append("BOT_PREDICTIONS_PER_QUESTION must be >= 1")
        if self.research_reports_per_question < 1:
            problems.append("BOT_RESEARCH_REPORTS_PER_QUESTION must be >= 1")
        if self.max_concurrent_llm_calls < 1:
            problems.append("BOT_MAX_CONCURRENT_LLM_CALLS must be >= 1")
        if self.web_search_max_uses < 1:
            problems.append("BOT_WEB_SEARCH_MAX_USES must be >= 1")
        if self.allowed_tries < 1:
            problems.append("BOT_LLM_ALLOWED_TRIES must be >= 1")
        if self.llm_timeout_seconds <= 0 or self.cheap_llm_timeout_seconds <= 0:
            problems.append("LLM timeouts must be > 0")
        if self.main_max_tokens < 1024 or self.cheap_max_tokens < 256:
            problems.append("max_tokens settings are too small to hold a forecast")
        if problems:
            raise ValueError("Invalid bot settings:\n  - " + "\n  - ".join(problems))

    def _collect_warnings(self) -> list[str]:
        warnings: list[str] = []
        for label, model in self.models_in_use().items():
            if "claude" not in model and "anthropic" not in model:
                warnings.append(
                    f"{label} model {model!r} is not an Anthropic model; only "
                    "ANTHROPIC_API_KEY is wired into the workflows, so this will "
                    "probably fail at runtime."
                )
            if price_for_model(model) is None:
                warnings.append(
                    f"{label} model {model!r} has no known price; costs will be "
                    "estimated at the most expensive tier."
                )
        return warnings

    def models_in_use(self) -> dict[str, str]:
        return {
            "main": self.main_model,
            "fallback": self.fallback_model,
            "research": self.research_model,
            "parser": self.parser_model,
            "summarizer": self.summarizer_model,
        }

    @property
    def web_search_enabled(self) -> bool:
        return self.research_mode == "web_search"

    def describe(self) -> str:
        return (
            f"main={self.main_model} (effort={self.effort}), fallback={self.fallback_model}, "
            f"research={self.research_model} mode={self.research_mode}, "
            f"parser={self.parser_model}, caps: ${self.max_cost_per_run_usd:.2f}/run "
            f"${self.max_cost_per_day_usd:.2f}/day, predictions/question={self.predictions_per_question}"
        )


def strip_provider_prefix(model: str) -> str:
    """'anthropic/claude-sonnet-5-5' -> 'claude-sonnet-5-5'."""
    return model.split("/")[-1]


def price_for_model(model: str) -> tuple[float, float] | None:
    """
    USD per 1M (input, output) tokens for a model, or None if unknown.
    Prefers litellm's price table, falls back to the table above.
    """
    bare = strip_provider_prefix(model)
    try:
        from litellm import model_cost  # imported lazily: litellm import is slow

        info = model_cost.get(bare) or model_cost.get(model)
        if info and info.get("input_cost_per_token") is not None:
            return (
                float(info["input_cost_per_token"]) * 1_000_000,
                float(info.get("output_cost_per_token") or 0) * 1_000_000,
            )
    except Exception:  # pragma: no cover - litellm not importable in some tests
        pass
    matches = [key for key in FALLBACK_PRICES_PER_MILLION if key in bare]
    if not matches:
        return None
    best = max(matches, key=len)
    return FALLBACK_PRICES_PER_MILLION[best]


def estimate_cost_usd(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    web_searches: int = 0,
    cost_per_search_usd: float = 0.01,
) -> float:
    """Token-based cost estimate, used when litellm reports no cost."""
    price = price_for_model(model) or UNKNOWN_MODEL_PRICE_PER_MILLION
    input_price, output_price = price
    cost = (
        max(prompt_tokens, 0) * input_price + max(completion_tokens, 0) * output_price
    ) / 1_000_000
    cost += max(web_searches, 0) * cost_per_search_usd
    return cost
