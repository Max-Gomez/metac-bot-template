"""
Claude-on-Anthropic LLM wrapper for the bot.

ClaudeLlm is a forecasting-tools GeneralLlm (so the ForecastBot machinery,
structure_output, cost tracking and the Metaculus comment metadata all keep
working) that adds:

- spend guard: every call is checked against the per-run / per-day caps
  before it starts and its real cost is recorded afterwards
- retry then fallback: GeneralLlm already retries `allowed_tries` times with
  backoff; if the primary model still fails (API error, timeout, empty
  answer) the call is repeated on a cheaper fallback model
- optional Anthropic web search for research, automatically dropped if the
  API rejects the tool so research degrades to knowledge-only instead of
  failing
- a shared concurrency limit so the caps cannot be overshot by a burst

Request shape (verified against litellm 1.80): no `temperature`, no
`thinking`/`reasoning_effort` (both are rejected or mis-mapped for Claude 5.x),
just `max_tokens` and `output_config.effort`.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import litellm
from forecasting_tools import GeneralLlm
from forecasting_tools.ai_models.ai_utils.response_types import TextTokenCostResponse

from bot_config import BotSettings, estimate_cost_usd
from spend_guard import BudgetExceededError, SpendGuard

logger = logging.getLogger(__name__)


class LlmThrottle:
    """Lazily-created semaphore shared by all ClaudeLlm instances in a loop."""

    def __init__(self, max_concurrent: int) -> None:
        self.max_concurrent = max_concurrent
        self._semaphores: dict[int, asyncio.Semaphore] = {}

    def semaphore(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        key = id(loop)
        if key not in self._semaphores:
            self._semaphores[key] = asyncio.Semaphore(self.max_concurrent)
        return self._semaphores[key]


class ClaudeLlm(GeneralLlm):
    def __init__(
        self,
        model: str,
        *,
        guard: SpendGuard,
        role: str,
        effort: str | None = None,
        max_tokens: int = 4000,
        timeout: float = 120,
        allowed_tries: int = 2,
        fallback: "ClaudeLlm | None" = None,
        throttle: LlmThrottle | None = None,
        web_search_tool: dict[str, Any] | None = None,
        web_search_cost_per_search_usd: float = 0.01,
        **litellm_kwargs: Any,
    ) -> None:
        kwargs = dict(litellm_kwargs)
        if effort:
            kwargs["output_config"] = {"effort": effort}
        if web_search_tool:
            kwargs["tools"] = [dict(web_search_tool)]
        super().__init__(
            model=model,
            allowed_tries=allowed_tries,
            temperature=None,
            timeout=timeout,
            max_tokens=max_tokens,
            **kwargs,
        )
        self.guard = guard
        self.role = role
        self.effort = effort
        self.fallback = fallback
        self.throttle = throttle
        self._web_search_tool = dict(web_search_tool) if web_search_tool else None
        self._web_search_cost_per_search_usd = web_search_cost_per_search_usd
        self.web_search_disabled = False

    # ------------------------------------------------------------------ invoke

    async def invoke(self, prompt: Any, system_prompt: str | None = None) -> str:
        if system_prompt is not None:
            if isinstance(prompt, list):
                raise ValueError("System prompt cannot be combined with a message list")
            prompt = self.model_input_to_message(prompt, system_prompt)
        try:
            return await self._guarded_invoke(prompt)
        except BudgetExceededError:
            raise
        except Exception as primary_error:
            if self.fallback is None:
                raise
            logger.warning(
                f"[{self.role}] {self.model} failed after {self.allowed_tries} tr"
                f"{'y' if self.allowed_tries == 1 else 'ies'} "
                f"({type(primary_error).__name__}: {str(primary_error)[:300]}). "
                f"Falling back to {self.fallback.model}."
            )
            self.guard.note_fallback(self.role, self.model, self.fallback.model)
            try:
                return await self.fallback.invoke(prompt)
            except BudgetExceededError:
                raise
            except Exception as fallback_error:
                raise RuntimeError(
                    f"[{self.role}] both {self.model} and fallback {self.fallback.model} "
                    f"failed. Primary: {type(primary_error).__name__}: "
                    f"{str(primary_error)[:300]}. Fallback: "
                    f"{type(fallback_error).__name__}: {str(fallback_error)[:300]}"
                ) from fallback_error

    async def _guarded_invoke(self, prompt: Any) -> str:
        self.guard.check()
        if self.throttle is not None:
            async with self.throttle.semaphore():
                response = await self._invoke_allowing_web_search_downgrade(prompt)
        else:
            response = await self._invoke_allowing_web_search_downgrade(prompt)
        cost = self._cost_of(response)
        self.guard.record(
            cost_usd=cost,
            model=self.model,
            role=self.role,
            prompt_tokens=response.prompt_tokens_used,
            completion_tokens=response.completion_tokens_used,
        )
        return response.data

    async def _invoke_allowing_web_search_downgrade(
        self, prompt: Any
    ) -> TextTokenCostResponse:
        try:
            return await self._invoke_with_request_cost_time_and_token_limits_and_retry(
                prompt
            )
        except litellm.BadRequestError as error:
            if not self.litellm_kwargs.get("tools"):
                raise
            logger.warning(
                f"[{self.role}] {self.model} rejected the request with web search tool "
                f"{self._web_search_tool!r} ({str(error)[:300]}). Disabling web search "
                "for this model and retrying without it."
            )
            self.litellm_kwargs.pop("tools", None)
            self.web_search_disabled = True
            return await self._invoke_with_request_cost_time_and_token_limits_and_retry(
                prompt
            )

    # -------------------------------------------------------------------- cost

    def _cost_of(self, response: TextTokenCostResponse) -> float:
        cost = float(response.cost or 0.0)
        if cost > 0:
            return cost
        searches = 0
        if self.litellm_kwargs.get("tools") and not self.web_search_disabled:
            searches = int(self._web_search_tool.get("max_uses", 1)) if self._web_search_tool else 1
        estimate = estimate_cost_usd(
            self.model,
            response.prompt_tokens_used,
            response.completion_tokens_used,
            web_searches=searches,
            cost_per_search_usd=self._web_search_cost_per_search_usd,
        )
        logger.debug(
            f"[{self.role}] litellm reported no cost for {self.model}; "
            f"estimated ${estimate:.5f} from tokens"
        )
        return estimate

    # ---------------------------------------------------------------- metadata

    @property
    def web_search_enabled(self) -> bool:
        return bool(self._web_search_tool) and not self.web_search_disabled

    def to_dict(self) -> dict[str, Any]:
        info = super().to_dict()
        info.pop("tools", None)
        info.update(
            {
                "role": self.role,
                "effort": self.effort,
                "fallback_model": self.fallback.model if self.fallback else None,
                "web_search": self.web_search_enabled,
            }
        )
        return info


def make_llms(settings: BotSettings, guard: SpendGuard) -> dict[str, GeneralLlm | str]:
    """Build the ForecastBot `llms` dict from settings."""
    throttle = LlmThrottle(settings.max_concurrent_llm_calls)

    def cheap(role: str, model: str) -> ClaudeLlm:
        return ClaudeLlm(
            model,
            guard=guard,
            role=role,
            effort=settings.cheap_effort,
            max_tokens=settings.cheap_max_tokens,
            timeout=settings.cheap_llm_timeout_seconds,
            allowed_tries=settings.allowed_tries,
            throttle=throttle,
        )

    def fallback_for(role: str) -> ClaudeLlm | None:
        if settings.fallback_model == settings.main_model:
            return None
        return ClaudeLlm(
            settings.fallback_model,
            guard=guard,
            role=f"{role}-fallback",
            effort=settings.effort,
            max_tokens=settings.main_max_tokens,
            timeout=settings.llm_timeout_seconds,
            allowed_tries=settings.allowed_tries,
            throttle=throttle,
        )

    default_llm = ClaudeLlm(
        settings.main_model,
        guard=guard,
        role="forecaster",
        effort=settings.effort,
        max_tokens=settings.main_max_tokens,
        timeout=settings.llm_timeout_seconds,
        allowed_tries=settings.allowed_tries,
        fallback=fallback_for("forecaster"),
        throttle=throttle,
    )

    researcher: GeneralLlm | str
    if settings.research_mode == "none":
        researcher = "no_research"
    else:
        web_search_tool = {
            "type": settings.web_search_tool_type,
            "name": "web_search",
            "max_uses": settings.web_search_max_uses,
        }
        researcher = ClaudeLlm(
            settings.research_model,
            guard=guard,
            role="researcher",
            effort=settings.effort,
            max_tokens=settings.main_max_tokens,
            timeout=settings.llm_timeout_seconds,
            allowed_tries=settings.allowed_tries,
            fallback=fallback_for("researcher"),
            throttle=throttle,
            web_search_tool=web_search_tool,
            web_search_cost_per_search_usd=settings.web_search_cost_per_search_usd,
        )

    return {
        "default": default_llm,
        "researcher": researcher,
        "parser": cheap("parser", settings.parser_model),
        "summarizer": cheap("summarizer", settings.summarizer_model),
    }
