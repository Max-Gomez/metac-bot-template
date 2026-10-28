import pytest

from bot_config import BotSettings, estimate_cost_usd, price_for_model


def test_defaults_are_claude_with_low_caps(clean_bot_env):
    s = BotSettings.from_env({})
    assert s.main_model == "anthropic/claude-sonnet-5-5"
    assert s.fallback_model == "anthropic/claude-haiku-5-5"
    assert s.parser_model == s.fallback_model
    assert s.research_model == s.main_model
    assert s.effort == "medium"
    assert s.research_mode == "web_search"
    assert 0 < s.max_cost_per_run_usd <= 5
    assert s.max_cost_per_run_usd <= s.max_cost_per_day_usd <= 15
    assert s.publish is True
    assert s.warnings == ()


def test_env_overrides(clean_bot_env):
    s = BotSettings.from_env(
        {
            "BOT_MAIN_MODEL": "anthropic/claude-opus-5-5",
            "BOT_FALLBACK_MODEL": "anthropic/claude-sonnet-5-5",
            "BOT_EFFORT": "HIGH",
            "BOT_RESEARCH_MODE": "none",
            "BOT_MAX_COST_PER_RUN_USD": "1.5",
            "BOT_MAX_COST_PER_DAY_USD": "9",
            "BOT_PREDICTIONS_PER_QUESTION": "5",
            "BOT_PUBLISH": "false",
            "BOT_MAIN_MAX_TOKENS": "",  # empty == unset
        }
    )
    assert s.main_model == "anthropic/claude-opus-5-5"
    assert s.parser_model == "anthropic/claude-sonnet-5-5"  # follows fallback
    assert s.research_model == "anthropic/claude-opus-5-5"  # follows main
    assert s.effort == "high"
    assert s.research_mode == "none"
    assert s.max_cost_per_run_usd == 1.5
    assert s.max_cost_per_day_usd == 9
    assert s.predictions_per_question == 5
    assert s.publish is False
    assert s.main_max_tokens == BotSettings().main_max_tokens


@pytest.mark.parametrize(
    "env",
    [
        {"BOT_EFFORT": "xhigh"},  # litellm 1.80 only maps low/medium/high
        {"BOT_RESEARCH_MODE": "perplexity"},
        {"BOT_MAX_COST_PER_RUN_USD": "0"},
        {"BOT_MAX_COST_PER_DAY_USD": "-1"},
        {"BOT_MAX_COST_PER_RUN_USD": "10", "BOT_MAX_COST_PER_DAY_USD": "5"},
        {"BOT_PREDICTIONS_PER_QUESTION": "0"},
        {"BOT_MAX_COST_PER_RUN_USD": "lots"},
        {"BOT_PUBLISH": "maybe"},
    ],
)
def test_invalid_settings_are_rejected(clean_bot_env, env):
    with pytest.raises(ValueError):
        BotSettings.from_env(env)


def test_non_anthropic_model_produces_warning(clean_bot_env):
    s = BotSettings.from_env({"BOT_MAIN_MODEL": "openrouter/openai/gpt-4o"})
    assert any("not an Anthropic model" in w for w in s.warnings)


def test_prices_known_for_default_models(clean_bot_env):
    s = BotSettings.from_env({})
    for model in s.models_in_use().values():
        assert price_for_model(model) is not None, model


def test_estimate_cost_uses_list_prices():
    # Sonnet 5.5: $2 in / $10 out per 1M tokens
    assert estimate_cost_usd("anthropic/claude-sonnet-5-5", 1_000_000, 0) == pytest.approx(2.0)
    assert estimate_cost_usd("anthropic/claude-sonnet-5-5", 0, 1_000_000) == pytest.approx(10.0)
    # Haiku 5.5: $0.10 in / $0.50 out
    assert estimate_cost_usd("anthropic/claude-haiku-5-5", 1_000_000, 1_000_000) == pytest.approx(0.6)
    # unknown model: priced at the most expensive tier so caps trip early
    assert estimate_cost_usd("anthropic/claude-mystery-9", 1_000_000, 0) == pytest.approx(10.0)
    # web search surcharge
    assert estimate_cost_usd("anthropic/claude-haiku-5-5", 0, 0, web_searches=3) == pytest.approx(0.03)
