"""
Pure functions that keep malformed LLM output from ever reaching Metaculus.

- binary probabilities are clipped to [1%, 99%]
- multiple-choice outputs must name every option exactly once, get a 1% floor
  per option, and are renormalised to sum to 1
- numeric / date percentile lists must be finite, strictly increasing in
  percentile, non-decreasing in value, and are clamped to closed bounds
- a forecast is refused if it has no real written reasoning to post with it

Anything that cannot be repaired safely raises ValueError, which makes that
single forecast sample fail (forecasting-tools then falls back to the other
samples, or fails the question loudly if too few remain).
"""
from __future__ import annotations

import math
import re
from typing import Iterable, Sequence

from forecasting_tools import Percentile, PredictedOptionList
from forecasting_tools.data_models.multiple_choice_report import PredictedOption

MIN_PROBABILITY = 0.01
MAX_PROBABILITY = 0.99
MIN_OPTION_PROBABILITY = 0.01
MIN_PERCENTILES = 3
MIN_REASONING_CHARS = 200
# Reasoning text produced by forecasting-tools itself for conditional
# questions whose parent/child already has a live forecast.
REAFFIRMED_PREFIX = "Already existing forecast reaffirmed"


def clip_probability(value: float) -> float:
    """Clip a binary probability to [0.01, 0.99]; reject non-finite input."""
    try:
        p = float(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Probability is not a number: {value!r}") from e
    if not math.isfinite(p):
        raise ValueError(f"Probability is not finite: {value!r}")
    if p < 0 or p > 1:
        raise ValueError(f"Probability {p} is outside [0, 1]")
    return min(MAX_PROBABILITY, max(MIN_PROBABILITY, p))


# "Option Red" -> "Red", "Option A: Red" / "Choice 2) Red" -> "Red"
_OPTION_PREFIXES = (
    re.compile(r"^(?:option|choice|answer)\s*[:.\-)]?\s+", re.I),
    re.compile(r"^(?:option|choice|answer)?\s*[a-z0-9]{1,2}\s*[:.)\-]\s+", re.I),
)


def normalize_option_name(name: str) -> str:
    """Case/whitespace/punctuation-insensitive key for matching option names."""
    text = str(name).strip().strip("\"'`*").strip()
    text = re.sub(r"\s+", " ", text)
    return text.casefold()


def _match_option(name: str, options: Sequence[str]) -> str | None:
    by_key = {normalize_option_name(o): o for o in options}
    key = normalize_option_name(name)
    if key in by_key:
        return by_key[key]
    # Tolerate a leading "Option A:" style prefix the parser may have kept.
    for pattern in _OPTION_PREFIXES:
        stripped = normalize_option_name(pattern.sub("", str(name).strip(), count=1))
        if stripped and stripped in by_key:
            return by_key[stripped]
    return None


def sanitize_multiple_choice(
    options: Sequence[str], predicted: PredictedOptionList
) -> PredictedOptionList:
    """
    Map predicted option names onto the question's exact option strings,
    require every option exactly once, floor each at 1%, renormalise to 1.
    """
    if len(options) < 2:
        raise ValueError("Multiple choice question needs at least 2 options")
    probabilities: dict[str, float] = {}
    for item in predicted.predicted_options:
        matched = _match_option(item.option_name, options)
        if matched is None:
            raise ValueError(
                f"Predicted option {item.option_name!r} is not one of {list(options)}"
            )
        if matched in probabilities:
            raise ValueError(f"Option {matched!r} was predicted more than once")
        p = float(item.probability)
        if not math.isfinite(p) or p < 0 or p > 1:
            raise ValueError(f"Probability for {matched!r} is invalid: {item.probability!r}")
        probabilities[matched] = p
    missing = [o for o in options if o not in probabilities]
    if missing:
        raise ValueError(f"Prediction is missing options: {missing}")

    normalized = _floor_and_normalize(
        [probabilities[o] for o in options], floor=MIN_OPTION_PROBABILITY
    )
    return PredictedOptionList(
        predicted_options=[
            PredictedOption(option_name=o, probability=p)
            for o, p in zip(options, normalized)
        ]
    )


def _floor_and_normalize(values: Sequence[float], floor: float) -> list[float]:
    """Scale to sum 1 while guaranteeing every entry >= floor (n * floor <= 1)."""
    n = len(values)
    if n * floor > 1:
        raise ValueError(f"Cannot give {n} options a {floor} floor each")
    total = sum(values)
    if total <= 0:
        return [1.0 / n] * n
    scaled = [v / total for v in values]
    fixed: set[int] = set()
    for _ in range(n + 1):
        newly_fixed = {i for i, v in enumerate(scaled) if i not in fixed and v < floor}
        if not newly_fixed:
            break
        fixed |= newly_fixed
        free = [i for i in range(n) if i not in fixed]
        remaining = 1.0 - floor * len(fixed)
        free_total = sum(scaled[i] for i in free)
        for i in fixed:
            scaled[i] = floor
        if free and free_total > 0:
            for i in free:
                scaled[i] = scaled[i] / free_total * remaining
        elif free:
            for i in free:
                scaled[i] = remaining / len(free)
    # absorb floating point drift into the largest entry
    drift = 1.0 - sum(scaled)
    scaled[max(range(n), key=lambda i: scaled[i])] += drift
    return scaled


def sanitize_percentiles(
    percentiles: Iterable[Percentile],
    *,
    lower_bound: float,
    upper_bound: float,
    open_lower_bound: bool,
    open_upper_bound: bool,
) -> list[Percentile]:
    """
    Clean a parsed percentile list for a numeric or date question (dates are
    passed as timestamps). Drops non-finite entries, de-duplicates percentiles,
    sorts, requires >= MIN_PERCENTILES usable points, requires non-decreasing
    values, and clamps values to closed bounds.
    """
    if not math.isfinite(lower_bound) or not math.isfinite(upper_bound):
        raise ValueError("Question bounds are not finite")
    if upper_bound <= lower_bound:
        raise ValueError("Question upper bound must exceed lower bound")

    cleaned: dict[float, float] = {}
    for item in percentiles:
        try:
            pct = float(item.percentile)
            val = float(item.value)
        except (TypeError, ValueError, AttributeError):
            continue
        if not (math.isfinite(pct) and math.isfinite(val)):
            continue
        if not 0 < pct < 1:
            continue
        pct = round(pct, 6)
        if pct in cleaned:
            continue
        cleaned[pct] = val
    if len(cleaned) < MIN_PERCENTILES:
        raise ValueError(
            f"Only {len(cleaned)} usable percentile(s) parsed; need at least {MIN_PERCENTILES}"
        )
    ordered = sorted(cleaned.items())
    values = [v for _, v in ordered]
    for earlier, later in zip(values, values[1:]):
        if later < earlier:
            raise ValueError(
                f"Percentile values are not non-decreasing: {values}. Refusing to guess."
            )
    span = upper_bound - lower_bound
    result: list[Percentile] = []
    for pct, val in ordered:
        if not open_lower_bound:
            val = max(val, lower_bound)
        if not open_upper_bound:
            val = min(val, upper_bound)
        # Reject distributions that live entirely far outside the question range.
        result.append(Percentile(percentile=pct, value=val))
    if all(v.value < lower_bound - 2 * span or v.value > upper_bound + 2 * span for v in result):
        raise ValueError("All percentile values are far outside the question range")
    return result


def validate_reasoning(reasoning: str, *, context: str = "") -> None:
    """
    Refuse to publish a forecast whose rationale is empty or trivially short.
    The bot's comment on Metaculus must be its real reasoning.
    """
    text = (reasoning or "").strip()
    if text.startswith(REAFFIRMED_PREFIX):
        return
    if len(text) < MIN_REASONING_CHARS:
        raise ValueError(
            f"Refusing to publish {context or 'forecast'}: reasoning is only "
            f"{len(text)} characters (minimum {MIN_REASONING_CHARS}). "
            "A forecast must be posted together with the bot's written rationale."
        )
