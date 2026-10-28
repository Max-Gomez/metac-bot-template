import math

import pytest
from forecasting_tools import Percentile, PredictedOptionList
from forecasting_tools.data_models.multiple_choice_report import PredictedOption

from forecast_guards import (
    clip_probability,
    sanitize_multiple_choice,
    sanitize_percentiles,
    validate_reasoning,
)


@pytest.mark.parametrize(
    "value, expected",
    [(0.0, 0.01), (0.001, 0.01), (0.5, 0.5), (0.99, 0.99), (0.999, 0.99), (1.0, 0.99)],
)
def test_clip_probability(value, expected):
    assert clip_probability(value) == pytest.approx(expected)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.5, "0.5x", None])
def test_clip_probability_rejects_garbage(value):
    with pytest.raises(ValueError):
        clip_probability(value)


def _pol(pairs):
    return PredictedOptionList(
        predicted_options=[PredictedOption(option_name=n, probability=p) for n, p in pairs]
    )


def test_multiple_choice_matches_names_and_normalizes():
    out = sanitize_multiple_choice(
        ["Red", "Blue", "Green"],
        _pol([("Option Red", 0.5), (" blue ", 0.3), ("GREEN", 0.21)]),
    )
    names = [o.option_name for o in out.predicted_options]
    probs = [o.probability for o in out.predicted_options]
    assert names == ["Red", "Blue", "Green"]  # exact question strings, question order
    assert sum(probs) == pytest.approx(1.0)
    assert probs[0] == pytest.approx(0.5 / 1.01)


@pytest.mark.parametrize(
    "raw", ["Red", "red", " Red ", "'Red'", "**Red**", "Option Red", "Option A: Red", "Choice 2) Red", "A. Red"]
)
def test_option_name_variants_match(raw):
    out = sanitize_multiple_choice(["Red", "Blue"], _pol([(raw, 0.6), ("Blue", 0.4)]))
    assert [o.option_name for o in out.predicted_options] == ["Red", "Blue"]


def test_option_prefix_stripping_does_not_eat_real_names():
    # "Answer" is itself an option here and must not be treated as a prefix.
    out = sanitize_multiple_choice(["Answer", "Other"], _pol([("Answer", 0.7), ("Other", 0.3)]))
    assert out.predicted_options[0].option_name == "Answer"
    with pytest.raises(ValueError):
        sanitize_multiple_choice(["Red", "Blue"], _pol([("Option", 0.6), ("Blue", 0.4)]))


def test_multiple_choice_floors_each_option_at_one_percent():
    out = sanitize_multiple_choice(
        ["A", "B", "C"], _pol([("A", 0.999), ("B", 0.0005), ("C", 0.0005)])
    )
    probs = {o.option_name: o.probability for o in out.predicted_options}
    assert probs["B"] == pytest.approx(0.01)
    assert probs["C"] == pytest.approx(0.01)
    assert probs["A"] == pytest.approx(0.98)
    assert sum(probs.values()) == pytest.approx(1.0)
    assert max(probs.values()) <= 0.99


def test_multiple_choice_rejects_unknown_missing_or_duplicate_options():
    with pytest.raises(ValueError, match="not one of"):
        sanitize_multiple_choice(["A", "B"], _pol([("A", 0.5), ("Z", 0.5)]))
    with pytest.raises(ValueError, match="missing"):
        sanitize_multiple_choice(["A", "B", "C"], _pol([("A", 0.5), ("B", 0.5)]))
    with pytest.raises(ValueError, match="more than once"):
        sanitize_multiple_choice(["A", "B"], _pol([("A", 0.3), ("a", 0.3), ("B", 0.4)]))


def _pcts(pairs):
    return [Percentile(percentile=p, value=v) for p, v in pairs]


def test_percentiles_cleaned_sorted_and_clamped():
    out = sanitize_percentiles(
        _pcts([(0.9, 120), (0.1, -5), (0.4, 40), (0.2, 10), (0.2, 11), (0.6, 60)]),
        lower_bound=0,
        upper_bound=100,
        open_lower_bound=False,
        open_upper_bound=False,
    )
    assert [p.percentile for p in out] == [0.1, 0.2, 0.4, 0.6, 0.9]
    assert out[0].value == 0  # clamped to closed lower bound
    assert out[-1].value == 100  # clamped to closed upper bound
    assert out[1].value == 10  # first duplicate percentile wins


def test_percentiles_keep_values_beyond_open_bounds():
    out = sanitize_percentiles(
        _pcts([(0.1, -5), (0.5, 50), (0.9, 120)]),
        lower_bound=0,
        upper_bound=100,
        open_lower_bound=True,
        open_upper_bound=True,
    )
    assert out[0].value == -5 and out[-1].value == 120


def test_percentiles_reject_too_few_after_dropping_garbage():
    with pytest.raises(ValueError, match="usable percentile"):
        sanitize_percentiles(
            _pcts([(0.1, 1), (0.2, float("nan")), (0.3, 3)]) + [Percentile(percentile=0.5, value=math.inf)],
            lower_bound=0,
            upper_bound=10,
            open_lower_bound=False,
            open_upper_bound=False,
        )


def test_percentiles_reject_decreasing_values():
    with pytest.raises(ValueError, match="non-decreasing"):
        sanitize_percentiles(
            _pcts([(0.1, 50), (0.5, 40), (0.9, 60)]),
            lower_bound=0,
            upper_bound=100,
            open_lower_bound=False,
            open_upper_bound=False,
        )


def test_percentiles_reject_distribution_far_outside_range():
    with pytest.raises(ValueError, match="far outside"):
        sanitize_percentiles(
            _pcts([(0.1, 5000), (0.5, 6000), (0.9, 7000)]),
            lower_bound=0,
            upper_bound=100,
            open_lower_bound=True,
            open_upper_bound=True,
        )


def test_validate_reasoning():
    validate_reasoning("x" * 200)
    validate_reasoning("Already existing forecast reaffirmed at 40%.")
    with pytest.raises(ValueError, match="Refusing to publish"):
        validate_reasoning("Probability: 70%")
    with pytest.raises(ValueError):
        validate_reasoning("")
