"""Fake LLM responses and a fake Metaculus client for the tests."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Callable

from forecasting_tools import (
    BinaryQuestion,
    DateQuestion,
    MetaculusClient,
    MultipleChoiceQuestion,
    NumericQuestion,
)
from forecasting_tools.ai_models.ai_utils.response_types import TextTokenCostResponse

from claude_llm import ClaudeLlm

LONG_RATIONALE = (
    "(a) About three months remain until resolution. (b) The status quo is that the "
    "event has not happened and current reporting suggests no imminent change. "
    "(c) A No scenario: the relevant deadline passes quietly as officials delay. "
    "(d) A Yes scenario: a sudden policy announcement triggers the criteria early. "
    "Weighing the base rate for similar announcements and the slow pace of the "
    "process, I put most weight on the status quo while leaving room for surprise.\n\n"
    "Probability: 72%"
)
SHORT_RATIONALE = "Probability: 70%"
RESEARCH_TEXT = (
    "Research rundown: recent reporting (dated this week) indicates the process is "
    "ongoing with no resolution yet. Sources: https://example.com/a https://example.com/b"
)
SUMMARY_TEXT = "The research says the process is ongoing with no resolution yet."


def binary_question(post_id: int = 101) -> BinaryQuestion:
    return BinaryQuestion(
        question_text="Will the thing happen before the end of the quarter?",
        id_of_post=post_id,
        id_of_question=post_id,
        page_url=f"https://www.metaculus.com/questions/{post_id}/",
        background_info="Some background.",
        resolution_criteria="Resolves Yes if the thing happens.",
        fine_print="",
        already_forecasted=False,
    )


def multiple_choice_question(post_id: int = 202) -> MultipleChoiceQuestion:
    return MultipleChoiceQuestion(
        question_text="Which colour will be chosen?",
        id_of_post=post_id,
        id_of_question=post_id,
        page_url=f"https://www.metaculus.com/questions/{post_id}/",
        background_info="Some background.",
        resolution_criteria="Resolves to the chosen colour.",
        fine_print="",
        options=["Red", "Blue", "Green"],
        already_forecasted=False,
    )


def numeric_question(post_id: int = 303) -> NumericQuestion:
    return NumericQuestion(
        question_text="How many widgets will be sold?",
        id_of_post=post_id,
        id_of_question=post_id,
        page_url=f"https://www.metaculus.com/questions/{post_id}/",
        background_info="Some background.",
        resolution_criteria="Resolves to the number sold.",
        fine_print="",
        unit_of_measure="widgets",
        upper_bound=100.0,
        lower_bound=0.0,
        open_upper_bound=False,
        open_lower_bound=False,
        zero_point=None,
        cdf_size=201,
        already_forecasted=False,
    )


def date_question(post_id: int = 404) -> DateQuestion:
    return DateQuestion(
        question_text="When will the thing happen?",
        id_of_post=post_id,
        id_of_question=post_id,
        page_url=f"https://www.metaculus.com/questions/{post_id}/",
        background_info="Some background.",
        resolution_criteria="Resolves to the date it happens.",
        fine_print="",
        upper_bound=datetime(2027, 12, 31, tzinfo=timezone.utc),
        lower_bound=datetime(2026, 1, 1, tzinfo=timezone.utc),
        open_upper_bound=True,
        open_lower_bound=False,
        zero_point=None,
        cdf_size=201,
        already_forecasted=False,
    )


def default_parser_answer(prompt: str) -> str:
    """Return the JSON the structure_output parser is expected to emit."""
    if "DatePercentile" in prompt:
        return json.dumps(
            [
                {"percentile": 0.1, "value": "2026-04-01T00:00:00Z"},
                {"percentile": 0.2, "value": "2026-06-01T00:00:00Z"},
                {"percentile": 0.4, "value": "2026-09-01T00:00:00Z"},
                {"percentile": 0.6, "value": "2027-01-01T00:00:00Z"},
                {"percentile": 0.8, "value": "2027-06-01T00:00:00Z"},
                {"percentile": 0.9, "value": "2027-10-01T00:00:00Z"},
            ]
        )
    if "Percentile" in prompt:
        return json.dumps(
            [
                {"percentile": 0.1, "value": 20},
                {"percentile": 0.2, "value": 30},
                {"percentile": 0.4, "value": 45},
                {"percentile": 0.6, "value": 55},
                {"percentile": 0.8, "value": 70},
                {"percentile": 0.9, "value": 80},
            ]
        )
    if "PredictedOptionList" in prompt:
        return json.dumps(
            {
                "predicted_options": [
                    {"option_name": "Option Red", "probability": 0.5},
                    {"option_name": "blue", "probability": 0.3},
                    {"option_name": "Green", "probability": 0.2},
                ]
            }
        )
    if "BinaryPrediction" in prompt:
        return json.dumps({"prediction_in_decimal": 0.72})
    raise AssertionError(f"Unexpected parser prompt: {prompt[:200]}")


def make_fake_llm_call(
    *,
    rationale: str = LONG_RATIONALE,
    parser: Callable[[str], str] = default_parser_answer,
    cost_per_call: float = 0.01,
    fail_roles: dict[str, Callable[[], Exception]] | None = None,
    calls: list[tuple[str, str]] | None = None,
):
    """
    Build an async replacement for ClaudeLlm._mockable_direct_call_to_model.
    Routes on the llm's role; `fail_roles` maps role -> exception factory.
    `calls` collects (role, model) for assertions.
    """
    fail_roles = fail_roles or {}

    async def fake_call(self: ClaudeLlm, prompt: Any) -> TextTokenCostResponse:
        if calls is not None:
            calls.append((self.role, self.model))
        if self.role in fail_roles:
            raise fail_roles[self.role]()
        text = prompt if isinstance(prompt, str) else json.dumps(prompt)
        if self.role.startswith("parser"):
            answer = parser(text)
        elif self.role.startswith("researcher"):
            answer = RESEARCH_TEXT
        elif self.role.startswith("summarizer"):
            answer = SUMMARY_TEXT
        else:
            answer = rationale
        return TextTokenCostResponse(
            data=answer,
            prompt_tokens_used=1000,
            completion_tokens_used=300,
            total_tokens_used=1300,
            model=self.model,
            cost=cost_per_call,
        )

    return fake_call


class FakeMetaculusClient(MetaculusClient):
    """Records everything that would have been posted; never touches the network."""

    questions: list = []

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(token="test-token-not-real")
        self.posted_binary: list[tuple[int, float]] = []
        self.posted_numeric: list[tuple[int, list[float]]] = []
        self.posted_mc: list[tuple[int, dict[str, float]]] = []
        self.comments: list[tuple[int, str]] = []
        self.requested_tournaments: list = []

    def get_all_open_questions_from_tournament(self, tournament_id, group_question_mode="unpack_subquestions"):
        self.requested_tournaments.append(tournament_id)
        return list(self.questions)

    def post_binary_question_prediction(self, question_id: int, prediction_in_decimal: float) -> None:
        if prediction_in_decimal < 0.001 or prediction_in_decimal > 0.999:
            raise ValueError("Prediction value must be between 0.001 and 0.999")
        self.posted_binary.append((question_id, prediction_in_decimal))

    def post_numeric_question_prediction(self, question_id: int, cdf_values: list[float]) -> None:
        self.posted_numeric.append((question_id, cdf_values))

    def post_multiple_choice_question_prediction(self, question_id: int, options_with_probabilities: dict[str, float]) -> None:
        self.posted_mc.append((question_id, options_with_probabilities))

    def post_question_comment(self, post_id: int, comment_text: str, is_private: bool = True, included_forecast: bool = True) -> None:
        self.comments.append((post_id, comment_text))
