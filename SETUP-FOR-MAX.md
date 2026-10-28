# Setup for Max (10 minutes, phone-friendly)

This fork runs the Metaculus FutureEval template bot on **Claude** with hard
spend caps. You add two secrets, press one button, then switch the schedule on.
Never paste keys anywhere except the GitHub Secrets page.

## 1. Add the two secrets

1. Open your fork on GitHub → **Settings** (tab at the top).
2. Left menu: **Secrets and variables → Actions**.
3. Green button **New repository secret**. Add these two, names exactly as written:
   - `METACULUS_TOKEN` — your **bot account** token from
     https://www.metaculus.com/futureeval/participate/
   - `ANTHROPIC_API_KEY` — from https://platform.claude.com/ → API keys.
     Use a key with a **monthly spend limit** set in the Anthropic console too.
4. Click **Add secret** after each one.

## 2. Run the Test Bot once

1. Top tabs: **Actions**.
2. If you see a yellow box "Workflows aren't being run on this forked repository",
   click **I understand my workflows, go ahead and enable them**.
   (The 20-minute schedule starts counting from this click, see step 3.)
3. Left list: **Test Bot** → right side **Run workflow** → pick branch
   `claude/live-ready` (or `main` after you merge) → green **Run workflow**.
   - Tick **Dry run** if you first want to see reasoning without posting anything.
4. Wait 5–15 minutes, then open the run. The **Summary** at the top tells you:
   ✅ how many forecasts posted, ❌ what failed, and **what it spent**.
5. Check Metaculus: log in → click your name → **Settings → My Forecasting Bots →
   Switch to bot account** → open a question from the summary. The forecast is on
   the graph and the bot's full reasoning is under the **Private** comments tab.

If the summary says "Bot not configured yet", a secret name is misspelled or missing.

## 3. Turn the schedule on (or make sure it is)

- **Actions → Forecast on new AI tournament questions**. If it says "This scheduled
  workflow is disabled", click **Enable workflow**. It then runs every 20 minutes,
  forecasts only new tournament + MiniBench questions, and skips ones already done.
- Also run it once by hand (**Run workflow**) so you do not wait for the clock.
- Fill in the Metaculus participation form once (required by the rules):
  https://forms.gle/aQdYMq9Pisrf1v7d8

To pause: same page → **…** (top right) → **Disable workflow**.

## What it costs

Rates used (Anthropic list prices): Claude Sonnet 5.5 $2 in / $10 out per million
tokens, Claude Haiku 5.5 $0.10 / $0.50, web search $0.01 per search.

| Item | Typical |
|---|---|
| One binary / multiple-choice / numeric question | **$0.15 – $0.35** |
| One conditional question (2–4 sub-forecasts) | $0.40 – $0.80 |
| The Test Bot run (all open test questions) | $2 – $4, may stop at the $4 per-run cap |
| A week of FutureEval + MiniBench (past seasons: ~50–100 new questions/week) | **$10 – $35** |
| Absolute ceiling per week from the caps | $84 ($12/day × 7) |

What one question buys: 1 research call with up to 3 web searches (Sonnet),
3 independent forecasts (Sonnet, medium thinking effort), parsing and a research
summary (Haiku).

## The safety rails (already on)

- **Spend caps:** $4 per run and $12 per UTC day by default. When a cap is reached
  the bot refuses to start any more LLM calls, the run is marked failed, and the
  summary says "Spend cap hit". Unforecasted questions are picked up next run.
  The daily total is remembered between runs via the GitHub Actions cache (best
  effort; the per-run cap always holds).
- **Clean outputs:** probabilities are clipped to 1–99%, multiple-choice answers
  must name every option exactly once with at least 1% each, numeric answers are
  checked and clamped to the question bounds. Anything malformed is rejected
  before it reaches Metaculus.
- **Retry then fallback:** every call is retried, then repeated on Claude Haiku 5.5
  if Sonnet errors or times out. If the web search tool is rejected the research
  silently degrades to knowledge-only instead of failing.
- **Reasoning always posted:** a forecast is refused if it has no written
  rationale; the comment on Metaculus is the bot's actual reasoning (no human in
  the loop anywhere).
- **Missing secrets:** scheduled runs stop with a clear notice and exit green;
  manual runs fail red so you notice.
- **Tests:** `Tests (no API calls)` workflow runs 70+ tests on every push with the
  network blocked.

## Tuning without code (optional)

**Settings → Secrets and variables → Actions → Variables → New repository variable.**
Unset means the default.

| Variable | Default | Notes |
|---|---|---|
| `BOT_MAX_COST_PER_RUN_USD` | `4` | hard cap per run |
| `BOT_MAX_COST_PER_DAY_USD` | `12` | hard cap per UTC day |
| `BOT_MAIN_MODEL` | `anthropic/claude-sonnet-5-5` | e.g. `anthropic/claude-opus-5-5` (2× price) |
| `BOT_FALLBACK_MODEL` | `anthropic/claude-haiku-5-5` | also does parsing/summaries |
| `BOT_EFFORT` | `medium` | `low` / `medium` / `high` thinking depth |
| `BOT_PREDICTIONS_PER_QUESTION` | `3` | upstream template uses 5 |
| `BOT_RESEARCH_MODE` | `web_search` | `none` = no web search, cheaper |
| `BOT_PUBLISH` | `true` | `false` = dry run, nothing posted |
| `METACULUS_CUP_ENABLED` | unset | `true` turns on the every-2-days Metaculus Cup workflow (costs extra, not FutureEval) |

## Reading a run

Actions → click a run → the **Summary** box:

- `✅ N forecast(s) posted to Metaculus.` all good, links listed.
- `⚠️ … failed` some questions failed; reasons listed; they retry next run.
- `❌ All … failed` / `Run crashed` open the "Run bot" step log; the last lines say why.
- `⚠️ Spend cap hit` working as designed; raise the cap variable if you want more.

Local run (optional): `cp .env.template .env`, fill the two keys,
`poetry install`, `poetry run pip install pytest pytest-asyncio`, `poetry run pytest -q`
(no API calls), then
`poetry run python main.py --mode test_questions`.
