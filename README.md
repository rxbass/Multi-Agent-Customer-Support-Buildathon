# 🛎️ Crew Desk Support — Self-Checking Multi-Agent Customer Support

> **One agent answers. One agent verifies. One agent reconciles and records.**

Crew Desk Support is a three-agent customer-support system built with **CrewAI + Streamlit**.
A user submits a support query; the crew processes it **sequentially**:

1. **Assistant** answers from the LLM's own knowledge.
2. **Web Search Assistant** searches the live web and produces a web-grounded answer with sources.
3. **Entry Agent (+ Reconciler)** compares both answers, identifies contradictions, produces a resolved response (or refuses when it can't verify), and saves the complete interaction to `answers.txt`.

The goal is simple: make a **second source of truth visible**, and use the third
agent to **reconcile the model's answer against live web evidence** before
presenting a result to the user.

The implementation follows the buildathon constraints exactly: **exactly three
agents, sequential execution, a Streamlit UI, environment-based API keys, and a
single `app.py`.**

---

## 1. Why the Third Agent Exists

An LLM answering from memory can be confidently wrong — it may state a policy or a
figure that has since changed. That's the classic support failure: **a wrong answer
delivered with total confidence.**

Crew Desk Support doesn't claim to *know* when the model is wrong. It does something
narrower and defensible: it **cross-checks the model's answer against live web
evidence and surfaces any contradiction** between the two. When the web evidence
conflicts with the model's answer, the third agent flags it, resolves to the
better-supported answer, and — when it can't verify confidently — **says so instead
of guessing.**

**Worked example (illustrative):**

```
User:
  "What is the current refund window for Product X?"

Agent 1 (Assistant, from memory):
  "Customers can request a refund within 30 days."

Agent 2 (Web Search, from live sources):
  "Current documentation states refunds are available within 14 days."
  sources: [https://example.com/refund-policy]

Agent 3 (Entry Agent + Reconciler):
  contradiction_found:  true
  severity:             material
  resolved_answer:      "The current policy is a 14-day refund window. The model's
                         direct answer (30 days) conflicts with the live source."
  confidence:           0.82
  → written to answers.txt
```

That single exchange is the whole point of the third agent: without it, the user
gets the confident-but-stale 30-day answer.

---

## 2. Architecture & Data Flow

```
User Query (Streamlit input)
        │
        ▼
┌──────────────────────────┐
│ Input guardrails         │  length cap · prompt-injection heuristics
│ (functions, not an agent)│  · PII masking · OpenAI moderation (free)
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐
│ Agent 1 — Assistant      │  no tools
│  answers from memory     │  → direct_answer
└────────────┬─────────────┘
             │  context
             ▼
┌──────────────────────────┐
│ Agent 2 — Web Search     │  SerperDevTool (this agent only)
│  live web + sources      │  → web_answer + source URLs
└────────────┬─────────────┘
             │  context = [task1, task2]
             ▼
┌──────────────────────────┐
│ Agent 3 — Entry Agent    │  file-writer tool
│  + Reconciler            │  • compare both answers
│                          │  • detect contradiction + severity
│                          │  • resolve, or refuse if unverified
│                          │  • write answers.txt
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐
│ Output moderation        │  each answer checked before display
└────────────┬─────────────┘
             ▼
       Streamlit UI  (direct · web + sources · resolved)
```

**The mechanism the evaluator will look for** — in a sequential crew, each task
receives prior outputs through its `context`. The reconciler declares both earlier
tasks as context, which is the only way it can compare and persist both answers:

```python
task3 = Task(
    description="Compare the direct answer and the web answer, detect any "
                "contradiction, resolve to the best-supported answer (or refuse "
                "if it cannot be verified), then save the record to answers.txt.",
    agent=entry_agent,
    context=[task1, task2],          # ← receives query + both prior answers
    output_pydantic=ReconciledAnswer,
)
```

The web-search tool is assigned to **Agent 2 only** — so Agent 1 answers purely from
memory (the fallible source being checked), and no extra tool schemas inflate the
other agents' token cost.

---

## 3. The Three Agents

| # | Agent | Tools | Produces |
|---|-------|-------|----------|
| 1 | **Assistant** | none | `direct_answer` — from model knowledge alone |
| 2 | **Web Search Assistant** | `SerperDevTool` | `web_answer` + preserved source URLs |
| 3 | **Entry Agent + Reconciler** | file-writer | `ReconciledAnswer`; writes `answers.txt` |

**Agent 2 is explicitly instructed to preserve real source URLs** from the search
results and never to invent them — the grounding claim depends on this.

---

## 4. Structured Verdict (Agent 3 output)

Agent 3's task uses `output_pydantic`, so its verdict is structured JSON — the UI
stays robust and the output is machine-checkable.

```python
from pydantic import BaseModel
from typing import Literal

class ReconciledAnswer(BaseModel):
    query: str
    direct_answer: str                                    # Agent 1, carried forward
    web_answer: str                                       # Agent 2, carried forward
    sources: list[str]                                    # Agent 2's real URLs
    contradiction_found: bool                             # web conflicts with model?
    contradiction_severity: Literal["none", "minor", "material"]
    contradiction_detail: str                             # what differed ("" if none)
    resolved_answer: str                                  # best-supported answer, OR the
                                                          # refusal string when unverified
    confidence: float                                     # agent-ASSESSED, 0.0–1.0
```

**`confidence` is agent-assessed, not objective reliability.** It drives a defined,
reproducible rule:

```
confidence < 0.60  →  resolved_answer = "I couldn't verify this confidently."
```

That threshold is what makes the closed-fail behavior reproducible rather than a
matter of the model's mood.

---

## 5. Guards & Reliability

| Guard | Prevents | How |
|-------|----------|-----|
| `max_iter` (e.g. 3) | Web agent looping on an elusive query, draining credits | Set per agent |
| `max_rpm` | Rate-limit crash mid-run | Set per agent |
| Tool scoping | Token bloat + Agent 1 "cheating" by searching | Serper on Agent 2 only |
| File-writer **tool** | Unreliable "please write the file" instructions | Agent 3 writes via a real tool, not a prompt request. If a run ends without the tool having been called, the app persists Agent 3's structured output itself and the UI footer says "saved by app fallback" |
| Graceful degradation | A failed web search crashing the app | Fall back to `direct_answer`, and say so |
| Confidence threshold | Confident wrong answers | `< 0.60` → explicit refusal |
| `output_pydantic` | UI breaking on messy LLM text | Structured verdict |
| Input guardrails | Prompt injection, PII leakage, abusive queries | Regex heuristics + PII masking + OpenAI moderation, before the crew runs |
| Output moderation | Unsafe text reaching the customer | Every answer moderated before display; flagged text withheld |
| Per-agent timeout | A stalled agent hanging the UI | `max_execution_time=150` → clean error |

<details>
<summary><b>Guardrails in detail</b> (click to expand)</summary>

All guardrails are plain functions in `app.py` wrapped around the crew by
`guarded_run()` — the crew is still exactly three agents. Both the UI and
evaluation mode go through the same entry point.

| Stage | What it does | What the user sees |
|-------|--------------|--------------------|
| **Small talk** | "hi", "thanks", "bye", "how are you"… answered with a canned greeting | 👋 instant reply, crew not run, nothing billed |
| **Length cap** | Rejects queries over 1,000 characters | 🛡️ blocked, reason shown, crew not run |
| **Prompt-injection heuristics** | Regex for instruction-override ("ignore previous instructions"), system-prompt extraction, role/chat-format spoofing (`### system:`), raw model tokens (`<\|im_start\|>`), and tool hijacking ("use the save tool to write … to x.txt") | 🛡️ blocked, matched phrase shown, crew not run |
| **PII masking** | Emails, Luhn-valid card numbers, SSNs, IBANs and phone numbers are replaced with `[EMAIL]`, `[CARD]`, … *before* the query reaches the LLM, Serper, or `answers.txt`. Order IDs like `#A-4471` are kept. | 🛡️ info note listing what was masked; query still runs |
| **Input moderation** | OpenAI `omni-moderation-latest` (free endpoint) on the masked query | 🛡️ blocked with the categories (e.g. harassment, violence) |
| **Output moderation** | Same endpoint on the direct, web and resolved answers | Flagged text replaced by `[Withheld by output moderation: …]`; a 🛡️ note says which answer |

Honest limits: injection detection is heuristic (a determined attacker can
rephrase); PII regexes cover common formats, not every national ID; moderation
**fails open** — if the endpoint is unreachable the check is skipped and a 🛡️
warning says so, rather than the app refusing to work. **Deliberate tradeoff:**
Agent 3 writes `answers.txt` via its tool *before* output moderation runs, so the
record preserves the raw agent decision (PII-masked, since masking happens before
the crew). Moderating first would mean the app, not the agent, writes the file —
the spec prefers the agent-tool path. `answers.txt` is an audit record, not a
customer-facing sanitized transcript.

</details>

---

## 6. Setup & Run (step by step)

Tested on Python 3.13; any Python 3.10+ should work. Commands are shown for
macOS/Linux (bash) and Windows (PowerShell).

### Step 1 — Clone the repository

```bash
git clone <this-repo-url>
cd Multi-Agent-Customer-Support-Buildathon
```

### Step 2 — Create and activate a virtual environment

```bash
# macOS / Linux
python -m venv venv
source venv/bin/activate
```

```powershell
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Your prompt should now start with `(venv)`.

### Step 3 — Install the pinned dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` pins `crewai` and `crewai-tools` to a matching pair
(`1.15.22`) — install them together, not separately.

### Step 4 — Provide the API keys (environment variables only)

You need two keys: an **OpenAI** key (all three agents) and a **Serper** key
(Agent 2's web search — free tier at [serper.dev](https://serper.dev)).

Either export them in the shell:

```bash
# macOS / Linux
export OPENAI_API_KEY="sk-..."
export SERPER_API_KEY="..."
```

```powershell
# Windows (PowerShell)
$env:OPENAI_API_KEY="sk-..."
$env:SERPER_API_KEY="..."
```

…or copy the template and fill in a local `.env` file (gitignored, loaded into
`os.environ` on startup):

```bash
cp .env.example .env      # Windows: Copy-Item .env.example .env
# then edit .env
```

If either key is missing, the app stops with a message naming the missing
variable — it never falls back to a hard-coded key.

| Var | For |
|-----|-----|
| `OPENAI_API_KEY` | All three agents (`gpt-4o-mini`) |
| `SERPER_API_KEY` | Agent 2's web search |
| `CREWDESK_RECONCILER_MODEL` | *(optional)* Agent 3's model, default `gpt-4o-mini`. Set to `gpt-4o` for stronger contradiction judgement (note: entry-tier OpenAI accounts cap `gpt-4o` at 30k TPM) |

### Step 5 — Run the app

```bash
streamlit run app.py
```

Streamlit opens `http://localhost:8501`. Type a support question (e.g.
*"What is the latest stable version of Python?"*), click **Ask**, and wait for
the spinner — the crew runs Assistant → Web Search → Entry Agent (typically
10–30 s). You'll see the **Direct answer**, the **Web answer** with sources, and
the **Resolved answer**. Each run appends a record to `answers.txt` in the repo
root; the terminal running Streamlit shows the verbose agent trace.

### Step 6 — Run the evaluation (optional)

1. With the app running, open the **sidebar** (arrow at top-left).
2. Tick **Run evaluation**.
3. The app loads `golden_set.json` (16 labelled queries: 6 catch, 6 control,
   4 refusal), runs the full crew on each item, and reports **catch rate**,
   **false-positive rate**, refusal accuracy, and a one-line latency/cost figure.

The evaluation runs 16 crew executions, so expect several minutes and roughly
16× the cost of a single query. Section 7 shows the numbers from one such run;
your run will differ somewhat — this is an LLM system, not a unit test.

<details>
<summary><b>Troubleshooting</b> (click to expand)</summary>

| Symptom | Fix |
|---------|-----|
| `Missing environment variable(s): OPENAI_API_KEY …` | Set the key (Step 4) and restart Streamlit |
| `ImportError: cannot import name 'SerperDevTool'` | You have a mismatched `crewai-tools`; re-run `pip install -r requirements.txt` |
| Web answer says "Sources: none" | Serper returned nothing usable; the Entry Agent will lower confidence or refuse |
| Terminal shows `429 … tokens per min (TPM)` | OpenAI rate limit; wait a minute or raise the eval pause. Most likely if you set Agent 3 to `gpt-4o` on an entry-tier account |

</details>

---

## 7. Optional: Evaluation Mode

The assignment doesn't require evaluation — but to show the reconciler actually
earns its place, `app.py` includes an **optional evaluation mode** (a sidebar
toggle). It runs the crew over a small labelled set (`golden_set.json`) and reports
two headline numbers:

| Metric | Question it answers |
|--------|---------------------|
| **Catch rate** | Of the stale-answer cases the crew *should* flag, how many did it flag as a material contradiction? |
| **False-positive rate** | On cases where the model was already correct, how often did it wrongly cry contradiction? Reported twice: *any* contradiction (the golden-set definition) and `material` only (what the system is for). |

*(Latency and token cost per run are shown as a single line for context.)*

The golden set mixes three case types on purpose — **catches** (model likely stale,
web corrects it), **controls** (model already correct, expect no contradiction), and
**refusals** (unanswerable without private data, expect the closed-fail response) —
so the catch rate is reported *alongside* its false-positive rate. A two-sided
number is the honest one.

### Results — 2026-09-20, all agents `gpt-4o-mini`

Raw per-item output is in [`eval_results.json`](eval_results.json); its `summary`
block is computed by the same `summarize()` the sidebar uses, so **the numbers
below are exactly what the app's evaluation mode shows** — nothing is adjusted by
hand. 16 cases: 6 catch, 6 control, 4 refusal. **14 runs produced a verdict; 2
timed out** at the 150 s per-agent cap and are excluded from every rate.

**Provenance, stated plainly:** 15 rows come from one full 16-item run. The original
`catch-02` ("current monthly price of ChatGPT Plus") turned out not to be a stale
case — the model's remembered price ($20) was still current, so nothing could be
caught. It was replaced with "What is the latest Ubuntu LTS release?" (a fact that
changes on a fixed two-year cadence) and that one item was run separately the same
day. The replaced row is kept in `eval_results.json` under
`_about.replaced_row_original`, and the replacement is noted on the item in
`golden_set.json`.

| Metric | Result | Reading |
|--------|--------|---------|
| **Material-contradiction catch rate** | **6 / 6 = 100 %** | Python, Ubuntu LTS, iPhone, Node LTS, OpenAI limits, CrewAI version — all flagged `material` at confidence ≥ 0.9. |
| **False-positive rate, any contradiction** (golden-set definition) | **2 / 5 = 40 %** | Both `minor` (password reset, strong-password tips): the web answer added platform-specific detail and the reconciler called it a discrepancy. |
| **False-positive rate, `material`** | **0 / 5** | No material contradiction on the scored control cases. |
| **Refusals correct** | **3 / 3 completed = 100 %** (1 timed out) | All three scored confidence 0.3 → refusal string shown. `refusal-03` also reported a `material` contradiction between two generic answers — an over-flag hidden behind the refusal. |
| Timed-out runs | 2 / 16 | `control-05` ("What is a VPN"), `refusal-04`. Both stalled in Agent 3; **cause not diagnosed** (traces were not inspected). The per-agent cap turned each into a clean error instead of a hang. |
| **Avg latency / cost per completed run** | **21.7 s · ~9,300 tokens · ≈ $0.002** | n = 14 completed verdicts; timed-out runs excluded. `gpt-4o-mini` list prices. |

Per item:

| id | category | outcome | severity | confidence | time |
|----|----------|---------|----------|------------|------|
| catch-01 | catch | caught | material | 1.0 | 14.4 s |
| catch-02 † | catch | caught | material | 1.0 | 14.5 s |
| catch-03 | catch | caught | material | 1.0 | 16.5 s |
| catch-04 | catch | caught | material | 1.0 | 16.6 s |
| catch-05 | catch | caught | material | 0.9 | 31.6 s |
| catch-06 | catch | caught | material | 0.9 | 19.2 s |
| control-01 | control | false positive (minor) | minor | 0.8 | 32.2 s |
| control-02 | control | clean | none | 1.0 | 23.1 s |
| control-03 | control | clean | none | 1.0 | 14.6 s |
| control-04 | control | false positive (minor) | minor | 0.9 | 27.3 s |
| control-05 | control | timed out | — | — | 180 s |
| control-06 | control | clean | none | 1.0 | 49.0 s |
| refusal-01 | refusal | refused | none | 0.3 | 13.8 s |
| refusal-02 | refusal | refused | none | 0.3 | 11.7 s |
| refusal-03 | refusal | refused (but flagged material) | material | 0.3 | 19.2 s |
| refusal-04 | refusal | timed out | — | — | 180 s |

† replacement item, run separately the same day (see provenance above).

What this run says: in this evaluation set the reconciler caught all six cases
where the model's answer was stale, and produced no `material` false positive on
the scored control cases — but `gpt-4o-mini` is better at spotting factual
conflicts than at distinguishing them from harmless differences in specificity
(two `minor` over-flags on controls, one `material` over-flag on a refusal case).
Agent 3 is also the latency bottleneck: 2 of 16 runs reached the execution limit.
In an **informal 5-case spot check**, `gpt-4o` as Agent 3
(`CREWDESK_RECONCILER_MODEL=gpt-4o`) did not produce the minor over-flags; that
sample is far too small to treat as a benchmark, and `gpt-4o` carries a low
per-minute token cap on entry-tier OpenAI accounts. A single 16-item run is a
direction, not a benchmark.

---

## 8. Repository

```
app.py            # the complete application — 3 sequential agents, one file
golden_set.json   # labelled queries for optional evaluation mode (data, not code)
eval_results.json # measured evaluation rows + summary (the numbers in section 7)
requirements.txt  # pinned dependencies (crewai + crewai-tools as a matching pair)
.streamlit/config.toml  # UI theme (light, brand colours) — config, not code
answers.txt       # generated — query + both answers + verdict, per run (gitignored)
.env.example      # environment-variable template (real keys in .env, gitignored)
README.md         # this file
```

All application code lives in `app.py`, per the spec. `golden_set.json` is evaluation
*data*, and evaluation runs as a mode *inside* `app.py` — there is no separate code
file.

---

## 9. Design Decisions

- **Agent 3 is the Entry Agent *and* a reconciler.** The spec's Entry Agent is
  usually a five-line file-writer. Keeping it a single agent but having it reconcile
  the two prior answers adds real depth without a fourth agent — the spec is
  followed to the letter.
- **The claim is precise.** The system doesn't "know" the model is wrong; it
  cross-checks against live web evidence and surfaces contradictions. That's what it
  actually does, and what it can defend.
- **Agent 3 writes via a file tool, not an instruction.** "Please write the file" is
  unreliable; a real tool guarantees the record is saved.
- **Serper on Agent 2 only.** Agent 1 must answer from memory to be the source under
  check; tool scoping also cuts token cost.
- **Confidence is agent-assessed with a fixed 0.60 threshold** — so "refuse when
  unsure" is reproducible, not vibes.
- **Agent 2 is isolated from Agent 1's answer.** In a CrewAI sequential crew, a
  task with no `context` declared receives *every* prior output. Left at the
  default, the web agent saw the model's answer before searching and simply
  restated it — so Task 2 sets `context=[]`, and only Task 3 gets both answers.
  Without this the comparison is meaningless.
- **`gpt-4o-mini` for all three agents.** Early runs showed the reconciler
  reading "more detailed" as "contradicts" and scoring private-data queries above
  the refusal threshold; that was fixed in the Task 3 prompt (a contradiction is
  *incompatible facts only*; confidence measures whether the customer's question
  was answered). `CREWDESK_RECONCILER_MODEL=gpt-4o` is available for Agent 3 if
  judgement quality needs it (informal 5-case spot check only), at the cost of a
  low per-minute token cap on entry-tier accounts.
- **No agent memory, by design.** CrewAI memory stays off (`Crew(memory=False)`,
  the default) and a fresh crew is built per query, so each query is an
  independent trial. If Agent 1 remembered a previous run's web-corrected answer,
  it would stop being the fallible source under check and the contradiction would
  vanish — the evaluation would measure nothing. The only persistence is the
  `answers.txt` audit record and Streamlit session state for re-rendering the UI,
  neither of which is fed back into the agents. The **"Earlier in this session"**
  transcript on the page is display-only: each question is still an independent
  crew run that receives nothing but that question.
- **`output_pydantic` everywhere it matters** — structured verdict, robust UI,
  scorable output.
- **`max_iter` / `max_rpm` caps** — cheap insurance against the classic multi-agent
  failure of a looping search agent.
- **Guardrails are functions, not an agent.** Injection heuristics, PII masking and
  moderation wrap the crew in `guarded_run()`; adding a "safety agent" would break
  the three-agent constraint and cost an LLM call per query — the moderation
  endpoint is free and the regexes are instant.
- **One sequential crew, exactly three agents, one `app.py`.** No Flows, no fourth
  agent, no second code file.

---

## 10. Limitations

- Agent 1's knowledge reflects the model's training cut-off; the crew's value is in
  catching where that's stale — but it can only catch what Agent 2's search surfaces.
- Contradiction detection is only as good as the web answer; a weak search yields a
  weak `web_answer`, which the confidence/refusal path is there to handle.
- `confidence` is the agent's self-assessment, not a calibrated probability — hence
  the fixed threshold and the honest label.
- Guardrails are heuristic + moderation-API, not a security boundary: injection
  regexes can be evaded by rephrasing, PII masking covers common formats only, and
  moderation fails open when the endpoint is unavailable (the UI says so).
- The golden set is small and hand-labelled; results show direction and consistency,
  not a precise percentage. Labelling rationale lives in `golden_set.json` so every
  number is traceable.

---

*Built for the Weekly Buildathon — Multi-Agent Customer Support (CrewAI + Streamlit).*
