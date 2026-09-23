"""
Crew Desk Support — a three-agent CrewAI customer-support system with a Streamlit UI.

    Agent 1  Assistant            answers from model knowledge only (no tools)
    Agent 2  Web Search Assistant answers from the live web (SerperDevTool)
    Agent 3  Entry Agent          reconciles both answers, resolves or refuses,
                                  and saves the record to answers.txt (file tool)

The three agents run sequentially (1 -> 2 -> 3). Task 3 receives the outputs of
Tasks 1 and 2 through `context`, which is what lets it compare and persist both.

All application code lives in this single file. API keys come from environment
variables only.
"""

import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Literal, Optional
from urllib.parse import urlparse

import streamlit as st
from pydantic import BaseModel, Field

# CrewAI's verbose logger prints emoji; on a Windows cp1252 console that raises
# 'charmap' codec errors in its event handlers. Force UTF-8 so the trace is clean.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# 1. Environment keys — read from os.environ, never hard-coded
# ---------------------------------------------------------------------------

try:  # optional convenience: populate os.environ from a local .env (gitignored)
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

REQUIRED_KEYS = {
    "OPENAI_API_KEY": "LLM for all three agents",
    "SERPER_API_KEY": "Agent 2's live web search",
}


def missing_keys() -> list[str]:
    """Return the names of any required keys absent from the environment."""
    return [k for k in REQUIRED_KEYS if not os.environ.get(k)]


MODEL_NAME = "gpt-4o-mini"           # all three agents, per spec
# Optional override for Agent 3 only (e.g. gpt-4o) if contradiction judgement needs it.
RECONCILER_MODEL = os.environ.get("CREWDESK_RECONCILER_MODEL", MODEL_NAME)
CONFIDENCE_THRESHOLD = 0.60          # below this Agent 3's verdict becomes a refusal
REFUSAL_TEXT = "I couldn't verify this confidently."
ANSWERS_FILE = "answers.txt"
AGENT_TIMEOUT_S = 90                 # hard cap per agent so a stuck run fails instead of hanging
# CrewAI retries a failed task up to max_retry_limit times (default 2), so a stalled
# agent would burn 3 x AGENT_TIMEOUT_S before the user sees anything. Allow one retry.
AGENT_RETRIES = 1

# ---------------------------------------------------------------------------
# 2. Structured verdict — Agent 3's output schema
# ---------------------------------------------------------------------------


class ReconciledAnswer(BaseModel):
    query: str
    # Agent 3 is told to leave these two empty: they are the prior agents' answers
    # verbatim, and app.py fills them from the actual task outputs after kickoff().
    # Making the model retype ~300 tokens of text it was given is the single
    # biggest latency cost in the crew, and a transcription risk on top.
    direct_answer: str = ""
    web_answer: str = ""
    sources: list[str]
    contradiction_found: bool
    contradiction_severity: Literal["none", "minor", "material"]
    contradiction_detail: str
    resolved_answer: str            # best-supported answer, or the refusal string
    confidence: float = Field(ge=0.0, le=1.0)   # agent-assessed 0.0–1.0


def apply_confidence_rule(verdict: ReconciledAnswer) -> ReconciledAnswer:
    """Fixed rule enforced in code, not just in the prompt:
    confidence < 0.60 -> resolved_answer becomes the refusal string."""
    if verdict.confidence < CONFIDENCE_THRESHOLD:
        verdict.resolved_answer = REFUSAL_TEXT
    return verdict


def write_record(query: str, direct_answer: str, web_answer: str, resolved_answer: str) -> str:
    """Append one interaction record to answers.txt. Used by Agent 3's file tool,
    and by the app as a fallback if the agent ever finishes without calling it."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    record = (
        f"=== {stamp} ===\n"
        f"QUERY: {query}\n\n"
        f"DIRECT ANSWER (Agent 1, model knowledge):\n{direct_answer}\n\n"
        f"WEB ANSWER (Agent 2, live search):\n{web_answer}\n\n"
        f"RESOLVED ANSWER (Agent 3, Entry Agent):\n{resolved_answer}\n\n"
    )
    with open(ANSWERS_FILE, "a", encoding="utf-8") as fh:
        fh.write(record)
    return f"Record appended to {ANSWERS_FILE}."


# ---------------------------------------------------------------------------
# 3. Agents + tools  (imported lazily so the key guard can run first)
# ---------------------------------------------------------------------------


def build_crew(task_callback: Optional[Callable] = None):
    from crewai import LLM, Agent, Crew, Process, Task
    from crewai.tools import tool
    from crewai_tools import SerperDevTool

    llm = LLM(model=MODEL_NAME, temperature=0)
    reconciler_llm = LLM(model=RECONCILER_MODEL, temperature=0)

    # --- tools ---------------------------------------------------------------
    search_tool = SerperDevTool(n_results=5)   # Agent 2 ONLY

    prior: dict = {}                   # filled with task1/task2 once they exist

    @tool("Save support record")
    def save_support_record(query: str, resolved_answer: str) -> str:
        """Append the support record to answers.txt: the customer query, BOTH prior
        answers, and your resolved answer. The two prior answers are attached
        automatically from the earlier tasks — do not paste them in. Call this
        exactly once, after you have decided on the resolved answer."""
        def raw(name: str) -> str:
            t = prior.get(name)
            out = getattr(t, "output", None)
            return (getattr(out, "raw", "") or "") if t is not None else ""
        return write_record(query, raw("task1"), raw("task2"), resolved_answer)

    # --- agents --------------------------------------------------------------
    assistant = Agent(
        role="Assistant",
        goal="Answer the customer's support query accurately from your own knowledge.",
        backstory=(
            "You are a knowledgeable customer-support assistant. You answer only "
            "from what you already know — you have no tools and must not pretend "
            "to have looked anything up. If the query needs private account data "
            "you cannot access, say so plainly."
        ),
        llm=llm,
        tools=[],                       # no tools: this is the source under check
        allow_delegation=False,
        max_iter=3,
        max_rpm=30,                     # see note above: low values cause a blind 60 s sleep
        max_execution_time=AGENT_TIMEOUT_S,
        max_retry_limit=AGENT_RETRIES,
        verbose=True,
    )

    web_searcher = Agent(
        role="Web Search Assistant",
        goal="Answer the customer's query using current information from the live web, citing real sources.",
        backstory=(
            "You are a research-minded support assistant. You use the web search "
            "tool to find the most current, authoritative information and you "
            "always keep the exact source URLs that the search returned."
        ),
        llm=llm,
        tools=[search_tool],            # Serper on Agent 2 ONLY
        allow_delegation=False,
        max_iter=3,                     # budget guard against tool loops
        max_rpm=30,
        max_execution_time=AGENT_TIMEOUT_S,
        max_retry_limit=AGENT_RETRIES,
        verbose=True,
    )

    entry_agent = Agent(
        role="Entry Agent",
        goal=(
            "Reconcile the direct answer with the web answer, surface any "
            "contradiction, resolve to the best-supported answer or refuse, and "
            "save the complete record to answers.txt."
        ),
        backstory=(
            "You are the Entry Agent: the final checkpoint before a response "
            "reaches the customer. You compare the model's answer against live web "
            "evidence, flag where they conflict, and never present an answer you "
            "cannot support. You always save the record with your file tool."
        ),
        llm=reconciler_llm,
        tools=[save_support_record],    # file writer only
        allow_delegation=False,
        max_iter=3,
        max_rpm=30,
        max_execution_time=AGENT_TIMEOUT_S,
        max_retry_limit=AGENT_RETRIES,
        verbose=True,
    )

    # --- tasks ---------------------------------------------------------------
    # NOTE: in a sequential crew, a task with no `context` declared automatically
    # receives EVERY previous task's output. Task 2 must NOT see Task 1's answer,
    # otherwise the web answer anchors on the model's answer and the comparison
    # is meaningless — so Task 2 sets context=[] explicitly. Only Task 3 gets both.
    task1 = Task(
        description=(
            "Today's date: {today}\n"
            "Customer query: {query}\n\n"
            "Answer this query from your own knowledge only. Be concise and "
            "specific. Do not claim to have searched anything. If the answer may "
            "have changed since your training data, still give your best answer."
        ),
        expected_output="A concise, direct answer to the query in plain prose.",
        agent=assistant,
    )

    task2 = Task(
        description=(
            "Today's date: {today}\n"
            "Customer query: {query}\n\n"
            "Use the web search tool and answer ONLY from what the search results "
            "say — do not rely on your own prior knowledge, which may be out of "
            "date. If the query asks for the latest/current/newest of something, "
            "include the current year in your search query and prefer the most "
            "recently dated result. Preserve the real source URLs exactly as they "
            "appear in the search results. NEVER invent, guess or modify a URL — "
            "if the search returned no usable source, say so and list no sources. "
            "If the query is about private account data that the public web "
            "cannot answer, say that clearly."
        ),
        expected_output=(
            "A concise answer grounded in the search results, followed by a line "
            "'Sources:' and a bulleted list of the real URLs used (or 'Sources: none')."
        ),
        agent=web_searcher,
        context=[],                     # isolated: must not see Agent 1's answer
    )

    task3 = Task(
        description=(
            "Today's date: {today}\n"
            "Customer query: {query}\n\n"
            "You have received the DIRECT ANSWER (from model knowledge) and the "
            "WEB ANSWER (from live search, with sources) as context. Judge them "
            "against today's date: a direct answer phrased 'as of <an earlier "
            "year>' for a 'latest/current' question is likely stale if the web "
            "answer names something newer.\n\n"
            "1. Compare the two answers. A contradiction exists ONLY when the two "
            "answers assert incompatible facts about the same thing (a different "
            "version number, price, date, limit, policy, or a yes vs. no). One "
            "answer being longer, more detailed, more platform-specific, "
            "differently worded, or covering extra examples is NOT a "
            "contradiction — that is severity 'none'. Rate severity: 'none' (no "
            "incompatible facts), 'minor' (a small factual discrepancy that would "
            "not change what the customer does), or 'material' (a fact, figure, "
            "version, price or policy differs in a way that would mislead the "
            "customer). Quote the specific conflicting facts in "
            "contradiction_detail, or leave it empty when severity is 'none'.\n"
            "2. Write the resolved answer: the best-supported answer, preferring "
            "the web answer when it is well-sourced and conflicts with the direct "
            "answer.\n"
            "3. Assess your confidence from 0.0 to 1.0 that the resolved answer "
            "actually answers the customer's question and is verifiable. Below "
            f"0.60 the customer will be shown '{REFUSAL_TEXT}' instead of your "
            "resolved answer. Confidence measures whether the question got "
            "answered — NOT how sure you are that it cannot be answered. If the "
            "query is about the customer's own order, refund, account, balance, "
            "warranty or device ('my order', 'my refund', 'my account'...), it "
            "needs private data that neither answer has; a generic explanation "
            "of how such things usually work does NOT answer it, so confidence "
            "MUST be 0.3 or lower.\n"
            "4. Call the 'Save support record' tool exactly once, passing only the "
            "customer's query and your resolved answer. Both prior answers are "
            "attached to the record automatically — do NOT paste them into the "
            "tool call.\n"
            "5. Return the final verdict as JSON matching the required schema. "
            "Set direct_answer and web_answer to empty strings \"\" — they are "
            "filled in from the prior task outputs, so do not retype them. Copy "
            "the source URLs from the web answer without changing them. Be "
            "concise: the only long field you write is resolved_answer."
        ),
        expected_output=(
            "A JSON object with fields: query, direct_answer, web_answer, sources, "
            "contradiction_found, contradiction_severity, contradiction_detail, "
            "resolved_answer, confidence — with direct_answer and web_answer left "
            "as empty strings."
        ),
        agent=entry_agent,
        context=[task1, task2],         # receives the query + both prior answers
        output_pydantic=ReconciledAnswer,
    )

    prior["task1"], prior["task2"] = task1, task2     # the file tool reads these

    crew = Crew(
        agents=[assistant, web_searcher, entry_agent],
        tasks=[task1, task2, task3],
        process=Process.sequential,     # 1 -> 2 -> 3
        task_callback=task_callback,    # lets the UI report per-agent progress
        verbose=True,
    )
    return crew


@dataclass
class RunResult:
    """Everything the UI needs from one crew run, including partial results."""
    query: str
    direct_raw: Optional[str] = None
    web_raw: Optional[str] = None
    verdict: Optional[ReconciledAnswer] = None
    error: Optional[str] = None            # set if the crew did not complete
    elapsed_s: float = 0.0
    total_tokens: int = 0
    saved_by: str = ""                     # who wrote answers.txt this run
    blocked_reason: str = ""               # set when input guardrails refused to run the crew
    small_talk_reply: str = ""             # greeting answered without running the crew
    masked_pii: Optional[dict] = None      # PII placeholders substituted into the query
    guardrail_notes: Optional[list] = None # output-moderation / skipped-check notes


def _file_size(path: str) -> int:
    return os.path.getsize(path) if os.path.exists(path) else 0


def run_query(query: str, on_task_done: Optional[Callable[[int, str], None]] = None) -> RunResult:
    """Run the crew for one query. Never raises: failures come back in `error`,
    together with whichever answers were produced before the failure."""
    completed = []

    def _task_done(output):
        completed.append(output)
        if on_task_done:
            on_task_done(len(completed), getattr(output, "raw", "") or "")

    crew = build_crew(task_callback=_task_done)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    size_before = _file_size(ANSWERS_FILE)
    t0 = time.perf_counter()

    try:
        result = crew.kickoff(inputs={"query": query, "today": today})
    except Exception as exc:                            # graceful degradation
        return RunResult(
            query=query,
            direct_raw=completed[0].raw if len(completed) > 0 else None,
            web_raw=completed[1].raw if len(completed) > 1 else None,
            error=f"{type(exc).__name__}: {exc}",
            elapsed_s=time.perf_counter() - t0,
        )

    direct_raw = result.tasks_output[0].raw
    web_raw = result.tasks_output[1].raw
    verdict = result.pydantic
    if verdict is None and result.json_dict:            # structured parse fallback
        try:
            verdict = ReconciledAnswer.model_validate(result.json_dict)
        except Exception:
            verdict = None
    if verdict is not None:
        # Provenance, not judgement: fill from the real task outputs so the verdict
        # always carries the exact prior answers (the model never retypes them).
        verdict.direct_answer = direct_raw
        verdict.web_answer = web_raw
        if not verdict.query:
            verdict.query = query
        verdict = apply_confidence_rule(verdict)

    # Agent 3 should have saved the record via its tool. If the file did not
    # grow, persist the structured output app-side so no interaction is lost.
    saved_by = "Agent 3 (file tool)"
    if _file_size(ANSWERS_FILE) == size_before:
        write_record(query, direct_raw, web_raw, verdict.resolved_answer if verdict else REFUSAL_TEXT)
        saved_by = "app fallback"

    usage = result.token_usage
    return RunResult(
        query=query,
        direct_raw=direct_raw,
        web_raw=web_raw,
        verdict=verdict,
        elapsed_s=time.perf_counter() - t0,
        total_tokens=getattr(usage, "total_tokens", 0) or 0,
        saved_by=saved_by,
    )


# ---------------------------------------------------------------------------
# 3b. Guardrails — plain functions, NOT a fourth agent. They wrap the crew:
#     input  : prompt-injection heuristics -> PII masking -> moderation (free API)
#     output : moderation on each answer before it is shown
# ---------------------------------------------------------------------------

MAX_QUERY_CHARS = 1000

# Greetings / thanks / farewells are answered instantly — no crew, no API call.
SMALL_TALK_RE = re.compile(
    r"^(?:hi|hello|hey|hiya|yo|howdy|good\s+(?:morning|afternoon|evening)|"
    r"thanks|thank\s+you|thx|ty|cheers|ok|okay|cool|great|nice|bye|goodbye|see\s+you|"
    r"how\s+are\s+you|what'?s\s+up|sup|test|testing|hello\s+there|hi\s+there)"
    r"(?:\s+(?:there|bot|crew|team|again|all|a\s+lot|so\s+much))?[\s!.,?]*$",
    re.IGNORECASE,
)
SMALL_TALK_REPLY = (
    "👋 Hi! I'm **Crew Desk Support**. Ask me a support question — for example a "
    "version, a price, a policy, or how to do something — and three agents will "
    "answer it, verify it against the live web, and reconcile the two. "
    "Try one of the examples above."
)

# Common jailbreak / instruction-override phrasings and raw chat-format markers.
INJECTION_PATTERNS = [
    r"\bignore\s+(all\s+|any\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules?)",
    r"\bdisregard\s+(all\s+|any\s+)?(previous|prior|above|earlier|your)\s+(instructions?|prompts?|rules?)",
    r"\b(reveal|print|show|repeat|output)\s+(me\s+)?(your\s+(system\s+)?(prompt|instructions?)|the\s+system\s+prompt)",
    r"\byou\s+are\s+now\s+(a|an|the|in)\b",
    r"\b(act|behave|pretend)\s+as\s+(if\s+you\s+are\s+)?(a|an|the)\s+\w+\s+(with|without)\s+(no\s+)?(restrictions?|rules?|limits?)",
    r"\b(jailbreak|do\s+anything\s+now|DAN|developer\s+mode)\b",
    r"\bnew\s+(system\s+)?instructions?\s*:",
    r"(^|\n)\s*(###\s*)?(system|assistant|user)\s*:",       # chat-format spoofing
    r"<\|im_start\|>|<\|endoftext\|>|\[INST\]|<<SYS>>",     # raw model tokens
    r"\b(call|use|invoke)\s+the\s+(save|file|search)\s+\w*\s*tool\b",   # tool hijacking
    r"\b(write|append|save)\s+.{0,40}\bto\s+\S+\.(txt|json|py|env)\b",
]
INJECTION_RE = re.compile("|".join(f"(?:{p})" for p in INJECTION_PATTERNS), re.IGNORECASE)

# PII that a support query never needs to send to an LLM or a search engine.
PII_PATTERNS = {
    "[EMAIL]": re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)+"),
    "[CARD]":  re.compile(r"\b(?:\d[ -]?){13,19}\b"),                          # validated with Luhn below
    "[SSN]":   re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "[IBAN]":  re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){2,7}(?:\s?[A-Z0-9]{1,4})?\b"),
    "[PHONE]": re.compile(r"(?<![\w#-])(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)\d{3,4}[\s.-]?\d{3,4}(?![\w-])"),
}


def _luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d = d * 2 - 9 if d > 4 else d * 2
        total, alt = total + d, not alt
    return total % 10 == 0


def mask_pii(text: str) -> tuple[str, dict[str, int]]:
    """Replace emails, card numbers (Luhn-valid), SSNs, IBANs and phone numbers with
    placeholders. Order IDs like #A-4471 and short numbers are left alone."""
    counts: dict[str, int] = {}

    def sub_card(m: re.Match) -> str:
        digits = re.sub(r"\D", "", m.group(0))
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            counts["[CARD]"] = counts.get("[CARD]", 0) + 1
            return "[CARD]"
        return m.group(0)

    text = PII_PATTERNS["[CARD]"].sub(sub_card, text)
    for tag in ("[EMAIL]", "[SSN]", "[IBAN]", "[PHONE]"):
        text, n = PII_PATTERNS[tag].subn(tag, text)
        if n:
            counts[tag] = counts.get(tag, 0) + n
    return text, counts


def moderate(text: str) -> tuple[bool, list[str], str]:
    """OpenAI moderation (free endpoint). Returns (flagged, categories, error).
    Fails OPEN on network/API errors — the caller shows that a check was skipped."""
    try:
        from openai import OpenAI

        res = OpenAI(max_retries=1, timeout=10).moderations.create(
            model="omni-moderation-latest", input=text[:4000]
        ).results[0]
        # the API reports some categories under two spellings (a/b and a_b); dedupe
        cats = list(dict.fromkeys(k.replace("_", "/") for k, v in res.categories.model_dump().items() if v))
        return bool(res.flagged), cats, ""
    except Exception as exc:
        return False, [], f"{type(exc).__name__}: {exc}"


@dataclass
class InputScreen:
    ok: bool                       # False -> do not run the crew
    query: str                     # cleaned query to send on
    blocked_reason: str = ""       # why it was blocked (shown to the user)
    masked: Optional[dict] = None  # {placeholder: count} of PII removed
    notes: Optional[list] = None   # non-blocking warnings (e.g. moderation skipped)
    reply: str = ""                # instant answer (small talk) — crew not needed


def screen_input(raw: str) -> InputScreen:
    """Input guardrails, in order: size -> prompt injection -> PII mask -> moderation."""
    q = raw.strip()
    notes: list[str] = []
    if not q:
        return InputScreen(False, q, "Empty query.")
    if SMALL_TALK_RE.match(q):
        return InputScreen(False, q, reply=SMALL_TALK_REPLY)
    if len(q) > MAX_QUERY_CHARS:
        return InputScreen(False, q, f"Query is too long ({len(q)} chars; limit {MAX_QUERY_CHARS}).")
    m = INJECTION_RE.search(q)
    if m:
        return InputScreen(False, q, f"Looks like a prompt-injection attempt (matched: “{m.group(0).strip()}”).")
    q, masked = mask_pii(q)
    flagged, cats, err = moderate(q)
    if err:
        notes.append(f"Input moderation skipped ({err[:80]}).")
    if flagged:
        return InputScreen(False, q, "Blocked by content moderation: " + ", ".join(cats) + ".")
    return InputScreen(True, q, masked=masked or None, notes=notes or None)


WITHHELD_TEXT = "[Withheld by output moderation: {cats}]"


def screen_output(res: "RunResult") -> "RunResult":
    """Output guardrail: moderate each answer; replace flagged text before display
    (the unmodified record has already been written to answers.txt by Agent 3)."""
    notes: list[str] = []

    def check(label: str, text: Optional[str]) -> Optional[str]:
        if not text:
            return text
        flagged, cats, err = moderate(text)
        if err:
            notes.append(f"{label}: output moderation skipped ({err[:60]}).")
            return text
        if flagged:
            notes.append(f"{label}: withheld ({', '.join(cats)}).")
            return WITHHELD_TEXT.format(cats=", ".join(cats))
        return text

    res.direct_raw = check("Direct answer", res.direct_raw)
    res.web_raw = check("Web answer", res.web_raw)
    if res.verdict is not None:
        res.verdict.resolved_answer = check("Resolved answer", res.verdict.resolved_answer)
    res.guardrail_notes = (res.guardrail_notes or []) + notes
    return res


def guarded_run(raw_query: str, on_task_done: Optional[Callable[[int, str], None]] = None) -> "RunResult":
    """The single entry point used by the UI and by evaluation mode:
    input guardrails -> crew -> output guardrails. Never raises."""
    screen = screen_input(raw_query)
    if not screen.ok:
        return RunResult(query=raw_query, blocked_reason=screen.blocked_reason, small_talk_reply=screen.reply)
    res = run_query(screen.query, on_task_done=on_task_done)
    res.masked_pii = screen.masked
    res.guardrail_notes = list(screen.notes or [])
    return screen_output(res)


# ---------------------------------------------------------------------------
# 4. Streamlit UI
# ---------------------------------------------------------------------------

URL_RE = re.compile(r"https?://[^\s\)\]>\"']+")

EXAMPLES = [
    "What is the latest stable version of Python?",
    "What is the latest Ubuntu LTS release?",
    "How do I reset my password?",
    "Why was my order #A-4471 delayed?",
]


def split_sources(web_raw: str) -> tuple[str, list[str]]:
    """Separate Agent 2's prose from its trailing 'Sources:' block."""
    parts = re.split(r"\n\s*\**\s*sources\s*\**\s*:", web_raw, maxsplit=1, flags=re.IGNORECASE)
    body = parts[0].strip()
    urls = URL_RE.findall(parts[1]) if len(parts) > 1 else []
    return body, urls


def pretty_url(url: str) -> str:
    p = urlparse(url)
    path = p.path.rstrip("/")
    return p.netloc.replace("www.", "") + (path if len(path) <= 40 else path[:37] + "…")


def render_sources(urls: list[str]) -> None:
    if not urls:
        st.caption("No sources returned by the search.")
        return
    seen = []
    for u in urls:
        if u not in seen:
            seen.append(u)
    for u in seen:
        st.markdown(f"🔗 [{pretty_url(u)}]({u})")


def render_result(res: RunResult, k: str = "") -> None:
    """`k` suffixes container keys so several results can be on one page (history)."""
    v = res.verdict

    # --- guardrails ------------------------------------------------------------
    if res.small_talk_reply:
        st.success(res.small_talk_reply)
        return
    if res.blocked_reason:
        st.error(f"🛡️ **Not sent to the crew.** {res.blocked_reason}")
        return
    if res.masked_pii:
        st.info("🛡️ Personal data was masked before the query was sent: "
                + ", ".join(f"{n}× `{tag}`" for tag, n in res.masked_pii.items()) + ".")
    for note in res.guardrail_notes or []:
        st.warning(f"🛡️ {note}")

    # --- headline verdict ----------------------------------------------------
    if res.error:
        st.error(
            "The crew did not finish. Showing whatever was produced before the "
            f"failure.\n\n`{res.error}`"
        )
    elif v is None:
        st.warning("The Entry Agent did not return a structured verdict; showing both raw answers.")
    elif v.resolved_answer == REFUSAL_TEXT:
        st.warning(
            f"**{REFUSAL_TEXT}** — agent-assessed confidence "
            f"{v.confidence:.0%} is below the {CONFIDENCE_THRESHOLD:.0%} threshold, "
            "so the answer is withheld rather than guessed."
        )
    elif v.contradiction_severity == "material":
        st.error("**Material contradiction** — the live web evidence disagrees with the model's answer.")
    elif v.contradiction_severity == "minor":
        st.info("**Minor difference** — wording or detail differs, but nothing that would mislead.")
    else:
        st.success("**Answers agree** — the web evidence supports the model's answer.")

    # --- resolved answer -----------------------------------------------------
    st.markdown("### 🛎️ Resolved answer")
    st.caption("Agent 3 · Entry Agent — reconciled from both answers below")
    with st.container(border=True, key=f"card_resolved{k}"):
        if v is not None:
            st.markdown(v.resolved_answer)
            st.progress(v.confidence, text=f"Agent-assessed confidence: {v.confidence:.0%}")
            if v.contradiction_found and v.contradiction_detail:
                st.markdown(f"**What differed:** {v.contradiction_detail}")
        elif res.direct_raw:
            st.markdown(res.direct_raw)
            st.caption("Web verification was unavailable — this is the model's unverified direct answer.")
        else:
            st.markdown(f"_{REFUSAL_TEXT}_")

    # --- the two answers side by side ---------------------------------------
    st.markdown("### The two answers")
    left, right = st.columns(2)

    with left:
        with st.container(border=True, key=f"card_direct{k}"):
            st.markdown("#### 🧠 Direct answer")
            st.caption("Agent 1 · Assistant — model knowledge only, no tools")
            if res.direct_raw:
                st.markdown(res.direct_raw)
            else:
                st.caption("Not produced.")

    with right:
        with st.container(border=True, key=f"card_web{k}"):
            st.markdown("#### 🌐 Web answer")
            st.caption("Agent 2 · Web Search Assistant — live search via Serper")
            if res.web_raw:
                body, parsed_urls = split_sources(res.web_raw)
                st.markdown(body)
                st.markdown("**Sources**")
                render_sources((v.sources if v and v.sources else None) or parsed_urls)
            else:
                st.warning("Web verification unavailable for this query.")

    # --- details -------------------------------------------------------------
    with st.expander("Reconciliation details"):
        if v is not None:
            c1, c2, c3 = st.columns(3)
            c1.metric("Contradiction", "Yes" if v.contradiction_found else "No")
            c2.metric("Severity", v.contradiction_severity)
            c3.metric("Confidence", f"{v.confidence:.2f}")
            st.json(v.model_dump())
        else:
            st.caption("No structured verdict available.")

    footer = f"⏱ {res.elapsed_s:.1f}s"
    if res.total_tokens:
        footer += f" · {res.total_tokens:,} tokens"
    if res.saved_by:
        footer += f" · saved to `{ANSWERS_FILE}` by {res.saved_by}"
    st.caption(footer)


def render_pipeline(done: int = 0) -> None:
    """Three compact, colour-coded agent cards; `done` marks how many finished."""
    cards = [
        ("🧠", "blue", "Assistant", "model knowledge · no tools"),
        ("🌐", "green", "Web Search", "live web · Serper"),
        ("🛎️", "orange", "Entry Agent", "reconciles · saves record"),
    ]
    # Keys must be unique within one script run; the cards are rendered before
    # AND after a run, so use a per-run render counter rather than `done`.
    n = st.session_state["_pipeline_renders"] = st.session_state.get("_pipeline_renders", 0) + 1
    cols = st.columns(3)
    for i, (icon, color, title, what) in enumerate(cards):
        with cols[i], st.container(border=True, key=f"agent{i + 1}_r{n}"):
            mark = " ✅" if done > i else ""
            st.markdown(f"{icon} :{color}[**{title}**]{mark}")
            st.caption(what)


def run_and_store(query: str) -> None:
    status = st.status("Running the crew…", expanded=True)
    steps = ["Assistant answering from memory", "Web Search Assistant searching the live web",
             "Entry Agent reconciling and saving the record"]
    lines = [status.empty() for _ in steps]
    for i, s in enumerate(steps):
        lines[i].markdown(f"{'⏳' if i == 0 else '◽'} {s}")

    # Partial answers are shown as they land, so the user reads Agent 1's answer
    # in a few seconds instead of staring at a spinner until Agent 3 finishes.
    early = st.container()
    shown: dict = {}

    def on_task_done(n: int, text: str) -> None:
        lines[n - 1].markdown(f"✅ {steps[n - 1]}")
        if n < len(steps):
            lines[n].markdown(f"⏳ {steps[n]}")
            status.update(label=f"Running the crew… ({n}/3 done)", expanded=True)
        if n == 1:
            with early:
                st.markdown("#### 🧠 Direct answer — not yet verified")
                shown["d"] = st.info(text)
        elif n == 2:
            with early:
                st.markdown("#### 🌐 Web answer — reconciling…")
                shown["w"] = st.info(text)

    res = guarded_run(query, on_task_done=on_task_done)
    if res.small_talk_reply:
        status.update(label="Greeting — no crew needed", state="complete", expanded=False)
    elif res.blocked_reason:
        status.update(label="Blocked by input guardrails — crew not run", state="error", expanded=False)
    elif res.error:
        status.update(label="Crew stopped early", state="error", expanded=False)
    else:
        status.update(label=f"Crew finished in {res.elapsed_s:.1f}s", state="complete", expanded=False)
    st.session_state["last_result"] = res
    st.session_state["clear_box"] = True         # empty the input on the next render
    if not res.small_talk_reply:                 # transcript is display-only; agents never see it
        st.session_state.setdefault("history", []).append(res)
    st.rerun()                                   # re-render: result shown, box emptied, cards ticked


# ---------------------------------------------------------------------------
# 5. Optional evaluation mode — runs golden_set.json through the same crew
# ---------------------------------------------------------------------------

GOLDEN_SET_FILE = "golden_set.json"


def load_golden_set() -> tuple[list[dict], str]:
    """Return (items, error). Never raises."""
    import json

    if not os.path.exists(GOLDEN_SET_FILE):
        return [], f"`{GOLDEN_SET_FILE}` not found next to app.py."
    try:
        with open(GOLDEN_SET_FILE, encoding="utf-8") as fh:
            data = json.load(fh)
        return list(data["items"]), ""
    except Exception as exc:
        return [], f"Could not read `{GOLDEN_SET_FILE}`: {exc}"


def score_item(item: dict, res: RunResult) -> dict:
    """Score one golden-set item against the labels in golden_set.json:
       catch   -> contradiction_found AND severity == 'material'
       control -> NOT contradiction_found (else a false positive)
       refusal -> resolved_answer is the refusal string (confidence < threshold)"""
    v = res.verdict
    cat = item["category"]
    if v is None:
        outcome, passed = "no verdict", False
    elif cat == "catch":
        passed = v.contradiction_found and v.contradiction_severity == "material"
        outcome = "caught" if passed else "missed"
    elif cat == "control":
        passed = not v.contradiction_found
        outcome = "clean" if passed else "false positive"
    else:  # refusal
        passed = v.resolved_answer == REFUSAL_TEXT
        outcome = "refused" if passed else "answered"
    return {
        "id": item["id"],
        "category": cat,
        "query": item["query"],
        "severity": v.contradiction_severity if v else "—",
        "confidence": round(v.confidence, 2) if v else None,
        "outcome": outcome,
        "pass": passed,
        "seconds": round(res.elapsed_s, 1),
        "tokens": res.total_tokens,
        "error": res.error or res.blocked_reason or ("small talk" if res.small_talk_reply else ""),
    }


def summarize(rows: list[dict]) -> dict:
    """Headline metrics from scored rows. Rates are None when a category is empty."""
    valid = [r for r in rows if not r["error"]]     # runs that produced a verdict

    def rate(cat: str, want_pass: bool) -> Optional[float]:
        sub = [r for r in valid if r["category"] == cat]
        if not sub:
            return None
        return sum(1 for r in sub if r["pass"] == want_pass) / len(sub)

    n = len(valid)
    return {
        "catch_rate": rate("catch", True),
        "false_positive_rate": rate("control", False),
        "refusal_correct": rate("refusal", True),
        "errors": len(rows) - n,                   # timeouts / crew failures, reported separately
        "avg_seconds": sum(r["seconds"] for r in valid) / n if n else 0.0,
        "avg_tokens": sum(r["tokens"] for r in valid) / n if n else 0.0,
        "counts": {c: sum(1 for r in valid if r["category"] == c) for c in ("catch", "control", "refusal")},
    }


def fmt_rate(x: Optional[float]) -> str:
    return "—" if x is None else f"{x:.0%}"


def render_eval() -> None:
    st.markdown("## 📊 Evaluation mode")
    st.markdown(
        "Runs every item in `golden_set.json` through the **same three-agent crew** "
        "and scores Agent 3's verdict against the hand-written label. Reported "
        "two-sided on purpose: a reconciler that flags everything has a perfect "
        "catch rate and is useless."
    )
    items, err = load_golden_set()
    if err:
        st.error(err)
        return

    counts = {c: sum(1 for i in items if i["category"] == c) for c in ("catch", "control", "refusal")}
    st.caption(
        f"{len(items)} items — {counts['catch']} catch (expect material contradiction) · "
        f"{counts['control']} control (expect none) · {counts['refusal']} refusal (expect the refusal string)."
    )

    c1, c2 = st.columns([2, 1])
    with c1:
        cats = st.multiselect("Categories to run", ["catch", "control", "refusal"],
                              default=["catch", "control", "refusal"])
    with c2:
        pause = st.number_input("Pause between items (s)", min_value=0, max_value=60, value=5,
                                help="Spaces out requests to stay under OpenAI per-minute rate limits.")
    selected = [i for i in items if i["category"] in cats]
    st.caption(f"{len(selected)} item(s) selected · roughly {len(selected) * (20 + pause) // 60 + 1} min and "
               f"{len(selected)}× the cost of a single query.")

    if st.button("▶ Run evaluation", type="primary", disabled=not selected):
        rows: list[dict] = []
        bar = st.progress(0.0, text="Starting…")
        live = st.empty()
        for n, item in enumerate(selected, start=1):
            bar.progress((n - 1) / len(selected), text=f"{n}/{len(selected)} · {item['id']} — {item['query']}")
            rows.append(score_item(item, guarded_run(item["query"])))
            live.dataframe(rows, use_container_width=True, hide_index=True)
            if n < len(selected) and pause:
                time.sleep(pause)
        bar.progress(1.0, text=f"Done — {len(rows)} item(s).")
        st.session_state["eval_rows"] = rows

    rows = st.session_state.get("eval_rows")
    if not rows:
        return

    s = summarize(rows)
    st.markdown("### Results")
    m1, m2, m3 = st.columns(3)
    m1.metric("Catch rate", fmt_rate(s["catch_rate"]),
              help="Share of 'catch' items where Agent 3 returned contradiction_found=true AND severity='material'.")
    m2.metric("False-positive rate", fmt_rate(s["false_positive_rate"]),
              help="Share of 'control' items where Agent 3 wrongly returned contradiction_found=true.")
    m3.metric("Refusals correct", fmt_rate(s["refusal_correct"]),
              help=f"Share of 'refusal' items where the resolved answer was '{REFUSAL_TEXT}'.")
    st.caption(
        f"Latency/cost: avg **{s['avg_seconds']:.1f} s** and **{s['avg_tokens']:,.0f} tokens** per query "
        f"across {len(rows) - s['errors']} completed run(s)"
        + (f" · **{s['errors']} run(s) errored/timed out** and are excluded from the rates." if s["errors"] else ".")
    )
    st.dataframe(rows, use_container_width=True, hide_index=True)

    import json
    st.download_button("Download results (JSON)", data=json.dumps({"summary": s, "rows": rows}, indent=2),
                       file_name="eval_results.json", mime="application/json")


# ---------------------------------------------------------------------------
# 6. Page
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Crew Desk Support", page_icon="🛎️", layout="centered")

# Cosmetic only: colour palette for the page, sidebar, hero banner and the cards.
st.markdown(
    """<style>
    #MainMenu, footer, .stAppDeployButton {visibility: hidden;}
    .block-container {padding-top: 1.6rem;}
    .stApp {background: linear-gradient(180deg, #EEF2FF 0%, #FDF2F8 100%);}
    [data-testid="stSidebar"] {background: linear-gradient(180deg, #312E81 0%, #4F46E5 100%);}
    [data-testid="stSidebar"] * {color: #EEF2FF;}
    [data-testid="stSidebar"] code {background: rgba(255,255,255,.18); color: #FDE68A;}
    [data-testid="stSidebar"] .stButton button {background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.55);}
    [data-testid="stSidebar"] .stButton button:hover {background: rgba(255,255,255,.28);}
    .hero {background: linear-gradient(90deg, #4F46E5 0%, #9333EA 55%, #EC4899 100%);
           border-radius: 18px; padding: 26px 30px; color: #fff; margin-bottom: 14px;
           box-shadow: 0 8px 24px rgba(79,70,229,.25);}
    .hero h1 {color: #fff; margin: 0; font-size: 2.2rem; line-height: 1.1;}
    .hero p  {margin: 8px 0 0; font-size: 1.05rem; opacity: .95;}
    [class*="st-key-agent1"] {background: #DBEAFE; border-left: 6px solid #2563EB; border-radius: 14px;}
    [class*="st-key-agent2"] {background: #DCFCE7; border-left: 6px solid #16A34A; border-radius: 14px;}
    [class*="st-key-agent3"] {background: #FFEDD5; border-left: 6px solid #EA580C; border-radius: 14px;}
    [class*="st-key-card_resolved"] {background: #FFF7ED; border-left: 6px solid #F59E0B; border-radius: 14px;}
    [class*="st-key-card_direct"]   {background: #EFF6FF; border-left: 6px solid #3B82F6; border-radius: 14px;}
    [class*="st-key-card_web"]      {background: #F0FDF4; border-left: 6px solid #22C55E; border-radius: 14px;}
    </style>""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 🛎️ Crew Desk Support")
    st.caption("Three agents, run in sequence. The third one cross-checks the first against live web evidence.")
    st.markdown(
        f"- **Model:** `{MODEL_NAME}`"
        + (f" · Agent 3: `{RECONCILER_MODEL}`" if RECONCILER_MODEL != MODEL_NAME else "")
        + f"\n- **Refusal threshold:** confidence < {CONFIDENCE_THRESHOLD:.2f}"
        + f"\n- **Record file:** `{ANSWERS_FILE}`"
    )
    n_records = 0
    if os.path.exists(ANSWERS_FILE):
        with open(ANSWERS_FILE, encoding="utf-8") as fh:
            n_records = sum(1 for line in fh if line.startswith("=== "))
    st.caption(f"{n_records} record(s) saved so far.")
    st.markdown("---")
    eval_mode = st.toggle("Run evaluation", help=f"Score the crew on the labelled queries in {GOLDEN_SET_FILE}.")
    if st.session_state.get("history") and st.button("Clear session history"):
        st.session_state.pop("history", None); st.session_state.pop("last_result", None); st.rerun()

st.markdown(
    """<div class="hero"><h1>🛎️ Crew Desk Support</h1>
    <p>One agent answers &middot; one agent verifies on the live web &middot; one agent reconciles and records.</p></div>""",
    unsafe_allow_html=True,
)

absent = missing_keys()
if absent:
    st.error(
        "Missing environment variable(s): "
        + ", ".join(f"`{k}` ({REQUIRED_KEYS[k]})" for k in absent)
        + ". Set them in your shell or a local `.env` file and restart."
    )
    st.stop()

if eval_mode:                       # evaluation replaces the single-query view
    render_eval()
    st.stop()

pipeline_slot = st.empty()          # re-rendered after a run so the cards show ✅


def show_pipeline() -> None:
    last = st.session_state.get("last_result")
    with pipeline_slot.container():
        render_pipeline(done=3 if last and not last.error else 0)


show_pipeline()

# Empty the box after a question has been answered (flag set by run_and_store;
# widget state may only be changed before the widget is created on the next run).
if st.session_state.pop("clear_box", False):
    st.session_state["ask_box"] = ""

with st.form("ask", clear_on_submit=False, border=False):   # keep the text visible while the crew runs
    c_in, c_btn = st.columns([5, 1], vertical_alignment="bottom")
    query = c_in.text_input(
        "Ask a support question",
        placeholder="e.g. What is the latest stable version of Python?",
        key="ask_box",
    )
    submitted = c_btn.form_submit_button("Ask", type="primary", use_container_width=True)

picked = st.pills("Try an example", EXAMPLES, selection_mode="single", key="example_pick")

if submitted and query.strip():
    run_and_store(query.strip())
elif picked and picked != st.session_state.get("last_example"):
    st.session_state["last_example"] = picked
    run_and_store(picked)

if "last_result" in st.session_state:
    res: RunResult = st.session_state["last_result"]
    st.markdown("")
    st.markdown(f"**Query:** {res.query}")
    render_result(res)

# Session transcript — display only. Each query is still an independent crew run;
# nothing here is passed back to the agents (see README: no agent memory).
earlier = [h for h in st.session_state.get("history", []) if h is not st.session_state.get("last_result")]
if earlier:
    st.markdown("---")
    st.markdown(f"#### Earlier in this session ({len(earlier)})")
    for i, old in enumerate(reversed(earlier)):
        v = old.verdict
        tag = ("🛡️ blocked" if old.blocked_reason else "⚠️ error" if old.error else
               "🟡 refused" if v and v.resolved_answer == REFUSAL_TEXT else
               f"🔴 {v.contradiction_severity}" if v and v.contradiction_severity == "material" else
               f"🔵 {v.contradiction_severity}" if v and v.contradiction_severity == "minor" else "🟢 agree")
        with st.expander(f"{tag} · {old.query}"):
            render_result(old, k=f"_h{i}")
