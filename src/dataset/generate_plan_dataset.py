"""Generate natural (API-name-free) queries with 10-15 step canonical plans via Gemini.

Discovery-stage dataset, v2 design:
  - The query is written the way an ordinary user would type it: no API, vendor,
    product or library names (fixes ToolBench's name leakage, see data/results/).
  - The plan is in the project's canonical step form
        {step_num, canonical_intent, primitive_function, primitive_id,
         parameters, requires_api, depends_on, ground_truth_tool}
    using the RWPL taxonomy (READ=0, WRITE=1, PROCESS=2, LOCAL=3), which replaces
    the earlier READ/WRITE/CHECK/PROCESS one: the three API labels say what the
    API call does, LOCAL marks every step that calls no API, so requires_api is
    derived from the label. Note the ids differ from the older datasets (CHECK=2).
  - Each accepted plan then goes through ground_plan.py: API steps get
    `ground_truth_tool` / `acceptable_tools` from master_registry.json and the
    record gets the gold `final_plan`. Only fully grounded samples are written to
    the dataset; the rest go to *.ungrounded.jsonl or *.rejects.jsonl.

Every sample passes a validator (structure, primitive consistency, intent form,
parameter leakage, duplicate steps, vendor names). Failures are retried with the
error list fed back to the model; samples that still fail go to a rejects file.

Usage:
    python3 src/dataset/generate_plan_dataset.py --n 5 [--workers 4]
Workers run samples concurrently; one shared limiter keeps all requests under 14/min.
"""
import argparse
import contextlib
import fcntl
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from google import genai
from google.genai import types

from ground_plan import CandidateIndex, ground

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

PROMPT_VERSION = "plan_v3.6"
TAXONOMY = "RWPL"
PRIMITIVE_IDS = {"READ": 0, "WRITE": 1, "PROCESS": 2, "LOCAL": 3}
API_PRIMITIVES = {"READ", "WRITE", "PROCESS"}

SYSTEM_PROMPT = """You build training data for a task-planning system that turns a plain user request into an ordered canonical plan. Some plan steps are later matched to real web APIs from a large public API marketplace by semantic search over API descriptions.

Given a DOMAIN, a TASK TYPE, a PERSONA, a STYLE and a DELIVERY, produce ONE sample.

=== THE QUERY ===
- What the persona would actually type, in the given style. 1-3 sentences, natural, somewhat underspecified.
- It must be a software / automation task of the given TASK TYPE that can genuinely be built by combining web APIs with ordinary program logic (not a purely offline or physical-world chore).
- It must be substantial enough that a real implementation needs 10+ steps: it should naturally involve at least 4 DIFFERENT external capabilities (e.g. search news + extract article text + translate + send email; not several calls to the same service). Even a terse query can ask for something substantial.
- NEVER name any API, vendor, product, platform, app or library (no "Slack", "Gmail", "Google", "Twilio", "OpenAI", "Shopify", ...). Use generic words ("email", "chat", "my online store", "my calendar").
- The end result goes to the given DELIVERY destination. Do NOT use a spreadsheet unless DELIVERY says so.
- Input data should come from sources public web APIs actually serve (web/news/social search, weather, maps, finance and market data, the user's online store, email inbox, bank, calendar, files they upload), NOT from reading the user's own spreadsheet or private database - those read APIs are rarely available.
- Concrete details the user would naturally give (file types, cities, frequencies, languages, thresholds) are welcome. If the query says "a few specific currencies" or "a genre" without naming them, the plan must NOT name them either.

=== THE CANONICAL PLAN: 10-15 steps (HARD LIMIT - count them; fewer than 10 is rejected) ===
Reach 10+ steps by decomposing at the granularity a developer would implement (input validation, fetch, parse, transform, decide, store, notify), never by padding with filler.
Each step:
- step_num: 1..N in order.
- canonical_intent: 4-8 words, Action-Object imperative, e.g. "Extract text content from PDF document".
    * Be SPECIFIC about the capability and data type: "Convert text into speech audio", NOT "Process the data". Avoid vague nouns (data, info, item, resource, content) when a specific noun exists. This text is used for retrieval, so it must describe the capability precisely.
    * For API steps (READ/WRITE/PROCESS), name the API CAPABILITY, not the payload being passed through it: "Send SMS text message to phone", NOT "Send summary report via text message"; "Translate text into another language", NOT "Translate headlines into target language"; "List recent orders from online store", NOT "List new patient appointment payments"; "Read rows from online spreadsheet", NOT "Read rows from security audit spreadsheet". Payload or domain words (summary, report, headlines, appointment, security) make search return the wrong kind of API.
    * A READ step must not be worded like a creation ("Create", "Add") and a WRITE step must not be worded like a lookup.
    * NEVER include concrete parameter values (no cities, dates, numbers, file names, keywords) - those go in `parameters`.
    * NEVER include vendor/product/API names.
- primitive_function: exactly one of four labels, decided by WHETHER the step calls an external web API and WHAT that call does (never by the surface verb):
    LOCAL   : NO external API call. Ordinary program logic: defining rules or config, validating input, parsing or extracting fields from data already in hand, comparing against a threshold, filtering/sorting/merging/counting lists, formatting messages, making decisions, reading or writing the program's own local state.
    READ    : API call that RETRIEVES data already stored in an external source, changing nothing: get weather, search news articles, list store orders, look up a stock price, get shipment tracking status, read spreadsheet rows, fetch an account balance, scrape a web page.
    WRITE   : API call that CHANGES STATE in an external system: send an email/SMS/push notification, post a chat message, publish a blog post, append a spreadsheet row, create a calendar event, upload a file, book or purchase, update or delete a remote record.
    PROCESS : API call that COMPUTES or TRANSFORMS the input you send and returns a new result, storing nothing: summarize text, translate, analyze sentiment, convert text to speech, extract text from a PDF or image, generate an image or chart, score readability, convert currency amounts, verify an email address is deliverable.
  Boundary cases: extracting article text from a web page / URL is READ (a scrape of existing content); extracting text from a PDF, image or audio you send is PROCESS. Writing a report or message with a text-generation API is PROCESS; assembling a report from numbers you already computed is LOCAL.
  Tie-breakers: (1) no API call means LOCAL, whatever the verb ("Check if delay exceeds threshold" is LOCAL). (2) READ returns data that already existed in the external source; PROCESS returns something newly computed from your input. (3) If the call leaves a lasting change outside the program it is WRITE, even if it also returns something.
  Use an API step only when a public API marketplace would plausibly offer that capability. Aim for 4-6 API steps (READ + WRITE + PROCESS), min 4, max 6; all other steps are LOCAL.
- parameters: list of {name, value} for USER-SPECIFIC values stated in or directly implied by the query that this step uses: cities, dates, times, frequencies, thresholds, amounts, keywords, topics, recipients, languages. Never invent values. Empty list if none.
    * Data types, formats and capability qualifiers (PDF, spreadsheet, audio, SMS, abstractive, sentiment) are NOT parameters - they belong in canonical_intent because they define WHICH API is needed.
    * A parameter value must never also appear in canonical_intent: "Schedule recurring digest delivery" + {"frequency": "daily"}, NOT "Schedule daily digest delivery".
    * Keep values exactly as the user gave them, including units ("5 kilowatts", not "5"). If the user did not give an actual value, OMIT the parameter - never write placeholders like "the person I track", "my region", "my working language", "specified".
- depends_on: step_nums of earlier steps whose output this step needs (empty list if none).

Plan quality rules:
- Steps follow the real execution order and together fully accomplish the request.
- If the query only needs 1-2 external capabilities, write a richer query instead of inventing API steps (no "retrieve origin details", "verify upload status", "move processed file" padding).
- Faithfulness: the plan must not add capabilities the user neither asked for nor strictly needs (no unrequested translation, email-address validation, logging, image generation, extra notifications; language detection only when translation or language filtering is requested). The 3+ capabilities must come from the QUERY itself.
- No duplicate or near-duplicate steps. No filler unless the user explicitly asked for it: no "log the activity", "perform final QA", "confirm success", "verify row was added", "monitor the system", "clear cache", "store prices in local memory", "initialize connection to document", "validate spreadsheet schema", "identify target website", "define input format", and NO infrastructure checks like "validate chat connection", "verify API availability", "check access permissions", "authenticate with service". Every step must do real work toward the user's goal.
- No loop-control steps ("check for remaining items", "repeat for next article"): applying a step to each item of a list is implicit.
- Include the LOCAL steps a real implementation needs (define input rules, validate, parse, chunk, merge, compare, format, decide).

=== EXAMPLE ===
DOMAIN: Text Analysis | TASK TYPE: build a small user-facing tool or app | PERSONA: graduate student | STYLE: direct build request | DELIVERY: email to the user or a recipient
{"query": "Design a text summarizer that takes PDFs and pasted text, creates an abstractive summary in English, translates it into Spanish for my co-author, and emails both versions to me.",
 "canonical_plan": [
  {"step_num": 1, "canonical_intent": "Define accepted input formats and limits", "primitive_function": "LOCAL", "parameters": [{"name": "input_formats", "value": "pdf, plain text"}], "depends_on": []},
  {"step_num": 2, "canonical_intent": "Validate uploaded file type and size", "primitive_function": "LOCAL", "parameters": [], "depends_on": [1]},
  {"step_num": 3, "canonical_intent": "Extract text content from PDF document", "primitive_function": "PROCESS", "parameters": [], "depends_on": [2]},
  {"step_num": 4, "canonical_intent": "Merge extracted and pasted input text", "primitive_function": "LOCAL", "parameters": [], "depends_on": [3]},
  {"step_num": 5, "canonical_intent": "Clean and normalize raw input text", "primitive_function": "LOCAL", "parameters": [], "depends_on": [4]},
  {"step_num": 6, "canonical_intent": "Split long text into token-sized chunks", "primitive_function": "LOCAL", "parameters": [], "depends_on": [5]},
  {"step_num": 7, "canonical_intent": "Generate abstractive summary for each chunk", "primitive_function": "PROCESS", "parameters": [{"name": "output_language", "value": "English"}], "depends_on": [6]},
  {"step_num": 8, "canonical_intent": "Combine chunk summaries into final summary", "primitive_function": "LOCAL", "parameters": [], "depends_on": [7]},
  {"step_num": 9, "canonical_intent": "Verify summary length within configured limit", "primitive_function": "LOCAL", "parameters": [], "depends_on": [8]},
  {"step_num": 10, "canonical_intent": "Translate text into another language", "primitive_function": "PROCESS", "parameters": [{"name": "target_language", "value": "Spanish"}], "depends_on": [9]},
  {"step_num": 11, "canonical_intent": "Format both summaries as email body", "primitive_function": "LOCAL", "parameters": [], "depends_on": [9, 10]},
  {"step_num": 12, "canonical_intent": "Send email message to recipient address", "primitive_function": "WRITE", "parameters": [], "depends_on": [11]}
 ]}

Output ONLY the JSON object."""

USER_TEMPLATE = ("DOMAIN: {domain} | TASK TYPE: {task_type} | PERSONA: {persona} | STYLE: {style}"
                 " | DELIVERY: {delivery}")

# Where the result ends up. Rotated so WRITE steps cover many endpoint families; the
# registry's only spreadsheet writer ('Add rows') has no parameter for the row data, and
# without this slot 71% of v3.5 samples ended in it.
DELIVERIES = [
    "email to the user or a recipient",
    "SMS text message",
    "message in a team chat channel",
    "push notification to a phone",
    "event on the user's calendar",
    "task or ticket in a project / helpdesk tool",
    "post published on social media",
    "file uploaded to cloud storage",
    "contact or lead record in a CRM",
    "booking, reservation or order placed",
    "post published on a blog or website",
    "shortened shareable link sent by email or SMS",
    "result shown to the user in the tool itself (the work is in the READ/PROCESS steps)",
]

TASK_TYPES = [
    "build a small user-facing tool or app",
    "one-off research / comparison / report",
    "content creation or media generation pipeline",
    "data enrichment, conversion or cleanup job",
    "monitoring with alerts or notifications",
    "booking, purchase or other transaction workflow",
    "personal or team productivity automation",
    "data analysis with a visual or written summary",
]

PERSONAS = [
    "small business owner", "graduate student", "freelance designer", "HR manager",
    "indie game developer", "high-school teacher", "journalist", "nonprofit coordinator",
    "real-estate agent", "fitness coach", "hobbyist investor", "startup founder",
    "data analyst", "travel blogger", "clinic administrator", "e-commerce seller",
    "backend developer", "marketing intern", "event organizer", "research scientist",
    "restaurant manager", "IT support technician", "content creator", "logistics coordinator",
]
STYLES = [
    "direct build request (e.g. 'Build/Design/Make a ...')",
    "casual problem description of a pain point, no explicit 'build' verb",
    "question form (e.g. 'Is there a way to ...' / 'Can you set up ...')",
    "goal statement with one or two explicit constraints",
    "terse one-liner, lowercase, slightly sloppy",
    "detailed but non-technical, 2-3 sentences of context",
]

# Vendor/product names a real user might mention but which must never appear.
BRANDS = [
    "google", "gmail", "slack", "twilio", "openai", "chatgpt", "gpt", "aws", "amazon",
    "azure", "microsoft", "outlook", "dropbox", "notion", "trello", "zapier", "stripe",
    "paypal", "shopify", "twitter", "facebook", "instagram", "linkedin", "youtube",
    "spotify", "whatsapp", "telegram", "discord", "salesforce", "hubspot", "mailchimp",
    "sendgrid", "github", "jira", "zoom", "airtable", "rapidapi", "yelp", "uber",
    "tiktok", "reddit", "pinterest", "openweather", "alpha vantage", "deepl",
]
BRAND_RE = re.compile(r"\b(" + "|".join(map(re.escape, BRANDS)) + r")\b", re.I)
FILLER_RE = re.compile(
    r"\b(log (the )?(activity|action|event)s?|audit trail|final (qa|quality)|"
    r"confirm .*(success|dispatch)|monitor (the )?system|clear .*cache|temporary staging|"
    r"(validate|verify|check|test) .*(connection|availability|access|permission|credential)s?|"
    r"authenticat\w*|credentials?|api keys?|(error|system|execution|run|audit|access) logs?|"
    r"(write|save|append|record|update) .*\blogs?|"
    r"(define|configure|establish|open|set up) .*(connection|access)|iterate|"
    r"confirm|(verify|check|identify|determine) .*\b(complet\w*|success\w*|saved|added|appended|sent|deliver(y|ed))|"
    r"(check|look) for (remaining|more|next)|repeat|initiali[sz]e|store .* (local )?memory|"
    r"define (input|output) format|data structure|validate .*schema|"
    r"identify (the )?(target|destination) (website|url|spreadsheet|sheet|document|folder|file))\b",
    re.I)

# For API steps: leading verbs whose API label is unambiguous. Ambiguous verbs
# ("extract", "check", "get") are absent: extracting can be a scrape (READ) or OCR (PROCESS).
VERB_PRIMITIVE = {
    **dict.fromkeys(["summarize", "translate", "convert", "generate", "rewrite", "analyze",
                     "detect", "classify", "transcribe", "score"], "PROCESS"),
    **dict.fromkeys(["send", "post", "publish", "append", "insert", "delete", "upload",
                     "create", "book", "schedule", "update", "notify"], "WRITE"),
    **dict.fromkeys(["fetch", "retrieve", "search", "query", "download", "look", "list"], "READ"),
}
# Verbs that only make sense as external calls, so they cannot be LOCAL.
EXTERNAL_VERBS = {"send", "post", "publish", "upload", "book", "notify", "fetch", "download",
                  "search", "scrape", "translate", "summarize", "transcribe"}
PLACEHOLDER_VALUES = {"specified", "specific", "target", "given", "custom", "user", "any",
                      "relevance", "selected", "desired", "chosen", "politician", "person",
                      "set", "significant", "certain", "some", "various", "stock", "ongoing",
                      "brand", "company", "product", "business"}
# Vague stand-ins the model writes instead of omitting an unknown value ("my region").
PLACEHOLDER_RE = re.compile(r"^(my|the|a|some|their|our)\b|\b(i'm|i am|i track|specified)\b", re.I)

SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "query": {"type": "STRING"},
        "canonical_plan": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "step_num": {"type": "INTEGER"},
                    "canonical_intent": {"type": "STRING"},
                    "primitive_function": {"type": "STRING", "enum": list(PRIMITIVE_IDS)},
                    "parameters": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {"name": {"type": "STRING"}, "value": {"type": "STRING"}},
                            "required": ["name", "value"],
                        },
                    },
                    "depends_on": {"type": "ARRAY", "items": {"type": "INTEGER"}},
                },
                "required": ["step_num", "canonical_intent", "primitive_function",
                             "parameters", "depends_on"],
            },
        },
    },
    "required": ["query", "canonical_plan"],
}


def _tokens(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


NUMBER_WORDS = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                "twelve", "twenty", "fifty", "hundred", "thousand", "million", "billion",
                "half", "quarter", "dozen", "k"}


def _stem(tok):
    return re.sub(r"(ing|ed|es|s|ly|er)$", "", tok) if len(tok) > 4 else tok


def invented(value, q_toks):
    """True if no word of a parameter value is in the query ('Bitcoin, Ethereum' for
    'a few digital currencies'). Stem/prefix matching tolerates 'weekly'/'week', 'rated'/'rating'."""
    toks = {t for t in _tokens(value) if len(t) >= 3 or t.isdigit()}
    if not toks:
        return False
    if all(t.isdigit() for t in toks):  # '1000000' for 'a million dollars' is fine
        return not (any(t.isdigit() for t in q_toks) or q_toks & NUMBER_WORDS)
    stems = {_stem(u) for u in q_toks if len(u) >= 3}
    return not any(s.startswith(u) or u.startswith(s)
                   for s in map(_stem, (t for t in toks if not t.isdigit()))
                   for u in stems if min(len(s), len(u)) >= 3)


def bad_value(name, value, q_toks=None):
    """'placeholder' / 'invented' (only checked when q_toks is given) / None."""
    v = value.lower()
    if (not v or v in (name.lower(), name.lower().replace("_", " "))
            or v in PLACEHOLDER_VALUES or PLACEHOLDER_RE.search(value)):
        return "placeholder"
    if q_toks is not None and invented(value, q_toks):
        return "invented"
    return None


def autofix(rec):
    """Drop placeholder / invented parameters in place - exactly what a retry would ask
    the model to do, without spending a request. Returns the list of fixes made."""
    q_toks, fixes = _tokens(rec["query"]), []
    for s in rec["canonical_plan"]:
        for k, v in list(s["parameters"].items()):
            if bad := bad_value(k, str(v).strip(), q_toks):
                del s["parameters"][k]
                fixes.append(f"step {s['step_num']}: dropped {bad} parameter {k}={str(v)!r}")
    return fixes


def normalize(raw):
    """Convert model output to the stored canonical record shape."""
    plan = []
    for s in raw["canonical_plan"]:
        plan.append({
            "step_num": s["step_num"],
            "canonical_intent": s["canonical_intent"].strip().rstrip("."),
            "primitive_function": s["primitive_function"],
            "primitive_id": PRIMITIVE_IDS[s["primitive_function"]],
            "parameters": {p["name"]: p["value"] for p in s.get("parameters", [])},
            "requires_api": s["primitive_function"] in API_PRIMITIVES,
            "depends_on": s.get("depends_on", []),
            "ground_truth_tool": None,
        })
    return {"query": raw["query"].strip(), "canonical_plan": plan}


def validate(rec, registry_titles, min_api=4, check_invented=True):
    """Return (errors, warnings). Errors trigger a retry; warnings are kept for review.
    min_api=3 / check_invented=False re-validate samples made before plan_v3.6."""
    errors, warnings = [], []
    q, plan = rec["query"], rec["canonical_plan"]

    if m := BRAND_RE.search(q):
        errors.append(f"query names a vendor/product: '{m.group(0)}'")
    if not 8 <= len(q.split()) <= 90:
        errors.append(f"query length {len(q.split())} words (want 8-90)")
    ql = f" {q.lower()} "
    leaked = [t for t in registry_titles if f" {t} " in ql]
    if leaked:
        warnings.append(f"query contains registry tool title(s): {leaked[:3]}")

    if not 10 <= len(plan) <= 15:
        errors.append(f"plan has {len(plan)} steps (want 10-15)")
    n_api = sum(s["requires_api"] for s in plan)
    if not min_api <= n_api <= 6:
        errors.append(f"plan has {n_api} API (READ/WRITE/PROCESS) steps (want {min_api}-6); "
                      f"the query must need at least {min_api} different external capabilities")
    q_toks = _tokens(q)

    seen = []
    for i, s in enumerate(plan, 1):
        intent = s["canonical_intent"]
        if s["step_num"] != i:
            errors.append(f"step_num {s['step_num']} out of order (expected {i})")
        if any(not (1 <= d < s["step_num"]) for d in s["depends_on"]):
            errors.append(f"step {i} depends_on {s['depends_on']} must reference earlier steps")
        if not 4 <= len(intent.split()) <= 8:
            errors.append(f"step {i} intent '{intent}' has {len(intent.split())} words (want 4-8)")
        if m := BRAND_RE.search(intent):
            errors.append(f"step {i} intent names a vendor/product: '{m.group(0)}'")
        verb = intent.split()[0].lower() if intent else ""
        prim = s["primitive_function"]
        if prim == "LOCAL" and verb in EXTERNAL_VERBS:
            errors.append(f"step {i} '{intent}' is labeled LOCAL but '{verb}' needs an "
                          f"external API (READ/WRITE/PROCESS)")
        expected = VERB_PRIMITIVE.get(verb)
        if prim in API_PRIMITIVES and expected and prim != expected:
            errors.append(f"step {i} '{intent}' is labeled {prim} "
                          f"but '{verb}' is a {expected} API action")
        for k, v in s["parameters"].items():
            v = str(v).strip()
            bad = bad_value(k, v, q_toks if check_invented else None)
            if bad == "placeholder":
                errors.append(f"step {i} parameter '{k}' has placeholder value '{v}'; omit it")
                continue
            if bad == "invented":
                errors.append(f"step {i} parameter '{k}' = '{v}' does not come from the query; "
                              f"never invent values - omit it")
                continue
            if len(v) < 3 or not re.search(rf"\b{re.escape(v.lower())}\b", intent.lower()):
                continue
            # Specific values (numbers, names, multi-word) are leaks; one generic word is a review flag.
            if re.search(r"\d", v) or " " in v or v[0].isupper():
                errors.append(f"step {i} intent leaks parameter value '{v}'")
            else:
                warnings.append(f"step {i} intent repeats parameter word '{v}'")
        if re.search(r"\d", intent):
            warnings.append(f"step {i} intent contains a digit: '{intent}'")
        if FILLER_RE.search(intent):
            errors.append(f"step {i} is filler/infrastructure, replace with real work: '{intent}'")
        toks = _tokens(intent)
        for j, prev in seen:
            if toks and len(toks & prev) / len(toks | prev) >= 0.8:
                errors.append(f"step {i} duplicates step {j}: '{intent}'")
        seen.append((i, toks))
    return errors, warnings


@contextlib.contextmanager
def locked(path):
    """Exclusive lock shared with apply_plan_review.py, which rewrites the dataset in place."""
    with open(path + ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


class QuotaExhausted(BaseException):
    """Daily quota hit. A BaseException so the per-call `except Exception` handlers
    (here and in ground_plan) let it through and every worker stops."""


class RateLimiter:
    """Spaces request starts across all worker threads (free tier: 15 RPM)."""

    def __init__(self, rpm):
        self.interval, self.next, self.lock = 60.0 / rpm, 0.0, threading.Lock()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            start = max(now, self.next)
            self.next = start + self.interval
        time.sleep(start - now)


LIMITER = RateLimiter(14)
STOP = threading.Event()


def call_model(client, model, contents, system=SYSTEM_PROMPT, schema=SCHEMA, temperature=0.9,
               max_transient=6, log=print):
    """One generation call; overload/rate-limit errors are retried here so they
    don't use up the quality-retry attempts."""
    for k in range(1, max_transient + 1):
        if STOP.is_set():
            raise QuotaExhausted()
        LIMITER.wait()
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=temperature,
                    response_mime_type="application/json",
                    response_schema=schema,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                ),
            )
        except Exception as e:
            if "429" in str(e) and "PerDay" in str(e):
                STOP.set()
                raise QuotaExhausted() from None
            if k == max_transient or not any(c in str(e) for c in ("429", "500", "503")):
                raise
            log(f"    transient error ({str(e)[:40]}...), retry in {15 * k}s")
            time.sleep(15 * k)


def report(path):
    recs = [json.loads(l) for l in open(path)] if os.path.exists(path) else []
    if not recs:
        return
    steps = [s for r in recs for s in r["canonical_plan"]]
    prim = {p: sum(s["primitive_function"] == p for s in steps) for p in PRIMITIVE_IDS}
    print(f"\n=== {path}: {len(recs)} samples, {len(steps)} steps ===")
    print(f"steps/sample  : {len(steps)/len(recs):.1f}")
    print(f"api steps     : {sum(s['requires_api'] for s in steps)} "
          f"({sum(s['requires_api'] for s in steps)/len(recs):.1f}/sample)")
    print(f"primitives    : {prim}")
    print(f"warnings      : {sum(len(r['qc']['warnings']) for r in recs)}")


def main():
    ap = argparse.ArgumentParser()
    root = f"{REPO}/data/processed"
    ap.add_argument("--registry", default=f"{root}/master_registry.json")
    ap.add_argument("--output", default=f"{root}/plan_dataset_v3.jsonl")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--model", default="gemini-3.1-flash-lite")
    ap.add_argument("--max_attempts", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    rejects_path = args.output.replace(".jsonl", ".rejects.jsonl")
    ungrounded_path = args.output.replace(".jsonl", ".ungrounded.jsonl")
    repair_path = args.output.replace(".jsonl", ".repair.jsonl")
    index = CandidateIndex(args.registry)

    registry = json.load(open(args.registry))
    categories = sorted({r["category"] for r in registry})
    # Multi-word tool titles only: single words ("Weather") are too generic to call leakage.
    registry_titles = {t.lower() for r in registry
                       if len((t := r.get("tool_title") or "").split()) >= 2}

    existing = [json.loads(l) for l in open(args.output)] if os.path.exists(args.output) else []
    # Samples waiting for a manual repair, with no API in the registry, or discarded as final
    # are not regenerated; other rejects are, so they get a fresh attempt under the current prompt.
    parked = [json.loads(l) for p in (repair_path, ungrounded_path) if os.path.exists(p)
              for l in open(p)]
    final_rejects = [r for r in (json.loads(l) for l in open(rejects_path))
                     if r.get("no_retry")] if os.path.exists(rejects_path) else []
    done_ids = {r["sample_id"] for r in existing + parked + final_rejects}
    seen_queries = {r["query"].lower() for r in existing}
    state_lock, print_lock = threading.Lock(), threading.Lock()

    # Slots are drawn up front, in index order, so they do not depend on worker timing.
    rng = random.Random(args.seed)
    cat_order = categories[:]
    rng.shuffle(cat_order)
    slots = []
    for i in range(args.n):
        persona, style = rng.choice(PERSONAS), rng.choice(STYLES)
        slots.append({"sample_id": f"planv3_{args.seed}_{i:05d}",
                      "domain": cat_order[i % len(cat_order)],
                      "task_type": TASK_TYPES[i % len(TASK_TYPES)],
                      "persona": persona, "style": style,
                      "delivery": DELIVERIES[i % len(DELIVERIES)]})
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    def append(path, rec):
        with state_lock, locked(args.output), open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def run(i):
        slot = slots[i]
        if STOP.is_set() or slot["sample_id"] in done_ids:
            return
        lines = [f"[{i}] {slot['domain']} | {slot['task_type']} | {slot['persona']} | "
                 f"{slot['style'].split(' (')[0]} | {slot['delivery']}"]
        say = lines.append
        try:
            process(i, slot, say)
        except QuotaExhausted:
            say("    stopped: daily quota exhausted")
        finally:
            with print_lock:
                print("\n".join(lines), flush=True)

    def process(i, slot, say):
        user_msg = USER_TEMPLATE.format(**{k: slot[k] for k in
                                           ("domain", "task_type", "persona", "style", "delivery")})
        feedback, rec, errors = "", None, ["no attempt"]
        for attempt in range(1, args.max_attempts + 1):
            try:
                resp = call_model(client, args.model, user_msg + feedback, log=say)
                rec = normalize(json.loads(resp.text))
                fixes = autofix(rec)
                errors, warnings = validate(rec, registry_titles)
                warnings += [f"autofix: {f}" for f in fixes]
                with state_lock:
                    if rec["query"].lower() in seen_queries:
                        errors.append("duplicate query")
            except Exception as e:  # malformed JSON, schema mismatch, or retries exhausted
                errors, warnings = [f"call/parse error: {str(e)[:200]}"], []
            if not errors:
                break
            say(f"    attempt {attempt}: {len(errors)} error(s): {errors[:3]}")
            feedback = ("\n\nYour previous output was rejected for these reasons; "
                        "produce a new sample that fixes all of them:\n- " + "\n- ".join(errors))

        meta = {**slot, "generator_model": args.model, "prompt_version": PROMPT_VERSION,
                "taxonomy": TAXONOMY, "attempts": attempt}
        if errors:
            # A skeleton that parsed goes to the repair queue (fixed with
            # repair_plan.py); only unparseable output is a plain reject.
            path, stage = (repair_path, "plan") if rec else (rejects_path, None)
            append(path, {**meta, **(rec or {}), "repair_stage": stage, "errors": errors})
            say(f"    {'REPAIR QUEUE' if rec else 'REJECTED'} after {attempt} attempts")
            return
        out = {**meta, **rec,
               "n_steps": len(rec["canonical_plan"]),
               "n_api_steps": sum(s["requires_api"] for s in rec["canonical_plan"]),
               "qc": {"errors": [], "warnings": warnings}}
        say(f"    plan ok: {out['n_steps']} steps, {out['n_api_steps']} api, "
            f"{len(warnings)} warning(s), attempt {attempt}")

        # Stage 2: ground API steps in the registry and write the gold final plan.
        out, status, g_errors = ground(
            out, index,
            lambda contents, system, schema: call_model(client, args.model, contents, system,
                                                        schema, temperature=0.2, log=say),
            judge_model=args.model)
        if status != "complete":
            # failed / plan_defect carry the candidate pools and the judge's last output,
            # so they can be fixed without spending more requests.
            path = ungrounded_path if status == "ungrounded" else repair_path
            append(path, {**out, "repair_stage": "grounding", "errors": g_errors})
            say(f"    GROUNDING {status.upper()}: {g_errors[:3]}")
            return
        with state_lock:
            seen_queries.add(rec["query"].lower())
        append(args.output, out)
        say(f"    grounded: {[s['ground_truth_tool']['api_name'] for s in out['canonical_plan'] if s['requires_api']]}")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(run, range(args.n)))
    if STOP.is_set():
        print("Daily request quota exhausted; stopping. Re-run tomorrow to resume.")
    report(args.output)


if __name__ == "__main__":
    main()
