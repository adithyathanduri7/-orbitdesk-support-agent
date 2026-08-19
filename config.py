
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_REVISION = "e4ce9877abf3edfe10b0d82785e83bdcb973e22e"

LLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LLM_REVISION = "fe8a4ea1ffedaf415f4da2f062534de366a451e6"

KB_DIR = "knowledge_base"
CASES_FILE = r"..\AI Engineer Internship - Assignment material\resolved_cases.json"

TOP_K_CHUNKS = 5
SIMILARITY_THRESHOLD = 0.35

MIN_CONFIDENCE_THRESHOLD = 0.4
MAX_RETRIES = 1

OUT_OF_SCOPE_KEYWORDS = [
    "refund",
    "legal advice",
    "cancel subscription",
    "billing",
    "payment",
    "medical",
    "ignore documentation",
    "ignore the supplied",
    "act as",
    "jailbreak",
    "override",
    "forget instructions"
]

CLARIFICATION_KEYWORDS = [
    "not working",
    "broken",
    "issue",
    "problem",
    "error",
    "doesn't work",
    "sync",
    "help me"
]

ESCALATION_KEYWORDS = [
    "render_failed",
    "consecutive",
    "twice",
    "escalate",
    "two runs",
    "failed twice",
    "repeated failure"
]