"""
CareConnect deterministic safety rules (SDK / notebook edition)
===============================================================

In the console/CLI build these rules lived in a Lambda function
(careconnect-deterministic-safety). In the SDK build we keep the identical
logic as a plain importable Python module, so the Verification agent and the
Supervisor can call it directly in-process — no Lambda round-trip needed.

The behaviour matches the original: detect clinical/dosage intent, urgent
symptoms, manipulation (prompt-injection) attempts, and PII; mask PII; and
sanitise instruction-like lines out of retrieved documents.
"""

import re

CLINICAL_PATTERNS = [
    r"\bdiagnose\b", r"\bdiagnosis\b",
    r"\bwhat disease do i have\b", r"\bwhat condition do i have\b",
    r"\bshould i take\b", r"\bshould i increase\b", r"\bshould i decrease\b",
    r"\bdouble (my|the) dose\b", r"\bchange (my|the) dose\b",
    r"\bhow much .* should i take\b",
]

URGENT_PATTERNS = [
    r"\bchest pain\b", r"\bsevere difficulty breathing\b",
    r"\bcan't breathe\b", r"\bcannot breathe\b",
    r"\bunconscious\b", r"\bsevere bleeding\b", r"\bpassed out\b",
]

MANIPULATION_PATTERNS = [
    r"\bignore previous instructions\b",
    r"\bignore all previous instructions\b",
    r"\boverride (the )?(rules|policy|policies|instructions)\b",
    r"\bignore (the )?(rules|policy|policies)\b",
    r"\breveal (the )?system prompt\b",
    r"\bshow (me )?(the )?system prompt\b",
    r"\bdisregard previous instructions\b",
]

PII_PATTERNS = {
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "mrn": r"\bMRN-\d{8}\b",
    "phone": r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b",
}

INSTRUCTION_LINE_PATTERNS = [
    r"^\s*ignore previous instructions",
    r"^\s*ignore all previous instructions",
    r"^\s*override .*instructions",
    r"^\s*disregard previous instructions",
    r"^\s*system\s*:", r"^\s*developer\s*:", r"^\s*assistant\s*:",
]


def matches(text, patterns):
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def find_pii(text):
    findings = {}
    for pii_type, pattern in PII_PATTERNS.items():
        values = re.findall(pattern, text, re.IGNORECASE)
        if values:
            findings[pii_type] = len(values)
    return findings


def mask_pii(text):
    masked = text
    replacements = {"email": "[EMAIL]", "ssn": "[SSN]",
                    "mrn": "[MRN]", "phone": "[PHONE]"}
    for pii_type, pattern in PII_PATTERNS.items():
        masked = re.sub(pattern, replacements[pii_type], masked, flags=re.IGNORECASE)
    return masked


def sanitize_retrieved(text):
    safe = []
    for line in text.splitlines():
        suspicious = any(re.search(p, line, re.IGNORECASE)
                         for p in INSTRUCTION_LINE_PATTERNS)
        if not suspicious:
            safe.append(line)
    return "\n".join(safe)


def classify(text):
    return {
        "clinical": matches(text, CLINICAL_PATTERNS),
        "urgent": matches(text, URGENT_PATTERNS),
        "injection": matches(text, MANIPULATION_PATTERNS),
        "pii": find_pii(text),
    }


def evaluate(text):
    """Convenience wrapper returning classification + routing flags,
    equivalent to the old Lambda handler's response shape."""
    c = classify(text)
    must_block = c["injection"]
    must_escalate = c["clinical"] or c["urgent"]
    return {
        "classification": c,
        "must_block": must_block,
        "must_escalate": must_escalate,
        "masked_text": mask_pii(text),
        "sanitized_text": sanitize_retrieved(text),
    }
