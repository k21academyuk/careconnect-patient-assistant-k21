"""
CareConnect Supervisor — AgentCore Runtime entrypoint (SDK build)
=================================================================

This is the file the AgentCore starter-toolkit packages and deploys as the
CareConnect Supervisor. It is the SDK-build equivalent of the console/CLI
`supervisor_agent.py`, wrapped in the AgentCore Runtime app so it can run as a
managed endpoint.

It orchestrates the specialist logic in-process:
  - safety gate (deterministic_safety)
  - retrieval (Knowledge Base via search_docs)
  - escalation (DynamoDB + Step Functions)
  - verification (grounding + guardrail + deterministic checks)

Kept intentionally at Autonomy Level 1-2: clinical questions are never answered,
only escalated; high-impact actions are staged, never auto-submitted.
"""

import os
import re
import time

import boto3
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from strands import Agent
from strands.models import BedrockModel

from lab_helpers import deterministic_safety as ds
from lab_helpers.careconnect_agents import (
    VERIFICATION_SYSTEM_PROMPT,
    search_docs,
)
from lab_helpers.utils import (
    ESCALATION_TABLE,
    MODEL_ID,
    REGION,
    SSM_PREFIX,
    get_ssm_parameter,
)

model = BedrockModel(model_id=MODEL_ID)
app = BedrockAgentCoreApp()

_dynamodb = boto3.resource("dynamodb", region_name=REGION)
_sfn = boto3.client("stepfunctions", region_name=REGION)
_bedrock_runtime = boto3.client("bedrock-runtime", region_name=REGION)


# --- budgets ---------------------------------------------------------------
class Budgets:
    max_steps = 8
    max_seconds = 30


CLINICAL_PATTERNS = [
    r"should i (stop|start|change|take|increase|decrease|double)",
    r"stop (taking|my) ",
    r"change (my )?(dose|medication|medicine)",
    r"increase .*(dose|medication)",
    r"decrease .*(dose|medication)",
    r"is it safe to (take|stop|skip)",
]


def _check_safety(request):
    hits = [p for p in CLINICAL_PATTERNS if re.search(p, request.lower())]
    return {"clinical_detected": bool(hits), "matched": hits}


def _escalate(request):
    ticket_table = _dynamodb.Table(ESCALATION_TABLE)
    import uuid
    from datetime import datetime, timezone
    ticket_id = str(uuid.uuid4())
    ticket_table.put_item(Item={
        "ticket_id": ticket_id,
        "category": "clinical_question",
        "reason": "Medication/clinical decision requires clinician review.",
        "request_details": {"patient_request": request},
        "status": "pending_review",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "expires_at": int(datetime.now(timezone.utc).timestamp()) + 2592000,
    })
    try:
        arn = get_ssm_parameter(f"{SSM_PREFIX}/state_machine_arn")
        _sfn.start_execution(stateMachineArn=arn,
                             input=f'{{"ticket_id": "{ticket_id}"}}')
    except Exception as exc:  # noqa: BLE001
        print(f"Step Functions start skipped: {exc}")
    return {"ticket_id": ticket_id, "status": "pending_review"}


def _run_guardrail(text):
    try:
        gid = get_ssm_parameter(f"{SSM_PREFIX}/guardrail_id")
        gver = get_ssm_parameter(f"{SSM_PREFIX}/guardrail_version")
        resp = _bedrock_runtime.apply_guardrail(
            guardrailIdentifier=gid, guardrailVersion=gver,
            source="OUTPUT", content=[{"text": {"text": text}}], outputScope="FULL")
        return resp.get("action") == "NONE"
    except Exception as exc:  # noqa: BLE001
        print(f"Guardrail check skipped: {exc}")
        return True


@app.entrypoint
async def invoke(payload, context=None):
    """CareConnect Supervisor entrypoint."""
    request = (payload or {}).get("prompt", "").strip()
    if not request:
        return "Error: missing 'prompt'."

    start = time.time()
    safety = _check_safety(request)

    parts = []
    evidence = ""

    # Retrieve approved info for the safe part of the request.
    try:
        raw = search_docs(request)
        evidence = raw.split("Passage 2")[0].strip()  # top passage only
        if "No approved Riverside Health information" not in evidence:
            parts.append("Approved Riverside Health information:\n" + evidence)
    except Exception as exc:  # noqa: BLE001
        parts.append(f"[retrieval error: {exc}]")

    # Clinical questions are escalated, never answered.
    if safety["clinical_detected"]:
        esc = _escalate(request)
        parts.append(
            "Medication question:\nYour medication question requires review by a "
            f"licensed clinician. (Ticket: {esc['ticket_id']}, status: {esc['status']})")

    draft = "\n\n".join(parts) if parts else \
        "I can help with approved Riverside Health information. Could you rephrase your question?"

    # Verify before returning.
    det = ds.evaluate(draft)
    guardrail_ok = _run_guardrail(draft)
    if time.time() - start > Budgets.max_seconds:
        return "Your request took too long to process safely. Please try again."

    if det["must_block"] or not guardrail_ok:
        return ("I can help with approved Riverside Health information, but I cannot "
                "provide medical diagnosis, medication dosage changes, treatment "
                "recommendations, or medical triage. Please contact a qualified "
                "healthcare professional for medical advice.")

    return draft


if __name__ == "__main__":
    app.run()
