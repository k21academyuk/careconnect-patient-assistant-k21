# CareConnect on Amazon Bedrock AgentCore — SDK / Notebook Edition

A notebook-driven build of **CareConnect**, a secure multi-agent patient-support
assistant for the fictional **Riverside Health** hospital network, built on Amazon
Bedrock AgentCore using the AWS SDK (boto3).

This is the **second build** of CareConnect. The first was done with the AWS Console and
CLI on an Ubuntu EC2 instance; this edition rebuilds the same system from **SageMaker
notebooks using boto3**, so the whole thing is scripted rather than clicked. It follows the
structure of AWS's official `amazon-bedrock-agentcore` customer-support tutorial (labs +
a shared `lab_helpers/` package).

> **Scope and status.** Educational reference that demonstrates a realistic, safety-first
> agentic architecture. It uses **synthetic data only** and is **not production-ready** as
> written. See [Limitations & production checklist](#limitations--production-checklist).

---

## Table of contents

1. [What you build](#what-you-build)
2. [Architecture](#architecture)
3. [Relationship to the console/CLI build](#relationship-to-the-consolecli-build)
4. [Repository layout](#repository-layout)
5. [Prerequisites](#prerequisites)
6. [Setup](#setup)
7. [Notebook run order](#notebook-run-order)
8. [Configuration](#configuration)
9. [Helper modules](#helper-modules)
10. [Safety model](#safety-model)
11. [Cost](#cost)
12. [Cleanup](#cleanup)
13. [Troubleshooting](#troubleshooting)
14. [Limitations & production checklist](#limitations--production-checklist)
15. [Credits](#credits)

---

## What you build

CareConnect answers routine patient questions (visiting hours, appointment prep,
prescription refills, insurance, billing) from **approved hospital documents**, with
citations — while ensuring that anything clinical (dosages, diagnoses, urgent symptoms) is
**never answered automatically** and is escalated to a licensed clinician. High-impact
actions such as a prescription refill are **staged for human approval**, never
auto-submitted.

By the end you have, running end to end and callable from a browser:

- a **RAG knowledge base** over approved documents (Amazon Bedrock Knowledge Base backed by
  Amazon S3 Vectors),
- a **seven-agent workflow** (one Supervisor + six specialists) built with the Strands
  Agents SDK,
- a **layered safety system** (deterministic rules + Amazon Bedrock Guardrails + an
  independent verification/grounding agent),
- a **human-in-the-loop escalation path** (Amazon DynamoDB + AWS Step Functions),
- a **managed deployment** on Amazon Bedrock AgentCore Runtime,
- an **API + web UI** (Amazon API Gateway + a Lambda proxy + a Streamlit frontend),
- an **observability + golden-scenario evaluation** step for release gating.

---

## Architecture

```
                       Browser (Streamlit UI)
                                |  HTTPS
                       Amazon API Gateway  --->  Lambda proxy
                                |
                Amazon Bedrock AgentCore Runtime
                                |
                        Supervisor Agent
        +---------------+-------+--------+---------------+
   Safety gate      Retrieval        Task/Tool        Escalation
 (deterministic   (Bedrock KB +     (AgentCore       (DynamoDB +
   + Guardrail)     S3 Vectors)      Gateway ->        Step Functions)
        |               |            mock Lambda)          |
        +-----------> Document-Processing --> Verification +
                                                (grounding + Guardrail
                                                 + deterministic checks)
```

**The seven agents**

| Agent | Responsibility |
|---|---|
| Supervisor / Orchestrator | Plans the request, routes sub-tasks, enforces step/time budgets, verifies before replying |
| Safety | Deterministic gate for clinical/urgent/manipulation intent and PII |
| Retrieval | Searches the approved documents and returns passages **with citations** |
| Document-Processing | Structures retrieved evidence (e.g. into checklists) without adding content |
| Task/Tool | Calls synthetic hospital tools via AgentCore Gateway; **stages** actions only |
| Verification | Independently checks grounding, citations, safety, and Guardrail before a reply ships |
| Escalation | Creates a durable ticket and starts the human-approval workflow |

**Autonomy level:** deliberately kept at **Level 1-2 (assistant / human-approved)**. The
system informs and prepares; it does not independently execute clinical or high-impact
actions.

---

## Relationship to the console/CLI build

This edition is designed to run **alongside** your original build without interfering with
it.

| Aspect | Console/CLI build | This SDK build |
|---|---|---|
| Run environment | Ubuntu EC2 | SageMaker notebooks |
| Resource creation | Console clicks + CLI | boto3 in notebooks |
| Naming | original names | every new resource suffixed **`-sdk`** |
| Documents S3 bucket | `careconnect-approved-docs` | **reused as-is** (Option B) |
| AgentCore deploy | `agentcore` Node CLI | `bedrock-agentcore-starter-toolkit` `Runtime()` (pure Python) |
| Deterministic safety | AWS Lambda | in-process Python module |

**No collisions.** The **only** shared resource is the documents S3 bucket, reused on
purpose so both builds use the same approved documents. Everything else is new and
`-sdk`-suffixed. Cleanup (lab-08) removes only the `-sdk` resources and leaves the bucket
and your original build untouched.

---

## Repository layout

```
careconnect-agentcore-sdk/
|-- README.md
|-- requirements.txt
|-- lab-00-prerequisites.ipynb        # config, IAM, KB + S3 Vectors, Guardrail
|-- lab-01-create-agents.ipynb        # Retrieval + Document-Processing agents
|-- lab-02-safety-and-verification.ipynb
|-- lab-03-gateway-tools.ipynb        # mock hospital tools via AgentCore Gateway
|-- lab-04-escalation.ipynb           # DynamoDB + Step Functions
|-- lab-05-supervisor-runtime.ipynb   # Supervisor + deploy to AgentCore Runtime
|-- lab-06-api-and-frontend.ipynb     # API Gateway + Lambda proxy + Streamlit UI
|-- lab-07-observability-eval.ipynb   # golden scenarios + CloudWatch traces
|-- lab-08-cleanup.ipynb              # delete ONLY the -sdk resources
`-- lab_helpers/
    |-- utils.py                      # config, naming, SSM, IAM, KB polling
    |-- careconnect_agents.py         # system prompts + retrieval tool + agent builders
    |-- deterministic_safety.py       # safety rules as an importable module
    |-- runtime_entrypoint.py         # the deployable Supervisor (AgentCore Runtime app)
    `-- frontend/
        |-- app.py                    # Streamlit patient chat UI
        `-- requirements.txt
```

---

## Prerequisites

- **AWS account** with Amazon Bedrock access, in **us-east-1 (N. Virginia)** — the same
  Region as your original build.
- **SageMaker Studio / notebook** environment running in us-east-1.
- **Bedrock model access enabled** for **Amazon Nova 2 Lite** (`us.amazon.nova-2-lite-v1:0`)
  and **Titan Text Embeddings V2** (`amazon.titan-embed-text-v2:0`).
- The existing **`careconnect-approved-docs`** bucket with the approved documents present
  under the `approved/` prefix.
- The **SageMaker execution role** must be able to act on: Bedrock, Bedrock AgentCore, S3,
  S3 Vectors, DynamoDB, Step Functions, Lambda, API Gateway, IAM (create role/policy),
  SSM Parameter Store, CloudWatch / X-Ray, and ECR (for the runtime container build).
- **Python 3.10+**.

---

## Setup

```bash
# from the project root, inside your SageMaker environment
pip install -r requirements.txt
```

Open **`lab-00-prerequisites.ipynb`** and run it top to bottom, then proceed through the
labs in order. Restart the kernel after the first `pip install` if prompted.

Before running lab-00, open **`lab_helpers/utils.py`** and confirm two values match your
account:

```python
EXISTING_DOCS_BUCKET = "careconnect-approved-docs"   # your real bucket name
DOCS_PREFIX          = "approved/"                    # your documents prefix
```

---

## Notebook run order

Run the notebooks in sequence — each depends on IDs the previous one wrote to SSM.

| # | Notebook | Creates / does |
|---|---|---|
| 0 | `lab-00-prerequisites` | config, IAM roles, reuse bucket, Knowledge Base + S3 Vectors, Guardrail (versioned) |
| 1 | `lab-01-create-agents` | Retrieval + Document-Processing agents |
| 2 | `lab-02-safety-and-verification` | deterministic safety module + Verification agent (grounding + Guardrail) |
| 3 | `lab-03-gateway-tools` | mock hospital Lambda -> AgentCore Gateway (three synthetic tools) |
| 4 | `lab-04-escalation` | DynamoDB ticket table (TTL) + Step Functions approval workflow |
| 5 | `lab-05-supervisor-runtime` | Supervisor orchestration + deploy to AgentCore Runtime |
| 6 | `lab-06-api-and-frontend` | Lambda proxy + API Gateway (POST /careconnect) + Streamlit UI |
| 7 | `lab-07-observability-eval` | five golden scenarios as sessions + CloudWatch traces |
| 8 | `lab-08-cleanup` | delete **only** the `-sdk` resources |

---

## Configuration

All configuration lives in **one place**, `lab_helpers/utils.py`:

- `RESOURCE_SUFFIX` — the `-sdk` suffix appended to every new resource.
- `EXISTING_DOCS_BUCKET` / `DOCS_PREFIX` — the reused documents bucket.
- `MODEL_ID` / `EMBED_MODEL_ID` — the Bedrock models.
- `SSM_PREFIX` — the Parameter Store namespace (`/app/careconnect/agentcore`).
- Canonical resource names (`KB_NAME`, `GUARDRAIL_NAME`, `ESCALATION_TABLE`, ...).

IDs generated at runtime (Knowledge Base id, Guardrail id/version, Gateway id/URL, Runtime
ARN, API URL) are written to **SSM Parameter Store** under `SSM_PREFIX`, so later notebooks
read them rather than relying on copy-paste. This is what keeps the labs consistent and
free of hard-coded identifiers.

---

## Helper modules

| Module | Purpose |
|---|---|
| `lab_helpers/utils.py` | Config, resource naming, SSM get/put/delete, IAM role creation, KB readiness polling |
| `lab_helpers/careconnect_agents.py` | System prompts, the `search_docs` retrieval tool, and builders for the Retrieval / Document-Processing / Verification agents |
| `lab_helpers/deterministic_safety.py` | Clinical/urgent/manipulation detection, PII masking, retrieved-document sanitising — as an importable module (was a Lambda in the console build) |
| `lab_helpers/runtime_entrypoint.py` | The deployable Supervisor wrapped as an AgentCore Runtime app |
| `lab_helpers/frontend/app.py` | Streamlit patient chat UI that calls the API endpoint |

---

## Safety model

CareConnect uses **defense in depth** rather than trusting any single control:

1. **Deterministic rules** (predictable regex-based detection of clinical/dosage intent,
   urgent symptoms, prompt-injection phrases, and PII).
2. **Amazon Bedrock Guardrails** (denied topics — Diagnosis, Dosage, Treatment, Triage —
   plus PII masking and a custom MRN regex).
3. **Retrieved-content sanitising** (instruction-like lines stripped from documents to
   blunt indirect prompt injection).
4. **Independent verification** (a separate agent confirms the draft is grounded in the
   retrieved evidence, correctly cited, and free of unsupported clinical/dosing claims).
5. **Human-in-the-loop** (clinical questions escalate; high-impact actions stage for
   approval and are never auto-submitted).
6. **Budgets** (the Supervisor caps steps and execution time to prevent loops / runaway
   cost).

**Design rule:** the assistant answers documented, logistical questions and escalates
anything clinical to a licensed human.

---

## Cost

The application itself is inexpensive per run — most services are pay-per-use and the
architecture deliberately avoids always-on infrastructure (which is why it uses **S3
Vectors** rather than a provisioned search cluster). Typical experimentation is a few
dollars.

Two things dominate real cost if you forget them:

- **Idle compute** — the SageMaker Studio space/app running notebooks. Stop it when done.
- **Undeleted resources** — run **lab-08** to remove the `-sdk` resources.

Keep an **AWS Budget** and **Cost Anomaly Detection** configured, and verify current
per-service pricing for your Region before a large test run.

---

## Cleanup

Run **`lab-08-cleanup.ipynb`** top to bottom. It deletes **only** the `-sdk` resources:
AgentCore Runtime + ECR image, API Gateway + proxy Lambda, Gateway + target + mock Lambda,
Knowledge Base + data source, DynamoDB table, Step Functions state machine, Guardrail, the
`-sdk` IAM roles, and the SSM parameters.

**Not deleted on purpose:**

- the shared **`careconnect-approved-docs`** bucket (used by both builds),
- anything **without** the `-sdk` suffix (your original build).

**Manual confirmation:** after deleting the Knowledge Base, verify in the **S3 Vectors**
console that the vector index/bucket is actually gone — a vector store can outlive the KB
that used it. Finally, **stop or delete your SageMaker space** to end notebook compute cost.

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `create_knowledge_base` rejects `storageConfiguration` | S3-Vectors KB request schema varies by `botocore` version. Print the exception and adjust to your installed schema. |
| `create_gateway` / runtime calls error on arguments | AgentCore control-plane APIs evolve; align the call with your installed `bedrock-agentcore` / `bedrock-agentcore-control` version. |
| Retrieval returns nothing | The ingestion job hadn't finished, or the KB id in SSM is stale. Re-run the sync cell and confirm `COMPLETE`. |
| Guardrail not blocking as expected | Confirm you published a numbered **version** (not "Working draft") and that `guardrail_version` in SSM matches it. |
| Evaluation shows 0 sessions | Traces take a few minutes to land; ensure **CloudWatch Transaction Search** is enabled in us-east-1 before evaluating. |
| Runtime deploy fails building the container | Ensure the SageMaker role can use ECR and that Bedrock model access is enabled. |
| IAM "not authorized" during a create step | The SageMaker execution role is missing a permission — see [Prerequisites](#prerequisites). |

---

## Limitations & production checklist

This project is a **learning reference**, not a production system. Before anything like it
could handle real patients, the following would be required — most of it is
**organizational/process work that cannot be scripted in a notebook**:

- **Infrastructure as Code.** Replace imperative notebook resource creation with **CDK or
  Terraform**, versioned in Git and deployed by pipeline. Notebooks are for exploration,
  not production deployment.
- **Authentication.** The dev Gateway uses `NONE` inbound auth and the API uses
  `authorizationType=NONE`. Production must use **Cognito/JWT or IAM** auth end to end.
- **Least-privilege IAM.** Remove every `"Resource": "*"` and wildcard action; scope each
  role to specific ARNs.
- **Encryption & network.** Customer-managed **KMS** keys on S3/DynamoDB/logs; enforced
  TLS; private networking where appropriate.
- **Edge protection.** Attach **AWS WAF** managed rule groups and rate-based rules to the
  API; do not treat the browser as a security boundary.
- **Observability.** Structured logging with **PII masking**, correlation IDs, log
  retention, dashboards, and alarms; a separate restricted audit trail.
- **CI/CD with a release gate.** A pipeline that runs unit tests **and** the golden-scenario
  evaluation and **blocks deploy on regression**.
- **Resilience.** Retries, idempotency, timeouts, and concurrency limits in agent and tool
  code.
- **Real integrations.** The hospital tools here are **synthetic**; production would
  integrate real systems behind the Gateway with their own auth and audit.
- **Compliance (human/legal).** A signed **AWS BAA**, verification that **every service in
  the path is HIPAA-eligible**, data-residency/retention/breach-notification policies, a
  defined **clinician review process** behind the escalation workflow, and an independent
  **security review / penetration test**. No code can make the system HIPAA-compliant by
  itself.

**API-shape caveat.** The exact request bodies for S3-Vectors Knowledge Bases and for
AgentCore Gateway/Runtime change between SDK versions. `requirements.txt` pins the versions
used by the reference AWS sample to reduce drift; cells that create those resources are
annotated where adjustment is most likely.

---

## Credits

Structure and deployment patterns adapted from AWS's `amazon-bedrock-agentcore`
customer-support tutorial. CareConnect scenario, safety model, and agent design based on the
K21Academy *Production Ready — CareConnect Patient Assistant* build guide. All data used is
synthetic.
