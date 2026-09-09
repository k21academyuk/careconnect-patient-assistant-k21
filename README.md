# CareConnect on Amazon Bedrock AgentCore — SDK / Notebook Edition

A notebook-driven build of **CareConnect**, a secure multi-agent patient-support assistant
for the fictional **Riverside Health** hospital network, built on **Amazon Bedrock
AgentCore** using the AWS SDK (boto3).

CareConnect answers routine patient questions (visiting hours, appointment prep, prescription
refills, insurance, billing) **only from approved hospital documents, with citations** —
while anything clinical (dosages, diagnoses, urgent symptoms) is **never answered
automatically** and is **escalated to a licensed clinician**. High-impact actions such as a
refill are **staged for human approval**, never auto-submitted.

This is the **second build** of CareConnect. The first used the AWS Console and CLI on an
Ubuntu EC2 instance; this edition rebuilds the same system from **SageMaker notebooks using
boto3**, so the whole thing is scripted rather than clicked.

> Companion repo (the Console / CLI build):
> [`careconnect-patient-assistant-k21-console`](https://github.com/k21academyuk/careconnect-patient-assistant-k21-console).

> **Scope & status.** Educational reference using **synthetic data only**. Not
> production-ready as written — see [Limitations](#limitations).

---

## Architecture

The complete CareConnect system — patient browser → API edge → the seven-agent AgentCore
runtime → knowledge/AI services → tools, escalation, and cross-cutting safety/observability.

![Full architecture](images/full-architecture.png)

**Request flow** (blue = request in, red = clinical question escalated to a human,
green = verified, guardrail-approved answer returning to the patient):

![Animated flow](images/architecture-flow-animated.gif)

---

## What you build

By the end you have, running end to end and callable from a browser:

- a **RAG knowledge base** over approved documents (Bedrock Knowledge Base + Amazon S3 Vectors),
- a **seven-agent workflow** (one Supervisor + six specialists) with the Strands Agents SDK,
- a **layered safety system** (deterministic rules + Bedrock Guardrails + an independent
  verification/grounding agent),
- a **human-in-the-loop escalation path** (Amazon DynamoDB + AWS Step Functions),
- a **managed deployment** on Amazon Bedrock AgentCore Runtime,
- an **API + web UI** (Amazon API Gateway + a Lambda proxy + a Streamlit frontend),
- an **observability + golden-scenario evaluation** step for release gating.

**The seven agents**

| Agent | Responsibility |
|---|---|
| Supervisor / Orchestrator | Plans the request, routes sub-tasks, enforces step/time budgets, verifies before replying |
| Safety | Deterministic gate for clinical/urgent/manipulation intent and PII |
| Retrieval | Searches approved documents and returns passages **with citations** |
| Document-Processing | Structures retrieved evidence without adding content |
| Task/Tool | Calls synthetic hospital tools via AgentCore Gateway; **stages** actions only |
| Verification | Independently checks grounding, citations, safety, and Guardrail before a reply ships |
| Escalation | Creates a durable ticket and starts the human-approval workflow |

**Autonomy level:** deliberately kept at **Level 1–2 (assistant / human-approved)**.

---

## Relationship to the Console / CLI build

This edition runs **alongside** the original build without interfering with it.

| Aspect | Console/CLI build | This SDK build |
|---|---|---|
| Run environment | Ubuntu EC2 | SageMaker notebooks |
| Resource creation | Console clicks + CLI | boto3 in notebooks |
| Naming | original names | every new resource suffixed **`-sdk`** |
| Documents S3 bucket | `careconnect-approved-docs` | **reused as-is** |
| AgentCore deploy | `agentcore` Node CLI | `bedrock-agentcore-starter-toolkit` `Runtime()` (pure Python) |
| Deterministic safety | AWS Lambda | in-process Python module |

**No collisions.** The only shared resource is the documents S3 bucket. Everything else is
new and `-sdk`-suffixed. Cleanup (lab-08) removes only the `-sdk` resources.

---

## Prerequisites

- **AWS account** with Amazon Bedrock access, in **us-east-1 (N. Virginia)**.
- **SageMaker Studio / notebook** environment in us-east-1.
- **Bedrock model access enabled** for **Amazon Nova 2 Lite** (`us.amazon.nova-2-lite-v1:0`)
  and **Titan Text Embeddings V2** (`amazon.titan-embed-text-v2:0`).
- The existing **`careconnect-approved-docs`** bucket with documents under `approved/`.
- **SageMaker execution role** able to act on: Bedrock, Bedrock AgentCore, S3, S3 Vectors,
  DynamoDB, Step Functions, Lambda, API Gateway, IAM (create role/policy), SSM, CloudWatch /
  X-Ray, and ECR.
- **Python 3.10+**.

---

## Setup

```bash
# from the project root, inside your SageMaker environment
pip install -r requirements.txt
```

Every notebook begins with two **bootstrap cells** that (1) locate the repo root so imports
work from any folder, and (2) check that `lab_helpers/` and `requirements.txt` are present.

```bash
# Confirm the layout — you MUST see lab_helpers/ next to the notebooks:
ls              # -> lab-00-...ipynb ... lab_helpers/  requirements.txt  README.md  images/
ls lab_helpers/ # -> __init__.py utils.py careconnect_agents.py deterministic_safety.py runtime_entrypoint.py frontend/
```

> If `ls lab_helpers/` shows nothing, the package was not uploaded — the notebooks cannot run
> until it is next to them. This is the single most common setup mistake.

Before running lab-00, open `lab_helpers/utils.py` and confirm:

```python
EXISTING_DOCS_BUCKET = "careconnect-approved-docs"   # your real bucket name
DOCS_PREFIX          = "approved/"                    # your documents prefix
```

---

## Notebook run order

Run the notebooks in sequence — each depends on IDs the previous one wrote to SSM.

| # | Notebook | Creates / does |
|---|---|---|
| 0 | `lab-00-prerequisites` | config, IAM roles, reuse bucket, Knowledge Base + S3 Vectors, Guardrail |
| 1 | `lab-01-create-agents` | Retrieval + Document-Processing agents |
| 2 | `lab-02-safety-and-verification` | deterministic safety module + Verification agent |
| 3 | `lab-03-gateway-tools` | mock hospital Lambda → AgentCore Gateway |
| 4 | `lab-04-escalation` | DynamoDB ticket table (TTL) + Step Functions workflow |
| 5 | `lab-05-supervisor-runtime` | Supervisor orchestration + deploy to AgentCore Runtime |
| 6 | `lab-06-api-and-frontend` | Lambda proxy + API Gateway (POST /careconnect) + Streamlit UI |
| 7 | `lab-07-observability-eval` | five golden scenarios + CloudWatch traces |
| 8 | `lab-08-cleanup` | delete **only** the `-sdk` resources |

---

## Step-by-step diagrams

Each diagram shows the full architecture in faded context with **that step's components
highlighted**, plus a zoomed-in detail panel. (The step numbers match the full lab guide.)

### Step 5 — Store Approved Documents in Amazon S3
![Step 5](images/step-05-s3-documents.png)

### Step 6 — Create the Knowledge Base with S3 Vectors
![Step 6](images/step-06-knowledge-base.png)

### Step 7 — Create the Patient Safety Guardrail
![Step 7](images/step-07-guardrail.png)

### Step 8 — Build & Test the Deterministic Safety Rules
![Step 8](images/step-08-deterministic-safety.png)

### Step 9 — Build the Retrieval Agent (Strands)
![Step 9](images/step-09-retrieval-agent.png)

### Step 10 — Build the Document-Processing Agent
![Step 10](images/step-10-document-processing.png)

### Step 11 — Build the Task/Tool Agent with AgentCore Gateway
![Step 11](images/step-11-task-tool-gateway.png)

### Step 12 — Build the Response Verification Agent
![Step 12](images/step-12-verification-agent.png)

### Step 13 — Build the Escalation Agent (DynamoDB + Step Functions)
![Step 13](images/step-13-escalation.png)

### Step 14 — Build the Supervisor Agent (Orchestration & Budgets)
![Step 14](images/step-14-supervisor.png)

### Step 15 — Deploy to Amazon Bedrock AgentCore Runtime
![Step 15](images/step-15-deploy-runtime.png)

### Step 16 — Create the API Gateway Endpoint
![Step 16](images/step-16-api-gateway.png)

### Step 17 — Build & Host the Frontend (S3 + CloudFront)
![Step 17](images/step-17-frontend-cloudfront.png)

### Step 18 — Evaluate, Monitor & Gate for Release
![Step 18](images/step-18-evaluate-monitor-gate.png)

### Step 19 — CareConnect, SDK / Notebook Edition
![Step 19](images/step-19-sdk-notebook-edition.png)

---

## Safety model

Defense in depth: (1) deterministic rules, (2) Bedrock Guardrails (denied topics + PII
masking + MRN regex), (3) retrieved-content sanitising, (4) independent verification
(grounding + citations + Guardrail), (5) human-in-the-loop (clinical questions escalate;
high-impact actions stage for approval), (6) Supervisor step/time budgets.

---

## Cost & cleanup

The application is inexpensive per run (pay-per-use; S3 Vectors avoids an always-on cluster).
Two things dominate real cost if forgotten: **idle SageMaker compute** (stop the space) and
**undeleted resources** (run **lab-08**). After deleting the Knowledge Base, verify in the
**S3 Vectors** console that the vector index/bucket is actually gone.

---

## Limitations

Educational reference, not production. Before handling real patients you would need:
Infrastructure-as-Code (CDK/Terraform), real authentication (Cognito/JWT or IAM),
least-privilege IAM (no `"Resource": "*"`), customer-managed KMS encryption, AWS WAF, a
masked/audited logging pipeline, CI/CD with the golden-scenario evaluation as a release gate,
real (authenticated) hospital-system integrations instead of the synthetic tools, and the
full compliance work (signed AWS BAA, HIPAA-eligibility review, a defined clinician review
process, and an independent security review). **No code can make the system HIPAA-compliant
by itself.**

---

## Credits

Structure and deployment patterns adapted from AWS's `amazon-bedrock-agentcore` customer-
support tutorial. CareConnect scenario, safety model, and agent design based on the
K21Academy *Production Ready — CareConnect Patient Assistant* build guide. All data is synthetic.
