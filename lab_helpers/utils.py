"""
CareConnect on AgentCore — shared utilities (SDK / notebook edition)
====================================================================

This module centralises configuration, SSM parameter helpers, and IAM role
creation for the CareConnect notebooks. It mirrors the structure of the AWS
`amazon-bedrock-agentcore` customer-support sample, adapted for the Riverside
Health CareConnect scenario.

KEY DESIGN RULE — no collisions with your existing console/CLI build:
  Every resource this SDK build creates is suffixed with RESOURCE_SUFFIX
  ("-sdk" by default). The ONE exception is the S3 documents bucket, which is
  reused on purpose (Option B) so both builds share the same approved
  Riverside Health documents.

Everything reads REGION from the notebook's boto3 session, so run these
notebooks in the same Region as your existing build (us-east-1 / N. Virginia).
"""

import json
import time
from typing import Optional

import boto3
from boto3.session import Session

# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------

# Suffix appended to every NEW resource so the SDK build never overwrites the
# console/CLI build. Change it if you want a second parallel copy.
RESOURCE_SUFFIX = "-sdk"

# Reused from the existing build (Option B). Documents are identical, so we
# point the new Knowledge Base at the same bucket + prefix.
EXISTING_DOCS_BUCKET = "careconnect-approved-docs"   # <-- your real bucket name
DOCS_PREFIX = "approved/"

# Models (match the sample: Nova 2 Lite for reasoning, Titan v2 for embeddings)
MODEL_ID = "us.amazon.nova-2-lite-v1:0"
EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"

# SSM namespace for everything this build stores (KB id, guardrail id, etc.)
SSM_PREFIX = "/app/careconnect/agentcore"

sts_client = boto3.client("sts")
REGION = boto3.session.Session().region_name or "us-east-1"


def get_aws_region() -> str:
    return Session().region_name or "us-east-1"


def get_aws_account_id() -> str:
    return boto3.client("sts").get_caller_identity()["Account"]


# ---------------------------------------------------------------------------
# Naming — single source of truth so notebooks never disagree on a name
# ---------------------------------------------------------------------------

def name(base: str) -> str:
    """Return a suffixed resource name, e.g. name('careconnect-kb') -> careconnect-kb-sdk."""
    return f"{base}{RESOURCE_SUFFIX}"


# Canonical resource names used across notebooks
KB_NAME = name("careconnect-kb")
GUARDRAIL_NAME = name("careconnect-patient-safety-guardrail")
ESCALATION_TABLE = name("careconnect-escalations")
STATE_MACHINE_NAME = name("careconnect-human-approval-workflow")
GATEWAY_NAME = name("careconnect-tools-gateway")
MOCK_TOOLS_LAMBDA = name("careconnect-mock-hospital-tools")
PROXY_LAMBDA = name("careconnect-agentcore-proxy")
RUNTIME_AGENT_NAME = "careconnect_supervisor_sdk"   # runtime names: no dashes
FRONTEND_BUCKET = name("careconnect-frontend-prod")


# ---------------------------------------------------------------------------
# SSM parameter helpers (same interface as the AWS sample)
# ---------------------------------------------------------------------------

def get_ssm_parameter(name: str, with_decryption: bool = True) -> str:
    ssm = boto3.client("ssm")
    response = ssm.get_parameter(Name=name, WithDecryption=with_decryption)
    return response["Parameter"]["Value"]


def put_ssm_parameter(
    name: str, value: str, parameter_type: str = "String", with_encryption: bool = False
) -> None:
    ssm = boto3.client("ssm")
    params = {"Name": name, "Value": value, "Type": parameter_type, "Overwrite": True}
    if with_encryption:
        params["Type"] = "SecureString"
    ssm.put_parameter(**params)


def delete_ssm_parameter(name: str) -> None:
    ssm = boto3.client("ssm")
    try:
        ssm.delete_parameter(Name=name)
    except ssm.exceptions.ParameterNotFound:
        pass


# ---------------------------------------------------------------------------
# IAM role helpers
# ---------------------------------------------------------------------------

def _create_role(role_name: str, assume_service: str, inline_policy: dict,
                 policy_name: str) -> str:
    """Create (or reuse) an IAM role with a single inline least-privilege policy.
    Returns the role ARN."""
    iam = boto3.client("iam")
    trust = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": assume_service},
            "Action": "sts:AssumeRole",
        }],
    }
    try:
        resp = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust),
            Description=f"CareConnect SDK build role: {role_name}",
        )
        role_arn = resp["Role"]["Arn"]
        print(f"Created role {role_name}")
    except iam.exceptions.EntityAlreadyExistsException:
        role_arn = iam.get_role(RoleName=role_name)["Role"]["Arn"]
        print(f"Reusing existing role {role_name}")

    iam.put_role_policy(
        RoleName=role_name,
        PolicyName=policy_name,
        PolicyDocument=json.dumps(inline_policy),
    )
    # IAM is eventually consistent; give it a moment before the role is used.
    time.sleep(8)
    return role_arn


def create_kb_execution_role() -> str:
    """Role Bedrock Knowledge Base assumes to read S3 + call the embed model."""
    account = get_aws_account_id()
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {"Effect": "Allow", "Action": ["s3:GetObject", "s3:ListBucket"],
             "Resource": [
                 f"arn:aws:s3:::{EXISTING_DOCS_BUCKET}",
                 f"arn:aws:s3:::{EXISTING_DOCS_BUCKET}/*",
             ]},
            {"Effect": "Allow", "Action": ["bedrock:InvokeModel"],
             "Resource": f"arn:aws:bedrock:{REGION}::foundation-model/{EMBED_MODEL_ID}"},
            {"Effect": "Allow",
             "Action": ["s3vectors:*"],
             "Resource": "*"},
        ],
    }
    return _create_role(
        name("CareConnectKBRole"), "bedrock.amazonaws.com", policy,
        name("CareConnectKBPolicy"),
    )


def create_agentcore_runtime_execution_role() -> str:
    """Role the deployed Supervisor runs as. Least-privilege for CareConnect."""
    account = get_aws_account_id()
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {"Effect": "Allow", "Action": [
                "bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream",
                "bedrock:Retrieve", "bedrock:ApplyGuardrail",
            ], "Resource": "*"},
            {"Effect": "Allow", "Action": [
                "dynamodb:PutItem", "dynamodb:GetItem",
            ], "Resource": f"arn:aws:dynamodb:{REGION}:{account}:table/{ESCALATION_TABLE}"},
            {"Effect": "Allow", "Action": ["states:StartExecution"],
             "Resource": f"arn:aws:states:{REGION}:{account}:stateMachine:{STATE_MACHINE_NAME}"},
            {"Effect": "Allow", "Action": [
                "bedrock-agentcore:*",
            ], "Resource": "*"},
            {"Effect": "Allow", "Action": [
                "logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents",
                "ecr:GetAuthorizationToken", "ecr:BatchGetImage",
                "ecr:GetDownloadUrlForLayer",
                "xray:PutTraceSegments", "xray:PutTelemetryRecords",
            ], "Resource": "*"},
            {"Effect": "Allow", "Action": ["ssm:GetParameter"],
             "Resource": f"arn:aws:ssm:{REGION}:{account}:parameter{SSM_PREFIX}/*"},
        ],
    }
    return _create_role(
        name("CareConnectRuntimeRole"),
        "bedrock-agentcore.amazonaws.com", policy,
        name("CareConnectRuntimePolicy"),
    )


def wait_for_kb_ready(kb_id: str, timeout: int = 600) -> None:
    """Poll a Knowledge Base until ACTIVE."""
    client = boto3.client("bedrock-agent")
    start = time.time()
    while time.time() - start < timeout:
        status = client.get_knowledge_base(knowledgeBaseId=kb_id)[
            "knowledgeBase"]["status"]
        print(f"KB status: {status}")
        if status == "ACTIVE":
            return
        if status in ("FAILED", "DELETE_UNSUCCESSFUL"):
            raise RuntimeError(f"Knowledge Base entered {status}")
        time.sleep(15)
    raise TimeoutError("Knowledge Base did not become ACTIVE in time")
