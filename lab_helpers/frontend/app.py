"""
CareConnect — Streamlit frontend (SDK build)
============================================

Minimal patient chat UI that calls the CareConnect API Gateway endpoint.
Mirrors the sample's lab-05 Streamlit approach, adapted for CareConnect.

Run:
    export CARECONNECT_API_URL=$(aws ssm get-parameter \
        --name /app/careconnect/agentcore/api_url \
        --query Parameter.Value --output text)
    streamlit run lab_helpers/frontend/app.py --server.port 8501
"""

import json
import os
import urllib.request

import streamlit as st

API_URL = os.environ.get("CARECONNECT_API_URL", "")

st.set_page_config(page_title="CareConnect — Riverside Health", page_icon="🏥")
st.title("🏥 CareConnect — Riverside Health Patient Assistant")
st.caption(
    "Answers routine questions from approved documents. Clinical questions are "
    "escalated to a licensed clinician; high-impact actions require human approval.")

if not API_URL:
    st.error("CARECONNECT_API_URL is not set. Export it before launching (see app.py docstring).")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Ask about visiting hours, appointment prep, refills, insurance…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("CareConnect is checking approved information…"):
            try:
                req = urllib.request.Request(
                    API_URL,
                    data=json.dumps({"prompt": prompt}).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                body = urllib.request.urlopen(req, timeout=300).read().decode()
                answer = json.loads(body).get("answer", body)
            except Exception as exc:  # noqa: BLE001
                answer = f"Sorry — the assistant could not be reached: {exc}"
            st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
