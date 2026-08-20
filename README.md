# OrbitDesk Support Agent

A local Retrieval-Augmented Generation (RAG) based AI support agent built with **Python, LangGraph, TinyLlama, Sentence Transformers, FAISS, and PyTorch**.

The system is designed to answer product-support questions using relevant knowledge-base documents and resolved support cases while providing controlled routing, evidence retrieval, response verification, retry handling, clarification, escalation, and safe-failure paths.

---

## Problem Statement

Customer-support systems need to provide accurate answers while avoiding unsupported or hallucinated information.

A simple LLM chatbot may generate an answer even when it does not have enough information. OrbitDesk addresses this problem by combining:

- Local LLM response generation
- Knowledge-base retrieval
- Evidence-based responses
- Deterministic request routing
- Response verification
- Retry handling
- Human escalation
- Safe failure for unsupported requests

The goal is to make the support workflow more **reliable, traceable, and controllable**.

---

## Key Features

- Local LLM inference using **TinyLlama**
- Retrieval-Augmented Generation (RAG)
- Semantic search using **Sentence Transformers**
- FAISS-based vector retrieval
- LangGraph state-machine workflow
- Request triage
- Knowledge-base retrieval
- Evidence/source tracking
- Response generation
- Response verification
- Automatic retry
- Clarification handling
- Human escalation
- Out-of-scope handling
- Safe-failure mechanism
- Automated testing

---

## System Architecture

```text
                         USER QUERY
                              |
                              v
                       +--------------+
                       |    TRIAGE    |
                       +------+-------+
                              |
              +---------------+---------------+
              |               |               |
              v               v               v
        ANSWERABLE       CLARIFICATION    ESCALATION /
              |           / OUT-OF-SCOPE   SPECIAL CASE
              v               |               |
        +-----------+         |               |
        | RETRIEVAL |         |               |
        +-----+-----+         |               |
              |               |               |
              +-------+-------+---------------+
                      |
                      v
                +-----------+
                | GENERATION|
                +-----+-----+
                      |
                      v
                +-----------+
                |VERIFICATION|
                +-----+-----+
                      |
             +--------+--------+
             |                 |
          PASSED             FAILED
             |                 |
             v                 v
          OUTPUT          RETRY GENERATION
                               |
                               v
                         VERIFICATION
                               |
                         +-----+-----+
                         |           |
                      PASSED      FAILED
                         |           |
                         v           v
                      OUTPUT    SAFE FAILURE
