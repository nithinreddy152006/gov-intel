---
trigger: always_on
---

# PRODUCT_CONTEXT.md

## Project Overview
**Gov-Intel** is a dual-mode Regulatory Intelligence Engine designed for the Ministry of Education (MoE). It provides an autonomous, multi-modal Retrieval-Augmented Generation (RAG) and Semantic Similarity pipeline to ingest, index, and reason over fragmented government regulations, gazettes, and visual schemes.

## Tech Stack
* **Frontend/App Layer:** Web Interface (Streamlit/React), Containerized via Docker / Google Cloud Run.
* **Ingestion & ETL:** Python, `Unstructured.io`, `LlamaParse`, Custom Web Crawlers.
* **Vision-Language Models (VLM):** `LLaVA-NeXT` (Local), `Gemini Vision` (Cloud).
* **Embedding Models:** `BAAI/bge-m3` (Dense semantic vectors), `Sparse Splade` (Sparse keyword vectors).
* **Vector Database:** `Qdrant` (Dockerized).
* **Retrieval & Reranking:** `BGE-Reranker-v2` (Cross-encoder).
* **Large Language Models (LLM):** `Llama-3-8B-Instruct` (Local via Ollama, 4-bit GGUF), `Gemini 1.5 Pro/GPT-4o` (Cloud fallback).

## Core Features
1.  **Multi-Modal RAG Chatbot:** Synthesizes direct answers to natural language queries with precise page-level citations, reasoning over both text and embedded images (e.g., flowcharts, scheme posters).
2.  **Semantic Similarity Search:** Evaluates uploaded draft policies against the entire regulatory knowledge base to detect conflicts, overlaps, and duplications (bypassing the LLM for raw document matching).
3.  **Automated Policy Scraping:** Cron-triggered ETL pipeline that autonomously fetches, parses, and indexes new notifications from MoE, UGC, and AICTE portals.

## User Stories
* **As a Government Official**, I want to upload a draft PDF regulation so that the system can automatically flag conflicts or high-percentage similarities with existing historical policies.
* **As a Stakeholder/Public User**, I want to query the chatbot in natural language about specific scheme criteria so that I receive an immediate, citation-backed answer without manually cross-referencing dozens of gazettes.
* **As a System Administrator**, I want the web scrapers to run periodically so that the vector database is updated with the latest circulars without manual data entry.
* **As an Auditor**, I want to view the exact source document and page number for every AI-generated claim so that I can verify the legal authenticity of the information.

## Architectural Decisions
* **Hybrid Local/Cloud Topology:** The core pipeline is designed to run completely air-gapped on local hardware (using quantized models and local Qdrant) to ensure absolute data sovereignty for sensitive drafts. Cloud APIs (Gemini, Cloud Run) serve as a configurable fallback/scaling tier.
* **Vision-First Ingestion:** Standard text extraction fails on government scheme posters and hierarchy diagrams. We route all extracted images through a VLM to generate dense semantic text descriptions *before* embedding, ensuring visual data is natively searchable.
* **Hybrid Search Implementation:** Standard dense embeddings often fail on exact legal terminology (e.g., "Section 12B"). We utilize Qdrant's hybrid search (Dense + Sparse) to capture both contextual semantic meaning and exact keyword matches.
* **Cross-Encoder Reranking:** To minimize LLM hallucination, initial retrieval results from the vector DB are strictly re-ordered by a dedicated reranking model (`BGE-Reranker-v2`) before being injected into the LLM prompt context window. 
* **Dual-Mode Execution (LLM Bypass):** "Mode A" (Similarity Search) intentionally bypasses the generative LLM to provide deterministically calculated match percentages between documents, saving compute resources and eliminating generative errors for pure document discovery tasks.