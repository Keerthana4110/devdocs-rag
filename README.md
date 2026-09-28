# Production RAG System with Multi-Strategy Retrieval Evaluation

A cited question-answering system over FastAPI's documentation and closed
GitHub issues, with an evaluation harness comparing retrieval strategies, a
small tool-using agent, and a deployed API.

Resume line: *Production RAG System with Multi-Strategy Retrieval Evaluation
— Built a cited Q&A system over open-source documentation and GitHub issues;
compared 3 retrieval strategies via an automated eval harness, then deployed
as a tool-using agent with guardrails via FastAPI/Docker.*

## Why this domain

FastAPI's docs describe current behavior; its closed issues describe real
bugs, version-specific quirks, and duplicate-question chains. That mismatch
is exactly what makes retrieval hard and an eval table meaningful — a clean
FAQ would not test the system at all.

## The 3-week plan

### Week 1: RAG core and eval set
- [ ] Day 1-2: fetch issues + docs, ingest, get cited answers working (`src/`)
- [ ] Day 3-4: write 30-40 eval questions with gold answers and source (doc page or issue #) (`eval/questions.jsonl`)
- [ ] Day 5-7: eval harness: retrieval recall@k, MRR, answer faithfulness

### Week 2: comparison and light agent
- [ ] Day 1-3: compare baseline vector vs hybrid (BM25 + vector) vs hybrid + reranker; record results table below
- [ ] Day 4-7: single agent with two tools (docs/issues retrieval, GitHub issue status lookup), retries, low-confidence guardrail, per-request token cap

### Week 3: ship and document
- [ ] Day 1-3: FastAPI endpoint, Dockerfile, deploy to free tier, live URL
- [ ] Day 4-5: log latency and cost per request
- [ ] Day 6-7: finish this README (architecture, eval table, cost, next steps); practice a 5-minute walkthrough

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add ANTHROPIC_API_KEY, GITHUB_TOKEN (optional but raises rate limit)
python src/fetch_issues.py tiangolo/fastapi --max 300
python src/fetch_docs.py
python src/ingest.py
python src/ask.py "Why do I get a 422 error on a POST endpoint?"
```

## Results table (fill in during week 2)

| Setup | Chunk size | recall@5 | MRR | Faithfulness | Latency (s) |
|-------|-----------|----------|-----|--------------|-------------|
| Vector only | 500 | | | | |
| Hybrid | 500 | | | | |
| Hybrid + rerank | 500 | | | | |

## Architecture (fill in during week 3)

TODO: diagram of fetch (docs + issues) -> parse -> chunk -> embed -> Chroma -> retrieve -> LLM -> cited answer.

## Notes and lessons

TODO: what surprised you, what failed, what you would do next.
