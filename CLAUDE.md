# Project context for Claude Code

Goal: portfolio project for ML/AI engineer interviews. A RAG system over
FastAPI's documentation and closed GitHub issues, with an eval table
comparing retrieval strategies, a light tool-using agent, and a deployed
FastAPI service. Timeline: 3 weeks.

Resume title: "Production RAG System with Multi-Strategy Retrieval Evaluation"

## Conventions
- Python 3.11+, source in `src/`, eval code and data in `eval/`.
- Change one variable at a time when running experiments; log every result.
- Never index documents used to write eval answers without keeping the gold
  source IDs (doc page slug or issue number), to avoid leakage between eval
  set and index.
- Keep functions small and typed; add a test for anything in the eval harness.
- Commit small, one logical change per commit.
- GitHub API calls should send a User-Agent and use GITHUB_TOKEN if set, to
  avoid the 60 req/hr unauthenticated rate limit.

## Priorities if time runs short
1. Eval table (most important)
2. Deployment with live URL
3. Agent layer (drop first if behind)
