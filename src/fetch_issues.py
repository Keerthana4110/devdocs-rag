"""Fetch closed GitHub issues (question + accepted resolution) for a repo.

Usage: python src/fetch_issues.py tiangolo/fastapi --max 300

Saves one JSON file per issue into data/raw_issues/, containing the issue
title, body, and top comments — this is the "messy real-world" half of the
knowledge base (docs describe current behavior; issues describe real bugs,
version quirks, and duplicate-question chains).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()
API_ROOT = "https://api.github.com"
OUT_DIR = Path("data/raw_issues")


def headers() -> dict:
    h = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "devdocs-rag-fetcher",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def check_rate_limit() -> None:
    r = requests.get(f"{API_ROOT}/rate_limit", headers=headers(), timeout=15)
    r.raise_for_status()
    core = r.json()["resources"]["core"]
    print(f"Rate limit: {core['remaining']}/{core['limit']} remaining")


def fetch_closed_issues(repo: str, max_count: int) -> list[dict]:
    issues = []
    page = 1
    per_page = min(100, max_count)
    while len(issues) < max_count:
        resp = requests.get(
            f"{API_ROOT}/repos/{repo}/issues",
            headers=headers(),
            params={
                "state": "closed",
                "per_page": per_page,
                "page": page,
                "sort": "comments",
                "direction": "desc",
            },
            timeout=30,
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        # Exclude pull requests -- the issues endpoint includes PRs too
        batch = [b for b in batch if "pull_request" not in b]
        issues.extend(batch)
        page += 1
        if len(batch) < per_page:
            break
        time.sleep(0.2)  # be polite to the API
    return issues[:max_count]


def fetch_comments(repo: str, issue_number: int, max_comments: int = 5) -> list[str]:
    resp = requests.get(
        f"{API_ROOT}/repos/{repo}/issues/{issue_number}/comments",
        headers=headers(),
        params={"per_page": max_comments},
        timeout=30,
    )
    if resp.status_code != 200:
        return []
    return [c["body"] for c in resp.json() if c.get("body")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo", help="owner/repo, e.g. tiangolo/fastapi")
    parser.add_argument("--max", type=int, default=300)
    args = parser.parse_args()

    check_rate_limit()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Fetching up to {args.max} closed issues from {args.repo}...")
    issues = fetch_closed_issues(args.repo, args.max)
    print(f"Got {len(issues)} closed issues (excluding PRs)")

    saved = 0
    for issue in issues:
        number = issue["number"]
        body = issue.get("body") or ""
        if len(body.strip()) < 20:
            continue  # skip near-empty issues, low signal
        comments = fetch_comments(args.repo, number) if issue["comments"] > 0 else []
        record = {
            "number": number,
            "title": issue["title"],
            "body": body,
            "comments": comments,
            "labels": [l["name"] for l in issue.get("labels", [])],
            "url": issue["html_url"],
            "closed_at": issue.get("closed_at"),
        }
        out_path = OUT_DIR / f"issue_{number}.json"
        out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        saved += 1
        time.sleep(0.1)

    print(f"Saved {saved} issues to {OUT_DIR}/")


if __name__ == "__main__":
    main()
