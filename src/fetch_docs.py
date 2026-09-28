"""Fetch FastAPI's documentation pages.

FastAPI's docs live as markdown files inside its own GitHub repo (not just on
the rendered website), so the simplest and most reliable source is a shallow
clone of the repo rather than scraping HTML. This also sidesteps rendering
quirks (JS-rendered nav, etc.) that HTML scraping would hit.

Usage: python src/fetch_docs.py [--repo-dir path/to/existing/clone]

If --repo-dir is not given, this clones tiangolo/fastapi into a temp-ish
local folder (data/_fastapi_repo/) with a shallow, docs-only checkout.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

OUT_DIR = Path("data/raw_docs")
DEFAULT_CLONE_DIR = Path("data/_fastapi_repo")
DOCS_SUBPATH = "docs/en/docs"  # English docs only, to keep the corpus focused


def clone_repo(dest: Path) -> None:
    if dest.exists():
        print(f"Using existing clone at {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Cloning tiangolo/fastapi (shallow) into {dest}...")
    subprocess.run(
        [
            "git", "clone", "--depth", "1",
            "https://github.com/tiangolo/fastapi", str(dest),
        ],
        check=True,
        env={"GIT_LFS_SKIP_SMUDGE": "1"},
    )


def extract_docs(repo_dir: Path) -> int:
    docs_root = repo_dir / DOCS_SUBPATH
    if not docs_root.exists():
        sys.exit(f"Expected docs folder not found: {docs_root}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    count = 0
    for md_file in docs_root.rglob("*.md"):
        rel_path = md_file.relative_to(docs_root)
        text = md_file.read_text(encoding="utf-8", errors="ignore")
        if len(text.strip()) < 50:
            continue  # skip stub/near-empty pages
        record = {
            "slug": str(rel_path.with_suffix("")),  # e.g. "tutorial/dependencies"
            "source_path": str(rel_path),
            "text": text,
        }
        # Flatten path separators for a safe filename
        safe_name = str(rel_path.with_suffix("")).replace("/", "__")
        out_path = OUT_DIR / f"{safe_name}.json"
        out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", type=Path, default=None,
                         help="Path to an existing fastapi clone (skips cloning)")
    args = parser.parse_args()

    repo_dir = args.repo_dir or DEFAULT_CLONE_DIR
    if args.repo_dir is None:
        clone_repo(repo_dir)

    n = extract_docs(repo_dir)
    print(f"Extracted {n} documentation pages to {OUT_DIR}/")


if __name__ == "__main__":
    main()
