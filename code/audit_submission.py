#!/usr/bin/env python3
"""Python-dependency-free structural audit for the JRTIP v10 submission."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import Counter
from pathlib import Path


def extract(text: str, pattern: str) -> str | None:
    match = re.search(pattern, text, flags=re.DOTALL)
    return match.group(1) if match else None


def pdf_pages(path: Path | None) -> int | None:
    if path is None or not path.is_file():
        return None
    try:
        result = subprocess.run(
            ["pdfinfo", str(path)], check=True, capture_output=True, text=True
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    match = re.search(r"^Pages:\s*(\d+)\s*$", result.stdout, flags=re.MULTILINE)
    return int(match.group(1)) if match else None


def read_tex_tree(path: Path, seen: set[Path] | None = None) -> str:
    """Read a manuscript and expand local ``\\input`` files for label audits."""

    resolved = path.resolve()
    visited = set() if seen is None else seen
    if resolved in visited:
        return ""
    visited.add(resolved)
    text = resolved.read_text(encoding="utf-8", errors="replace")

    def replace(match: re.Match[str]) -> str:
        child = Path(match.group(1))
        if child.suffix == "":
            child = child.with_suffix(".tex")
        if not child.is_absolute():
            child = resolved.parent / child
        if not child.is_file():
            return match.group(0)
        return read_tex_tree(child, visited)

    return re.sub(r"\\input\{([^}]+)\}", replace, text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tex", type=Path, required=True)
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--max-pages", type=int, default=12)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    text = read_tex_tree(args.tex)
    abstract = extract(text, r"\\abstract\{(.*?)\}\s*\\keywords")
    keywords = extract(text, r"\\keywords\{(.*?)\}")
    abstract_words = len(re.findall(r"\b[\w$%-]+\b", abstract or ""))
    keyword_items = [item.strip() for item in re.split(r"[;,]", keywords or "") if item.strip()]
    labels = re.findall(r"\\label\{([^}]+)\}", text)
    references = set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", text))
    label_counts = Counter(labels)
    duplicate_labels = sorted(label for label, count in label_counts.items() if count > 1)
    unreferenced_figures = sorted(label for label in labels if label.startswith("fig:") and label not in references)
    unreferenced_tables = sorted(label for label in labels if label.startswith("tab:") and label not in references)
    log_text = args.log.read_text(encoding="utf-8", errors="replace") if args.log and args.log.is_file() else ""
    pages = pdf_pages(args.pdf)

    checks = {
        "documentclass_iicol": bool(re.search(r"\\documentclass\[[^]]*iicol[^]]*\]\{(?:sn-jnl|svjour3)\}", text)),
        "abstract_150_to_250_words": 150 <= abstract_words <= 250,
        "keywords_4_to_6": 4 <= len(keyword_items) <= 6,
        "page_limit": pages is not None and pages <= args.max_pages,
        "no_duplicate_labels": not duplicate_labels,
        "all_figures_referenced": not unreferenced_figures,
        "all_tables_referenced": not unreferenced_tables,
        "no_undefined_reference_warning": not bool(re.search(r"undefined references?|Citation .* undefined", log_text, flags=re.IGNORECASE)),
        "data_availability": "Data availability" in text,
        "code_availability": "Code availability" in text,
        "competing_interests": "Competing interests" in text,
        "author_contributions": "Author contributions" in text,
        "funding_statement": "Funding" in text,
        "ethics_statement": "Ethics approval" in text,
    }
    payload = {
        "schema_version": "jrtip_submission_audit_v10_v1",
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "abstract_words": abstract_words,
        "keyword_count": len(keyword_items),
        "keywords": keyword_items,
        "pages": pages,
        "max_pages": args.max_pages,
        "duplicate_labels": duplicate_labels,
        "unreferenced_figures": unreferenced_figures,
        "unreferenced_tables": unreferenced_tables,
    }
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if payload["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
