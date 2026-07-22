#!/usr/bin/env python3
"""Fail-closed structural audit for the JVCIR v11 submission workspace."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--allow-pending-evidence", action="store_true")
    args = parser.parse_args()
    root = args.workspace.resolve()
    tex_path = root / "manuscript" / "manuscript_jvcir_v11.tex"
    tex = tex_path.read_text(encoding="utf-8")
    failures: list[str] = []
    warnings: list[str] = []

    abstract_match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)
    abstract_words = len(re.findall(r"\b[\w-]+\b", re.sub(r"\\[A-Za-z]+|[{}$]", " ", abstract_match.group(1) if abstract_match else "")))
    if not abstract_match or not 1 <= abstract_words <= 250:
        failures.append(f"abstract word count invalid: {abstract_words}")

    keyword_match = re.search(r"\\begin\{keywords\}(.*?)\\end\{keywords\}", tex, re.S)
    keyword_count = (keyword_match.group(1).count(r"\sep") + 1) if keyword_match else 0
    if not 1 <= keyword_count <= 7:
        failures.append(f"keyword count invalid: {keyword_count}")

    highlights = [line.lstrip("• ").strip() for line in (root / "manuscript" / "highlights.txt").read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    highlight_lengths = [len(item) for item in highlights]
    if not 3 <= len(highlights) <= 5:
        failures.append(f"highlight count invalid: {len(highlights)}")
    if any(length > 85 for length in highlight_lengths):
        failures.append(f"highlight exceeds 85 characters: {highlight_lengths}")

    pending = "Evidence gate:" in (root / "manuscript" / "same_pipeline_offline_results.tex").read_text(encoding="utf-8") or "Evidence gate:" in (root / "manuscript" / "same_pipeline_replay_results.tex").read_text(encoding="utf-8")
    if pending and not args.allow_pending_evidence:
        failures.append("visible evidence-gate marker remains")

    required = [
        root / "manuscript" / "cas-sc.cls", root / "manuscript" / "cas-common.sty",
        root / "figures" / "Fig1.pdf", root / "figures" / "graphical_abstract.pdf",
        root / "manuscript" / "AI_DISCLOSURE.txt", root / "manuscript" / "DATA_AVAILABILITY.txt",
    ]
    missing = [str(path.relative_to(root)) for path in required if not path.exists()]
    if missing:
        failures.append(f"missing required files: {missing}")

    log_path = root / "manuscript" / "manuscript_jvcir_v11.log"
    if log_path.exists():
        log = log_path.read_text(encoding="utf-8", errors="ignore")
        if re.search(r"undefined citations|undefined references|Fatal error", log, re.I):
            failures.append("LaTeX log contains undefined citation/reference or fatal error")
        overfull = len(re.findall(r"Overfull \\hbox", log))
        if overfull:
            warnings.append(f"LaTeX log contains {overfull} overfull boxes")
    else:
        failures.append("LaTeX log missing")

    payload = {
        "audit_pass": not failures,
        "allow_pending_evidence": args.allow_pending_evidence,
        "abstract_words": abstract_words,
        "keyword_count": keyword_count,
        "highlight_count": len(highlights),
        "highlight_lengths": highlight_lengths,
        "pending_evidence": pending,
        "failures": failures,
        "warnings": warnings,
    }
    output = root / "qa" / "JVCIR_AUDIT_RESULT.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
