#!/usr/bin/env python3
"""Fail-closed structural and evidence audit for the JVCIR v12 package."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", type=Path, required=True)
    args = ap.parse_args()
    root = args.workspace.resolve()
    tex_path = root / "manuscript" / "manuscript_jvcir_v12.tex"
    vi_path = root / "manuscript" / "manuscript_vi_jvcir_v12.md"
    tex = tex_path.read_text(encoding="utf-8")
    vi = vi_path.read_text(encoding="utf-8")
    failures: list[str] = []
    warnings: list[str] = []

    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)
    abstract_words = len(re.findall(r"\b[A-Za-z0-9-]+\b", re.sub(r"\\[A-Za-z]+|[{}$]", " ", abstract.group(1) if abstract else "")))
    if not abstract or not 1 <= abstract_words <= 250:
        failures.append(f"abstract word count invalid: {abstract_words}")
    keywords = re.search(r"\\begin\{keywords\}(.*?)\\end\{keywords\}", tex, re.S)
    keyword_count = keywords.group(1).count(r"\sep") + 1 if keywords else 0
    if not 1 <= keyword_count <= 7:
        failures.append(f"keyword count invalid: {keyword_count}")

    highlights = [x.strip() for x in (root / "manuscript" / "highlights.txt").read_text(encoding="utf-8-sig").splitlines() if x.strip()]
    if not 3 <= len(highlights) <= 5:
        failures.append(f"highlight count invalid: {len(highlights)}")
    highlight_lengths = [len(x) for x in highlights]
    if any(n > 85 for n in highlight_lengths):
        failures.append(f"highlight exceeds 85 characters: {highlight_lengths}")

    cf = json.loads((root / "evidence" / "counterfactual" / "frozen_counterfactual_summary.json").read_text(encoding="utf-8"))
    replay = json.loads((root / "evidence" / "replay_v12" / "replay_v12_summary.json").read_text(encoding="utf-8"))
    if (cf["bypass"]["correct"], cf["route_on"]["correct"], cf["invocation"]["invoked_videos"]) != (502, 494, 489):
        failures.append("frozen counterfactual endpoints changed")
    if cf["bypass"]["n"] != 526 or cf["route_on"]["n"] != 526:
        failures.append("frozen counterfactual denominator is not 526")
    if replay["run_count"] != 18 or replay["candidate_equivalence_pairs"] != 9:
        failures.append("same-pipeline replay is not the sealed 18-run/9-pair campaign")
    for rel, expected in (
        ("evidence/replay_raw/matched_replay_summary.json", replay["input_hashes"]["matched_replay_summary_sha256"]),
        ("evidence/replay_raw/candidate_equivalence_audit.json", replay["input_hashes"]["candidate_equivalence_audit_sha256"]),
    ):
        actual = sha256(root / rel)
        if actual != expected:
            failures.append(f"hash mismatch: {rel}")

    required_tokens = {
        "English": (tex, ("502/526", "494/526", "489/526", "95.44", "93.92", "21.52--22.29", "77.71--78.48", "49.07--52.40")),
        "Vietnamese": (vi, ("502/526", "494/526", "489/526", "95,44", "93,92", "21,52–22,29", "77,71–78,48", "49,07–52,40")),
    }
    for language, (content, tokens) in required_tokens.items():
        missing = [t for t in tokens if t not in content]
        if missing:
            failures.append(f"{language} manuscript lacks verified tokens: {missing}")

    prohibited = ("stage4_prime_paired_accuracy_analysis.json", "deprecated_final_t50")
    for token in prohibited:
        if token in tex or token in vi:
            failures.append(f"excluded evidence appears in manuscript: {token}")

    log_path = root / "manuscript" / "manuscript_jvcir_v12.log"
    if not log_path.exists():
        failures.append("LaTeX log missing")
    else:
        log = log_path.read_text(encoding="utf-8", errors="ignore")
        if re.search(r"undefined citations|undefined references|multiply defined|Fatal error", log, re.I):
            failures.append("LaTeX log contains an undefined/duplicate/fatal error")
        overfull = len(re.findall(r"Overfull \\hbox", log))
        if overfull:
            warnings.append(f"LaTeX log contains {overfull} overfull box (CAS highlights internals)")

    required = [
        "manuscript/manuscript_jvcir_v12.pdf", "manuscript/manuscript_vi_jvcir_v12.pdf",
        "supplementary/supplementary_jvcir_v12.pdf", "figures/Fig1.pdf", "figures/Fig6.pdf",
        "figures/Fig7.pdf", "figures/graphical_abstract.pdf", "manuscript/AI_DISCLOSURE.txt",
        "manuscript/DATA_AVAILABILITY.txt",
    ]
    missing_files = [x for x in required if not (root / x).exists()]
    if missing_files:
        failures.append(f"missing required files: {missing_files}")

    payload = {
        "audit_pass": not failures,
        "abstract_words": abstract_words,
        "keyword_count": keyword_count,
        "highlight_count": len(highlights),
        "highlight_lengths": highlight_lengths,
        "counterfactual": {"bypass": "502/526", "route_on": "494/526", "invoked": "489/526"},
        "replay": {"runs": replay["run_count"], "candidate_equivalence_pairs": replay["candidate_equivalence_pairs"]},
        "failures": failures,
        "warnings": warnings,
    }
    out = root / "qa" / "JVCIR_V12_AUDIT_RESULT.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
