# Reproduction guide

Run commands from the package root with Python 3.13.9 and the versions in
`requirements-locked.txt`. Replace angle-bracketed paths with local paths. Raw
third-party videos are not included.

## Source-wise offline endpoints

```text
python code/generate_sourcewise.py \
  --manifest data/manifests/corpus_lineage.csv \
  --m1 data/predictions/m1_v2_predictions.csv \
  --m3 data/predictions/m3_v2_predictions.csv \
  --routed data/predictions/routed_v2_predictions.csv \
  --external data/predictions/external_paired_predictions.csv \
  --output-dir reproduced/sourcewise
```

The generator checks the canonical 526 IDs, manifest labels, prediction hashes,
and three external seeds before writing the cohort, per-endpoint results,
external mean/sample-SD summary, JSON provenance, and LaTeX supplementary file.

## External-model endpoints

The exact experiment entry points are in `code/model_comparison_scripts/`, and
the shared architectures and cached-feature evaluators are in
`code/training_code/`. The `*.ps1` launch records now require an explicit
repository root and optionally a Python executable; adapt reconstructed dataset,
cache, and output locations as needed. Compare reproduced endpoint JSON against
`analysis/offline/final_model_comparison_evidence_v1.json`; do not substitute
pre-amendment accuracy files for the frozen final predictions.

## Matched replay

Place the three hash-matched workloads under
`code/result/streaming_2k/workloads_v1_10m/`. Keep the filenames recorded in
`data/workload_manifest.json`. The two trained checkpoint files are included in
`checkpoints/`; verify them against `data/checkpoint_manifest.json`. The runner
accepts these files directly through the two checkpoint arguments below.

If licensed source clips are available, reconstruct the controlled media with
`code/validation_code/build_streaming_workloads.py`,
`data/workloads/builder_input_manifest.csv`, seed 50900, a 0.10 mixed-workload
violence-duration target, and the FFmpeg settings embedded in that script.
The original construction used FFmpeg/ffprobe 8.1 full build (gyan.dev).
Compare the selection order and encoded event timeline against
`data/workloads/`. The builder manifest predates the final semantic-group split;
the composition summary records the later membership of every selected ID.
The package intentionally omits the derived MP4 files because they contain
third-party media.

```text
python code/run_matched_replay.py \
  --repo-root code \
  --python <python-executable> \
  --output-root reproduced/matched_raw \
  --analysis-dir reproduced/matched_analysis \
  --m1-checkpoint checkpoints/m1_dense_s1_best.pt \
  --m3-checkpoint checkpoints/m3_gated_best.pt \
  --warmup-sec 60 \
  --duration-sec 600
```

The wrapper executes three process replicates for each combination of two modes
and three workloads. It passes `--loop-source` so the approximately 600-s file
is rewound once to supply 60 s warm-up plus a complete 600-s measured interval.
Runtime state is retained across that rewind.
Do not treat frames as replicates. After the completion
marker is present, validate hashes and protocol cells:

```text
python code/validate_matched_replay.py \
  --analysis-dir reproduced/matched_analysis \
  --raw-root reproduced/matched_raw \
  --output reproduced/MATCHED_REPLAY_QA.json
```

On the evaluation host, create the release-safe evidence copy only after that
QA file passes:

```text
python code/sanitize_matched_replay_release.py \
  --analysis-dir reproduced/matched_analysis \
  --raw-root reproduced/matched_raw \
  --validation reproduced/MATCHED_REPLAY_QA.json \
  --output-root reproduced/matched_release
```

This step preserves traces, telemetry, hashes, and resource samples while
removing hostname and machine-specific absolute paths. It does not modify the
internal evidence.

Then reproduce the verified aggregate, table, and figure:

```text
python code/render_matched_replay.py \
  --summary reproduced/matched_raw/matched_replay_summary.json \
  --raw-root reproduced/matched_raw \
  --figure reproduced/Fig6.pdf \
  --table reproduced/matched_replay_table.tex \
  --supplementary reproduced/supplementary_matched_replay.tex \
  --verified-csv reproduced/matched_replay_verified.csv
```

The energy output integrates NVIDIA GPU board-power telemetry over the
decode/analysis loop, including source-time warm-up and excluding model
initialization. The renderer requires at least 90% process-wall coverage and no
telemetry gap above 5 s; joules per update use all analyzed warm-up and measured
updates as the denominator. It is not total-host energy.

## Manuscript structural audit

After compiling the manuscript twice, run the Python-dependency-free structural
audit against the final LaTeX source, PDF, and second-pass log. The `pdfinfo`
executable must be available on `PATH` for the page-limit check:

```text
python code/audit_submission.py \
  --tex <manuscript_final.tex> \
  --pdf <manuscript_final.pdf> \
  --log <manuscript_final.log> \
  --max-pages 12 \
  --output <JRTIP_AUDIT_RESULT.json>
```

This check covers the abstract and keyword counts, `[iicol]` template option,
page limit, labels/references, undefined-reference warnings, and required
declaration headings. It supplements rather than replaces visual inspection.

After rendering the matched table, verify that locked offline counts and all six
workload-level table rows agree with the sealed JSON evidence:

```text
python code/audit_numeric_consistency.py \
  --tex <manuscript_final.tex> \
  --vi <manuscript_vi.md> \
  --sourcewise analysis/sourcewise/sourcewise_summary.json \
  --matched benchmark/matched_replay_verified.json \
  --matched-table benchmark/matched_replay_table.tex \
  --output <NUMERIC_CONSISTENCY_QA.json>
```
