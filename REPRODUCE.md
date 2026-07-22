# Reproduction guide for v1.2.0

Run from the package root with Python 3.13.9 and the versions listed in
`requirements-locked.txt`. Raw third-party videos are not included.

## Frozen-prediction counterfactual

```text
python code/run_frozen_counterfactual_v12.py \
  --m3-predictions data/predictions/m3_v2_predictions.csv \
  --router-predictions data/predictions/routed_v2_predictions.csv \
  --manifest data/manifests/canonical_test_manifest.csv \
  --output-dir reproduced/counterfactual_v12 \
  --bootstrap 10000 --seed 50900
```

The script fails closed unless the two prediction hashes match the sealed
values, the manifest contains the same 526 unique test IDs, labels agree, and
the endpoints reproduce 502 bypass-correct, 494 route-on-correct, and 489
invoked videos. Compare the output hashes with
`analysis/counterfactual_v12/FROZEN_COUNTERFACTUAL_COMPLETE.json`.

## Same-pipeline replay

The sealed completed run is under `benchmark/same_pipeline_replay_v12/`. It
contains 18 process runs: two policies × three controlled workloads × three
independent processes. Each run uses 60 s warm-up and 600 s measured source
time at a target 8-Hz analyzed-update schedule. Candidate IDs and input-tensor
SHA-256 values are recorded before the policy branch.

To audit/post-process the sealed run:

```text
python code/summarize_replay_v12.py \
  --control-dir benchmark/same_pipeline_replay_v12 \
  --raw-dir benchmark/same_pipeline_replay_v12/raw \
  --output-dir reproduced/replay_v12
```

The measurement boundary is `analyzed=true` and `source_time_sec >= 60`, which
must yield 4,800 measured updates in every process. The process, not a frame or
candidate, is the replicate. Aggregates use mean, sample SD, minimum, and
maximum across `n=3` processes. No inferential p-value is reported for these
three-run runtime summaries.

To rerun the expensive campaign, reconstruct the three controlled media using
`DATA_SOURCES.md` and `data/workloads/`, then invoke
`code/run_matched_replay.py` with the included M3 checkpoint and
`code/validation_code/benchmark_streaming_2k_v11.py`. Both policies must use
the same workload bytes, precision, timer boundary, checkpoint, threshold, and
candidate builder. A replay is valid only if all nine candidate-pair audits
pass.

## Source-wise and contextual endpoints

Use `code/generate_sourcewise.py` with `data/manifests/corpus_lineage.csv` and
the prediction files under `data/predictions/`. Source-wise values are
descriptive because strata are unequal; they do not estimate external-domain
generalization.

## Integrity

```text
python build_manifest.py
```

The command rebuilds the deterministic SHA-256 manifest while excluding Git
metadata and the manifest files themselves. The failed final-T50 diagnostic is
not part of this release and must not be used as manuscript evidence.
