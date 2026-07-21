# Conditional Routing for 2K Violence Detection - reproducibility package

This package accompanies the JRTIP v10 manuscript. It contains the frozen
derived evidence and scripts needed to reproduce the source-wise tables and to
audit the matched replay campaign. It does not redistribute third-party videos.

## Contents

- `code/`: source-wise generator, matched replay wrapper/renderer and validator, structural and numeric manuscript audits, streaming benchmark, runtime implementation, MoViNet/preprocessing modules, and the exact external-baseline protocol/training/evaluation scripts.
- `data/manifests/`: canonical corpus lineage and split assignment.
- `data/predictions/`: frozen offline predictions for the dense, crowd-centric, routed, and external endpoints.
- `data/workloads/`: sanitized replay selection order, encoded timeline sidecars, composition counts, and input/output hashes; host-specific paths are removed.
- `checkpoints/` and `data/checkpoint_manifest.json`: the two author-trained task checkpoints, their roles, and SHA-256 values; the detector remains an acquired upstream dependency.
- `data/offline_endpoint_protocol.json`: separates the crowd-cache, offline routed evaluator, and causal replay configurations, including the retrospective `full_clip_on_gate` boundary.
- `data/external_dependency_manifest.json`: acquisition and hash boundary for upstream code/weights that cannot be redistributed here.
- `data/workload_manifest.json`: frozen replay IDs, media hashes, and timing protocol; third-party video bytes are not redistributed.
- `analysis/sourcewise/`: machine-readable source-wise results and hash summary.
- `analysis/offline/`: frozen external-model validation/test and model-core timing evidence.
- `benchmark/`: sealed 18-run evidence, verified aggregates, the matched figure/table, and Supplementary Tables S3--S4 after campaign completion.
- `requirements-locked.txt`, `environment-5090-packages.txt`, and `ENVIRONMENT.md`: direct dependencies, complete version inventory, and the exact host boundary recorded for the matched campaign.
- `REPRODUCE.md`: executable source-wise, matched replay, validation, and rendering commands.
- `DATA_SOURCES.md`: source links, license/availability boundaries, internal-key mapping, and reconstruction instructions.
- `LICENSE_SCOPE.md`: distinguishes authored code, vendored code, derived evidence, model weights, and third-party media rights.

## Reproduce source-wise results

Run `code/generate_sourcewise.py` with the canonical manifest and prediction
paths. The script verifies that all endpoint IDs equal the 526-video test set,
that labels match the manifest, and that the external baselines contain exactly
seeds 50900--50902. It performs no inference and no threshold selection.

## Matched replay protocol

The benchmark compares `m3_gated` with `m1_dense_s1` on identical normal,
mixed, and kinetic-rich media. Each mode/workload combination has three
independent process runs. Inputs are 2560x1440 at 30 FPS, sampled to an 8-Hz
analyzed-update schedule after 60 s warm-up and over 600 s measured source time.
Because each frozen file is approximately 600 s long, the v10 runner rewinds
that same hash-verified file once near its end to complete the requested total;
the continuous and routed configurations therefore see the same circular replay.
Runtime state is retained across the rewind, making the boundary an explicit
part of the frozen workload.
Both modes use FP32 (`--no-amp`) on the same RTX 5090 host. Labels are forbidden
inside the runtime.

From the package root, `code/run_matched_replay.py --repo-root code ...` invokes
the exact runner at `code/validation_code/benchmark_streaming_2k_v10.py`. The two
trained checkpoint paths and workload paths must be supplied explicitly; their
expected SHA-256 values are recorded in `data/checkpoint_manifest.json` and the
sealed benchmark provenance.

The exact workload builder is
`code/validation_code/build_streaming_workloads.py` (seed 50900). Its frozen
input cache manifest contained 585 entries marked `split=test` at workload
construction time. It selected 153 non-violence segments for `normal_only`,
142 non-violence plus 14 violence segments for `mixed_controlled`, and 149
violence segments for `kinetic_rich`. The builder-input manifest, selection,
encoded timeline, and later semantic-group split membership are recorded under
`data/workloads/`. Labels were used only to construct those frozen media and
were forbidden to the runtime. The workload mix is controlled execution
evidence, not an accuracy cohort or an estimate of operational prevalence.

The included matched-replay evidence was added only after the internal 18-run campaign passed
the protocol validator. The release-safe copy was produced with
`code/sanitize_matched_replay_release.py`: it retains numerical traces,
telemetry, resource samples, and hashes while removing hostname and
machine-specific absolute paths. Untouched internal evidence remains the audit
source.

## External-model protocol provenance

The external-model experiment entry points and parameterized PowerShell launch
records are retained under `code/model_comparison_scripts/`; shared model
definitions and cached-feature evaluation code are under `code/training_code/`.
The launch records require an explicit repository root and may need adaptation
to a reconstructed dataset location. The frozen
machine-readable endpoint evidence used by the manuscript is
`analysis/offline/final_model_comparison_evidence_v1.json`.

## Data and weights

The source datasets remain under their original licenses. Stable source IDs,
split assignments, raw-video SHA-256 values for all 3,516 retained items, and
reconstruction instructions are provided instead of raw media. The two
author-trained MoViNet task checkpoints are included and hash-verified; upstream
detector acquisition remains documented separately.

## Release gate

The completed benchmark ledger and task checkpoints are present in the public
repository at <https://github.com/VuVietDuc2203/conditional-routing-2k-violence-detection>
and archived at <https://doi.org/10.5281/zenodo.21465979>.
