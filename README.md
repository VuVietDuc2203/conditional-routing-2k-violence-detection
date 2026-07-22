# Kinematic Routing for Violence Recognition — reproducibility package v1.2.0

This package accompanies the JVCIR manuscript *Kinematic Routing for Violence
Recognition: Accuracy–Invocation Trade-offs and Same-Pipeline Replay*. It
separates two evidence tracks that answer different questions:

1. `analysis/counterfactual_v12/` is a 526-video post-hoc counterfactual built
   from frozen M3 predictions and frozen router decisions. It does not rerun
   inference or claim candidate-level implementation equivalence.
2. `benchmark/same_pipeline_replay_v12/` is an 18-process matched replay. Route
   on and route bypassed execute identical detection, tracking, grouping,
   history, T50 candidate construction, M3 checkpoint, threshold, and tensors;
   only the call decision differs. All nine candidate-equivalence pairs pass.

Third-party source videos are not redistributed. Stable IDs, hashes, source
links, split/group assignments, workload selections, and reconstruction
instructions are supplied instead.

## Headline endpoints

- Frozen bypass: 502/526 correct (95.44%).
- Frozen route on: 494/526 correct (93.92%).
- Offline invoked videos: 489/526 (92.97%).
- Replay route-on calls: 21.52–22.29% of eligible candidates.
- Replay process-CPU reduction: 49.07–52.40% across three workloads.
- Throughput and deadline-miss rates do not improve consistently. GPU-board
  energy is not total-system energy.

## Contents

- `code/`: analysis, replay, validation, rendering, and audit scripts.
- `data/manifests/`: canonical held-out manifest and corpus reconstruction data.
- `data/predictions/`: frozen endpoint and external-baseline predictions.
- `checkpoints/`: author-trained task checkpoints with hashes and scope notes.
- `analysis/counterfactual_v12/`: predictions, source strata, statistics, and
  completion hashes for the v12 counterfactual.
- `analysis/sourcewise/` and `analysis/offline/`: inherited locked recognition
  evidence used for contextual tables.
- `benchmark/same_pipeline_replay_v12/`: control records, raw process ledgers,
  candidate audit, process-level aggregates, and post-processed tables.
- `REPRODUCE.md`, `ENVIRONMENT.md`, and `requirements-locked.txt`: executable
  instructions and environment boundary.
- `ARTIFACT_MANIFEST.json` and `ARTIFACT_SHA256.txt`: deterministic file index.

## Public locations

- Repository: <https://github.com/VuVietDuc2203/conditional-routing-2k-violence-detection>
- Existing archived release v1.0.0: <https://doi.org/10.5281/zenodo.21465979>

The v1.2.0 files must be published as a new Zenodo version before the manuscript
describes them as archived at the DOI. The package itself is upload-ready and
does not assert that this newer version already exists.
