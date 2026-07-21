# License scope

The root `LICENSE` applies to the author-created software in this artifact. The
vendored MoViNet-PyTorch module retains its own included MIT license and notice.

The manifest and frozen prediction files are distributed as research evidence;
they do not grant rights to the underlying third-party videos. Dataset names,
stable identifiers, hashes, and reconstruction metadata remain subject to the
source owners' terms described in `DATA_SOURCES.md`. Raw third-party media are
not included.

Ultralytics software and model weights are not relicensed by this package. The
`yolo11n.pt` detector is acquired separately under the terms published by
Ultralytics. Trained task checkpoints, if included in the final deposit, are
provided for research reproducibility without conveying rights to reconstruct
or redistribute source video content.

The official JOSENet source is likewise not redistributed because its upstream
repository does not state a license. `data/external_dependency_manifest.json`
records the upstream URL and evaluated file hashes needed to verify a separately
obtained copy.
