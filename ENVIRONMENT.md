# Frozen benchmark environment

The matched replay campaign was executed with the following recorded host and
software boundary:

- OS: Windows 11 Pro, build 10.0.26200
- CPU: Intel Core Ultra 9 285K, 24 physical / 24 logical cores
- Installed RAM: 68,143,677,440 bytes (63.46 GiB)
- GPU: NVIDIA GeForce RTX 5090, 32,607 MiB reported VRAM
- NVIDIA driver: 595.79
- Python: 3.13.9
- FFmpeg/ffprobe used to construct the frozen media: 8.1 full build (gyan.dev)
- PyTorch: 2.11.0+cu130
- CUDA reported by PyTorch: 13.0
- Precision: FP32 (`--no-amp`)
- Analyzed-update target: 8 Hz from 30-FPS, 2560x1440 sources
- Per-process timing: 60 s source-time warm-up followed by 600 s measured source time

Direct Python package versions are recorded in `requirements-locked.txt`, and
the complete clean version inventory is `environment-5090-packages.txt`. A raw
`pip freeze --all` export was intentionally excluded from the public package
because Conda entries contained non-portable local build paths. The
campaign's machine-readable system manifest and per-process resource ledgers
are included under `benchmark/` after the campaign is sealed. The reported
GPU-board power boundary is NVIDIA board-power telemetry; it is not a
measurement of total host energy.

The PowerShell launch records under `code/model_comparison_scripts/` are kept as
exact execution provenance and therefore contain the original host paths. They
are not portable commands; use the corresponding Python entry points with
paths adapted to the reconstructed environment.
