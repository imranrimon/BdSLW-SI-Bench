# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-10-25

Initial public release accompanying the benchmark paper (under review).

### Added
- Signer-independent BdSLW60 benchmark: canonical signer split, 11 LOSO folds, five
  matched signer-mixed splits (`splits/`), and the BdSLW401 S15-validation protocol.
- MediaPipe Holistic 27-keypoint pose pipeline and bundlers for BdSLW60, BdSLW401, AUTSL and
  ASL Citizen (`preprocessing/`), including missing-joint-aware bones and cue perturbations.
- Training/evaluation harness (`main.py`) with eight pose architectures (SLGTFormer, BlockGCN,
  Adaptive GNN / CTR-GCN-style, Attention GNN, ST-GCN, GNN+Bi-LSTM, GNN+Transformer, Pose-LSTM),
  prior-method re-implementations (Attention Bi-LSTM, SPOTER family), a feature-isolation
  temporal transformer, RGB baselines (S3D, I3D, VideoMAE), a GRL signer adversary and masked
  pose SSL pre-training.
- Signer-normalized input recipe (`feeders/feeder_signernorm.py`, `config/signernorm/`).
- Analysis tools: completeness gate, pose-geometry check, metric suite, significance tests
  (clip and signer-clustered bootstrap, Holm), LOSO variance decomposition, decodability
  probe, test-time adaptation.
- Paper table generator with provenance (`tools/p2_make_tables.py`), archived run log,
  run manifest, resolved-source cache and reference table fragments (`results/`).
- Croissant 1.0 + RAI metadata and the datasheet for the derived pose releases.
- Packaging and tooling: `pyproject.toml`, `requirements.txt`, `Makefile`, ruff and
  pre-commit configuration, CPU-only GitHub Actions CI.
