# BdSLW401 with signer-disjoint validation (held-out signer S15)

Bundle `data/bdslw401_si_sdval` (`preprocessing/make_w401_sdval_bundle.py`, `splits/bdslw401_sdval.json`): train = old train+val minus S15 (15 signers, 39,315 clips); val = all S15 clips (3,950; 400/401 classes); test = unchanged S04/S08 (7,833). Configs `config/*_w401*_sdval.yaml` differ from the reported ones only in data paths. Jobs: `docs/w401_sdval/jobids.txt` (`scripts/hpc/slurm_w401_sdval_{build,train}.sbatch`). TEST = best-val checkpoint scored once on S04/S08. ± = population SD over seeds.

## Recognition: old (seen-signer val) vs new (S15 val)

| Model | old val per seed | new val (S15) per seed | old TEST per seed | **new TEST per seed** | old TEST | **new TEST** | new macro-F1 | paper |
|---|---|---|---|---|--:|--:|--:|--:|
| CTR-GCN (lr 0.1) | 99.16 / 99.13 / 99.13 | 82.68 / 83.49 / 81.34 | 87.51 / 87.03 / 86.39 | 85.15 / 85.68 / 85.84 | 86.98 ± 0.46 | **85.56 ± 0.29** | 84.91 ± 0.34 | 86.98±0.46 |
| ST-GCN (lr 0.1) | 98.86 / 98.84 / 98.88 | 83.54 / 82.53 / 83.70 | 87.37 / 86.99 / 86.25 | 86.26 / 85.61 / 86.29 | 86.87 ± 0.47 | **86.05 ± 0.31** | 85.39 ± 0.26 | 86.87±0.47 |
| BlockGCN (lr 0.1) | 81.66 / 84.39 / 98.75 | 47.82 / 76.61 / 47.90 | 52.90 / 51.19 / 80.51 | 48.68 / 80.89 / 50.35 | 61.53 ± 13.43 | **59.97 ± 14.81** | 58.87 ± 14.95 | 61.53±13.44 |
| BlockGCN (lr 0.01) | 98.97 / 98.95 / 98.91 | 76.38 / 76.38 / 77.87 | 81.46 / 81.58 / 81.32 | 79.70 / 81.28 / 81.60 | 81.45 ± 0.10 | **80.86 ± 0.83** | 80.07 ± 0.88 | — (job) |

## BlockGCN LR decision on the signer-disjoint val (supersedes the seen-signer check, job)

S15-val mean: lr 0.1 = 57.44, lr 0.01 = 76.88 → **select lr 0.01**.

## Signer decodability on the new checkpoints (probe on each model's own 15-signer training clips)

cc = class-conditioned (per-class mean removed), random 8000-clip sample, chance 1/15; wc = within-class on all training clips. Not directly comparable to the old 16-signer values (different signer set / chance).

| Model | cc per seed | cc | wc per seed | wc | old (16 signers, seen-signer-val ckpt) |
|---|---|--:|---|--:|---|
| CTR-GCN (lr 0.1) | 0.710 / 0.703 / 0.734 | 0.716 ± 0.013 | 0.993 / 0.992 / 0.994 | 0.993 ± 0.001 | cc 0.693±0.004 / wc 0.992 |
| ST-GCN (lr 0.1) | 0.658 / 0.656 / 0.661 | 0.658 ± 0.002 | 0.988 / 0.989 / 0.988 | 0.988 ± 0.000 | — |
| BlockGCN (lr 0.1) | 0.337 / 0.601 / 0.277 | 0.405 ± 0.141 | 0.927 / 0.985 / 0.915 | 0.942 ± 0.031 | cc 0.427±0.127 / wc 0.953 |
| BlockGCN (lr 0.01) | 0.640 / 0.635 / 0.614 | 0.630 ± 0.011 | 0.989 / 0.988 / 0.988 | 0.988 ± 0.000 | — |

## Other W401 numbers in P2 that used the seen-signer val (not re-run here)

- SMG(k) BdSLW401 curve in Fig. 4b and tab:smgk (`work_dir/bdsl_ctr_gcn_w401_divk*`, `data/bdslw401_si_div`): selects on the official (seen-signer) val.
- The W401 decodability numbers in P2 §leak and App. xds (0.69 / 0.43) come from the seen-signer-val checkpoints.
- The W401 mixed split (jobs) is a separate signer-mixed protocol and stays as is.
- VideoMAE 81.04 (W401 S04/S08, cited from the dataset paper) is not re-run; under S15-val selection CTR-GCN/ST-GCN still beat it, but val-selected BlockGCN (lr 0.01, 80.86) no longer does.
