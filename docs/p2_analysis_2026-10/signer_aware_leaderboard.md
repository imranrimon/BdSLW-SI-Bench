# P2 analysis 4: signer-aware leaderboard statistics (held-out LOSO)

Script: `tools/p2_analysis/signer_aware_leaderboard.py`. Numbers: `docs/p2_analysis_2026-10/signer_aware_leaderboard.json`.

## Data

Held-out-signer Top-1 comes from `docs/loso_heldout/<file>.csv`: 11 signers (U01, 02, 04, 05, 06, 08, 09, 11, 12, 13, 15) × 3 seeds × 8 archs = 264 runs. The file choice is the paper's (`tools/arch_significance.py` `LOSO_FILE`):

- **GNN+Transformer = `gnn_transformer_clip5.csv`**, the grad-norm-clip-5.0 re-run. The paper reports this run and marks it ‡.
  - The unclipped `gnn_transformer.csv` has collapsed seeds: U04 seed 2 = 1.73%, U11 seed 1 = 1.69% (chance = 1.67%), and U06 seed 1 = 20.86%.
  - It is shown below only as a sensitivity row.
- **BlockGCN = `block_gcn_lr01.csv`** (validation-selected base LR 0.01). The old lr 0.1 file is a sensitivity row.

The arch means equal `docs/loso_heldout/derived/arch_significance_blockgcn_lr01.txt` panel (B), which is the source of the paper's `tab:sig` (B).

No rliable install was needed. Its estimators are re-implemented in the script:

- IQM is `scipy.stats.trim_mean(·, 0.25)` over pooled runs.
- Probability of improvement is the per-task Mann–Whitney statistic, averaged over tasks.
- The stratified bootstrap resamples runs within each task.

## Signer-aware statistics (Top-1 %, 95% percentile CIs, 10k resamples)

- **strat** is the rliable-style stratified bootstrap. The 11 signers are fixed as "tasks" and the 3 seeds are resampled within each signer. It measures init noise only.
- **clus** is the signer-clustered bootstrap. It resamples signers, then seeds. This is the CI for generalisation to new signers.

| Arch | Mean [strat] [clus] | IQM (33 runs) [strat] [clus] | Worst signer [strat] | P10 (33 runs) [strat] [clus] | P10 (signer means) | Median signer |
|---|---|---|---|---|--:|--:|
| Adaptive GNN | 80.23 [79.8, 80.7] [71.2, 88.0] | 84.94 [84.3, 85.6] [71.6, 90.2] | 49.64 (U01) [48.1, 52.0] | 56.09 [55.9, 57.2] [48.2, 81.0] | 56.69 | 88.07 |
| ST-GCN | 78.65 [78.2, 79.0] [69.0, 86.9] | 83.11 [82.4, 83.7] [69.9, 89.3] | 44.00 (U01) [43.0, 45.1] | 58.35 [58.3, 60.3] [43.2, 80.8] | 59.03 | 86.38 |
| SLGTFormer | 78.52 [77.2, 79.7] [68.7, 87.1] | 82.99 [81.4, 84.5] [69.2, 90.2] | 48.24 (U01) [37.1, 49.5] | 50.45 [46.6, 56.5] [38.9, 76.6] | 48.96 | 84.52 |
| GNN+Transformer‡ | 77.10 [76.5, 77.6] [67.6, 85.7] | 81.81 [81.0, 82.5] [66.2, 88.5] | 47.45 (U01) [46.8, 48.2] | 52.30 [52.2, 53.0] [46.9, 78.6] | 52.62 | 85.65 |
| Attention GNN | 75.74 [75.2, 76.3] [65.9, 84.4] | 80.02 [79.2, 80.8] [66.1, 86.8] | 41.74 (U01) [39.7, 43.0] | 51.52 [51.1, 55.0] [40.3, 76.3] | 53.03 | 82.09 |
| BlockGCN (lr 0.01) | 74.37 [73.3, 75.4] [64.7, 83.0] | 78.33 [76.6, 79.8] [64.1, 86.0] | 42.63 (U01) [41.5, 44.5] | 49.95 [49.9, 53.3] [41.6, 74.4] | 51.09 | 81.42 |
| GNN+LSTM | 73.69 [72.2, 75.0] [63.8, 82.6] | 77.80 [75.3, 80.0] [62.4, 85.4] | 45.13 (U01) [42.0, 45.7] | 45.90 [44.4, 49.2] [42.4, 73.2] | 45.75 | 78.78 |
| Pose LSTM | 68.72 [68.1, 69.4] [59.3, 77.2] | 72.56 [72.0, 73.4] [58.5, 79.9] | 39.83 (U01) [37.5, 41.2] | 41.92 [41.4, 46.6] [38.2, 68.0] | 44.02 | 77.05 |
| *sens.* BlockGCN lr 0.1 (old) | 65.09 | 67.50 | 28.75 (U01) [18.3, 36.4] | 37.65 | 45.24 | 66.16 |
| *sens.* GNN+Transformer unclipped | 58.01 | 61.47 | 35.04 (U01) [1.7, 37.8] | 28.08 [5.6, 39.3] | 42.96 | 55.51 |

Notes on the worst-signer column:

- The worst-signer CI uses the stratified bootstrap only. Under signer resampling, a minimum over resampled signers is not a meaningful estimand.
- **U01 is the worst signer for all 8 archs.** See analysis 1: U01 is the predominantly left-handed signer.

**Rankings change with the statistic** (the full lists are in the JSON under `rankings`):

- Mean and IQM agree: Adaptive > ST-GCN > SLGTFormer > GNN+T > Attn > BlockGCN > GNN+LSTM > Pose-LSTM.
- **Worst signer:** Adaptive (49.6) > SLGTFormer (48.2) > GNN+T (47.5) > **GNN+LSTM (45.1)** > ST-GCN (44.0) > BlockGCN (42.6) > Attn (41.7) > Pose-LSTM (39.8).
  - ST-GCN drops from 2nd to 5th, and GNN+LSTM rises from 7th to 4th.
  - All 8 worst-signer values lie within 10 pp, versus a 49 pp range across signers.
- **P10 (runs):** ST-GCN (58.4) > Adaptive (56.1) > GNN+T > Attn > SLGTFormer > BlockGCN > GNN+LSTM > Pose-LSTM.
  - SLGTFormer drops to 5th because of its seed spread at U01/U12.

## Probability of improvement (adjacent-by-mean pairs)

The full 28-pair matrix is in the JSON.

| Pair | P(X > Y) | CI strat | CI clus |
|---|--:|---|---|
| Adaptive GNN > ST-GCN | 0.62 | [0.53, 0.70] | [0.36, 0.86] |
| ST-GCN > SLGTFormer | 0.40 | [0.26, 0.53] | [0.20, 0.62] |
| SLGTFormer > GNN+T‡ | 0.75 | [0.62, 0.87] | [0.56, 0.90] |
| GNN+T‡ > Attention GNN | 0.67 | [0.59, 0.76] | [0.46, 0.87] |
| Attention GNN > BlockGCN | 0.67 | [0.52, 0.80] | [0.46, 0.83] |
| BlockGCN > GNN+LSTM | 0.50 | [0.39, 0.62] | [0.27, 0.73] |
| GNN+LSTM > Pose LSTM | 0.84 | [0.73, 0.94] | [0.70, 0.96] |

Selected pairs against the leader:

| Pair | P(X > Y) | CI strat | CI clus |
|---|--:|---|---|
| Adaptive > SLGTFormer | 0.54 | [0.42, 0.67] | [0.34, 0.75] |
| Adaptive > GNN+T | 0.81 | [0.71, 0.92] | [0.65, 0.97] |
| Adaptive > Attention GNN | 0.94 | — | [0.82, 1.00] |
| Adaptive > BlockGCN | 0.96 | — | [0.85, 1.00] |
| Adaptive > Pose LSTM | 1.00 | — | — |

ST-GCN vs SLGTFormer has P = 0.40 although ST-GCN has the higher mean. SLGTFormer wins more signer×seed match-ups, and loses on mean through its low-seed outliers.

## Mixed models (statsmodels MixedLM, REML, 264 runs)

| Adjacent pair (by mean) | Δ (pp) | M1: (1\|signer)+(1\|seed) CI, p_Holm | M2: (1\|signer)+(1\|signer:arch) CI, p_Holm | Paper `tab:sig` (B), signer-clustered bootstrap: Δ, CI, p_Holm |
|---|--:|---|---|---|
| Adaptive > ST-GCN | +1.59 | [−0.04, +3.21], 0.166 | [−0.35, +3.53], 0.32 | [−0.08, +3.33], 0.155 |
| ST-GCN > SLGTFormer | +0.12 | [−1.50, +1.75], 0.44 | [−1.81, +2.06], 0.49 | [−2.12, +2.68], 0.478 |
| SLGTFormer > GNN+T‡ | +1.42 | [−0.20, +3.05], 0.22 | [−0.51, +3.36], 0.37 | [−0.50, +3.81], 0.260 |
| GNN+T‡ > Attention GNN | +1.35 | [−0.27, +2.98], 0.22 | [−0.58, +3.29], 0.37 | [−0.29, +3.05], 0.231 |
| **Attention GNN > BlockGCN** | +1.38 | [−0.25, +3.00], 0.22 | [−0.56, +3.31], 0.37 | **[+0.32, +2.71], 0.014 \*** |
| BlockGCN > GNN+LSTM | +0.68 | [−0.95, +2.30], 0.41 | [−1.26, +2.61], 0.49 | [−1.15, +2.53], 0.475 |
| GNN+LSTM > Pose LSTM | +4.97 | [+3.34, +6.60], <1e-4 | [+3.03, +6.91], <1e-4 | [+3.48, +6.43], <1e-4 |

**Variance components:**

- M1: signer var = 251.8 (SD 15.9 pp), seed var = 0.19 (SD 0.43), residual = 11.4.
- M2: signer var = 251.6, signer×arch var = 2.16, residual = 9.65.

In M1 the seed index carries almost no variance. That is expected: seed k of arch A and seed k of arch B are unrelated runs, so seed is not truly a crossed factor. M2 is the structurally correct model, with the 3 seeds as replicates within an arch×signer cell.

**Comparison with the paper's significance:**

- The point differences are identical, because the design is balanced.
- Every pair except one agrees in verdict with the paper's signer-clustered bootstrap.
- The exception is **Attention GNN > BlockGCN**, the only significant adjacent pair in the paper's panel (B) besides GNN+LSTM > Pose LSTM. It is **not** significant under either mixed model (p_Holm 0.22 / 0.37).
  - The paired bootstrap uses the pair-specific per-signer difference SD, which is small for this pair.
  - The mixed model pools the arch×signer interaction over all 8 archs.
  - The claim is therefore method-sensitive. In the paper it should be stated as "significant under the paired signer bootstrap, not under a pooled-interaction mixed model".
- `docs/loso_heldout/derived/signer_clustered_significance_7arch.txt` (named in the task) is **stale**. It uses old BlockGCN lr 0.1 (61.87; that file is itself inconsistent with the 65.09 in `block_gcn.csv`) and omits GNN+Transformer.
- `signer_clustered_significance_8arch.txt` is also stale (BlockGCN lr 0.1 = 65.09).
- The live comparison is `arch_significance_blockgcn_lr01.txt` panel (B), as used above.

## Headline reading

- The leaderboard order by mean and by IQM is identical.
- The signer-clustered CIs (width about 17 pp) overlap for all 8 archs. Only paired comparisons resolve them, and even then only GNN+LSTM > Pose-LSTM is robust to the analysis choice. Adaptive GNN's advantage shows up as P(improvement) ≥ 0.81 against 5 of the 7 other archs.
- A "worst-signer" column adds information. Every arch fails on the same signer (U01), the spread there is 39.8–49.6, and the order differs from the mean order (ST-GCN falls from 2nd to 5th).
