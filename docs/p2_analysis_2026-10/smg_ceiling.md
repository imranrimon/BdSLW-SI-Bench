# P2 analysis 3: is the SMG ordering a ceiling artifact?

Script: `tools/p2_analysis/smg_ceiling.py`. Numbers: `docs/p2_analysis_2026-10/smg_ceiling.json`.

## Inputs

The per-seed values are the same sources as the paper's `tab:smg8` rows. They are resolved live through `tools/p2_make_tables.py`, using the `POSE` spec's `mixed` and `test` sources, so no accuracy is recomputed differently from the paper. The per-arch source strings are stored in the JSON under `rows[*].mixed_src` and `si_src`:

| Arch | Mixed source (sd TEST, best-on-sd-val) | SI source (Protocol-A TEST) |
|---|---|---|
| Pose-LSTM, GNN+Bi-LSTM, Attention GNN, ST-GCN, Adaptive GNN | `results_final.csv`: `bdsl_<arch>_sd_noclip_test_seed{0,1,2}` | `results_final.csv`: `bdsl_<arch>_si_seed{0,1,2}_test` |
| BlockGCN (lr 0.01) | `docs/blockgcn_lr01_switch/raw/sd.jsonl [sd_test]` | `docs/blockgcn_lr01_switch/raw/protoA.jsonl [test_top1]` |
| SLGTFormer | `docs/audit_reruns_2026-10/raw/r5_slgtformer_sdtest_seed{s}.pkl` (**grad-clip 5.0 run; the paper's TODO**) | `results_final.csv`: `bdsl_slgtformer_si_seed{0,1,2}_test` |
| GNN+Transformer (clip 5) | `docs/audit_reruns_2026-10/raw/r5_gnn_transformer_sdtest_seed{s}.pkl` | `work_dir/bdsl_gnn_transformer_si_clip5_seed{s}_test/eval_results/best_acc.pkl` |

The SMG, Mixed and SI means reproduce the generated `tab:smg8` rows exactly (`paper_p2_overleaf/generated/tab_smg8_rows.tex`).

## Metrics

- SMG = mean(A_mixed) − mean(A_SI), in pp. This is the paper's metric.
- Error ratio = (100 − A_SI) / (100 − A_mixed): how many times more errors the model makes on unseen signers.
- Log-odds gap = logit(A_mixed) − logit(A_SI). This is the ceiling-robust scale.

CIs come from an exact seed bootstrap: all 3^3 × 3^3 = 729 resamples of the 3 Mixed and 3 SI seeds, resampled independently because seed i of the sd run and seed i of the SI run are unrelated trainings. The CIs are percentile 95%. With n = 3 per split they under-cover, so read them as optimistic.

## Results

| Arch | Mixed | SI | SMG pp [95% CI] | Error ratio [CI] | Log-odds gap [CI] |
|---|--:|--:|--:|--:|--:|
| Pose-LSTM | 94.20±0.54 | 61.64±1.21 | 32.56 [31.3, 34.1] | 6.6 [6.0, 7.5] | 2.31 [2.20, 2.45] |
| BlockGCN | 97.98±0.07 | 68.82±0.56 | 29.16 [28.6, 29.9] | 15.4 [14.8, 16.0] | 3.09 [3.04, 3.13] |
| GNN+Bi-LSTM | 97.24±0.21 | 71.57±3.34 | 25.67 [22.2, 30.1] | 10.3 [8.8, 12.0] | 2.64 [2.44, 2.85] |
| SLGTFormer* | 98.44±0.23 | 73.26±4.77 | 25.18 [21.5, 31.9] | 17.1 [13.4, 21.7] | 3.14 [2.85, 3.46] |
| Attention GNN | 98.24±0.22 | 73.97±1.01 | 24.28 [23.0, 25.3] | 14.8 [12.8, 16.9] | 2.98 [2.83, 3.12] |
| ST-GCN | 98.51±0.17 | 74.51±0.39 | 24.01 [23.5, 24.4] | 17.2 [15.6, 20.5] | 3.12 [3.02, 3.30] |
| GNN+Transformer‡ | 99.00±0.15 | 75.56±1.26 | 23.44 [22.4, 25.2] | 24.4 [20.9, 30.3] | 3.47 [3.30, 3.69] |
| Adaptive GNN | 98.73±0.21 | 79.07±0.89 | 19.66 [18.7, 20.8] | 16.5 [13.7, 20.3] | 3.03 [2.84, 3.24] |

\*Mixed is the grad-clip-5.0 sd run (paper TODO `sd_noclip_slgt`). ‡ is clipped on both splits.

**Rankings (largest gap first):**

- SMG: Pose-LSTM > BlockGCN > GNN+Bi-LSTM > SLGTFormer > Attention GNN > ST-GCN > GNN+Transformer > Adaptive GNN.
- Error ratio: GNN+Transformer > ST-GCN > SLGTFormer > Adaptive GNN > BlockGCN > Attention GNN > GNN+Bi-LSTM > Pose-LSTM.
- Log-odds gap: GNN+Transformer > SLGTFormer > ST-GCN > BlockGCN > Adaptive GNN > Attention GNN > GNN+Bi-LSTM > Pose-LSTM.

**Coupling to SI accuracy (Spearman, n = 8 archs):**

- SMG vs SI = **−1.00**. SMG is a monotone re-expression of SI accuracy, because Mixed is near-constant at 97–99 for 7 of 8 archs.
- Log-odds gap vs SI = +0.52; vs Mixed = +0.79.
- Error ratio vs SI = +0.71; vs Mixed = +0.90.

The ceiling-robust gaps are driven mainly by how close Mixed is to 100. They are not driven by SI weakness.

**Does "the gap is largest for the weakest models" survive?** This was tested as the Kendall τ between the gap and SI weakness, over 4,000 Monte-Carlo seed-bootstrap draws:

| Metric | τ(gap, weakness) mean [95%] | P(τ > 0) |
|---|---|--:|
| SMG | +0.80 [+0.50, +1.00] | 1.00 |
| Error ratio | −0.58 [−0.79, −0.36] | 0.00 |
| Log-odds gap | −0.46 [−0.71, −0.21] | 0.00 |

The direction of the claim reverses on both ceiling-robust scales.

**Rank stability (20k MC draws).**

- Log-odds: GNN+Transformer has the largest gap with P = 0.95, and Pose-LSTM the smallest with P = 0.997. The middle five (BlockGCN, SLGTFormer, ST-GCN, Adaptive GNN, Attention GNN) lie within 2.98–3.14 logits, with overlapping rank CIs (ranks 2–6).
- SMG: Pose-LSTM is largest (P = 0.99) and Adaptive GNN smallest (P = 1.00).

## Reading

On the log-odds scale the signer-mixing inflation is close to architecture-invariant. Five of 8 archs sit at about 3.0–3.1 logits, which is about a 20–23× odds inflation (exp(2.98) = 19.7, exp(3.14) = 23.1). The extremes are GNN+Transformer, which inflates most (3.47; its Mixed is the highest at 99.00), and the two recurrent models, which inflate least (Pose-LSTM 2.31, GNN+Bi-LSTM 2.64). Pose-LSTM is the only model clearly below the ceiling on Mixed (94.2).

The paper's appendix sentence "the gap is largest for the weakest models (Pose-LSTM, BlockGCN)---signer mixing most flatters the models that generalize least" is therefore a scale artifact. In pp it is true by construction (Spearman −1.0 with SI). In odds it is false; Pose-LSTM is flattered least. The main-text "Scope correction" paragraph (§SMG) already notes the mechanical anti-correlation, so the two passages are currently inconsistent.

The headline SMG numbers themselves are unaffected. They are correct pp differences.

## Caveats

- n = 3 seeds per split; the bootstrap CIs are optimistic.
- SLGTFormer's Mixed is still the grad-clip-5.0 run. Its log-odds gap may move when the unclipped sd run lands.
- The error ratio is unstable when Mixed error is about 1% (GNN+Transformer: 1.00% error, CI 20.9–30.3×). Prefer the log-odds gap.
