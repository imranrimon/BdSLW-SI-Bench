# P2 fold-in: BlockGCN lr 0.01 switch + BdSLW401 SMG(k) on signer-disjoint val (2026-10-10)

Scope: every BdSLW60 BlockGCN number in P2 moved from base LR 0.1 to the validation-selected 0.01.
BdSLW401 SMG(k) now selects on held-out signer S15. Three P1 sentences that cite P2's BlockGCN LOSO were updated.
Both P2 copies (`paper_p2_overleaf/main.tex`, `paper_p2_ieee/main.tex`) got the **same 123 string replacements**
from one script; each replacement was asserted to match exactly once (or the stated count) in each file.
After the edit, the two copies still differ only in layout lines (documentclass, table/table*, figure widths, D\&B wording).
Braces, `$` and `\begin`/`\end` balance in both P2 copies and in P1. Nothing was committed.

**Left unchanged on purpose** (jobs still running or out of scope): BlockGCN on BdSLW401 / AUTSL / ASL Citizen;
Table 7 panel (b); the Mixed/SMG cells of Adaptive GNN, ST-GCN, Attention GNN, SLGTFormer, GNN+Bi-LSTM, Pose-LSTM;
the VideoMAE/S3D/Uni-Sign/SHuBERT/prior-method rows.

## Sources

| key | file |
|---|---|
| S1 | `docs/blockgcn_lr01_switch/01_protocolA_metric_suite.md` (+ `tools/p2_metric_suite.metrics_for("bdsl_block_gcn_si_lr01")`, recomputed) |
| S2 | `docs/blockgcn_lr01_switch/02_sd_mixed.md` |
| S3 | `docs/blockgcn_lr01_switch/03_matched_splits.md` |
| S4 | `docs/blockgcn_lr01_switch/04_loso.md`, `docs/loso_heldout/block_gcn_lr01.csv` (stats recomputed: mean 74.37, CI [64.72, 82.91], 42.6–91.3, sample SD 16.07) |
| S5 | `docs/blockgcn_lr01_switch/05_grl.md`, `06_leak_probe.md` |
| S6 | `docs/blockgcn_lr01_switch/07_cues.md` |
| S7 | `docs/blockgcn_lr01_switch/08_enrollment.md` |
| S8 | `docs/blockgcn_lr01_switch/09_diversity_grid.md` |
| S9 | `docs/blockgcn_lr01_switch/10_smg_k.md` |
| S10 | `docs/blockgcn_lr01_switch/11_paired_smg.md`, `raw/fragility_lr01.md` |
| S11 | `docs/blockgcn_lr01_switch/12_keypoint_*.md`, `13_mirror.md` |
| S12 | `docs/blockgcn_lr01_switch/14_two_stream.md` |
| S13 | `docs/blockgcn_lr01_switch/15_rtmpose.md` |
| S14 | `docs/blockgcn_lr01_switch/16_ssl_init.md` |
| S15 | `docs/improvement_runs_2026-10/item5_p1_pretrain_seeds.md` plus a new hierarchical bootstrap of mono − scratch (`tools/p1_pretrain_seed_analysis.hier_boot`, scratch = `bdsl_block_gcn_si_lr01_f10_scratch_seed*`) |
| S16 | `docs/improvement_runs_2026-10/item4_lr_sweep_7arch.md` |
| S17 | `docs/improvement_runs_2026-10/item2_w401_smgk_sdval.md` |
| S18 | **recomputed**: `tools/variance_decompose.py --csv docs/loso_heldout/derived/heldout_as_results_blockgcn_lr01.csv` → `docs/loso_heldout/derived/variance_decomp_blockgcn_lr01.txt` |
| S19 | **recomputed**: `tools/arch_significance.py` (now reads BlockGCN lr 0.01 and GNN+Transformer clip-5) → `docs/loso_heldout/derived/arch_significance_blockgcn_lr01.txt` |

`tools/lr01_switch_report.py` was re-run once by accident (it ignores `--help`). It regenerated `docs/blockgcn_lr01_switch/02–16_*.md` from the raw evals with identical content, so the per-item numbers reproduce from data.

The variance-decomposition and significance tools were first run on the old inputs. They reproduce P2's old values exactly (15.05/4.96/3.37/4.13; 14.85/6.12/5.27/3.94; 14.77/7.57; panel A +4.69/+7.18). The new values come from the same code with only BlockGCN swapped.

## P2 changes (both copies)

| # | location | old → new | source |
|---|---|---|---|
| 1 | Abstract, SMG | 65.23 → 98.23 (was 95.18), range [90.2,97.5] → [98.1,98.4], +29.95 → +29.41; TEST anchor 65.23 → 68.82 | S1, S3 |
| 2 | Abstract, LOSO | 65.09 (CI [54.94,74.31], 28.8–86.6, SD 17.17) → 74.37 (CI [64.72,82.91], 42.6–91.3, SD 16.07) | S4 |
| 3 | Abstract + Conclusion | L_norm 0.76–0.91 → 0.82–0.91 (table now spans 0.824–0.913) | S5 |
| 4 | C2 | paired gap +27pp → +30pp | S10 |
| 5 | C3 | 3.0× architecture, 3.6× seed → 4.4×, 6.0×; enrollment 1 clip +10–21pp, 5 → ~95% → +20–25pp, 96–98% | S18, S7 |
| 6 | §Shared recipe | "base LR … not always at its validation optimum" → BlockGCN uses val-selected 0.01; others not guaranteed (single-seed check: argmax for 4/7) | S16 |
| 7 | Related work + App. audit takeaways | BlockGCN +29.95 → +29.41 (2×) | S3 |
| 8 | Fig. 1 caption + `fig1_smg.pdf` | +29.95 → +29.41; BlockGCN point 65.23/95.18 [90.2,97.5] → 68.82/98.23 [98.1,98.4] | S3 |
| 9 | §smg paired SMG + footnote | +27.25 (U02 +23.3, U13 +31.2; SD 8.8/8.3; min +11.6) → +30.36 (U02 +25.4, U13 +35.3; SD 3.2/3.8; min +20.9) | S10 |
| 10 | tab:smg | 65.23 / 95.18 / [90.2,97.5] / +29.95 → 68.82 / 98.23 / [98.1,98.4] / +29.41 | S3 |
| 11 | §smg para | +29.95 → +29.41; "most of the mixed-split accuracy is an artifact" → "nearly a third" (29.4 of 98.2) | S3 |
| 12 | §autsl signer-pool trend | BdSLW60 +29.95 → +29.41 | S3 |
| 13 | Fig. 2 caption + `fig2_loso.pdf` | 28.8→86.6, mean 65.1, CI [54.9,74.3] → 42.6→91.3, 74.4, [64.7,82.9]; figure now reads `block_gcn_lr01.csv` | S4 |
| 14 | LOSO table (§loso) | 65.09 / [54.94,74.31] / 28.8–86.6 / 17.17 → 74.37 / [64.72,82.91] / 42.6–91.3 / 16.07 | S4 |
| 15 | §loso para | spread ~58pp (28.8→86.6) → ~49pp (42.6→91.3) (2×); U02/U13 "~73/61 Protocol-A, ~68/60 LOSO" → "~73/64 (3-seed mean), ~74/64" | S4; per-signer from TEST pkls |
| 16 | §loso variance decomposition | 8-arch: signer 15.05, 3.0× arch (4.96), 4.5× interaction (3.37), 3.6× seed (4.13) → 15.14, 4.4× (3.42), 7.3× (2.07), 6.0× (2.54). 4-arch: 14.85/6.12/5.27/3.94 → signer 14.97, seed 2.56, arch 2.17, interaction 1.91 ("ordering is robust" → "signer dominance is robust; only the minor components reorder"). 2-arch: 14.77 vs 7.57 → 14.74 vs 2.93 | S18 |
| 17 | Table 7 (tab:arch) BlockGCN row | 65.23±2.25 / F1 64.18 / Mixed 97.59±0.16 / SMG +32.36 / LOSO 65.1±17.2 → 68.82±0.56 / 67.04 / 97.98±0.07 / +29.16 / 74.4±16.1 | S1, S2, S4 |
| 18 | §archtable significance, per-instance | GNN+LSTM > BlockGCN +4.7 → +4.3 (Holm p=0.001); BlockGCN > Pose LSTM +7.2 → +7.5 | S19 |
| 19 | §archtable significance, signer-clustered | "two tiers; bottom tier Pose-LSTM + BlockGCN (65.1), 3.6pp n.s." → near-continuous chain; significant steps Attention GNN > BlockGCN (+1.4, p=0.014) and GNN+LSTM > Pose-LSTM (+5.0, p<0.001); BlockGCN 74.4 vs GNN+LSTM 73.7 n.s. (+0.7, CI [−1.2,+2.5]) although GNN+LSTM wins per-instance; "only the bottom two separate" → "only Pose-LSTM falls clearly below" | S19 |
| 20 | §metrics exemplar table + caption | ens Top-1 70.62→71.43, Top-5 92.89→91.28, F1 69.18→69.67, MRR 0.80 (=), R@10 97.14→95.53, ECE 2.73%→8.04%, NLL 1.09→1.27, Brier 0.41 (=); caption "ensembling helps this high-variance model" → "ensembling helps" | S1 |
| 21 | §metrics para | 70.62 → 71.43; single 65.23±2.25 → 68.82±0.56 | S1 |
| 22 | Efficiency table | BlockGCN 65.23 → 68.82 | S1 |
| 23 | §eff RTMPose (**claim flips**) | "68.52±9.12, within seed noise of MediaPipe 65.23 … advantage is cross-lingual pretraining, not keypoint density" → 78.10±0.60, +9.3pp over MediaPipe 68.82±0.56; extractor explains ~9.3 of Uni-Sign's 14.8pp lead; remaining ~5.5pp = architecture + 133-kp input + pretraining, not separated; landmark quality limits the MediaPipe-27 leaderboard (extractor gain ≈ BlockGCN→Adaptive GNN gap, +10.3). Footnote rewritten. | S13 |
| 24 | Leak table + caption | supervised BlockGCN 0.763 / 0.968 → (3 seeds) 0.853±0.021 / 0.975±0.002; caption notes other rows are single encoders | S5 |
| 25 | §leak para (**claim removed**) | "supervised BlockGCN lower still at 0.763 … the two supervised backbones bracket the SSL pools" → BlockGCN sits in the SSL band (seed range 0.824–0.871; seed 0 ties DG-XL); added caution that per-encoder gaps of a few hundredths are within seed variation; "measurably reshapes" → "suggesting … reshapes" | S5 |
| 26 | §leak decoupling | 79.07 vs 65.23 → vs 68.82; 0.911 vs 0.763 → vs 0.853±0.021 (above BlockGCN's top seed 0.871) | S1, S5 |
| 27 | Cue table caption + rows | control 63.3±1.0 (2 seeds) → 68.9±0.8 (3 seeds); static 33.7±1.2→44.1±0.2; speed 39.6±10.8→56.0±1.3; shuffle 49.2±5.9→60.3±1.8; body-only 33.3±19.5→48.3±1.2 ("high variance" → "the hands carry most of the signal"); hands-only 44.0±25.3→76.9±1.1 ("high variance" → "dropping the body helps"); root-centred 67.7±2.2→70.3±1.1 | S6 |
| 28 | §cues para + footnote (**reading changes**) | removed "we do not read a hands-vs-body balance (high-variance)"; now all SD ≤1.8, motion essential (static −24.7, speed −12.9, shuffle −8.6), position not load-bearing (root +1.4, "does not hurt"), hands −20.5 when dropped, body nodes removal +8.1 → body hurts unseen-signer transfer (stated as "consistent with" body geometry carrying signer identity); footnote's high-variance sentence removed | S6, S11 |
| 29 | Interventions, SSL bullet (**claims flip**) | "+4.8pp at 10% labels (hier. CI over ≥3 subset draws)" [P1 VAL, single pretraining run] → VAL +8.2 [+4.7,+11.7] but TEST +1.5 [−1.2,+3.7], pooled over 3 pretraining × 3 fine-tune seeds, per-run TEST gains −0.3 to +3.5. "72.8±10.3 held-out TEST, above from-scratch 65.23" [was VAL] → 65.35±2.89 TEST (val 80.15±1.07, one pretraining run), 3.5pp below from-scratch 68.82; conclusion → no reliable SSL benefit on unseen signers | S14, S15 |
| 30 | Interventions, two-stream bullet | "same clip5 recipe", +8.35 [+6.5,+10.3], 73.1→81.5, +8.1 over bone → same recipe, +7.03 [+5.13,+8.94], 71.7→78.8, +9.6 over bone | S12 |
| 31 | Fig. 3 caption + `fig3_pareto.pdf` (**claim changes**) | "GRL barely moves leakage … 0.790→0.722 at best" → "GRL does not reduce signer leakage … 0.804–0.832 vs control 0.853 (seed range 0.824–0.871)"; the ΔL≈0.07 bracket removed and replaced by the control's seed-range error bar | S5 |
| 32 | tab:grl + caption | control 68.47/0.790 (clipped, 3-seed) → 68.82/0.853 (headline recipe, 3-seed; seed-0 control 68.06/0.824 in caption); λ 0.1: 48.86/0.760 → 50.70/0.804; 0.5: 41.90/0.722 → 43.81/0.832; 1.0: 51.65/0.732 → 45.27/0.832; "GRL runs clip at 5.0" → no clipping, arms single-seed | S5 |
| 33 | §grl para (**0.07 claim removed**) | "L_norm falls only 0.790→0.722 (Δ0.07)" → L_norm does not fall: λ=0.5/1.0 at 0.832 above seed-0 control 0.824 and inside control seed range; λ=0.1 0.804 (0.02 below); accuracy 68.82→43.81, non-monotonic (50.70, 45.27) | S5 |
| 34 | Fig. 4 caption (a) + `fig4_diversity.pdf` panel (a) | +20–23pp, clips +4–5pp, "6–8-signer cells high-variance" → +27–28pp, clip doubling −2 to +13pp (+4 avg), "though not monotonically" | S8 |
| 35 | Fig. 4 caption (b) + panel (b) | BdSLW401 83.2→12.2, k=2–16 → 77.0→14.3, k=2–15 (label "Bangla, 15"); BdSLW60 59.9→32.5 (k=2–6) then 37.1 at k=8 → 61.6→30.4, k=2–8 (now monotone) | S17, S9 |
| 36 | tab:div | 29.9±1.4/35.1±0.4; 47.2±1.4/51.3±1.1; 47.2±4.0/42.9±15.3; 50.0±8.7/58.4±9.9 → 26.9±1.6/30.2±0.7; 40.9±2.5/38.5±3.6; 37.8±0.5/50.7±0.4; 54.6±1.6/57.6±3.4 | S8 |
| 37 | §diversity para (**caveats change**) | +20.1/+23.3, "+4–5 per doubling", "monotonic except 6/1500", "6–8-signer cells ±4–15 high-variance" → +27.7/+27.3, doubling −2.4 to +13.0 (+4.2 avg); non-monotone cells named (4>6 signers at n=800; 4 signers 800>1500); SD 0.4–3.6 (init only), so reversals likely reflect subset composition (single draw per cell); "diversity beats clip count" and "still climbing at 8 signers" kept | S8 |
| 38 | SMG(k) footnote | "selects on the official VAL" → adds BdSLW401 exception (held-out training signer S15 removed from pool, so largest k = 15) | S17 |
| 39 | §diversity SMG(k) text | BdSLW60 "falls to ~32–37pp" → "~30pp"; BdSLW401 "83pp at k=2 … solved almost entirely … sixteen signers … 12pp" → "77pp … solved largely … fifteen training signers … 14pp" | S9, S17 |
| 40 | tab:pers | U02 73.5 / 83.4±1.2 / 92.3±1.0 / 94.8±0.4 → 69.8 / 89.4±1.1 / 96.0±0.6 / 96.5±1.1; U13 61.4 / 82.3±1.1 / 94.4±0.5 / 95.9±0.1 → 66.0 / 90.6±0.6 / 97.4±0.4 / 97.8±0.7 | S7 |
| 41 | §pers para | +10–21pp, ~95% (73.5→94.8, 61.4→95.9) → +20–25pp, 96–98% (69.8→96.5, 66.0→97.8) | S7 |
| 42 | §pers ceiling para | 87.8% vs 60.9%, +26.9, +27.25 → 96.5% vs 66.6%, +29.9, +30.36; "enrollment exceeds it" → "reaches it"; 18 signs ≥40pp (tattha, chal, bon, aam, juta), ~12 at ceiling, IQR [+5,+44] → 23 signs (tattha +91.7, dal +80.0, chokh utha +75.8, boroi +71.8, cake +69.4), 9 at ceiling, IQR [+8,+49] | S10 |
| 43 | Conclusion | +29.95 (95.18 vs 65.23) → +29.41 (98.23 vs 68.82); LOSO 65.09 [54.94,74.31] 28.8–86.6 → 74.37 [64.72,82.91] 42.6–91.3 | S3, S4 |
| 44 | App. smg8 footnote | "slightly smaller … (BlockGCN +29.95 …)" → "similar … (BlockGCN +29.41 …)" (matched gaps are now larger than the single-split ones for both models) | S2, S3 |
| 45 | tab:smg8 | BlockGCN 97.59±0.16 / 65.23±2.25 / +32.36 → 97.98±0.07 / 68.82±0.56 / +29.16 (+19.7 to +33.2 range unchanged; endpoints are other archs) | S2 |
| 46 | tab:xds | BdSLW60 95.18 / 65.23±2.3 / +29.95 → 98.23 / 68.82±0.6 / +29.41 | S3 |
| 47 | tab:smgk | BdSLW401 2–16 / 83.2 / 12.2 → 2–15 / 77.0 / 14.3; BdSLW60 59.9 / 37.1 → 61.6 / 30.4 | S17, S9 |
| 48 | App. losovar text + tab:losovar | mean 65.09, SD 17.17 → 74.37, 16.07; components 15.05/4.96/3.37/4.13 → 15.14/3.42/2.07/2.54 | S4, S18 |
| 49 | tab:losofolds | U01..U08 order + values (28.8 … 86.6) → U01 42.6, U12 51.1, U13 63.7, U04 69.2, U02 74.3, U15 81.4, U06 84.6, U11 85.3, U08 86.5, U09 88.0, U05 91.3 | S4 |
| 50 | App. bone text + tab:bone | +8.35 → +7.03; BlockGCN 65.23±2.25 / 66.71±1.74 → 68.82±0.56 / 65.54±1.63 | S12 |
| 51 | App. fragility text + tab:fragility | median +25.0, IQR [+5.1,+44.4], 18/60, 12 → +25.3, [+8.3,+49.3], 23/60, 9; fragile rows → tattha, dal, chokh utha, boroi, cake, tshirt, bhai; robust rows → tv, chikissha, chadar, chachi (chosma −3.3 no longer in the list) | S10 |
| 52 | tab:sig (A) | GNN+LSTM>BlockGCN +4.69 → +4.32 (0.001 **); BlockGCN>Pose-LSTM +7.18 → +7.55 (<0.001 ***) | S19 |
| 53 | tab:sig (B) | Holm p 0.170/0.489/0.232/0.232 → 0.155/0.478/0.260/0.231 (family changed); last three rows "Attn>GNN+LSTM +2.06 n.s.; GNN+LSTM>Pose +4.97 ***; Pose>BlockGCN +3.63 n.s." → "Attn>BlockGCN +1.38 (0.014 *); BlockGCN>GNN+LSTM +0.68 (0.475 n.s.); GNN+LSTM>Pose +4.97 ***" | S19 |
| 54 | tab:suite BlockGCN | 65.23±2.25 / 70.62 / 64.18 / 0.761 / 95.12 / 1.606 / 0.524 → 68.82±0.56 / 71.43 / 67.04 / 0.785 / 95.19 / 1.621 / 0.480 | S1 |
| 55 | App. suite calibration para | "BlockGCN's ensemble ECE 2.73% vs Adaptive 7.75%" → "Attention GNN's ensemble ECE 3.42% vs Adaptive 7.75%" (BlockGCN ens ECE is now 8.04, which would preserve the order; the claim still holds via Attention GNN) | S1 (recomputed for all archs) |
| 56 | App. suite per-signer para | BlockGCN ~6pp → ~9pp (U13 66.7 / U2 75.4, ensemble) | S1 |
| 57 | App. suite hardest classes | "class 42 at 0%, classes 22/28/13 below 20%" → "class 42 at 0%, classes 54/24 below 20%" | S1 (recomputed) |
| 58 | tab:tracks + text | BlockGCN Track A 65.23±2.25, Δ +7.27 → 68.82±0.56, Δ +3.68 | S1 |
| 59 | App. itemc keypoint text (**sign flips**) | "drops stable (±≤2.6); removing body costs only 3.35pp; body-only craters to 48.94; body not inert" → hands-only raises accuracy +4.51 (73.33±0.49 vs 68.82±0.56), body-only 54.04±4.13; for BlockGCN body nodes cost unseen-signer accuracy; added note that the 27-node input is kept for comparability | S11 |
| 60 | tab:kpdrop | full 65.23±2.25; hands 61.88±0.97 (−3.35); body 48.94±2.59 (−16.29) → 68.82±0.56; **73.33±0.49 (+4.51)**; 54.04±4.13 (−14.78) | S11 |
| 61 | App. itemc mirror text + caption + tab:mirroraug | BlockGCN +6.63 mean / +10.08 median, bimodal over 7 seeds (one collapse to 58.0 at lr 0.1) → +7.56 (68.82→76.38±1.99), 3 seeds, stable (74.8–79.2); bimodality caption removed | S11 |
| 62 | Audit table | BlockGCN 65.23±2.25 → 68.82±0.56; backbone row 65.23 → 68.82 | S1 |
| 63 | App. recipe text (**honesty rewrite**) | "BlockGCN base LR 0.1 is not validation-best (0.01 scores 77.97 vs 74.25)" → BlockGCN switched to val-selected 0.01 (77.96 vs 74.25), all BlockGCN results use 0.01. Other 7: single-seed check finds 0.01 is the val argmax for 4 (ST-GCN, SLGTFormer, GNN+LSTM, GNN+Transformer). 0.1 is higher for Adaptive GNN / Attention GNN / Pose-LSTM by 2.3/2.8/0.6pp vs seed 0, but by only +1.0/+1.1/−0.5 vs the 3-seed lr-0.01 mean. A 3-seed check is in progress; no claim that every arch is val-optimal. Config path for BlockGCN added. | S16, `docs/audit_reruns_2026-10/r3_blockgcn_lr.md` |
| 64 | tab:recipe | BlockGCN base LR 0.1 → 0.01 (all other settings identical; `diff` of configs) | config |
| 65 | App. smg matched-split method | added BlockGCN config path, "max over saved checkpoints" estimator note, and the val-selected alternative 98.12 → SMG +29.30 | S3 |
| 66 | App. smg SMG(k) method | adds BdSLW401 S15-selection exception | S17 |
| 67 | App. fusion method | "both from the _clip5 configurations (clipping at 5.0)" → both lr 0.01, no clipping (bone config `config/lr01_switch/bdsl_block_gcn_si_bone_vec_lr01.yaml`) | S12 |
| 68 | App. GRL method | base LR 0.1, clipping 5.0, `bdsl_block_gcn_si_signeradv.yaml` → lr 0.01, no clipping, `config/lr01_switch/bdsl_block_gcn_si_signeradv_lr01.yaml` | S5, config |
| 69 | App. cues method | config path → `config/lr01_switch/bdsl_cue_<cue>_lr01.yaml`; "body-only/hands-only drops are unstable" → "all cells stable (SD ≤1.8); keypoint-group ablation gives same direction" | S6 |

## P1 changes (`paper_p1_pami/main.tex`)

| location | old → new | source |
|---|---|---|
| C4 (line ~72) | LOSO variance 65.1±17.2 → 74.4±16.1 | S4 |
| §test, Headline variance (line ~338) | 65.1±17.2, CI [54.9,74.3], range [28.8,86.6], easiest U08 86.6, hardest U01 28.8 → 74.4±16.1, CI [64.7,82.9], range [42.6,91.3], easiest **U05** 91.3, hardest U01 42.6 | S4 |
| Limitations, statistical footing (line ~410) | (65.1, CI [54.9,74.3]) → (74.4, CI [64.7,82.9]) | S4 |

Nothing else in P1 was changed. P1's own +4.2 / +4.8 headline issues (item 5) are untouched.

## Figures (regenerated with `tools/make_paper_figures.py`, byte-identical copies in both P2 dirs)

- `fig1_smg.pdf`: BlockGCN 68.82 → 98.23 [98.1, 98.4], +29.4.
- `fig2_loso.pdf`: reads `docs/loso_heldout/block_gcn_lr01.csv`. The mean/CI note moved to the top-left with a white box so it no longer overlaps the value column.
- `fig3_pareto.pdf`: lr-0.01 sweep. The control is the 3-seed mean with an L_norm seed-range error bar, and the ΔL≈0.07 bracket was removed.
- `fig4_diversity.pdf`: (a) lr-0.01 grid; (b) BdSLW60 lr-0.01 SMG(k), and BdSLW401 relabelled "(Bangla, 15)" with the S15-selected curve.
- `figA_aslc_density.pdf`: not regenerated (no data change).

## Tool / data changes

- `tools/make_paper_figures.py`: updated the data for figs 1–4 and the fig-2/fig-3 label layout.
- `tools/arch_significance.py`: added `PKL_STEM`, so BlockGCN TEST pkls come from `bdsl_block_gcn_si_lr01` and GNN+Transformer from `_si_clip5`. The second matches what P2 panel (A) already reported. `LOSO_FILE["block_gcn"] = "block_gcn_lr01"`.
- New: `docs/loso_heldout/derived/heldout_as_results_blockgcn_lr01.csv` (heldout_as_results.csv with BlockGCN rows swapped to lr 0.01), `variance_decomp_blockgcn_lr01.txt`, `arch_significance_blockgcn_lr01.txt`.

## Open items / not resolved here

1. **VideoMAE text vs table (pre-existing, out of scope)**: §eff ("The VideoMAE baseline that matches pose-level accuracy ($78.76$, Table…)") and the Conclusion ("reaches only $78.76$") still say 78.76, while Table 7 / abstract say 80.93±2.11.
2. **Adaptive GNN uses only 2 TEST pkls** in `tools/p2_metric_suite.py` and `tools/arch_significance.py` (single 78.94, ensemble 80.00 / 80.59). Table 7 reports 79.07±0.89 over 3 seeds. Pre-existing and not touched.
3. The other 7 archs' Mixed/SMG cells (unclipped re-runs) and the BlockGCN BdSLW401/AUTSL/ASL Citizen rows are pending. When they land, re-check the "+19.7 to +33.2" range, the "largest for the weakest models (Pose-LSTM, BlockGCN)" sentence, and the §autsl/tab:xds numbers.
4. The 3-seed LR check for Adaptive GNN / Attention GNN / Pose-LSTM is running. The recipe appendix says "in progress", so update it when the check lands.
5. The App. cue footnote/§cues point to "paired A_SI/L_signer per cue in App. cues", but that appendix does not contain them. This is pre-existing.
6. GRL arms are single-seed (seed 0), and the full-label SSL init uses one pretraining run. Both are disclosed in the text.
7. Matched random splits: the new BlockGCN estimator is the max over all saved checkpoints. The old runs used every-5-epoch test evals, which Adaptive GNN still uses. This is disclosed in the SMG appendix, with the val-selected alternative (+29.30).
