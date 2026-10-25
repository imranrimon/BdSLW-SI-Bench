# P2 fold-in: unclipped BdSLW60 mixed split, BdSLW401 with S15 validation, AUTSL BlockGCN lr 0.01 (2026-10-10)

Both P2 copies (`paper_p2_overleaf/main.tex`, `paper_p2_ieee/main.tex`) got the **same 29 string replacements** (36 changed lines per file) from one script. Each replacement was checked to match exactly once in each file before anything was written.

Checks after the edit:
- The two copies still differ only in the template/layout lines they differed in before. The diff hunks are identical; only the line offsets moved.
- Braces, `$` and `\begin`/`\end` balance in both copies, and every changed line balances on its own.
- Nothing was committed. No jobs were submitted.

## Sources

| key | source |
|---|---|
| A | BdSLW60 signer-dependent split, unclipped re-run (job). Configs: `config/bdsl_<arch>_sd_noclip.yaml`. Selection: best on sd val, scored once on sd TEST. 3 seeds, population SD. |
| B | `docs/w401_sdval/README.md`: BdSLW401 trained on 15 signers, validated on held-out signer S15, tested on S04/S08. Includes the S15-val per-seed values, the lr decision for BlockGCN, and the 15-signer decodability probe. The W401 mixed numbers use the symmetric rule (best on mixed val, scored on mixed TEST). |
| C | AUTSL BlockGCN at lr 0.01: SI 92.25±0.32 (val 93.5 vs 84.85 at lr 0.1); random split 98.53±0.10; SMG +6.28. |

Derived here: the S15 val mean ± population SD from the README's per-seed values are CTR 82.50±0.89, ST 83.26±0.52, BlockGCN lr 0.01 76.88±0.70.

## A. BdSLW60 Mixed / SMG (unclipped)

| # | location | old → new |
|---|---|---|
| A1 | Table 7 (`tab:arch`) Adaptive GNN | Mixed 98.76±0.26 / SMG +19.69 → 98.73±0.21 / +19.66 |
| A2 | Table 7 ST-GCN | 98.73±0.03 / +24.22 → 98.51±0.17 / +24.00 |
| A3 | Table 7 Attention GNN | 98.12±0.12 / +24.15 → 98.24±0.22 / +24.27 |
| A4 | Table 7 GNN+Bi-LSTM | 98.05±0.34 / +26.48 → 97.24±0.21 / +25.67 |
| A5 | Table 7 Pose-LSTM | 94.83±0.28 / +33.19 → 94.20±0.54 / +32.56 |
| A6 | Table 7 SLGTFormer | **unchanged** (98.44±0.23 / +25.18); `% TODO SLGTFormer unclipped mixed pending (job sd_noclip_slgt)` added at the end of the row |
| A7 | Table 7 caption | Mixed definition gains "; same recipe as the TEST runs". A TODO comment follows the caption, because this is not yet true for SLGTFormer. |
| A8 | `tab:smg8` rows | Same five updates as A1–A5. Rows re-sorted by SMG (descending): **Attention GNN (+24.27) now sits above ST-GCN (+24.00)**. The full order is Pose-LSTM, BlockGCN, GNN+Bi-LSTM, SLGTFormer, Attention GNN, ST-GCN, GNN+Transformer, Adaptive GNN. The SLGTFormer row has the TODO comment plus "(re-sort rows by SMG when it lands)". |
| A9 | App. smg8 text + `tab:smg8` caption | range "+19.7 to +33.2" → "+19.7 to +32.6" (min Adaptive GNN +19.66, max Pose-LSTM +32.56) |
| A10 | App. smg8 footnote | "For GNN+Transformer both the disjoint-TEST and the mixed runs use grad-norm clipping at 5.0" → "The mixed runs use the same recipe as the disjoint-TEST runs, without gradient clipping, except GNN+Transformer, whose disjoint-TEST and mixed runs both use grad-norm clipping at 5.0". A TODO comment notes that SLGTFormer's Mixed cell is still the clip-5.0 run. |
| A11 | App. smg, "Single signer-dependent split" method | Added the configs: `config/bdsl_<arch>_sd_noclip.yaml`, BlockGCN `config/lr01_switch/bdsl_block_gcn_sd_lr01.yaml`, GNN+Transformer `config/bdsl_gnn_transformer_sd_clip5.yaml` (clipped on both splits). TODO comment added. |

Checked and still true: "largest for the weakest models (Pose-LSTM, BlockGCN)". The two largest SMGs are still Pose-LSTM +32.56 and BlockGCN +29.16, and these are the two lowest TEST rows. The smg8 footnote's "matched multi-split controls give similar gaps (BlockGCN +29.41, Adaptive GNN +19.90)" still holds (+29.16 / +19.66). No other prose quoted the single-split Mixed values. The abstract, Fig. 1 and `tab:smg` use the matched 5-split numbers, which are unaffected.

## B. BdSLW401 with S15 validation

| # | location | old → new |
|---|---|---|
| B1 | Table 7 panel (b) rows | CTR-GCN **86.98±0.46** / F1 86.40 / Mixed -- / SMG -- → 85.56±0.29 / 84.91 / 98.78±0.07 / +13.22. ST-GCN 86.87±0.47 / 86.15 → **86.05±0.31 / 85.39** / 98.42±0.10 / +12.37. BlockGCN 61.53±13.44 / 60.46 → 80.86±0.83 / 80.07 / 98.50±0.13 / +17.64. **Bold moves from CTR-GCN to ST-GCN.** Rows re-ordered by TEST (ST-GCN, CTR-GCN, BlockGCN). The Params/GFLOPs and LOSO (--) cells are unchanged. |
| B2 | Table 7 caption, (b) | Added: checkpoints selected on held-out training signer S15 (signer-disjoint val; 15 training signers); BlockGCN at its S15-selected base LR 0.01; Mixed is a signer-mixed split of BdSLW401 under the same selection rule. |
| B3 | §leak, 401-class sentence (**claim changed**) | Old: "16 training signers; CTR-GCN L_norm=0.69±0.00 (TEST 86.98; +65pp above 1/16 chance), BlockGCN 0.43±0.13 (within-class 0.99 / 0.95) … and again the stronger backbone encodes more". New: "15 training signers of the S15-selected runs: CTR-GCN 0.716±0.013 (TEST 85.56), ST-GCN 0.658±0.002 (86.05), BlockGCN 0.630±0.011 (80.86), within-class 0.99 for all three; the three are close (0.63–0.72) and their order does not follow TEST accuracy, so L_norm measures availability rather than generalization". **Removed:** "BlockGCN encodes much less" and "the stronger backbone encodes more" at 401 classes. The "+65pp above 1/16 chance" parenthetical was dropped. |
| B4 | §w401 lead-in | "alongside the validation number used for selection" → "… and the signer-mixed accuracy" |
| B5 | §w401 table (**redefined**) | The first column was the official (seen-signer) VAL: 99.14 / 98.86 / 88.27, with a val→TEST gap of −12.2 / −12.0 / −26.7. New columns: S15 val Top-1 (83.26±0.52 / 82.50±0.89 / 76.88±0.70), held-out TEST (86.05±0.31 / 85.56±0.29 / 80.86±0.83), signer-mixed TEST (98.42±0.10 / 98.78±0.07 / 98.50±0.13), and SMG (+12.4 / +13.2 / +17.6pp). The README reports no S15-run equivalent of a seen-signer val, so the "seen" comparison is now the symmetric mixed TEST, and the gap column is now the SMG (mixed − TEST). The BlockGCN dagger was removed. Row order is now ST, CTR, BlockGCN. The caption was rewritten to say what each column is. |
| B6 | §w401 paragraph (**claims changed**) | Removed: "val→TEST drop 12–27pp reflects selection optimism + seen→unseen", "CTR/ST to ~87%, BlockGCN to ~62%", "selection-set optimism … recurs at 401-class scale; a fully signer-disjoint val split is future work", and the dagger note that BlockGCN is unstable because it is the weaker backbone. Added: the S15 protocol (official VAL saturates near 99%); S15 val tracks TEST within 4pp; BlockGCN lr 0.01 selected on S15 (val 76.9 vs 57.4; at lr 0.1 TEST is 59.97±14.81), consistent with BdSLW60; W401 SMG +12.4 to +17.6pp (mixed 98.4–98.8), largest for BlockGCN. **VideoMAE claim narrowed:** ST-GCN (86.05) and CTR-GCN (85.56) beat VideoMAE 81.04 by +5.0 / +4.5pp; BlockGCN (80.86±0.83) is within seed noise of it. |
| B7 | Limitations, BdSLW401 bullet | "val is seen-signer, so the val→TEST drop (12–27pp) mixes selection optimism … a fully signer-disjoint validation split remains to be built" → "we select on a single held-out training signer (S15), so model selection rests on one validation signer". Removed "BlockGCN is high-variance on the 401-class TEST (a weaker-backbone-under-scale caveat)". |
| B8 | App. xds paragraph | W401 decodability "(CTR-GCN 0.69±0.00, TEST 86.98; BlockGCN 0.43±0.13)" → "(S15-selected, 15 training signers: CTR-GCN 0.716±0.013, TEST 85.56; ST-GCN 0.658±0.002, TEST 86.05; BlockGCN 0.630±0.011, TEST 80.86)" |

Not touched, as instructed or because already consistent:
- The SMG(k) BdSLW401 curve (Fig. 4b, `tab:smgk`).
- The §diversity footnote and the App. SMG(k) method, which already carve BdSLW401 out to S15.
- The "VideoMAE, as reported" 81.04 row in panel (b).

VideoMAE-on-W401 claims: §w401 was the only place pose was said to beat the 81.04 VideoMAE. The abstract, C1, the conclusion and the limitations never claimed it.

## C. AUTSL BlockGCN at val-selected lr 0.01

| # | location | old → new |
|---|---|---|
| C1 | Related work | "cross-language run on AUTSL (+9.8pp)" → "(+6.3pp)" |
| C2 | §autsl, first sentence | BlockGCN random 94.11±3.2 vs SI 84.29±10.6, SMG +9.8 → 98.53±0.10 vs 92.25±0.32, SMG +6.3 |
| C3 | §autsl, backbone argument (**claim changed**) | Old: "The weak, high-variance BlockGCN SI number (84.29) is a backbone artifact … CTR-GCN … stable (SD 0.13 vs 10.6)". New: BlockGCN runs at its validation-selected base LR (0.01; AUTSL val 93.5 vs 84.9 at 0.1, where its SI seeds were unstable, 84.29±10.6), as on BdSLW60. CTR-GCN 94.78±0.13 remains the strongest pose model on AUTSL, competitive with the >96% RGB winners. The instability is now attributed to the LR, not the backbone. |
| C4 | §autsl signer-pool trend | "AUTSL (31 train signers) +9.8" → "+6.3". The order is still monotone: 29.41 > 6.3 > 3.40. |
| C5 | Limitations, cross-dataset bullet | "replacing the high-variance BlockGCN 84.29±10.6" → "BlockGCN 92.25±0.32 at its validation-selected base LR" |
| C6 | Conclusion | "on AUTSL (+9.8pp)" → "(+6.3pp)" |
| C7 | `tab:xds` AUTSL row | 94.11±3.2 / 84.29±10.6 / +9.8 → 98.53±0.10 / 92.25±0.32 / +6.28 |
| C8 | App. xds paragraph | "A stable CTR-GCN reaches 94.78±0.13 … (the 84.29 BlockGCN entry is a high-variance backbone artifact)" → "BlockGCN, at its validation-selected base LR 0.01, reaches 92.25±0.32 held-out TEST and CTR-GCN 94.78±0.13" |

The abstract, contributions and figure captions do not quote the AUTSL BlockGCN numbers. The Fig. 4b caption's AUTSL text ("~5pp, unseen-signer accuracy climbing to 94") comes from the CTR-GCN SMG(k) curve and is unaffected.

## Figures

No figure was regenerated:
- **AUTSL SMG(k) curve:** built with **CTR-GCN** (`scripts/hpc/slurm_autsl_diversity_grid.sbatch` → `config/autsl_si_ctrgcn.yaml`, `work_dir/autsl_si_ctrgcn_divk*`), not with lr-0.1 BlockGCN. Unaffected.
- **BdSLW401 SMG(k) curve:** already on S15 (CTR-GCN).
- **Other figures:** `tools/make_paper_figures.py` contains none of the changed values. Fig. 1 is the BdSLW60 matched-split result, and figs 2/3 are BdSLW60 BlockGCN lr 0.01.

## P1 (`paper_p1_pami/main.tex`), checked only, not edited

There are no hits for the W401/AUTSL BlockGCN numbers 84.29, 94.11, 86.98, 86.87, 61.53 or 81.04. The pattern hits that do appear are all coincidental:
- "+9.8pp" at lines 72, 283 and 358 is P1's xling vs mono 54.4 vs 44.6 result.
- "+9.8" at line 214 and "+9.85" at line 226 are the KArSL G_data value.
- "0.432" at line 247 is a P1 xling metric.
- "0.439" at line 347 is a BdSLW60 per-signer value.
- "84.3" at line 631 is an AUTSL 25%-label cell in a P1 table.

## Open items

1. **SLGTFormer unclipped mixed is pending** (job sd_noclip_slgt). Its Mixed/SMG cells (Table 7 and `tab:smg8`) are still the clip-5.0 run, marked with `% TODO` comments. When it lands, re-check the following:
   - the row order of `tab:smg8`;
   - the +19.7 to +32.6 range (only an issue if the new SMG falls outside 19.66–32.56);
   - the "largest for the weakest models" sentence.

   The text now states that the mixed runs are unclipped (smg8 footnote, App. smg method, Table 7 caption). This is not yet true for SLGTFormer, and TODO comments flag the gap.
2. The BdSLW401 W401-mixed numbers (98.78 / 98.42 / 98.50) come from the separate signer-mixed W401 split. Its exact composition (which signers, and whether S15 is included) is not described in P2. The caption only says "a signer-mixed split of BdSLW401 under the same selection rule".
3. The ASL Citizen BlockGCN numbers (R@1 73.94 / 77.35, SMG +3.40) were not part of this fold-in. Per the previous changelog, they are still from the original-LR runs.
