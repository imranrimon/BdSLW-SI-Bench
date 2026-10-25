#!/usr/bin/env python
"""Aggregate the P2 BlockGCN lr-0.01 switch: old (lr 0.1) vs new (lr 0.01), one .md per experiment.

Reads results_final.csv (max Top-1 per run unless stated), the side-effect-free evals in
docs/blockgcn_lr01_switch/raw/ (tools/lr01_switch_eval.py, tail scripts) and
docs/loso_heldout/. The same code computes the old and the new column, so the old column
checks the aggregation against the numbers in the paper.
  python tools/lr01_switch_report.py
"""
import csv, collections, json, re, statistics as st
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "blockgcn_lr01_switch"
RAW = OUT / "raw"
SEEDS = (0, 1, 2)

rows = collections.defaultdict(list)          # exp -> [(epoch, top1%)]
for r in csv.reader(open(ROOT / "results_final.csv")):
    if len(r) < 4: continue
    try: rows[r[1]].append((int(r[2]), float(r[3]) * 100))
    except ValueError: pass


def best(exp): return max(v for _, v in rows[exp]) if rows.get(exp) else None
def final(exp, ep=99):
    v = [a for e, a in rows.get(exp, []) if e == ep]
    return v[-1] if v else None


def ms(xs):
    xs = [x for x in xs if x is not None]
    if not xs: return "--"
    return f"{st.mean(xs):.2f} ± {st.pstdev(xs):.2f}" if len(xs) > 1 else f"{xs[0]:.2f}"


def per(xs): return " / ".join("--" if x is None else f"{x:.2f}" for x in xs)


def raw(section):
    p = RAW / f"{section}.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else []


def write(name, title, lines):
    (OUT / name).write_text(f"# {title}\n\n" + "\n".join(lines) + "\n")
    print("wrote", name)


def row3(label, f_old, f_new):
    o = [f_old(s) for s in SEEDS]; n = [f_new(s) for s in SEEDS]
    return f"| {label} | {per(o)} | {ms(o)} | {per(n)} | **{ms(n)}** |"


HDR = ["| | old lr 0.1 per seed | old mean ± SD | new lr 0.01 per seed | new mean ± SD |",
       "|---|---|--:|---|--:|"]


def sd_mixed():
    new = {r["seed"]: r for r in raw("sd")}
    old_val = [best(f"bdsl_block_gcn_sd_seed{s}") for s in SEEDS]
    nt = [new.get(s, {}).get("sd_test") for s in SEEDS]
    nv = [new.get(s, {}).get("sd_val_best") for s in SEEDS]
    si_new = 68.82
    L = ["Signer-mixed `data/bdsl_sd`, BlockGCN. Current P2 protocol: checkpoint selected on sd val, scored on sd test "
         "(docs/audit_reruns_2026-10/r5_mixed_selection.md). Old lr-0.1 sd-test values from R5; old sd-val-best from CSV.", "",
         "| | per seed | mean ± SD |", "|---|---|--:|",
         "| old lr 0.1, Mixed (sd test, R5) | 97.44 / 97.51 / 97.81 | 97.59 ± 0.16 |",
         f"| old lr 0.1, sd val best (pre-R5 estimator) | {per(old_val)} | {ms(old_val)} |",
         f"| **new lr 0.01, Mixed (sd test)** | {per(nt)} | **{ms(nt)}** |",
         f"| new lr 0.01, sd val best | {per(nv)} | {ms(nv)} |", ""]
    if all(x is not None for x in nt):
        L.append(f"SMG = Mixed − TEST: old 97.59 − 65.23 = +32.36; **new {st.mean(nt):.2f} − {si_new} = {st.mean(nt) - si_new:+.2f}**.")
    write("02_sd_mixed.md", "2 — Signer-mixed split (Mixed column, tab:smg8 / Table 7)", L)


def rand_splits():
    new = {r["split"]: r for r in raw("rand")}
    L = ["Five matched random signer-mixed splits `data/bdsl_rand42..46` (one model per split, seed 0). "
         "New runs select the checkpoint on each split's own val (`config/lr01_switch/bdsl_block_gcn_rand4X_lr01.yaml`). "
         "Old estimator = best epoch on the split's own test (old runs evaluated test every 5 epochs; for the new runs it is "
         "recomputed as the max over all saved checkpoints, a slightly larger set).", "",
         "| split | old lr 0.1 best-on-own-test | new lr 0.01 best-on-own-test (all ckpts) | new lr 0.01 val-selected → test |",
         "|---|--:|--:|--:|"]
    o, a, v = [], [], []
    for r in range(42, 47):
        ob = best(f"bdsl_block_gcn_rand{r}_seed0"); o.append(ob)
        nr = new.get(r, {}); a.append(nr.get("old_estimator_max_test")); v.append(nr.get("val_selected_test"))
        L.append(f"| rand{r} | {ob if ob is None else round(ob, 2)} | {a[-1] if a[-1] is None else round(a[-1], 2)} | {v[-1] if v[-1] is None else round(v[-1], 2)} |")
    def rng(x):
        x = [y for y in x if y is not None]
        return f"{st.mean(x):.2f} [{min(x):.1f}, {max(x):.1f}]" if x else "--"
    L += [f"| **mean [range]** | {rng(o)} | {rng(a)} | **{rng(v)}** |", "",
          "Headline SMG (tab:smg, abstract): old 95.18 − 65.23 = +29.95."]
    if all(x is not None for x in a):
        L.append(f"New, same estimator as the paper: {st.mean(a):.2f} − 68.82 = **{st.mean(a) - 68.82:+.2f}**; "
                 f"val-selected: {st.mean(v):.2f} − 68.82 = {st.mean(v) - 68.82:+.2f}.")
    write("03_matched_splits.md", "3 — Matched random splits (headline SMG, tab:smg, Fig. 1)", L)


def loso():
    def load(f):
        d = collections.defaultdict(dict)
        p = ROOT / "docs" / "loso_heldout" / f
        if not p.exists(): return d
        for r in csv.DictReader(open(p)):
            if r["test_acc"]: d[r["heldout_signer"]][int(r["seed"])] = float(r["test_acc"])
        return d
    L = ["Held-out-signer Top-1 per fold (fold value = mean over seeds); stats over the 11 folds; CI = signer-clustered "
         "percentile bootstrap over folds (10k, seed 1234), as docs/baselines_rerun/item3_loso_all8.md. "
         "New per-fold CSV: docs/loso_heldout/block_gcn_lr01.csv.", "",
         "| | mean | 95% CI | min–max | SD (sample) | folds | seed-runs |", "|---|--:|--:|--:|--:|--:|--:|"]
    per_fold = {}
    for tag, f in [("old lr 0.1", "block_gcn.csv"), ("**new lr 0.01**", "block_gcn_lr01.csv")]:
        d = load(f)
        if not d: L.append(f"| {tag} | -- | | | | | |"); continue
        fm = {s: np.mean(list(v.values())) for s, v in d.items()}; x = np.array(list(fm.values()))
        rng = np.random.default_rng(1234); bs = [rng.choice(x, len(x)).mean() for _ in range(10000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        L.append(f"| {tag} | {x.mean():.2f} | [{lo:.2f}, {hi:.2f}] | {x.min():.1f}–{x.max():.1f} | {x.std(ddof=1):.2f} | {len(x)} | {sum(len(v) for v in d.values())} |")
        per_fold[tag] = d
    L += ["", "Per fold (seed 0/1/2):", ""]
    for tag, d in per_fold.items():
        L.append(f"- {tag}: " + "; ".join(f"{s}: " + "/".join(f"{d[s][k]:.1f}" for k in sorted(d[s]))
                                         for s in sorted(d, key=lambda z: int(z[1:]))))
    write("04_loso.md", "4 — LOSO, 11 held-out signers (tab:losofolds, Fig. 2, variance decomposition, Table 7 LOSO)", L)


def probes():
    p = RAW / "probes.log"
    txt = p.read_text() if p.exists() else ""
    vals = collections.defaultdict(dict)
    for line in txt.splitlines():
        m = re.search(r"\[([^\]|]+)\|(cc|mrg|wc)\]", line) or re.search(r"tag=?([^\s|]+)\|(cc|mrg|wc)", line)
        n = re.search(r"L_norm(?:\^cond)?\s*=\s*([0-9.]+)", line)
        if m and n: vals[m.group(1)][m.group(2)] = float(n.group(1))
    L = ["Same probe settings as docs/audit_reruns_2026-10/r4_lnorm.md (cc = class-conditioned n=4000; marginal n=6000; "
         "within-class n=6000). Old supervised BlockGCN = `bdsl_block_gcn_si_seed0_model-69.pt`.", "",
         "| encoder | cc (P2 table) | marginal | within-class |", "|---|--:|--:|--:|",
         "| old supervised BlockGCN lr 0.1 (seed 0) | 0.763 | 0.625 | 0.968 |"]
    for k in sorted(vals):
        v = vals[k]; L.append(f"| {k} | {v.get('cc', '--')} | {v.get('mrg', '--')} | {v.get('wc', '--')} |")
    if not vals: L.append("| (probes pending) | | | |")
    sup = [vals[k] for k in sorted(vals) if k.startswith("sup_blockgcn_lr01_s")]
    if sup:
        L.append("| **supervised BlockGCN lr 0.01, mean ± SD (3 seeds)** | "
                 + " | ".join("**{:.3f} ± {:.3f}**".format(st.mean([v[m] for v in sup]), st.pstdev([v[m] for v in sup]))
                              for m in ("cc", "mrg", "wc")) + " |")
    L += ["", "Raw log: docs/blockgcn_lr01_switch/raw/probes.log"]
    write("06_leak_probe.md", "6 — Signer decodability of the lr-0.01 BlockGCN (leak table) + GRL probes", L)


def grl():
    p = RAW / "grl_sweep_lr01.md"
    L = ["GRL sweep at base LR 0.01, no clipping (`config/lr01_switch/bdsl_block_gcn_si_signeradv_lr01.yaml`), seed 0 as "
         "the original sweep; Top-1 on TEST and class-conditioned L_norm from `tools/eval_grl_sweep.py`. λ=0 control = "
         "the lr-0.01 headline (TEST 68.82; its L_norm in 06_leak_probe.md).", "",
         "Old (lr 0.1, GRL clip 5.0): λ=0 (clipped control) 68.47 / 0.790; 0.1 → 48.86 / 0.760; 0.5 → 41.90 / 0.722; "
         "1.0 → 51.65 / 0.732.", "", "New:", "", p.read_text() if p.exists() else "(pending)"]
    write("05_grl.md", "5 — Signer-adversarial (GRL) sweep (tab:grl, Fig. 3)", L)


def cues():
    ctl = {r["seed"]: r["final_epoch_test"] for r in raw("cuectl")}
    L = ["Final-epoch Top-1 on the held-out TEST signers 2/13 (the cue configs' test feeder), as P2's cue table. "
         "Control = unperturbed BlockGCN final-epoch checkpoint on TEST (old: 64.25/62.34, seed 0 final ckpt pruned).", ""] + HDR
    L.append(f"| none (control) | -- / 64.25 / 62.34 | 63.30 ± 0.96 | {per([ctl.get(s) for s in SEEDS])} | **{ms([ctl.get(s) for s in SEEDS])}** |")
    for c in ["static", "speednorm", "shuffle", "bodyonly", "handsonly", "rootcenter"]:
        L.append(row3(c, lambda s: final(f"bdsl_cue_{c}_seed{s}"), lambda s: final(f"bdsl_cue_{c}_lr01_seed{s}")))
    write("07_cues.md", "7 — Controlled cue perturbations (§cues table)", L)


def pers():
    L = ["Few-shot enrollment (tools/run_personalization.py; adaptation recipe unchanged: 30 epochs, lr 0.01). "
         "Init = SI BlockGCN seed 0 (old lr 0.1 model-69 / new lr 0.01 best). k=0 = init model on the query; "
         "k>0 = best epoch on the query (as the paper).", ""] + HDR
    for f in (2, 13):
        for k in (0, 1, 3, 5):
            if k == 0:
                fo = lambda s, f=f: best(f"bdsl_block_gcn_si_pers_U{f:02d}_k0_seed{s}_test")
                fn = lambda s, f=f: best(f"bdsl_block_gcn_si_lr01_pers_U{f:02d}_k0_seed{s}_test")
            else:
                fo = lambda s, f=f, k=k: best(f"bdsl_block_gcn_si_pers_U{f:02d}_k{k}_seed{s}")
                fn = lambda s, f=f, k=k: best(f"bdsl_block_gcn_si_lr01_pers_U{f:02d}_k{k}_seed{s}")
            L.append(row3(f"U{f:02d} k={k}", fo, fn))
    write("08_enrollment.md", "8 — Few-shot signer enrollment (tab:pers)", L)


def grid():
    L = ["2D diversity grid (data/bdsl_grid_s*_n*), final-epoch val Top-1, as the paper (tab:div reproduces exactly).", ""] + HDR
    for sg in (2, 4, 6, 8):
        for n in (800, 1500):
            L.append(row3(f"{sg} signers, n={n}", lambda s: final(f"bdsl_grid_s{sg}_n{n}_seed{s}"),
                          lambda s: final(f"bdsl_grid_s{sg}_n{n}_lr01_seed{s}")))
    write("09_diversity_grid.md", "9 — 2D signer-diversity grid (tab:div, Fig. 4a)", L)


def smgk():
    L = ["BdSLW60 SMG(k) (tools/run_diversity_grid.py, seed 0): per draw SMG = seen-holdout Top-1 − unseen TEST Top-1; "
         "mean ± SD over draws (k=8 = single full-pool draw).", "",
         "| k | old lr 0.1 SMG mean ± SD (draws) | new lr 0.01 SMG mean ± SD (draws) |", "|---|--:|--:|"]
    for k, ds in [(2, range(5)), (4, range(5)), (6, range(5)), (8, [0])]:
        out = []
        for stem in ("bdsl_block_gcn_si", "bdsl_block_gcn_si_lr01"):
            g = [best(f"{stem}_divk{k}_d{d}_seed0_seentest") - best(f"{stem}_divk{k}_d{d}_seed0_unseentest")
                 for d in ds if best(f"{stem}_divk{k}_d{d}_seed0_seentest") is not None
                 and best(f"{stem}_divk{k}_d{d}_seed0_unseentest") is not None]
            out.append(f"{ms(g)} ({len(g)})")
        L.append(f"| {k} | {out[0]} | **{out[1]}** |")
    write("10_smg_k.md", "10 — BdSLW60 SMG(k) curve (Fig. 4b, tab:smgk)", L)


def paired():
    L = ["Paired-instance SMG (tools/run_paired_smg.py): TEST Top-1 of each condition on the identical frozen clips of the "
         "focus signer; gap = seen_matched − unseen.", ""] + HDR
    for f in (2, 13):
        for cond in ("unseen", "seen_matched", "seen_aug"):
            L.append(row3(f"U{f:02d} {cond}", lambda s: best(f"bdsl_block_gcn_si_pairsmg_U{f:02d}_{cond}_seed{s}_test"),
                          lambda s: best(f"bdsl_block_gcn_si_lr01_pairsmg_U{f:02d}_{cond}_seed{s}_test")))
    for stem, tag in (("bdsl_block_gcn_si", "old"), ("bdsl_block_gcn_si_lr01", "new")):
        g = {}
        for f in (2, 13):
            gs = [best(f"{stem}_pairsmg_U{f:02d}_seen_matched_seed{s}_test") - best(f"{stem}_pairsmg_U{f:02d}_unseen_seed{s}_test")
                  for s in SEEDS if best(f"{stem}_pairsmg_U{f:02d}_seen_matched_seed{s}_test") is not None
                  and best(f"{stem}_pairsmg_U{f:02d}_unseen_seed{s}_test") is not None]
            g[f] = gs
        allg = g[2] + g[13]
        L.append(f"\n{tag}: paired gap U02 {ms(g[2])}, U13 {ms(g[13])}, overall {ms(allg)} (n={len(allg)})")
    write("11_paired_smg.md", "11 — Paired-instance SMG (§smg (ii), fragility table)", L)


def test_rows(name, title, base, old_fmt):
    L = ["Validation-selected checkpoint scored on TEST signers 2/13.", ""] + HDR
    L.append(row3(base, lambda s: best(old_fmt.format(s=s)), lambda s: best(f"bdsl_block_gcn_si_{base}_lr01_test_seed{s}")))
    write(name, title, L)


def ssl():
    old = {r["seed"]: r["test_top1"] for r in raw("sslold")}
    L = ["Native pose-SSL init (BdSL-only SHuBERT-style encoder `bdsl_shubert_pretrain_bdsl_only/pretrained_epoch30.pt`), "
         "full labels, BlockGCN backbone. **The paper's 72.83 ± 10.31 (sample SD) is best-epoch VALIDATION (U15) accuracy, not "
         "TEST**: the J4 runs' test feeder was val_data and no TEST eval existed. Old TEST below = their best-val checkpoints "
         "scored on TEST (tools/lr01_switch_eval.py --sections sslold).", "",
         "| | per seed | mean ± SD |", "|---|---|--:|",
         f"| old lr 0.1, val (as in paper) | {per([best(f'bdsl_ssl_bdslonly_si_ft_seed{s}') for s in SEEDS])} | {ms([best(f'bdsl_ssl_bdslonly_si_ft_seed{s}') for s in SEEDS])} |",
         f"| old lr 0.1, TEST | {per([old.get(s) for s in SEEDS])} | {ms([old.get(s) for s in SEEDS])} |",
         f"| new lr 0.01, val | {per([best(f'bdsl_ssl_bdslonly_si_ft_lr01_seed{s}') for s in SEEDS])} | {ms([best(f'bdsl_ssl_bdslonly_si_ft_lr01_seed{s}') for s in SEEDS])} |",
         f"| **new lr 0.01, TEST** | {per([best(f'bdsl_ssl_bdslonly_si_ft_lr01_test_seed{s}') for s in SEEDS])} | **{ms([best(f'bdsl_ssl_bdslonly_si_ft_lr01_test_seed{s}') for s in SEEDS])}** |",
         "", "Comparison point: from-scratch BlockGCN TEST old 65.23 ± 2.24 / new 68.82 ± 0.56."]
    write("16_ssl_init.md", "16 — Native pose-SSL init at full labels (Interventions bullet)", L)


def fusion():
    p = RAW / "fusion_lr01_vec.md"
    L = ["Joint (lr-0.01 Protocol A dumps) + bone (lr 0.01, no clipping) score-sum fusion, 3-seed ensemble, paired clip "
         "bootstrap (tools/fusion_bootstrap.py with lr-0.01 paths).", "",
         "Old (clip-5 recipe, both streams): joint 73.1 → fused 81.5 ensemble, fused − joint +8.35 pp, CI [+6.5, +10.3]; "
         "single-stream TEST joint 68.47 / bone 66.71 ± 1.74 (tab:bone used joint 65.23 / bone 66.71).", ""] + HDR
    L.append(row3("bone stream TEST", lambda s: best(f"bdsl_block_gcn_si_bone_clip5_seed{s}_test"),
                  lambda s: best(f"bdsl_block_gcn_si_bone_vec_lr01_test_seed{s}")))
    L += ["", "Bone runs: `config/lr01_switch/bdsl_block_gcn_si_bone_vec_lr01.yaml` (feeder `is_vector: true`, as the old clip5 "
          "bone training runs). A first attempt, `bdsl_block_gcn_si_bone_lr01_seed*`, inherited `is_vector: false` from the current "
          "`config/bdsl_block_gcn_si_bone.yaml` (root-centring + random shift applied to bone vectors) and reached only "
          "28.84 ± 0.93 TEST; it is a config error, discarded (raw/fusion_lr01.md keeps its fusion for the record).", "",
          p.read_text() if p.exists() else "(fusion pending)"]
    write("14_two_stream.md", "14 — Two-stream joint + bone (Interventions, tab:bone)", L)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    sd_mixed(); rand_splits(); loso(); probes(); grl(); cues(); pers(); grid(); smgk(); paired(); ssl(); fusion()
    test_rows("12_keypoint_handsonly.md", "12a — Keypoint drop: hands-only (tab:kpdrop)", "handsonly", "bdsl_block_gcn_si_handsonly_test_seed{s}")
    test_rows("12_keypoint_bodyonly.md", "12b — Keypoint drop: body-only (tab:kpdrop)", "bodyonly", "bdsl_block_gcn_si_bodyonly_test_seed{s}")
    test_rows("13_mirror.md", "13 — Mirror augmentation (tab:mirroraug; old had 7 seeds, first 3 shown)", "mirror", "bdsl_block_gcn_si_mirror_test_seed{s}")
    test_rows("15_rtmpose.md", "15 — RTMPose-133 BlockGCN (§eff)", "rtm", "bdsl_block_gcn_si_rtm_seed{s}_test")
