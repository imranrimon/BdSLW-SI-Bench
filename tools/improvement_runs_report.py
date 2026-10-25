#!/usr/bin/env python
"""Reports for the 2026-10 P2 improvement runs (docs/improvement_runs_2026-10/).

item2: BdSLW401 SMG(k) on the signer-disjoint val (S15) vs the old seen-signer-val grid.
item4: base-LR check {0.1, 0.01, 0.001} for the 7 non-BlockGCN archs on BdSLW60 Protocol A
       (selection = best-epoch val Top-1 on signer 15, as logged in results_final.csv).
All numbers from results_final.csv (max Top-1 over a run's logged rows; a test-phase run
logs one row).
  python tools/improvement_runs_report.py
"""
import collections, csv, re, statistics as st, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "improvement_runs_2026-10"
sys.path.insert(0, str(ROOT / "tools"))

rows = collections.defaultdict(list)
for r in csv.reader(open(ROOT / "results_final.csv")):
    if len(r) < 4: continue
    try: rows[r[1]].append((int(r[2]), float(r[3]) * 100))
    except ValueError: pass


def best(e): return max(v for _, v in rows[e]) if rows.get(e) else None
def best_ep(e): return max(rows[e], key=lambda x: x[1])[0] if rows.get(e) else None
def final(e, ep=99):
    v = [a for x, a in rows.get(e, []) if x == ep]
    return v[-1] if v else None
def f(x): return "--" if x is None else f"{x:.2f}"
def ms(xs):
    xs = [x for x in xs if x is not None]
    if not xs: return "--"
    return f"{st.mean(xs):.2f} ± {st.pstdev(xs):.2f}" if len(xs) > 1 else f"{xs[0]:.2f}"


def nan_in_log(e):
    p = ROOT / "work_dir" / e / "log.txt"
    if not p.exists(): return False
    return bool(re.search(r"loss[^\n]*\bnan\b", p.read_text(errors="ignore"), re.I))


# ---------------------------------------------------------------- item 2
def item2():
    import build_w401_sdval_divgrid as nb
    old_pool = [1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    import run_diversity_grid as rdg
    def old_subset(k, d):
        rdg.POOL = old_pool; return rdg.subset_for(k, d)
    def new_subset(k, d):
        rdg.POOL = nb.POOL; return rdg.subset_for(k, d)

    ost, nst = "bdsl_ctr_gcn_w401_divk{k}_d{d}_seed0", "bdsl_ctr_gcn_w401_sdval_divk{k}_d{d}_seed0"
    old_ks = [(2, range(5)), (4, range(5)), (8, range(5)), (12, range(5)), (16, [0])]
    new_ks = [(2, range(5)), (4, range(5)), (8, range(5)), (12, range(5)), (15, [0])]

    def cell(tmpl, k, d):
        e = tmpl.format(k=k, d=d)
        s, u = best(e + "_seentest"), best(e + "_unseentest")
        return dict(val=best(e), ep=best_ep(e), seen=s, unseen=u,
                    smg=None if s is None or u is None else s - u)

    def agg(tmpl, ks):
        out = {}
        for k, ds in ks:
            cs = [cell(tmpl, k, d) for d in ds]
            out[k] = {m: [c[m] for c in cs] for m in ("val", "seen", "unseen", "smg")}
        return out
    O, N = agg(ost, old_ks), agg(nst, new_ks)

    L = ["# Item 2 — BdSLW401 SMG(k) on the signer-disjoint validation split", "",
         "Old grid: `scripts/hpc/slurm_w401_diversity_grid.sbatch` (CTR-GCN `config/bdsl_ctr_gcn_w401.yaml`, lr 0.1, seed 0; "
         "16-signer train pool; checkpoint selected on the official **seen-signer** val; data/bdslw401_si_div). "
         "New grid: `scripts/hpc/slurm_w401_sdval_smgk.sbatch` (job; first submission <job> cancelled after OOM from over-packing) + `tools/build_w401_sdval_divgrid.py` — identical recipe "
         "(`config/bdsl_ctr_gcn_w401_sdval.yaml`, only data paths differ; configs in `config/w401_sdval_div/`), but the "
         "training-signer pool is the 15 signers of `data/bdslw401_si_sdval` (S15 removed) and the best checkpoint is selected "
         "on **S15** (all 3,950 clips; held out from every subset). Unseen TEST = S04/S08 (7,833 clips, unchanged); seen TEST = "
         "the subset's 20% class-stratified holdout (same holdout seeding as before). SMG = seen − unseen Top-1 of the one "
         "best-val checkpoint. Bundles: `data/bdslw401_si_sdval_div/k<k>_d<r>`.", "",
         "**Design changes forced by excluding S15** (document in the paper footnote):",
         "- The largest k is **15** (full 15-signer pool), not 16 — k=16 is infeasible once S15 is the val signer.",
         "- Draws are re-sampled from the 15-signer pool with the same `subset_for` seeding, so the per-k subsets differ from the old grid "
         "(the old pool included S15). Per-draw subsets are listed below.",
         "- Each subset's clips now come from old train+val (the sdval train), so a signer contributes all its clips rather than its old-train share.", "",
         "## SMG(k), mean ± SD over draws (pp)", "",
         "| k (old / new) | old SMG | **new SMG** | Δ | old unseen TEST | new unseen TEST | old seen | new seen | old val (seen-signer) | new val (S15) | draws old/new |",
         "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|"]
    pairs = [(2, 2), (4, 4), (8, 8), (12, 12), (16, 15)]
    for ko, kn in pairs:
        o, n = O[ko], N[kn]
        os_, ns_ = [x for x in o["smg"] if x is not None], [x for x in n["smg"] if x is not None]
        d = f"{st.mean(ns_) - st.mean(os_):+.2f}" if os_ and ns_ else "--"
        L.append(f"| {ko} / {kn} | {ms(o['smg'])} | **{ms(n['smg'])}** | {d} | {ms(o['unseen'])} | {ms(n['unseen'])} | "
                 f"{ms(o['seen'])} | {ms(n['seen'])} | {ms(o['val'])} | {ms(n['val'])} | {len(os_)}/{len(ns_)} |")
    L += ["", "Fig. 4b plot values (one decimal, as `tools/make_paper_figures.py` line ~176 expects: k list, mean SMG, SD):", ""]
    for name, R, ks in (("old", O, [2, 4, 8, 12, 16]), ("new", N, [2, 4, 8, 12, 15])):
        mu = [st.mean([x for x in R[k]["smg"] if x is not None]) if any(x is not None for x in R[k]["smg"]) else float("nan") for k in ks]
        sd = [st.pstdev([x for x in R[k]["smg"] if x is not None]) if sum(x is not None for x in R[k]["smg"]) > 1 else 0.0 for k in ks]
        L.append(f"- {name}: k={ks}, SMG={[round(x, 1) for x in mu]}, SD={[round(x, 1) for x in sd]}")
    L += ["", "## Per draw", "",
          "| cell | old subset | old val | old seen | old unseen | old SMG | new subset | new val (S15) | best ep | new seen | new unseen | **new SMG** |",
          "|---|---|--:|--:|--:|--:|---|--:|--:|--:|--:|--:|"]
    for ko, kn in pairs:
        ds = range(5) if ko != 16 else [0]
        for d in ds:
            o, n = cell(ost, ko, d), cell(nst, kn, d)
            L.append(f"| k{ko}/{kn} d{d} | {old_subset(ko, d)} | {f(o['val'])} | {f(o['seen'])} | {f(o['unseen'])} | {f(o['smg'])} | "
                     f"{new_subset(kn, d)} | {f(n['val'])} | {n['ep'] if n['ep'] is not None else '--'} | {f(n['seen'])} | {f(n['unseen'])} | **{f(n['smg'])}** |")
    # cells whose subset is identical in both grids -> isolates the selection/val change
    same = []
    for ko, kn in pairs[:-1]:
        for d in range(5):
            if old_subset(ko, d) == new_subset(kn, d):
                o, n = cell(ost, ko, d), cell(nst, kn, d)
                if o["smg"] is not None and n["smg"] is not None:
                    same.append((ko, d, o["smg"], n["smg"]))
    L += ["", "## Same-subset cells (isolates the val/selection change from the re-drawn subsets)", "",
          "| cell | subset | old SMG | new SMG | Δ |", "|---|---|--:|--:|--:|"]
    for ko, d, a, b in same:
        L.append(f"| k{ko} d{d} | {new_subset(ko, d)} | {a:.2f} | {b:.2f} | {b - a:+.2f} |")
    if same:
        L.append(f"| mean | | {st.mean(x[2] for x in same):.2f} | {st.mean(x[3] for x in same):.2f} | "
                 f"{st.mean(x[3] - x[2] for x in same):+.2f} |")
    L += ["", "## What changes in P2", "",
          "- **Fig. 4b, BdSLW401 curve** (`tools/make_paper_figures.py` ~l.176): label `BdSLW401 (Bangla, 15)`; k `[2, 4, 8, 12, 15]`; "
          "SMG / SD from the 'new' line above.",
          "- **tab:smgk, BdSLW401 row**: k range `2--15`; SMG at min-k and max-k = the new k=2 and k=15 means.",
          "- **§div text** (\\S near Fig. 4b): \"83pp at k=2 ... with sixteen signers, drops to 12pp\" → the new k=2 value and \"with fifteen "
          "training signers, drops to\" the new k=15 value. The footnote's \"selects on the official VAL\" no longer holds for BdSLW401: "
          "it selects on the held-out signer S15 (BdSLW60/AUTSL/LSFB/ASL-Citizen curves unchanged).",
          "- Qualitative conclusion (monotone decay with signer count; BdSLW401 starts highest and ends near ~12–14pp) is unchanged; "
          "the k=4 point is now *higher* than old and k=2 lower, so the min-k endpoint shrinks by ~6pp and the max-k endpoint rises by ~2pp.",
          "- Unseen-TEST accuracy at the full pool drops slightly (86.19 → 84.34) with one signer fewer and S15-based selection."]
    return L


# ---------------------------------------------------------------- item 4
ARCHS = [("adaptive_gnn", "Adaptive GNN (CTR-GCN-style)", "bdsl_adaptive_gnn_si"),
         ("st_gcn_vanilla", "ST-GCN (vanilla)", "bdsl_st_gcn_vanilla_si"),
         ("slgtformer", "SLGTFormer", "bdsl_slgtformer_si"),
         ("gnn", "Attention GNN", "bdsl_gnn_si"),
         ("gnn_lstm", "GNN+LSTM", "bdsl_gnn_lstm_si"),
         ("pose_lstm", "Pose LSTM", "bdsl_pose_lstm_si"),
         ("gnn_transformer", "GNN+Transformer (clip 5.0)", "bdsl_gnn_transformer_si_clip5")]


def item4():
    L = ["# Item 4 — base-LR check for the 7 non-BlockGCN architectures (BdSLW60 Protocol A)", "",
         "Configs `config/lrsweep/<reported>_lr0p1.yaml` / `_lr0p001.yaml` = the reported `config/bdsl_<arch>_si.yaml` "
         "(GNN+Transformer: `config/bdsl_gnn_transformer_si_clip5.yaml`, grad clip 5.0 kept) with only `base_lr` changed; "
         "seed 0, 100 epochs, no grad clipping except GNN-T. Job (`scripts/hpc/slurm_lrsweep_7arch.sbatch`). "
         "Selection metric = best-epoch **validation** Top-1 on signer 15 (max over the every-5-epoch evals logged in "
         "results_final.csv); lr 0.01 = the existing reported seed-0 run (`<reported>_seed0`). No TEST scoring.", "",
         "| Architecture | val @ lr 0.1 | val @ lr 0.01 (reported) | val @ lr 0.001 | val argmax | 0.01 − best | flags |",
         "|---|--:|--:|--:|---|--:|---|"]
    notes, complete = [], True
    for key, name, stem in ARCHS:
        exps = {0.1: f"{stem}_lr0p1_seed0", 0.01: f"{stem}_seed0", 0.001: f"{stem}_lr0p001_seed0"}
        v = {lr: best(e) for lr, e in exps.items()}
        flags = []
        for lr, e in exps.items():
            n_ep = len({x for x, _ in rows.get(e, [])})
            if v[lr] is not None and n_ep < 20: flags.append(f"lr {lr}: only {n_ep}/20 evals")
            if nan_in_log(e): flags.append(f"lr {lr}: NaN loss")
            if v[lr] is not None and v[lr] < 5: flags.append(f"lr {lr}: chance-level ({v[lr]:.2f})")
            fe = final(e)
            if v[lr] is not None and fe is not None and v[lr] - fe > 10: flags.append(f"lr {lr}: final-epoch val {fe:.2f} (collapsed late)")
        have = {lr: x for lr, x in v.items() if x is not None}
        if len(have) == 3:
            am = max(have, key=have.get)
            gap = v[0.01] - have[am]
            if am != 0.01:
                notes.append(f"- **{name}**: val argmax is lr {am} ({have[am]:.2f}) vs lr 0.01 ({v[0.01]:.2f}), "
                             f"{have[am] - v[0.01]:.2f} pp better.")
            ams, gaps = f"{am}" + ("" if am == 0.01 else " **≠ 0.01**"), f"{gap:+.2f}"
        else:
            ams, gaps, complete = "pending", "--", False
        L.append(f"| {name} | {f(v[0.1])} | {f(v[0.01])} | {f(v[0.001])} | {ams} | {gaps} | {'; '.join(flags) or '—'} |")
    L += ["", "## Archs where 0.01 is not the validation argmax", ""] + (notes or (["- none: lr 0.01 is the val argmax for all 7 architectures."] if complete else ["- pending (not all runs finished)."]))
    L += ["", "## Seed-noise calibration (lr 0.01, existing reported runs)", "",
          "The 0.1-vs-0.01 comparison is single-seed. Val best Top-1 of the reported lr-0.01 runs over seeds 0/1/2 "
          "(val = 655 S15 clips, 1 clip = 0.15 pp); a gap smaller than the lr-0.01 seed spread is not a decision signal.", "",
          "| Architecture | lr 0.01 val seeds 0/1/2 | mean ± SD | lr-0.1 (seed 0) − lr-0.01 3-seed mean | final-epoch val 0.1 / 0.01 / 0.001 |",
          "|---|---|--:|--:|---|"]
    for key, name, stem in ARCHS:
        sv = [best(f"{stem}_seed{i}") for i in range(3)]
        have = [x for x in sv if x is not None]
        b01 = best(f"{stem}_lr0p1_seed0")
        d = f"{b01 - st.mean(have):+.2f}" if b01 is not None and have else "--"
        fe = " / ".join(f(final(e)) for e in (f"{stem}_lr0p1_seed0", f"{stem}_seed0", f"{stem}_lr0p001_seed0"))
        L.append(f"| {name} | {' / '.join(f(x) for x in sv)} | {ms(sv)} | {d} | {fe} |")
    L += ["", "## Divergence / NaN", "",
          "- **SLGTFormer lr 0.1 diverged.** Val 47.2 (ep 4) → 68.1 (ep 9) → 53.3 → collapse to 5.3 at ep 19 and stays at chance "
          "(9.5, 4.6); weight max-abs grows from 1.5e3 (ep-9 ckpt) to 4.5e13 (ep-19/29 ckpts). At epoch 35 the run crashed with a CUDA "
          "device-side assert in `torch.bernoulli` (`0 <= p <= 1`, i.e. NaN/inf probabilities in the DropT/DropSke masks) — "
          "a NaN failure, not an infrastructure fault (job_0 FAILED after 4h21m; all 6 other runs of that task finished). "
          "It was **not** resumed: the latest checkpoint (ep 29) is already blown up (|w| ~ 4e13, chance-level val), so finishing the "
          "remaining 65 epochs cannot change the decision (best pre-divergence val 68.09 vs 86.87 at lr 0.01). "
          "Resume command if a full-budget row is wanted anyway: `python tools/run_resumable_seed.py --config "
          "config/lrsweep/bdsl_slgtformer_si_lr0p1.yaml --seed 0 --exp bdsl_slgtformer_si_lr0p1_seed0`.",
          "- **GNN+Transformer lr 0.1 (grad clip 5.0) collapsed** to chance: best val 4.43 (ep 4), 1.5–1.7 thereafter; train loss stays at "
          "~ln 60 = 4.09 through epoch 100. No NaN in the log (clipping keeps it finite), but it never learns.",
          "- No NaN/divergence in any other run (all 13 others completed 100 epochs).",
          "", "Best-epoch per run (epoch index, 0-based, of the max logged val Top-1):", ""]
    for key, name, stem in ARCHS:
        L.append(f"- {name}: " + ", ".join(f"lr {lr}: ep {best_ep(e)}" for lr, e in
                                          ((0.1, f'{stem}_lr0p1_seed0'), (0.01, f'{stem}_seed0'), (0.001, f'{stem}_lr0p001_seed0'))))
    return L


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "item2_w401_smgk_sdval.md").write_text("\n".join(item2()) + "\n")
    (OUT / "item4_lr_sweep_7arch.md").write_text("\n".join(item4()) + "\n")
    print("wrote", OUT)
