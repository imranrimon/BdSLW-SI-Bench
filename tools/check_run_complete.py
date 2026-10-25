#!/usr/bin/env python
"""Completeness check for main.py training runs, and a sweep of every run that feeds P2.

A main.py training run is COMPLETE when its log reached the final epoch of its own saved
config (work_dir/<run>/config.yaml num_epoch): main.py always evaluates at the last epoch,
so `max("Eval epoch: N") == num_epoch` (fallback: the final-epoch checkpoint
`<run>_model-<num_epoch-1>.pt` exists). Checkpoints are pruned to best/last in this repo,
so checkpoint presence alone is not a completeness signal.

A `--phase test` run (TEST-split re-score of a training checkpoint) is traced to its training
run through the `Load weights from <ckpt>.` line of its log; it is flagged when that training
run is incomplete, when the scored checkpoint is not `<run>_model-best.pt` (best-on-val
selection not applied), or when the score dump is older than the checkpoint (stale).

Read errors are fatal (no silent skips; see docs/scratch_outage_2026-10-13/).

  python tools/check_run_complete.py work_dir/bdsl_slgtformer_si_seed2 [...]    # one-line status each
  python tools/check_run_complete.py --p2-sweep --md docs/p2_rerun_fixes_2026-10/incomplete_runs.md
  python tools/check_run_complete.py --loso-csv docs/loso_heldout/slgtformer.csv   # fold/seed coverage

Library use (held-out LOSO gate): `from check_run_complete import train_status`;
`train_status(wd)["complete"]`.
"""
import argparse
import collections
import csv
import glob
import json
import os
import re
import sys
import time

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WD = os.path.join(ROOT, "work_dir")
EVAL_RE = re.compile(r"\] Eval epoch: (\d+)")
TRAIN_RE = re.compile(r"\] Training epoch: (\d+)")
LOAD_RE = re.compile(r"Load weights from (.+?)\.?\s*$", re.M)
NAN_RE = re.compile(r"Loss: nan", re.I)
EVAL_TOP1_RE = re.compile(r"\] Eval epoch: (\d+)\n(?:.*\n){0,3}?.*\tTop1: ([\d.]+)%")


def val_curve(log):
    """{epoch (1-based): val Top-1 %}; a resumed run's later evals override earlier ones."""
    return {int(e): float(a) for e, a in EVAL_TOP1_RE.findall(log)}


def best_val_epoch(curve):
    """main.py keeps the FIRST epoch reaching the max (strict >)."""
    if not curve:
        return None
    m = max(curve.values())
    return min(e for e, a in curve.items() if a == m)


def _read(path):
    with open(path, errors="replace") as f:   # OSError (EIO) propagates on purpose
        return f.read()


def local_dir(path):
    """Map a logged checkpoint path (Windows / old /users clone / relative) to work_dir/<run>."""
    p = path.replace("\\", "/").rstrip("/")
    parts = p.split("/")
    if "work_dir" in parts:
        i = len(parts) - 1 - parts[::-1].index("work_dir")
        if i + 1 < len(parts):
            return os.path.join(WD, parts[i + 1])
    return os.path.join(WD, os.path.basename(os.path.dirname(p)))


def ckpts(wd):
    out = {}
    for p in glob.glob(os.path.join(wd, "*_model-*.pt")):
        m = re.search(r"_model-(\d+)\.pt$", p)
        if m:
            out[int(m.group(1))] = p
    return out


def train_status(wd):
    """Status dict for one main.py work_dir (train or test phase)."""
    wd = os.path.abspath(wd)
    st = dict(run=os.path.basename(wd), wd=wd, exists=os.path.isdir(wd))
    if not st["exists"]:
        return dict(st, status="MISSING", complete=False)
    cfgp = os.path.join(wd, "config.yaml")
    if not os.path.isfile(cfgp):
        # score-only dir written by an external evaluator (eval_results/best_acc.pkl, no config/log):
        # judge the training run of the same name without the _test suffix
        m = re.match(r"^(.*_seed\d+)_test$", st["run"])
        if m and os.path.isfile(os.path.join(wd, "eval_results", "best_acc.pkl")):
            cands = [m.group(1), re.sub(r"_seed(\d+)$", r"_scratch_seed\1", m.group(1))]  # lr01 Protocol A: *_lr01_scratch_seed<S>
            par = next((c for c in cands if os.path.isdir(os.path.join(os.path.dirname(wd), c))), cands[0])
            ps = train_status(os.path.join(os.path.dirname(wd), par))
            return dict(st, phase="test", status="SCORE_ONLY(parent " + ps["status"] + ")", parent=par,
                        parent_status=ps, complete=ps.get("complete"), flags=[])
        return dict(st, status="NO_CONFIG", complete=None)
    cfg = yaml.safe_load(_read(cfgp)) or {}
    st["phase"] = cfg.get("phase", "train")
    st["num_epoch"] = int(cfg.get("num_epoch", 0) or 0)
    logp = os.path.join(wd, "log.txt")
    log = _read(logp) if os.path.isfile(logp) else ""
    st["log_mtime"] = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(logp))) if log else ""
    loads = LOAD_RE.findall(log)
    if st["phase"] == "test" and loads and os.path.normpath(local_dir(loads[-1].strip())) == os.path.normpath(wd):
        # a --phase test re-score run in place over a training dir (config.yaml overwritten): judge the training log
        st["phase"] = "train"
        st["note"] = "config.yaml overwritten by an in-place --phase test run"
    if st["phase"] == "test":
        st["weights"] = loads[-1].strip() if loads else None
        pk = os.path.join(wd, "eval_results", "best_acc.pkl")
        st["has_score"] = os.path.isfile(pk)
        if not st["weights"]:
            return dict(st, status="TEST_NO_WEIGHTS_LINE", complete=None)
        parent = local_dir(st["weights"])
        st["parent"] = os.path.basename(parent)
        st["uses_best"] = st["weights"].replace("\\", "/").endswith("_model-best.pt")
        ps = train_status(parent)
        st["parent_status"] = ps
        lw = os.path.join(parent, os.path.basename(st["weights"].replace("\\", "/")))
        st["stale"] = bool(st["has_score"] and os.path.isfile(lw)
                           and os.path.getmtime(pk) < os.path.getmtime(lw))
        flags = []
        if ps.get("complete") is False:
            flags.append(f"PARENT_{ps['status']}")
        if not st["uses_best"]:
            flags.append("NOT_BEST_CKPT")
        if st["stale"]:
            flags.append("STALE")
        if not st["has_score"]:
            flags.append("NO_SCORE")
        m = re.search(r"_model-(\d+)\.pt$", st["weights"])
        curve = val_curve(_read(os.path.join(parent, "log.txt"))) if os.path.isfile(os.path.join(parent, "log.txt")) else {}
        if not st["uses_best"] and m and curve:
            ep = int(m.group(1)) + 1
            be = best_val_epoch(curve)
            st["scored_epoch"], st["best_epoch"] = ep, be
            st["scored_val"], st["best_val"] = curve.get(ep), curve[be]
            if curve.get(ep) == curve[be]:
                flags.remove("NOT_BEST_CKPT")   # the retained per-epoch checkpoint IS the best-val epoch
                st["uses_best"] = True
            else:
                flags[flags.index("NOT_BEST_CKPT")] = (f"NOT_BEST_CKPT(scored ep{ep} val {curve.get(ep)}; "
                                                       f"best ep{be} val {curve[be]})")
        st["flags"] = flags
        st["complete"] = ps.get("complete")
        return dict(st, status="TEST_OK" if not flags else "TEST_" + "+".join(flags))
    ev = [int(x) for x in EVAL_RE.findall(log)]
    tr = [int(x) for x in TRAIN_RE.findall(log)]
    ck = ckpts(wd)
    st.update(last_eval=max(ev) if ev else 0, last_train=max(tr) if tr else 0,
              n_eval=len(ev), ckpt_epochs=sorted(ck), last_ckpt=max(ck) if ck else None,
              has_best=os.path.isfile(os.path.join(wd, f"{st['run']}_model-best.pt")),
              nan=bool(NAN_RE.search(log)))
    n = st["num_epoch"]
    st["complete"] = bool(n) and (st["last_eval"] >= n or (n - 1) in ck)
    if st["complete"]:
        st["status"] = "COMPLETE"
    elif not log and not ck:
        st["status"] = "EMPTY"
    else:
        st["status"] = "INCOMPLETE"
        if log and time.time() - os.path.getmtime(logp) < 1800:
            st["status"] = "IN_PROGRESS"   # log written in the last 30 min: a job is (probably) training it
    # resume point (tools/run_resumable_seed.py semantics): latest checkpoint + 1, else fresh
    st["resume_from"] = (max(ck) + 1) if ck else 0
    return st


def short(st):
    if st.get("phase") == "test":
        ps = st.get("parent_status", {})
        return (f"{st['run']}: {st['status']}  weights={os.path.basename(str(st.get('weights')))}"
                f"  parent={st.get('parent')} [{ps.get('status')} {ps.get('last_eval', '?')}/{ps.get('num_epoch', '?')}]")
    if st["status"] in ("MISSING", "NO_CONFIG"):
        return f"{st['run']}: {st['status']}"
    return (f"{st['run']}: {st['status']}  eval {st['last_eval']}/{st['num_epoch']}  train-ep {st['last_train']}"
            f"  ckpts {st['ckpt_epochs']}  best.pt {'y' if st['has_best'] else 'n'}"
            f"{'  NaN-loss' if st['nan'] else ''}")


# =============================================================================
# P2 enumeration
# =============================================================================
LOSO_ARCHES = ["adaptive_gnn", "block_gcn", "st_gcn_vanilla", "slgtformer", "gnn_lstm",
               "gnn_transformer", "pose_lstm", "gnn"]   # longest-prefix match below


def loso_run_stem(csv_name):
    """docs/loso_heldout/<csv_name>.csv -> run-name format with {f},{s}."""
    for a in sorted(LOSO_ARCHES, key=len, reverse=True):
        if csv_name == a or csv_name.startswith(a + "_"):
            suf = csv_name[len(a):]
            return f"bdsl_{a}_si{suf}_loso_test{{f}}_seed{{s}}"
    raise ValueError(csv_name)


def loso_cells(csv_name):
    """{(fold, seed): heldout acc or None} from the CSV, plus every dir on disk of that family."""
    stem = loso_run_stem(csv_name)
    cells = {}
    path = os.path.join(ROOT, "docs/loso_heldout", csv_name + ".csv")
    if os.path.isfile(path):
        with open(path) as f:
            for r in csv.DictReader(f):
                cells[(r["fold"].zfill(2), int(r["seed"]))] = r["test_acc"] or None
    pat = re.compile("^" + re.escape(stem).replace(r"\{f\}", r"(\d+)").replace(r"\{s\}", r"(\d+)") + "$")
    for d in os.listdir(WD):
        m = pat.match(d)
        if m:
            cells.setdefault((m.group(1).zfill(2), int(m.group(2))), "NOT_IN_CSV")
    return stem, cells


def _csv_workdirs():
    latest = {}
    with open(os.path.join(ROOT, "results_final.csv"), newline="") as f:
        for i, r in enumerate(csv.DictReader(f)):
            latest[r["Experiment"]] = (r.get("WorkDir") or "").replace("\\", "/")
    return latest


def p2_sources():
    """[(table-use label, run dir)] for every run the P2 table generator reads."""
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    cwd = os.getcwd()
    import p2_make_tables as T   # chdirs to ROOT
    os.chdir(cwd)
    wdmap = _csv_workdirs()
    out = []

    def from_src(label, src):
        k = src["kind"]
        seeds = src.get("seeds", [0, 1, 2])
        if k == "csv":
            for s in seeds:
                e = src["exp"].format(s=s)
                w = wdmap.get(e)
                out.append((label, os.path.join(WD, os.path.basename(w)) if w else os.path.join(WD, e)))
        elif k == "suite":
            for s in seeds:
                out.append((label, os.path.join(WD, f"{src['stem']}_seed{s}_test")))
        elif k in ("pkl", "txt"):
            for s in seeds:
                p = src["files"].format(s=s)
                if p.startswith("work_dir/"):
                    out.append((label, os.path.join(ROOT, os.path.dirname(p)).replace("/eval_results", "")))
                else:
                    m = re.search(r"r5_(\w+?)_sdtest_seed(\d)", p)   # docs/audit_reruns_2026-10/r5_mixed_selection.md
                    if m:
                        out.append((label, os.path.join(WD, f"bdsl_{m.group(1)}_sd_seed{m.group(2)}")))
        elif k == "jsonl":
            with open(os.path.join(ROOT, src["path"])) as f:
                for line in f:
                    o = json.loads(line) if line.strip() else {}
                    if o.get("ckpt"):
                        out.append((label, local_dir(o["ckpt"])))
            if "sd.jsonl" in src["path"]:   # docs/blockgcn_lr01_switch: sd runs bdsl_block_gcn_sd_lr01_seed<S>
                for s in seeds:
                    out.append((label, os.path.join(WD, f"bdsl_block_gcn_sd_lr01_seed{s}")))
        elif k == "loso":
            stem, cells = loso_cells(src["file"])
            for (f_, s) in sorted(cells):
                out.append((label, os.path.join(WD, stem.format(f=f_, s=s))))

    for arch, spec in T.POSE.items():
        active = spec.get("recipe")
        for metric, src in spec["src"].items():
            from_src(f"P2 {arch}.{metric}", src)
        for rname, rsrc in (spec.get("recipes") or {}).items():
            tag = "ACTIVE recipe" if rname == active else "pending recipe"
            for metric, src in rsrc.items():
                from_src(f"P2 {arch}.{metric} [{tag} {rname}]", src)
    for row in T.ARCH_B:
        for metric in ("test", "f1", "mixed"):
            from_src(f"P2 tab:arch(b) {row['key']}.{metric}", row[metric])
    # tab:arch(a) non-pose rows: JSON summaries of main.py runs (docs/baselines_rerun)
    for fam, lab in [("bdsl_attn_bilstm_author_*_seed[0-9]", "P2 tab:arch(a) attn_bilstm_author"),
                     ("bdsl_attn_bilstm_author_si_test_seed[0-9]", "P2 tab:arch(a) attn_bilstm_author"),
                     ("bdsl_spoter_si_seed[0-9]", "P2 tab:arch(a) spoter_bdsl"),
                     ("bdsl_spoter_sd_seed[0-9]", "P2 tab:arch(a) spoter_bdsl"),
                     ("bdsl_spoter_orig_si_seed[0-9]", "P2 tab:arch(a) spoter_orig"),
                     ("bdsl_spoter_orig_si_test_seed[0-9]", "P2 tab:arch(a) spoter_orig"),
                     ("bdsl_spoter_orig_sd_seed[0-9]", "P2 tab:arch(a) spoter_orig"),
                     ("bdsl_spoter_si_test_seed[0-9]", "P2 tab:arch(a) spoter_bdsl")]:
        for d in sorted(glob.glob(os.path.join(WD, fam))):
            out.append((lab, d))
    # tab:sig (A) per-instance: Protocol-A TEST dumps (tools/arch_significance._pkl_for)
    import arch_significance as AS
    for a in AS.ARCHES:
        for s in AS.SEEDS:
            out.append(("P2 tab:sig(A)", os.path.join(WD, f"bdsl_{AS.PKL_STEM.get(a, a + '_si')}_seed{s}_test")))
    return out


# Prose / appendix runs that P2 cites outside the generated tables (audit list + recipe controls).
P2_PROSE_FAMILIES = [
    ("P2 prose: grad-clip recipe control (clip 1 / clip 5)", "bdsl_*_si_clip[15]_seed[0-9]"),
    ("P2 prose: grad-clip recipe control TEST", "bdsl_*_si_clip[15]_seed[0-9]_test"),
    ("P2 app:slgtabl SLGTFormer ablations", "bdsl_no_lgrpe*_seed[0-9]*"),
    ("P2 app:slgtabl SLGTFormer ablations", "bdsl_no_ttsa*_seed[0-9]*"),
    ("P2 app:slgtabl SLGTFormer ablations", "bdsl_no_paf*_seed[0-9]*"),
    ("P2 app:slgtabl SLGTFormer ablations", "bdsl_slgtformer_*_seed[0-9]*"),
    ("P2 mirror ablation", "bdsl_*_si_mirror*_seed[0-9]*"),
    ("P2 Protocol-A seeds (all pose archs)", "bdsl_*_si_seed[0-9]"),
    ("P2 bone stream", "bdsl_*_bone*_seed[0-9]*"),
]


def p2_runs():
    """OrderedDict run_dir -> sorted list of P2 uses (tables first, then prose families)."""
    uses = collections.defaultdict(set)
    for lab, d in p2_sources():
        uses[os.path.normpath(d)].add(lab)
    for lab, pat in P2_PROSE_FAMILIES:
        for d in glob.glob(os.path.join(WD, pat)):
            if ".partial_" in d or ".bak" in d or not os.path.isdir(d):
                continue
            uses[os.path.normpath(d)].add(lab)
    return collections.OrderedDict(sorted((d, sorted(u)) for d, u in uses.items()))


def all_train_dirs():
    for d in sorted(os.listdir(WD)):
        p = os.path.join(WD, d)
        if os.path.isdir(p) and not d.startswith("_") and ".partial_" not in d and os.path.isfile(os.path.join(p, "config.yaml")):
            yield p


def sweep_md(out_md):
    runs = p2_runs()
    rows, test_rows, seen_train = [], [], set()
    for d, uses in runs.items():
        st = train_status(d)
        if st.get("phase") == "test":
            ps = st.get("parent_status") or {}
            if ps.get("wd"):
                seen_train.add(os.path.normpath(ps["wd"]))
                if not ps.get("complete"):
                    rows.append((ps, uses + [f"via TEST dir {st['run']}"]))
            if st["status"] != "TEST_OK" and not (st["status"].startswith("SCORE_ONLY") and ps.get("complete")):
                test_rows.append((st, uses))
        else:
            seen_train.add(os.path.normpath(d))
            if st.get("complete") is not True:
                rows.append((st, uses))
    # merge duplicate training rows
    merged = collections.OrderedDict()
    for st, uses in rows:
        k = st.get("wd", st["run"])
        merged.setdefault(k, [st, set()])[1].update(uses)
    # everything else in work_dir
    other = []
    n_all = 0
    for d in all_train_dirs():
        n_all += 1
        if os.path.normpath(d) in seen_train or os.path.normpath(d) in runs:
            continue
        st = train_status(d)
        if st.get("phase") == "train" and st.get("complete") is False:
            other.append(st)

    L = [f"# Incomplete-run sweep ({time.strftime('%Y-%m-%d %H:%M')})", "",
         "Generated by `python tools/check_run_complete.py --p2-sweep --md <this file>`.",
         "Rule: a main.py training run is complete when `max(Eval epoch) == num_epoch` of its saved "
         "`config.yaml` (or the final-epoch checkpoint exists). A `--phase test` dir is traced to its training "
         "run through the `Load weights from` line of its log.", "",
         f"P2-linked run dirs checked: {len(runs)} (generated-table sources via `tools/p2_make_tables.py` SPEC, "
         f"held-out LOSO CSV cells + every LOSO dir of the same family, tab:sig TEST dumps, baseline-rerun JSON runs, "
         f"and the prose families listed at the end). All main.py dirs in work_dir scanned: {n_all}.", "",
         "## 1. Incomplete TRAINING runs linked to P2", "",
         "| run | status | eval epoch / num_epoch | last train epoch | checkpoints | resume from | P2 uses |",
         "|---|---|---|---|---|---|---|"]
    for k, (st, uses) in sorted(merged.items(), key=lambda kv: kv[1][0]["run"]):
        L.append(f"| `{st['run']}` | {st['status']}{' (NaN loss)' if st.get('nan') else ''} | "
                 f"{st.get('last_eval', '-')}/{st.get('num_epoch', '-')} | {st.get('last_train', '-')} | "
                 f"{st.get('ckpt_epochs', '-')} | "
                 f"{('epoch ' + str(st['resume_from'])) if st.get('ckpt_epochs') else 'FRESH (no ckpt)'} | "
                 f"{'; '.join(sorted(uses))} |")
    L += ["", f"Total: {len(merged)}", "",
          "## 2. TEST-phase dirs with a problem (incomplete parent / non-best checkpoint / stale / no score)", "",
          "| test dir | flags | scored checkpoint | parent status | P2 uses |", "|---|---|---|---|---|"]
    for st, uses in sorted(test_rows, key=lambda x: x[0]["run"]):
        ps = st.get("parent_status") or {}
        L.append(f"| `{st['run']}` | {' '.join(st.get('flags', [st['status']]))} | "
                 f"`{os.path.basename(str(st.get('weights')))}` | {ps.get('status')} "
                 f"{ps.get('last_eval', '?')}/{ps.get('num_epoch', '?')} | {'; '.join(uses)} |")
    L += ["", f"Total: {len(test_rows)}", "",
          "## 3. Other incomplete main.py training runs in work_dir (not traced to P2)", "",
          "| run | eval epoch / num_epoch | checkpoints | log last modified |", "|---|---|---|---|"]
    for st in other:
        L.append(f"| `{st['run']}` | {st['last_eval']}/{st['num_epoch']} | {st['ckpt_epochs']} | {st['log_mtime']} |")
    L += ["", f"Total: {len(other)}", "", "## Prose families scanned", ""]
    L += [f"- {lab}: `work_dir/{pat}`" for lab, pat in P2_PROSE_FAMILIES]
    txt = "\n".join(L) + "\n"
    if out_md:
        os.makedirs(os.path.dirname(os.path.abspath(out_md)), exist_ok=True)
        open(out_md, "w").write(txt)
    print(txt)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="*")
    ap.add_argument("--p2-sweep", action="store_true")
    ap.add_argument("--md")
    ap.add_argument("--loso-csv", nargs="*", help="docs/loso_heldout/<name>.csv: per-cell completeness")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    rc = 0
    for d in a.dirs:
        st = train_status(d)
        print(json.dumps({k: v for k, v in st.items() if k != "parent_status"}, default=str) if a.json else short(st))
        rc |= 0 if st.get("complete") else 1
    for c in a.loso_csv or []:
        name = os.path.splitext(os.path.basename(c))[0]
        stem, cells = loso_cells(name)
        bad = 0
        for (f_, s), acc in sorted(cells.items()):
            st = train_status(os.path.join(WD, stem.format(f=f_, s=s)))
            if not st.get("complete"):
                bad += 1
                print(f"  {name}: {short(st)}  csv_acc={acc}")
        # staleness: a CSV written while a fold run was still training scores an older model-best.pt
        csvp = os.path.join(ROOT, "docs/loso_heldout", name + ".csv")
        stale = [stem.format(f=f_, s=s) for (f_, s) in cells
                 if os.path.isfile(csvp) and os.path.isfile(os.path.join(WD, stem.format(f=f_, s=s), stem.format(f=f_, s=s) + "_model-best.pt"))
                 and os.path.getmtime(os.path.join(WD, stem.format(f=f_, s=s), stem.format(f=f_, s=s) + "_model-best.pt")) > os.path.getmtime(csvp)]
        for r in stale:
            print(f"  {name}: STALE (model-best.pt newer than the CSV) {r}")
        print(f"{name}: {len(cells) - bad}/{len(cells)} cells complete, {len(stale)} stale")
        bad += len(stale)
        rc |= 1 if bad else 0
    if a.p2_sweep:
        sweep_md(a.md)
    sys.exit(rc)


if __name__ == "__main__":
    main()
