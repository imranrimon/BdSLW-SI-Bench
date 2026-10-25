# P2 results tables: generated, not hand-typed

`tools/p2_make_tables.py` writes the body rows of the P2 results tables from the
result files. Both paper copies pull them in with `\input{generated/<name>}`:
`paper_p2_overleaf/main.tex` and `paper_p2_ieee/main.tex`. Captions, column
headers and `\multicolumn` group labels stay in `main.tex`. The numbers do not.

```bash
python tools/p2_make_tables.py               # regenerate generated/*.tex in BOTH paper dirs
python tools/p2_make_tables.py --check       # diff the paper against the generator, cell by cell (exit 1 on mismatch)
python tools/p2_make_tables.py --check --rerun-sig    # also re-run tools/arch_significance.py against tab:sig
python tools/p2_make_tables.py --check --live         # preview: ignore pins, show the latest data everywhere
python tools/p2_make_tables.py --check --use-recipe adaptive_gnn=lr0p1   # preview a recipe switch
python tools/p2_make_tables.py --dry-run --provenance # print every cell with its source file
```

**Overleaf:** after regenerating, upload `paper_p2_overleaf/generated/` (or
`paper_p2_ieee/generated/`) next to `main.tex`. LaTeX cannot find the rows
without that folder.

## What is generated

| fragment | table |
|---|---|
| `tab_arch_a_{pose,rgb,foundation,prior_rerun,prior_reported}`, `tab_arch_b_w401` | Table 7 / `tab:arch`, panels (a) and (b) |
| `tab_smg8_rows` | `tab:smg8` |
| `tab_eff_rows` | efficiency table (§eff) |
| `tab_suite_rows` | `tab:suite` |
| `tab_bone_rows` | `tab:bone` |
| `tab_losoboot_rows`, `tab_losofolds_rows`, `tab_losovar_rows` | LOSO bootstrap table (§loso), `tab:losofolds`, `tab:losovar` |
| `tab_sig_a_rows`, `tab_sig_b_rows` | `tab:sig` |

Prose numbers are **not** generated. When a table changes, grep the prose for the old value.

## Where the numbers come from

All sources are declared in the `SPEC` section at the top of the script:

- `POSE`: the eight pose architectures. They feed tab:arch (a), smg8, eff, suite and bone, so one change updates every table.
- `ARCH_A_OTHER`: RGB, foundation and prior-method re-runs (`docs/baselines_rerun/*.json`).
- `ARCH_B`: BdSLW401 with S15-selected checkpoints (`docs/w401_sdval/`).
- `docs/p2_tables/manual_values.yaml`: values that have no result file (Params/GFLOPs and the "as reported" literature rows). Each entry has a comment saying where it came from.

Source kinds:

- `CSV(exp)`: latest `results_final.csv` row per seed.
- `SUITE(stem)`: `tools/p2_metric_suite.metrics_for`.
- `PKL(files, labels)`: score dumps.
- `JSONL`, `JSON`, `TXT` (`top1=` files).
- `LOSO(name)`: `docs/loso_heldout/<name>.csv`, the held-out signer. Never the val-signer LOSO rows in the CSV.

Conventions (the same ones the paper uses):

- ± is the population SD over seeds.
- LOSO is the fold mean over seeds, then the mean ± *sample* SD over the 11 folds.
- SMG is the printed Mixed minus the printed TEST, so a reader can check it.
- Pose rows are sorted by TEST, and `tab:smg8` by SMG.
- **Bold** is computed: the column maximum within the group.

## Add a row

1. Add an entry to the right list in the SPEC. Give it a display name exactly as it should appear in LaTeX, with sources for `test` / `f1` / `mixed` (plus `suite` and `loso` for a `POSE` architecture).
2. If a value has no result file, put it in `manual_values.yaml` with a source comment.
3. Run `python tools/p2_make_tables.py`, then `--check`.

A row in a group that has no fragment yet needs a new `FRAGMENTS` entry: an anchor string in the table plus the line that precedes the rows. It also needs a matching `BUILDERS` entry and an `\input{generated/<name>}` line in **both** `main.tex` files.

## Switch a row's recipe (e.g. Adaptive GNN / Attention GNN to lr 0.1)

The alternative sources are already declared under `POSE[arch]["recipes"]`:

- `lr0p1` for `adaptive_gnn` and `gnn`
- `mixed_noclip` for `slgtformer`, the unclipped SD run from job `sd_noclip_slgt`

To switch:

1. Preview it: `python tools/p2_make_tables.py --check --use-recipe adaptive_gnn=lr0p1`. The script refuses ("cannot generate ...") while any seed or source of the recipe is missing. The "Pending recipes" block of `--check` shows what is ready.
2. Make it permanent: set `"recipe": "lr0p1"` in `POSE["adaptive_gnn"]`, then regenerate.
   - Pins never apply to a row running on a non-base recipe.
   - Rows re-sort and bold moves automatically.
3. Update the prose and captions by hand. Check: the 79.07 / 78.09 mentions, the "top six" significance claims, the efficiency text, and the smg8 range "+19.7 to +32.6".
4. Re-run `tools/arch_significance.py --lr0p1` and save its output. Then point `SIG_FILE` at the new output, since tab:sig is read from that file.

For SLGTFormer, once `bdsl_slgtformer_sd_noclip_test_seed{0,1,2}` are all in `results_final.csv`:

1. Set `"recipe": "mixed_noclip"` and drop the TODO `comments`.
2. Remove the TODO comments in the captions and in the "Single signer-dependent split" paragraph.

## Pins (paper value kept on purpose)

`POSE[...]["pins"]` and the `pins` of other rows keep the value currently printed when the data disagrees. `--check` lists every pin whose live value differs, in two groups:

- **data newer than paper**: for example, the Adaptive GNN suite was computed from 2 seeds before the seed-2 dump landed.
- **rounding**: for example, GNN+Transformer 75.55 vs the exact 75.56.

To publish the newer value, delete the pin and regenerate. `--live` previews all pins removed at once.
