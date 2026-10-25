# Contributing

Issues and pull requests are welcome. Please keep changes focused and reproducible.

## Development setup

```bash
conda env create -f environment.yml && conda activate bdsl_graph   # or: pip install -r requirements.txt
make setup-dev        # pytest, ruff, pre-commit hook
make check            # lint + tests + repository validation + table check (what CI runs)
```

## Guidelines

- **New models** must accept `(N, C, T, V, M)` input and return `(N, num_class)` logits, and be
  constructible as `Model(num_class, num_point, num_person, graph, graph_args, in_channels, ...)`.
  Add a case to `tests/test_project_smoke.py::test_dummy_forward_pass_per_model_family`.
- **New config keys** must be declared as `main.py` argparse flags first; unknown YAML keys
  are rejected.
- **Configs are loaded by dotted path** (`model: model.block_gcn.Model`,
  `feeder: feeders.feeder.Feeder`); keep module paths stable or update every config.
- **Evaluation protocol:** select checkpoints and hyper-parameters on validation and report on
  TEST; never tune on the reported split. Use the signer-independent split for headline numbers.
- **Results in tables are generated, never typed:** change a row's source in the `SPEC` of
  `tools/p2_make_tables.py`, then run `make tables-check`.
- **Data hygiene:** never commit data, checkpoints, score dumps or anything derived from the
  source videos (`.gitignore` covers `data/`, `work_dir/`, `*.npy`, `*.pkl`, `*.pt`).
- Tests must run on CPU without data (skip cleanly when a dataset is absent).
