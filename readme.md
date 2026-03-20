# ModelRepo — Git-like Versioning for ML Models

A lightweight, local-first version control system for machine learning models. Hash weights, track lineage, roll back to any version, diff metadata across experiments, and detect environment drift — all from the command line with no external services required.

Built entirely on Python's standard library + NumPy. No MLflow, no DVC, no cloud dependency.

---

## Table of Contents

- [Why ModelRepo](#why-modelrepo)
- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Command Reference](#command-reference)
- [File Structure](#file-structure)
- [Advanced Usage](#advanced-usage)
- [Phase Overview](#phase-overview)
- [Design Decisions](#design-decisions)
- [Roadmap](#roadmap)

---

## Why ModelRepo

ML experiments are hard to reproduce. The typical workflow looks like this:

```
model_v1.pt
model_v1_final.pt
model_v1_final_ACTUALLY_FINAL.pt
model_v2_lr001.pt
model_v2_lr001_retrained.pt
```

There is no record of which model came from which training run, what hyperparameters were used, which dataset it was trained on, or what the environment looked like at training time. When something breaks in production, you have no reliable way to roll back.

ModelRepo solves this by treating model weights exactly the way git treats source code — every version is immutable, content-addressed, and linked to its parent. You get a full audit trail, fast search across runs, and one-command rollback.

---

## How It Works

Every time you commit a model, ModelRepo:

1. **Hashes the weights file** (SHA-256, chunked for large files) and stores it in a content-addressable object store — identical weights are stored exactly once
2. **Creates a commit object** (a JSON blob) that records the weights hash, parent commit hash, message, timestamp, metadata (metrics, hyperparameters), and environment snapshot
3. **Stores the commit blob** in the same object store — the commit hash is both its ID and its integrity check
4. **Updates a SQLite index** for fast querying — search by accuracy, loss, dataset, date, or any metadata field without scanning blobs
5. **Advances HEAD** (or the current branch pointer) to the new commit

Rollback is just copying the weights blob for a given commit hash back to disk. Nothing is ever deleted.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        cli.py                           │
│              Command-line interface                      │
└────────────────────────┬────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
   ┌─────▼──────┐  ┌─────▼──────┐  ┌───▼────────┐
   │  commit.py │  │  index.py  │  │ env_lock.py│
   │  DAG layer │  │SQLite index│  │  Env lock  │
   └─────┬──────┘  └─────┬──────┘  └───┬────────┘
         │               │             │
         └───────────────┼─────────────┘
                         │
              ┌──────────▼──────────┐
              │   object_store.py   │
              │  Content-addressable│
              │    blob storage     │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │      dedup.py       │
              │  Chunk-level dedup  │
              └─────────────────────┘
```

### The six modules

| File | Phase | Responsibility |
|---|---|---|
| `object_store.py` | 1 | Immutable blob storage, SHA-256 hashing, content-addressable layout |
| `commit.py` | 2 | Commit objects, DAG traversal, branches, tags, HEAD, checkout |
| `index.py` | 3 | SQLite index for fast log, search, stats, lineage queries |
| `cli.py` | 4 | Full command-line interface |
| `dedup.py` | 5 | Chunk-level deduplication — stores only changed chunks across fine-tunes |
| `env_lock.py` | 6 | Environment snapshots, drift detection, venv restoration scripts |

---

## Installation

### Requirements

- Python 3.10+
- NumPy (`pip install numpy`)
- Everything else is Python stdlib (`hashlib`, `sqlite3`, `json`, `pathlib`, `subprocess`)

### Setup

```bash
# Clone the repo
git clone https://github.com/yourusername/modelrepo.git
cd modelrepo

# Install dependencies
pip install numpy

# Optional: add an alias so you can type `model` instead of `python cli.py`
# On Windows (PowerShell):
Set-Alias model "python $PWD\cli.py"

# On macOS/Linux (add to ~/.bashrc or ~/.zshrc):
alias model="python /path/to/cli.py"
```

### Verify installation

```bash
python cli.py --help
```

---

## Quick Start

```bash
# 1. Initialise the repo (creates ~/.modelrepo/)
python cli.py init

# 2. Train your model, then commit it
python cli.py commit model.pt \
  -m "baseline ResNet50" \
  --accuracy 0.87 --loss 0.42 --lr 0.01 --epochs 10 --dataset imagenet

# 3. Fine-tune and commit again
python cli.py commit model_v2.pt \
  -m "lower learning rate" \
  --accuracy 0.91 --loss 0.31 --lr 0.001 --epochs 10 --dataset imagenet

# 4. See the history
python cli.py log

# 5. Something broke — roll back instantly
python cli.py checkout <hash> --out restored.pt
```

---

## Command Reference

### `init`

Initialise a new model repo at `~/.modelrepo`. Safe to run multiple times.

```bash
python cli.py init
```

---

### `commit`

Store a weights file and create a new versioned commit.

```bash
python cli.py commit <weights_file> -m <message> [options]
```

**Options:**

| Flag | Type | Description |
|---|---|---|
| `-m`, `--message` | string | **Required.** Commit message |
| `--accuracy` | float | Accuracy metric |
| `--loss` | float | Loss metric |
| `--lr` | float | Learning rate |
| `--epochs` | int | Number of epochs |
| `--dataset` | string | Dataset identifier |
| `--model-arch` | string | Architecture name |
| `--meta KEY=VALUE` | repeatable | Any extra metadata |

**Examples:**

```bash
# Basic commit
python cli.py commit model.pt -m "baseline"

# With metrics
python cli.py commit model.pt -m "v2" --accuracy 0.94 --loss 0.21 --lr 0.001

# With extra metadata
python cli.py commit model.pt -m "v3" --accuracy 0.95 \
  --meta batch_size=64 --meta optimizer=adam --meta scheduler=cosine
```

---

### `log`

Show commit history, newest first.

```bash
python cli.py log [--branch <name>] [--limit N] [--json]
```

**Examples:**

```bash
python cli.py log                          # history from HEAD
python cli.py log --branch experiment-aug  # history from a branch
python cli.py log --limit 5               # last 5 commits
python cli.py log --json                  # machine-readable output
```

**Sample output:**

```
────────────────────────────────────────────────────────────────
  Log  HEAD  [main]
────────────────────────────────────────────────────────────────

  ●  bc60fc28  2026-03-18 10:44:11
     more epochs, imagenet-v2
     accuracy=0.94  loss=0.21  lr=0.001  epochs=20  dataset=imagenet-v2
     parent: ce46a82f...

  ○  ce46a82f  2026-03-18 10:43:55
     lower learning rate
     accuracy=0.91  loss=0.31  lr=0.001  epochs=10  dataset=imagenet
     parent: 51a238af...
```

---

### `show`

Print full details of a single commit.

```bash
python cli.py show <ref> [--json]
```

```bash
python cli.py show HEAD
python cli.py show bc60fc28
python cli.py show v1.0       # by tag name
python cli.py show experiment # by branch name
```

---

### `checkout`

Restore the weights file for a given commit to disk.

```bash
python cli.py checkout <ref> [--out <path>] [--detach]
```

| Flag | Description |
|---|---|
| `--out` | Destination path (default: `model_checkout.pt`) |
| `--detach` | Move HEAD directly to this commit without affecting branch pointer |

```bash
python cli.py checkout bc60fc28                    # restore to model_checkout.pt
python cli.py checkout bc60fc28 --out restored.pt  # custom output path
python cli.py checkout v1.0                        # checkout by tag
python cli.py checkout HEAD~                       # not yet supported — use hash
python cli.py checkout bc60fc28 --detach           # inspect without moving branch
```

---

### `diff`

Compare metadata and weights between two commits, branches, or tags.

```bash
python cli.py diff <ref_a> <ref_b> [--json]
```

```bash
python cli.py diff main experiment-aug
python cli.py diff bc60fc28 ce46a82f
python cli.py diff v1.0 HEAD
```

**Sample output:**

```
────────────────────────────────────────────────────────────────
  diff  bc60fc28  ↔  19ca9c39
────────────────────────────────────────────────────────────────
  A: more epochs, imagenet-v2  (2026-03-18 10:44:11)
  B: data augmentation         (2026-03-18 10:45:03)
  Ancestor: ce46a82f...
  Weights:  CHANGED

  Metadata changes:
    accuracy:  0.94  →  0.96
    epochs:    20    →  30
    loss:      0.21  →  0.15
────────────────────────────────────────────────────────────────
```

---

### `branch`

Create, list, switch, or delete branches.

```bash
python cli.py branch --list
python cli.py branch --create <name> [--from <ref>]
python cli.py branch --switch <name>
python cli.py branch --delete <name>
```

```bash
python cli.py branch --list
python cli.py branch --create experiment-lr --from HEAD
python cli.py branch --switch experiment-lr
python cli.py branch --delete experiment-lr
```

---

### `tag`

Create immutable named pointers to commits. Use for releases and production deployments.

```bash
python cli.py tag <name> [--ref <ref>] [--message <msg>]
python cli.py tag --list
```

```bash
python cli.py tag v1.0 --message "first production release"
python cli.py tag v1.1 --ref bc60fc28
python cli.py tag --list
```

---

### `search`

Search commits by metadata. All filters are optional and combine with AND logic.

```bash
python cli.py search [filters] [--limit N] [--json]
```

| Filter | Description |
|---|---|
| `--min-accuracy N` | accuracy >= N |
| `--max-loss N` | loss <= N |
| `--lr N` | exact learning rate match |
| `--dataset X` | exact dataset name match |
| `--model-arch X` | exact architecture match |
| `--message X` | case-insensitive substring in message |
| `--since ISO` | timestamp lower bound |
| `--until ISO` | timestamp upper bound |

```bash
python cli.py search --min-accuracy 0.92
python cli.py search --dataset imagenet-v2 --min-accuracy 0.90
python cli.py search --message "augmentation" --since 2026-01-01
python cli.py search --max-loss 0.25 --limit 10
```

---

### `best`

Show the top N commits ranked by a metric.

```bash
python cli.py best [--metric accuracy|loss|lr|epochs] [--top N] [--json]
```

```bash
python cli.py best --metric accuracy --top 5
python cli.py best --metric loss --top 3     # lower is better for loss
```

---

### `stats`

Aggregate statistics across the entire repo.

```bash
python cli.py stats
```

**Sample output:**

```
──────────────────────────────────────────────────
  Model Repo — Statistics
──────────────────────────────────────────────────
  Total commits    : 4
  Unique datasets  : 2
  Best accuracy    : 0.96
  Best loss        : 0.15
  Avg accuracy     : 0.9175
  First commit     : 2026-03-18 10:43:40
  Latest commit    : 2026-03-18 10:45:03

  Per-dataset best accuracy:
    imagenet-v2          best=0.96  (2 commits)
    imagenet             best=0.91  (2 commits)

  Branches:
    experiment-aug       → 19ca9c39...
    main                 → bc60fc28...
──────────────────────────────────────────────────
```

---

### `lineage`

Show the full ancestor chain of a commit back to the root.

```bash
python cli.py lineage <ref> [--json]
```

```bash
python cli.py lineage HEAD
python cli.py lineage experiment-aug
python cli.py lineage bc60fc28
```

---

### `rebuild`

Regenerate the SQLite index by scanning the object store. Use if the index is deleted, corrupted, or out of sync. The object store is always the source of truth — nothing is lost.

```bash
python cli.py rebuild
```

---

### `verify`

Re-hash every object in the store and report any corruption.

```bash
python cli.py verify
```

---

## File Structure

```
~/.modelrepo/
├── objects/              # Content-addressable blob store
│   ├── a3/               # First 2 chars of hash = subdirectory
│   │   └── f9b2c1d4...   # Remaining 62 chars = filename
│   └── ff/
│       └── 04ab8c2e...
├── refs/
│   ├── branches/
│   │   ├── main          # Contains commit hash
│   │   └── experiment    # Contains commit hash
│   └── tags/
│       └── v1.0          # Contains JSON: {commit, message}
├── HEAD                  # "ref: branches/main" or raw commit hash
└── index.db              # SQLite index (rebuildable)
```

---

## Advanced Usage

### Full pipeline with environment locking

Use `commit_with_env()` for production workflows — it combines chunked deduplication (Phase 5) with full environment snapshotting (Phase 6):

```python
from object_store import init_repo
from index import init_index
from env_lock import commit_with_env

init_repo()
init_index()

hash = commit_with_env(
    "model.pt",
    "fine-tuned on v3 dataset",
    metadata={
        "accuracy": 0.96,
        "loss":     0.14,
        "lr":       0.0001,
        "epochs":   30,
        "dataset":  "imagenet-v3",
    }
)
print(f"Committed: {hash[:8]}...")
```

### Detecting environment drift on checkout

```python
from env_lock import check_env_on_checkout

# After checking out a commit:
check_env_on_checkout(commit_hash)
```

Output if drift is detected:

```
[env] ⚠  Environment drift detected for commit bc60fc28...

  ──────────────────────────────────────────────────────
  Environment drift detected
  ──────────────────────────────────────────────────────
  ~  torch                        saved=2.1.0  →  current=2.3.1
  ✗  albumentations               saved=1.3.0  MISSING

  Summary: version changed: torch; missing: albumentations
  ──────────────────────────────────────────────────────

[env] To restore the exact environment:
      python env_lock.py restore bc60fc28 --venv
```

### Restoring an exact environment

```python
from env_lock import restore_env_for_commit

# Writes requirements_<hash>.txt and restore_env_<hash>.sh
restore_env_for_commit("bc60fc28", venv=True)
```

Then run the generated script:

```bash
bash restore_env_bc60fc28.sh
source .venv_bc60fc28/bin/activate
```

### Querying the index directly

```python
from index import search, best_model, accuracy_over_time, compare_datasets

# Find all runs with accuracy > 0.92 on a specific dataset
results = search(min_accuracy=0.92, dataset="imagenet-v2")

# Top 5 models by accuracy
top = best_model("accuracy", top_n=5)

# Accuracy trend over time (for plotting)
trend = accuracy_over_time()

# Best accuracy per dataset
by_dataset = compare_datasets()
```

### JSON output for scripting

Every command supports `--json` for piping into other tools:

```bash
python cli.py log --json | python -c "
import json, sys
commits = json.load(sys.stdin)
for c in commits:
    print(c['hash'][:8], c['accuracy'], c['message'])
"
```

---

## Phase Overview

The system was built in six incremental phases, each adding a layer on top of the previous:

### Phase 1 — Object Store (`object_store.py`)

Content-addressable blob storage. Every file is stored by its SHA-256 hash, split into a two-level directory structure (`objects/ab/cdef...`). Identical files are stored exactly once. Core operations: `store_object`, `retrieve_object`, `object_exists`, `verify_all`.

### Phase 2 — Commit Layer (`commit.py`)

Gives hashes meaning. A commit is a JSON blob connecting weights hash + parent hash + message + metadata + environment info. Supports full branch management (`create_branch`, `switch_branch`), immutable tags, `checkout` with environment drift warning, and `diff` between any two refs. HEAD can point to a branch (normal mode) or a raw hash (detached mode).

### Phase 3 — SQLite Index (`index.py`)

A fast query layer over the object store. Mirrors commit metadata into typed columns (`accuracy`, `loss`, `lr`, `epochs`, `dataset`) for SQL queries. The index is never the source of truth — `rebuild_index()` regenerates it entirely from the object store. Provides recursive CTE-based `fast_log`, `search`, `best_model`, `lineage`, and aggregate `stats`.

### Phase 4 — CLI (`cli.py`)

Thin argparse wrapper over the three layers. All business logic stays in the modules. Supports `--json` output on every command for scripting.

### Phase 5 — Deduplication (`dedup.py`)

Replaces whole-file storage with chunk-level storage. Files are split into 64 MB chunks, each chunk hashed and stored independently. A manifest records the chunk sequence. Fine-tuned models that share 90% of their weights with the base share 90% of their storage. Drop-in replacement for `store_object` / `retrieve_object`.

### Phase 6 — Environment Locking (`env_lock.py`)

Captures a full environment snapshot on every commit: Python version, `pip freeze` output, key library versions (PyTorch, NumPy, Transformers, etc.), CUDA info. Stored as a blob — immutable and content-addressed. On checkout, compares saved vs current environment and reports drift. Generates `requirements.txt` and venv creation scripts for exact environment restoration.

---

## Design Decisions

**Why SHA-256 and not MD5/SHA-1?**
SHA-256 is collision-resistant enough that two different models will never produce the same hash in practice. MD5 and SHA-1 have known collision vulnerabilities. The 64-character hex string is a small cost for the guarantee.

**Why store commits in the object store instead of a separate database?**
The same reason git does it — the commit hash is both its ID and its integrity check. If a commit blob is modified, its hash changes, and every child commit that references it becomes detectable as invalid. The commit graph is self-verifying.

**Why SQLite instead of just scanning blobs for queries?**
Scanning blobs for `log` or `search` requires loading and deserialising every commit object — O(n) blob reads for n commits. SQLite makes it O(1) for indexed queries. The index is a pure optimisation layer with no new information — `rebuild_index()` can regenerate it at any time.

**Why not use MLflow or DVC?**
MLflow requires a server. DVC requires git and a remote storage backend. ModelRepo is intentionally zero-dependency and single-machine — the entire system is six Python files. It's also a learning project: building it from scratch makes every design decision explicit.

**Why chunk at 64 MB instead of smaller?**
64 MB balances deduplication granularity against manifest overhead. Smaller chunks (e.g. 1 MB) give finer-grained dedup but produce large manifests and many small files. 64 MB works well for typical PyTorch model files where the natural unit of change is a layer (usually tens to hundreds of MB).

---

## Roadmap

- [ ] `pyproject.toml` — install as a proper `model` CLI command via `pip install -e .`
- [ ] Remote object store — swap local filesystem for S3, GCS, or Azure Blob Storage
- [ ] Tensor-level diff — compare actual weight values layer by layer using `safetensors`
- [ ] Web UI — Flask/FastAPI server rendering log, stats, and diff in a browser
- [ ] PyTorch/safetensors integration — canonical serialisation before hashing for framework-stable hashes
- [ ] Merge commits — combine two experimental branches back into main
- [ ] `model stash` — save work-in-progress weights without creating a full commit
- [ ] Export — package a specific commit (weights + requirements + config) into a portable bundle

---

## Running the Tests

Each module has a self-test at the bottom. Run them in order:

```bash
python object_store.py   # Phase 1
python commit.py         # Phase 2
python index.py          # Phase 3
python dedup.py          # Phase 5
python env_lock.py       # Phase 6
```

Each should print `All tests passed.` at the end.

---

## License

MIT License. See `LICENSE` for details.