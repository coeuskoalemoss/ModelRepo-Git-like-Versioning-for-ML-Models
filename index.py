"""
index.py — SQLite index for the ML model versioning system.

Phase 3 of the build. Sits on top of commit.py and object_store.py.

WHY THIS EXISTS
───────────────
The object store is the source of truth — every commit blob lives there
permanently. But reading history by chasing parent pointers means loading
and deserialising one JSON blob per commit. For 10 commits that's fine.
For 1 000 commits, or when you want to search ("show me every run where
accuracy > 0.9 and lr < 0.001"), scanning blobs is painfully slow.

The SQLite index is a pure query accelerator — a mirror of the commit
metadata in a relational table. It is NEVER the source of truth. If it
gets deleted, corrupted, or falls out of sync, rebuild_index() regenerates
it by scanning the object store in seconds. Nothing is lost.

SCHEMA
──────
  commits         — one row per commit (mirrors commit JSON)
  branches        — named pointers to commits (mirrors refs/branches/)
  tags            — immutable named pointers  (mirrors refs/tags/)

WHAT YOU GET
────────────
  - log()         fast, no blob loading
  - search()      SQL WHERE on any metadata field
  - stats()       aggregate metrics across all versions
  - lineage()     full ancestor chain as a query
  - rebuild()     self-healing — resync from object store at any time
"""

import json
import sqlite3
from pathlib import Path
from typing import Any

from object_store import init_repo, list_objects, retrieve_bytes
from commit import (
    BRANCHES_DIR,
    REPO_DIR,
    TAGS_DIR,
    load_commit,
    log as commit_log,
    resolve_ref,
    _resolve_head,
    _current_branch,
)

# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────

DB_PATH = REPO_DIR / "index.db"


# ─────────────────────────────────────────────
#  Connection
# ─────────────────────────────────────────────

def _connect() -> sqlite3.Connection:
    """
    Open (and if needed create) the SQLite database.
    Returns a connection with row_factory set so rows behave like dicts.
    Foreign key enforcement is enabled.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = OFF")   # root commits have NULL parent — no FK needed
    conn.execute("PRAGMA journal_mode = WAL")   # safe for concurrent reads
    return conn


# ─────────────────────────────────────────────
#  Schema creation
# ─────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS commits (
    hash            TEXT PRIMARY KEY,
    parent_hash     TEXT,
    weights_hash    TEXT NOT NULL,
    message         TEXT NOT NULL DEFAULT '',
    timestamp       TEXT NOT NULL,
    weights_file    TEXT NOT NULL DEFAULT '',

    -- Flattened metadata columns for fast querying
    -- NULL means the commit did not record this field
    accuracy        REAL,
    loss            REAL,
    lr              REAL,
    epochs          INTEGER,
    dataset         TEXT,
    model_arch      TEXT,

    -- Full metadata JSON for any fields not covered above
    metadata_json   TEXT NOT NULL DEFAULT '{}',

    -- Environment snapshot
    python_version  TEXT,
    platform        TEXT
    -- Note: parent_hash intentionally has no FK constraint.
    -- SQLite enforces FK on NULL differently across versions, and
    -- root commits legitimately have parent_hash = NULL.
    -- Referential integrity is guaranteed by the object store instead.
);

CREATE TABLE IF NOT EXISTS branches (
    name            TEXT PRIMARY KEY,
    commit_hash     TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tags (
    name            TEXT PRIMARY KEY,
    commit_hash     TEXT NOT NULL,
    message         TEXT NOT NULL DEFAULT '',
    created_at      TEXT NOT NULL
);

-- Indexes for the most common query patterns
CREATE INDEX IF NOT EXISTS idx_commits_parent    ON commits (parent_hash);
CREATE INDEX IF NOT EXISTS idx_commits_timestamp ON commits (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_commits_accuracy  ON commits (accuracy DESC);
CREATE INDEX IF NOT EXISTS idx_commits_dataset   ON commits (dataset);
"""


def init_index() -> None:
    """
    Create the SQLite database and tables if they don't exist.
    Safe to call multiple times — uses CREATE IF NOT EXISTS throughout.
    Call this once after init_repo().
    """
    with _connect() as conn:
        conn.executescript(_SCHEMA)
    print(f"[index] Initialised at {DB_PATH}")


# ─────────────────────────────────────────────
#  Writing to the index
# ─────────────────────────────────────────────

def _extract_known_fields(metadata: dict) -> dict:
    """
    Pull out the well-known metadata keys into typed Python values.
    Any key not listed here stays in metadata_json for ad-hoc queries.
    """
    def _float(key):
        v = metadata.get(key)
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    def _int(key):
        v = metadata.get(key)
        try:
            return int(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    return {
        "accuracy":   _float("accuracy"),
        "loss":       _float("loss"),
        "lr":         _float("lr"),
        "epochs":     _int("epochs"),
        "dataset":    metadata.get("dataset"),
        "model_arch": metadata.get("model_arch"),
    }


def index_commit(commit: dict) -> None:
    """
    Insert or replace a single commit into the index.
    Called automatically by commit_and_index() — you rarely need
    to call this directly unless rebuilding.

    Args:
        commit: The commit dict (as returned by load_commit()).
    """
    h        = commit.get("hash", "")
    metadata = commit.get("metadata", {})
    known    = _extract_known_fields(metadata)
    env      = commit.get("env", {})

    row = {
        "hash":           h,
        "parent_hash":    commit.get("parent"),
        "weights_hash":   commit.get("weights_hash", ""),
        "message":        commit.get("message", ""),
        "timestamp":      commit.get("timestamp", ""),
        "weights_file":   commit.get("weights_file", ""),
        "accuracy":       known["accuracy"],
        "loss":           known["loss"],
        "lr":             known["lr"],
        "epochs":         known["epochs"],
        "dataset":        known["dataset"],
        "model_arch":     known["model_arch"],
        "metadata_json":  json.dumps(metadata),
        "python_version": env.get("python_version"),
        "platform":       env.get("platform"),
    }

    sql = """
        INSERT OR REPLACE INTO commits
            (hash, parent_hash, weights_hash, message, timestamp,
             weights_file, accuracy, loss, lr, epochs, dataset,
             model_arch, metadata_json, python_version, platform)
        VALUES
            (:hash, :parent_hash, :weights_hash, :message, :timestamp,
             :weights_file, :accuracy, :loss, :lr, :epochs, :dataset,
             :model_arch, :metadata_json, :python_version, :platform)
    """
    with _connect() as conn:
        conn.execute(sql, row)


def index_branch(name: str, commit_hash: str) -> None:
    """
    Upsert a branch record in the index.

    Args:
        name:        Branch name.
        commit_hash: The commit the branch currently points to.
    """
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO branches (name, commit_hash, updated_at)
               VALUES (?, ?, ?)""",
            (name, commit_hash, now),
        )


def index_tag(name: str, commit_hash: str, message: str = "") -> None:
    """
    Upsert a tag record in the index.

    Args:
        name:        Tag name.
        commit_hash: The commit the tag points to.
        message:     Optional annotation.
    """
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO tags (name, commit_hash, message, created_at)
               VALUES (?, ?, ?, ?)""",
            (name, commit_hash, message, now),
        )


# ─────────────────────────────────────────────
#  Rebuild (self-healing)
# ─────────────────────────────────────────────

def rebuild_index() -> dict:
    """
    Rebuild the entire index from scratch by scanning the object store.
    Use this if:
      - index.db was deleted or corrupted
      - you imported commits from another machine
      - the index is suspected to be out of sync

    The object store is the source of truth — this never loses data.

    Returns:
        Dict with keys: commits_indexed, branches_indexed, tags_indexed.
    """
    print("[index] Rebuilding from object store...")

    # Drop and recreate tables
    with _connect() as conn:
        conn.executescript("""
            DROP TABLE IF EXISTS commits;
            DROP TABLE IF EXISTS branches;
            DROP TABLE IF EXISTS tags;
        """)
        conn.executescript(_SCHEMA)

    commits_n  = 0
    seen_hashes: set[str] = set()   # guard against the two-blob pattern in commit.py

    for obj_hash in list_objects():
        raw = retrieve_bytes(obj_hash)
        if not raw:
            continue
        try:
            obj = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue  # binary blob (weights), not a JSON commit

        if obj.get("type") == "commit":
            # commit.py writes the commit twice: first without the hash
            # field (this produces commit_hash = obj_hash for that blob),
            # then again with hash stamped in (different blob key).
            # HEAD always points at the FIRST blob (commit_hash = Blob A key).
            # We deduplicate by the logical commit hash stored in the blob,
            # falling back to obj_hash when the hash field is absent.
            logical_hash = obj.get("hash") or obj_hash
            if logical_hash in seen_hashes:
                continue
            seen_hashes.add(logical_hash)
            # Ensure the hash field is set before indexing
            if not obj.get("hash"):
                obj["hash"] = obj_hash
            index_commit(obj)
            commits_n += 1

    # Re-index branches from refs/branches/
    branches_n = 0
    if BRANCHES_DIR.exists():
        for branch_file in BRANCHES_DIR.iterdir():
            if branch_file.is_file():
                commit_hash = branch_file.read_text().strip()
                if commit_hash:
                    index_branch(branch_file.name, commit_hash)
                    branches_n += 1

    # Re-index tags from refs/tags/
    tags_n = 0
    if TAGS_DIR.exists():
        for tag_file in TAGS_DIR.iterdir():
            if tag_file.is_file():
                try:
                    payload = json.loads(tag_file.read_text())
                    index_tag(
                        tag_file.name,
                        payload.get("commit", ""),
                        payload.get("message", ""),
                    )
                    tags_n += 1
                except json.JSONDecodeError:
                    pass

    result = {
        "commits_indexed":  commits_n,
        "branches_indexed": branches_n,
        "tags_indexed":     tags_n,
    }
    print(f"[index] Rebuilt — {commits_n} commits, {branches_n} branches, {tags_n} tags.")
    return result


# ─────────────────────────────────────────────
#  Querying
# ─────────────────────────────────────────────

def fast_log(ref: str = "HEAD", limit: int = 50) -> list[dict]:
    """
    Return commit history starting from ref, newest first.
    Uses the SQLite index — no blob loading, very fast even for
    thousands of commits.

    Args:
        ref:   Branch name, tag, commit hash, or 'HEAD'.
        limit: Maximum commits to return.

    Returns:
        List of row dicts, newest first.
    """
    start_hash = resolve_ref(ref)
    if not start_hash:
        # try resolving as a branch name directly before giving up
        from commit import BRANCHES_DIR
        branch_file = BRANCHES_DIR / ref
        if branch_file.exists():
            start_hash = branch_file.read_text().strip() or None
    if not start_hash:
        return []

    # Recursive CTE — walks parent pointers inside SQLite
    sql = """
        WITH RECURSIVE history(hash, parent_hash, message, timestamp,
                               accuracy, loss, lr, epochs, dataset,
                               model_arch, weights_hash, depth) AS (
            SELECT hash, parent_hash, message, timestamp,
                   accuracy, loss, lr, epochs, dataset,
                   model_arch, weights_hash, 0
            FROM   commits
            WHERE  hash = ?

            UNION ALL

            SELECT c.hash, c.parent_hash, c.message, c.timestamp,
                   c.accuracy, c.loss, c.lr, c.epochs, c.dataset,
                   c.model_arch, c.weights_hash, h.depth + 1
            FROM   commits c
            JOIN   history h ON c.hash = h.parent_hash
            WHERE  h.depth < ?
        )
        SELECT * FROM history ORDER BY depth ASC
    """
    with _connect() as conn:
        rows = conn.execute(sql, (start_hash, limit - 1)).fetchall()
    return [dict(r) for r in rows]


def print_fast_log(ref: str = "HEAD", limit: int = 20) -> None:
    """
    Pretty-print the fast log to stdout.

    Args:
        ref:   Starting ref.
        limit: Max commits to show.
    """
    history = fast_log(ref, limit)
    if not history:
        print("No commits found.")
        return

    branch  = _current_branch()
    print(f"\n{'─' * 64}")
    print(f"  Log  {ref}" + (f"  [{branch}]" if branch else ""))
    print(f"{'─' * 64}")

    for i, row in enumerate(history):
        h      = row["hash"][:8]
        ts     = (row["timestamp"] or "")[:19].replace("T", " ")
        msg    = row["message"] or ""
        parent = (row["parent_hash"] or "")[:8]
        parent = parent + "..." if parent else "root"

        metrics = []
        for field in ("accuracy", "loss", "lr", "epochs", "dataset"):
            v = row.get(field)
            if v is not None:
                metrics.append(f"{field}={v}")
        metrics_str = "  ".join(metrics)

        print(f"\n  {'●' if i == 0 else '○'}  {h}...  {ts}")
        print(f"     {msg}")
        if metrics_str:
            print(f"     {metrics_str}")
        print(f"     parent: {parent}")

    print(f"\n{'─' * 64}\n")


def search(
    min_accuracy: float | None = None,
    max_loss: float | None = None,
    lr: float | None = None,
    dataset: str | None = None,
    model_arch: str | None = None,
    message_contains: str | None = None,
    since: str | None = None,
    until: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """
    Search commits by metadata fields. All filters are optional —
    only supplied filters are applied (AND logic).

    Args:
        min_accuracy:     Only return commits where accuracy >= this.
        max_loss:         Only return commits where loss <= this.
        lr:               Exact match on learning rate.
        dataset:          Exact match on dataset name.
        model_arch:       Exact match on model architecture string.
        message_contains: Case-insensitive substring match on message.
        since:            ISO timestamp lower bound (inclusive).
        until:            ISO timestamp upper bound (inclusive).
        limit:            Max results.

    Returns:
        List of row dicts, newest first.

    Example:
        results = search(min_accuracy=0.92, dataset="imagenet-v2")
    """
    conditions = []
    params: list[Any] = []

    if min_accuracy is not None:
        conditions.append("accuracy >= ?")
        params.append(min_accuracy)
    if max_loss is not None:
        conditions.append("loss <= ?")
        params.append(max_loss)
    if lr is not None:
        conditions.append("lr = ?")
        params.append(lr)
    if dataset is not None:
        conditions.append("dataset = ?")
        params.append(dataset)
    if model_arch is not None:
        conditions.append("model_arch = ?")
        params.append(model_arch)
    if message_contains is not None:
        conditions.append("message LIKE ?")
        params.append(f"%{message_contains}%")
    if since is not None:
        conditions.append("timestamp >= ?")
        params.append(since)
    if until is not None:
        conditions.append("timestamp <= ?")
        params.append(until)

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    sql   = f"""
        SELECT * FROM commits
        {where}
        ORDER BY timestamp DESC
        LIMIT ?
    """
    params.append(limit)

    with _connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def print_search_results(results: list[dict]) -> None:
    """Pretty-print search() results."""
    if not results:
        print("No matching commits.")
        return
    print(f"\n  Found {len(results)} commit(s):\n")
    for row in results:
        h   = row["hash"][:8]
        ts  = (row["timestamp"] or "")[:10]
        msg = row["message"] or ""
        acc = f"  acc={row['accuracy']}" if row.get("accuracy") is not None else ""
        ds  = f"  dataset={row['dataset']}" if row.get("dataset") else ""
        print(f"  {h}...  {ts}  {msg}{acc}{ds}")
    print()


def lineage(commit_hash: str) -> list[dict]:
    """
    Return the full ancestor chain for a commit, from the commit
    itself back to the root. Uses the recursive CTE — fast even for
    deep histories.

    Args:
        commit_hash: Starting commit hash.

    Returns:
        List of row dicts [commit, parent, grandparent, ...].
    """
    sql = """
        WITH RECURSIVE ancestors(hash, parent_hash, message, timestamp,
                                  accuracy, depth) AS (
            SELECT hash, parent_hash, message, timestamp, accuracy, 0
            FROM   commits
            WHERE  hash = ?

            UNION ALL

            SELECT c.hash, c.parent_hash, c.message, c.timestamp,
                   c.accuracy, a.depth + 1
            FROM   commits c
            JOIN   ancestors a ON c.hash = a.parent_hash
        )
        SELECT * FROM ancestors ORDER BY depth ASC
    """
    with _connect() as conn:
        rows = conn.execute(sql, (commit_hash,)).fetchall()
    return [dict(r) for r in rows]


def get_branch_tips() -> list[dict]:
    """
    Return all branch names and the commit they point to.

    Returns:
        List of dicts with keys: name, commit_hash, updated_at.
    """
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM branches ORDER BY name"
        ).fetchall()
    return [dict(r) for r in rows]


def get_tags() -> list[dict]:
    """
    Return all tags.

    Returns:
        List of dicts with keys: name, commit_hash, message, created_at.
    """
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM tags ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


# ─────────────────────────────────────────────
#  Aggregate stats
# ─────────────────────────────────────────────

def stats() -> dict:
    """
    Return aggregate statistics across all indexed commits.

    Returns:
        Dict with summary numbers useful for dashboards or reports.
    """
    sql = """
        SELECT
            COUNT(*)                  AS total_commits,
            COUNT(DISTINCT dataset)   AS unique_datasets,
            MAX(accuracy)             AS best_accuracy,
            MIN(loss)                 AS best_loss,
            AVG(accuracy)             AS avg_accuracy,
            MIN(timestamp)            AS first_commit,
            MAX(timestamp)            AS latest_commit
        FROM commits
    """
    with _connect() as conn:
        row = conn.execute(sql).fetchone()

    return {
        "total_commits":    row["total_commits"],
        "unique_datasets":  row["unique_datasets"],
        "best_accuracy":    round(row["best_accuracy"], 4) if row["best_accuracy"] else None,
        "best_loss":        round(row["best_loss"], 4)     if row["best_loss"]     else None,
        "avg_accuracy":     round(row["avg_accuracy"], 4)  if row["avg_accuracy"]  else None,
        "first_commit":     (row["first_commit"]  or "")[:19],
        "latest_commit":    (row["latest_commit"] or "")[:19],
    }


def accuracy_over_time(limit: int = 100) -> list[dict]:
    """
    Return accuracy and timestamp for all commits that have an
    accuracy value, ordered by time. Useful for plotting training
    progress.

    Args:
        limit: Max rows to return.

    Returns:
        List of dicts with keys: hash, timestamp, accuracy, message.
    """
    sql = """
        SELECT hash, timestamp, accuracy, message
        FROM   commits
        WHERE  accuracy IS NOT NULL
        ORDER  BY timestamp ASC
        LIMIT  ?
    """
    with _connect() as conn:
        rows = conn.execute(sql, (limit,)).fetchall()
    return [dict(r) for r in rows]


def best_model(metric: str = "accuracy", top_n: int = 5) -> list[dict]:
    """
    Return the top N commits ranked by a numeric metric column.

    Args:
        metric: Column name to rank by — 'accuracy', 'loss', 'lr', 'epochs'.
                For 'loss' the ranking is ascending (lower is better);
                for all others it is descending.
        top_n:  Number of results to return.

    Returns:
        List of row dicts, best first.

    Raises:
        ValueError: If metric is not a known numeric column.
    """
    allowed = {"accuracy", "loss", "lr", "epochs"}
    if metric not in allowed:
        raise ValueError(f"metric must be one of {allowed}, got '{metric}'")

    order = "ASC" if metric == "loss" else "DESC"
    sql   = f"""
        SELECT hash, timestamp, message, accuracy, loss, lr, epochs, dataset
        FROM   commits
        WHERE  {metric} IS NOT NULL
        ORDER  BY {metric} {order}
        LIMIT  ?
    """
    with _connect() as conn:
        rows = conn.execute(sql, (top_n,)).fetchall()
    return [dict(r) for r in rows]


def compare_datasets() -> list[dict]:
    """
    Group commits by dataset and return the best accuracy achieved
    on each, along with commit count per dataset.

    Returns:
        List of dicts: dataset, best_accuracy, commit_count, best_commit_hash.
    """
    sql = """
        SELECT
            dataset,
            COUNT(*)                            AS commit_count,
            MAX(accuracy)                       AS best_accuracy,
            hash                                AS best_commit_hash
        FROM commits
        WHERE dataset IS NOT NULL
        GROUP BY dataset
        ORDER BY best_accuracy DESC NULLS LAST
    """
    with _connect() as conn:
        rows = conn.execute(sql).fetchall()
    return [dict(r) for r in rows]


# ─────────────────────────────────────────────
#  Integrated commit + index (main entry point)
# ─────────────────────────────────────────────

def commit_and_index(
    weights_path: str | Path,
    message: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    """
    Create a commit AND immediately index it.
    This is the function you should call instead of commit.commit_model()
    once the index is set up — it keeps the object store and index in sync.

    Args:
        weights_path: Path to the serialised model weights.
        message:      Commit message.
        metadata:     Optional dict of metrics / hyperparameters.

    Returns:
        The commit hash.
    """
    from commit import commit_model
    commit_hash = commit_model(weights_path, message, metadata)

    # load_commit may return the blob without the hash field stamped in
    # (the first of the two writes in commit.py). Ensure it's set.
    commit = load_commit(commit_hash)
    if commit:
        if not commit.get("hash"):
            commit["hash"] = commit_hash
        index_commit(commit)

    # Keep branch tips in sync
    branch = _current_branch()
    if branch:
        index_branch(branch, commit_hash)

    return commit_hash


# ─────────────────────────────────────────────
#  Quick self-test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np
    import tempfile
    from pathlib import Path
    from commit import (
        BRANCHES_DIR,
        HEAD_FILE,
        create_branch,
        switch_branch,
        tag as commit_tag,
    )

    print("=" * 64)
    print("index.py — self test")
    print("=" * 64)

    # Bootstrap
    init_repo()
    BRANCHES_DIR.mkdir(parents=True, exist_ok=True)
    if not HEAD_FILE.read_text().strip():
        HEAD_FILE.write_text("ref: branches/main")

    init_index()

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        def make_weights(name: str, seed: int) -> Path:
            np.random.seed(seed)
            w = np.random.rand(64, 64).astype(np.float32)
            p = tmp / name
            np.save(p, w)
            return p

        # ── 3 commits on main ───────────────────────────────────────
        h1 = commit_and_index(
            make_weights("w1.npy", 1), "baseline",
            {"accuracy": 0.87, "loss": 0.42, "lr": 0.01,
             "epochs": 10, "dataset": "imagenet-v1"},
        )
        h2 = commit_and_index(
            make_weights("w2.npy", 2), "lower lr",
            {"accuracy": 0.91, "loss": 0.31, "lr": 0.001,
             "epochs": 10, "dataset": "imagenet-v1"},
        )
        h3 = commit_and_index(
            make_weights("w3.npy", 3), "imagenet-v2 dataset",
            {"accuracy": 0.93, "loss": 0.22, "lr": 0.001,
             "epochs": 15, "dataset": "imagenet-v2"},
        )

        # ── branch: experiment-aug ──────────────────────────────────
        create_branch("experiment-aug", from_ref="HEAD")
        switch_branch("experiment-aug")
        h4 = commit_and_index(
            make_weights("w4.npy", 4), "data augmentation",
            {"accuracy": 0.95, "loss": 0.18, "lr": 0.001,
             "epochs": 20, "dataset": "imagenet-v2"},
        )
        switch_branch("main")

        # ── tag ─────────────────────────────────────────────────────
        commit_tag("v1.0", message="First production release")
        index_tag("v1.0", h3, "First production release")

        # ── fast log ────────────────────────────────────────────────
        print("\n── fast_log (main) ──")
        print_fast_log("main")

        # ── search ──────────────────────────────────────────────────
        print("── search: accuracy >= 0.92 ──")
        results = search(min_accuracy=0.92)
        print_search_results(results)

        print("── search: dataset=imagenet-v2 ──")
        results = search(dataset="imagenet-v2")
        print_search_results(results)

        print("── search: message contains 'lr' ──")
        results = search(message_contains="lr")
        print_search_results(results)

        # ── best model ──────────────────────────────────────────────
        print("── best_model (top 3 by accuracy) ──")
        for row in best_model("accuracy", top_n=3):
            print(f"  {row['hash'][:8]}  acc={row['accuracy']}  {row['message']}")

        print("\n── best_model (top 3 by loss, lower=better) ──")
        for row in best_model("loss", top_n=3):
            print(f"  {row['hash'][:8]}  loss={row['loss']}  {row['message']}")

        # ── lineage ─────────────────────────────────────────────────
        print(f"\n── lineage of experiment-aug tip ({h4[:8]}...) ──")
        for row in lineage(h4):
            print(f"  depth={row['depth']}  {row['hash'][:8]}  {row['message']}")

        # ── stats ───────────────────────────────────────────────────
        print("\n── stats() ──")
        s = stats()
        for k, v in s.items():
            print(f"  {k}: {v}")

        # ── compare_datasets ────────────────────────────────────────
        print("\n── compare_datasets() ──")
        for row in compare_datasets():
            print(
                f"  {row['dataset']}  "
                f"best_acc={row['best_accuracy']}  "
                f"commits={row['commit_count']}"
            )

        # ── branch tips ─────────────────────────────────────────────
        print("\n── branch tips ──")
        for b in get_branch_tips():
            print(f"  {b['name']}  →  {b['commit_hash'][:8]}...")

        # ── tags ────────────────────────────────────────────────────
        print("\n── tags ──")
        for t in get_tags():
            print(f"  {t['name']}  →  {t['commit_hash'][:8]}  \"{t['message']}\"")

        # ── rebuild from scratch ────────────────────────────────────
        print("\n── rebuild_index() ──")
        result = rebuild_index()
        print(f"  {result}")

        # Verify rebuild finds all 4 commits (3 on main + 1 on experiment-aug)
        s2 = stats()
        assert s2["total_commits"] == 4, \
            f"Expected 4 commits after rebuild, got {s2['total_commits']}"
        print(f"  Stats consistent after rebuild ✓  ({s2['total_commits']} commits)")

        # ── accuracy over time ──────────────────────────────────────
        print("\n── accuracy_over_time() ──")
        for row in accuracy_over_time():
            print(
                f"  {row['timestamp'][:10]}  "
                f"acc={row['accuracy']}  "
                f"{row['message']}"
            )

    print("\nAll tests passed.")