"""
commit.py — Git-like commit layer for ML model versioning.

Sits directly on top of object_store.py. A commit is a JSON blob
that ties together:
  - a pointer to the stored model weights (weights_hash)
  - a pointer to its parent commit (parent_hash)
  - metadata: message, timestamp, metrics, hyperparameters, etc.

The commit itself is stored as a blob in the object store, so its
hash is both its ID and an integrity check.

Typical flow:
    1. Train / fine-tune your model → produces a weights file
    2. commit_model("model.pt", message="...", metadata={...})
    3. Repeat — each commit records its parent automatically
    4. Use log(), checkout(), diff() to navigate history
"""

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from object_store import (
    get_head,
    init_repo,
    retrieve_object,
    set_head,
    store_bytes,
    store_object,
    retrieve_bytes,
    object_exists,
)

# ─────────────────────────────────────────────
#  Repo paths
# ─────────────────────────────────────────────

REPO_DIR     = Path.home() / ".modelrepo"
BRANCHES_DIR = REPO_DIR / "refs" / "branches"
TAGS_DIR     = REPO_DIR / "refs" / "tags"
HEAD_FILE    = REPO_DIR / "HEAD"          # may hold a hash OR "ref: branches/<name>"


# ─────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────

def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _env_snapshot() -> dict:
    """
    Capture a lightweight snapshot of the current Python environment.
    Stored in every commit so you can detect environment drift on checkout.
    """
    return {
        "python_version": sys.version,
        "platform":       platform.platform(),
    }


def _serialise_commit(commit: dict) -> bytes:
    """
    Serialise a commit dict to canonical UTF-8 JSON bytes.
    Keys are sorted so the same logical commit always produces
    the same bytes (and therefore the same hash).
    """
    return json.dumps(commit, sort_keys=True, indent=2).encode("utf-8")


def _read_branch_file(branch_name: str) -> Path:
    """Return the Path to a branch ref file (may or may not exist yet)."""
    return BRANCHES_DIR / branch_name


def _read_tag_file(tag_name: str) -> Path:
    """Return the Path to a tag ref file."""
    return TAGS_DIR / tag_name


# ─────────────────────────────────────────────
#  HEAD resolution
# ─────────────────────────────────────────────

def _head_is_branch() -> bool:
    """
    Return True if HEAD is a symbolic ref pointing to a branch
    (normal mode), False if it holds a raw commit hash (detached mode).
    """
    content = HEAD_FILE.read_text().strip()
    return content.startswith("ref: ")


def _current_branch() -> str | None:
    """
    Return the name of the currently checked-out branch, or None
    if HEAD is detached (pointing directly at a commit hash).
    """
    content = HEAD_FILE.read_text().strip()
    if content.startswith("ref: "):
        return content[len("ref: branches/"):]
    return None


def _resolve_head() -> str | None:
    """
    Walk HEAD → branch ref → commit hash and return the current
    commit hash, or None if the repo is empty.
    """
    content = HEAD_FILE.read_text().strip()
    if not content:
        return None
    if content.startswith("ref: "):
        branch_name = content[len("ref: branches/"):]
        branch_file = _read_branch_file(branch_name)
        if not branch_file.exists():
            return None  # branch exists in HEAD but has no commits yet
        return branch_file.read_text().strip() or None
    return content  # raw commit hash (detached HEAD)


def _advance_head(commit_hash: str) -> None:
    """
    After a new commit, advance the current branch pointer (or HEAD
    directly if detached) to the new commit hash.
    """
    content = HEAD_FILE.read_text().strip()
    if content.startswith("ref: "):
        branch_name = content[len("ref: branches/"):]
        branch_file = _read_branch_file(branch_name)
        branch_file.parent.mkdir(parents=True, exist_ok=True)
        branch_file.write_text(commit_hash)
    else:
        HEAD_FILE.write_text(commit_hash)


# ─────────────────────────────────────────────
#  Commit creation
# ─────────────────────────────────────────────

def commit_model(
    weights_path: str | Path,
    message: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    """
    Store model weights and create a new commit pointing to them.

    The new commit's parent is automatically set to the current HEAD.
    HEAD (or the current branch pointer) is advanced to the new commit.

    Args:
        weights_path: Path to the serialised model weights file
                      (.pt, .npy, safetensors, etc.).
        message:      Human-readable description of this version,
                      e.g. "fine-tuned on v2 dataset, lr=0.001".
        metadata:     Optional dict of arbitrary key-value pairs —
                      metrics, hyperparameters, dataset info, etc.
                      Example: {"accuracy": 0.94, "epochs": 10}

    Returns:
        The commit hash (64-char hex string). Keep this — it's your
        permanent, unforgeable reference to this exact model version.

    Raises:
        FileNotFoundError: If weights_path does not exist.
        RuntimeError:      If the repo has not been initialised.
    """
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    if not REPO_DIR.exists():
        raise RuntimeError("Repo not initialised. Call init_repo() first.")

    # 1. Store the weights blob
    weights_hash = store_object(weights_path)

    # 2. Build the commit object
    parent_hash = _resolve_head()
    commit = {
        "type":          "commit",
        "weights_hash":  weights_hash,
        "parent":        parent_hash,      # None for the very first commit
        "message":       message,
        "timestamp":     _now_iso(),
        "weights_file":  weights_path.name,
        "metadata":      metadata or {},
        "env":           _env_snapshot(),
    }

    # 3. Serialise → hash → store the commit blob
    commit_bytes = _serialise_commit(commit)
    commit_hash  = store_bytes(commit_bytes)

    # 4. Stamp the hash back into the object (so loading it later gives
    #    you the hash without needing to re-derive it)
    commit["hash"] = commit_hash
    store_bytes(_serialise_commit(commit))   # re-store with hash field included

    # 5. Advance HEAD / branch pointer
    _advance_head(commit_hash)

    branch = _current_branch()
    branch_label = f" [{branch}]" if branch else " [detached]"
    print(f"[commit] {commit_hash[:8]}...{branch_label} — {message}")
    return commit_hash


# ─────────────────────────────────────────────
#  Commit loading
# ─────────────────────────────────────────────

def load_commit(commit_hash: str) -> dict | None:
    """
    Load and deserialise a commit object by its hash.

    Args:
        commit_hash: The 64-char hex hash of the commit.

    Returns:
        The commit dict, or None if the hash is not in the store.
    """
    raw = retrieve_bytes(commit_hash)
    if raw is None:
        return None
    return json.loads(raw.decode("utf-8"))


def resolve_ref(ref: str) -> str | None:
    """
    Resolve a human-readable ref to a commit hash.
    Accepts: full commit hash, branch name, tag name, or 'HEAD'.

    Args:
        ref: The reference to resolve.

    Returns:
        A 64-char commit hash, or None if the ref is unknown.
    """
    # HEAD
    if ref.upper() == "HEAD":
        return _resolve_head()

    # Full or partial commit hash — check object store directly
    if len(ref) >= 7 and all(c in "0123456789abcdef" for c in ref.lower()):
        if len(ref) == 64 and object_exists(ref):
            return ref
        # Partial hash: scan for prefix match
        from object_store import list_objects
        matches = [h for h in list_objects() if h.startswith(ref.lower())]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            print(f"[resolve] Ambiguous prefix '{ref}' matches {len(matches)} objects.")
            return None

    # Branch ref
    branch_file = _read_branch_file(ref)
    if branch_file.exists():
        content = branch_file.read_text().strip()
        return content or None

    # Tag ref
    tag_file = _read_tag_file(ref)
    if tag_file.exists():
        return tag_file.read_text().strip() or None

    return None


# ─────────────────────────────────────────────
#  History walking
# ─────────────────────────────────────────────

def log(ref: str = "HEAD", limit: int = 50) -> list[dict]:
    """
    Return the commit history starting from ref, following parent
    pointers backward (newest first).

    Args:
        ref:   Starting point — branch name, tag, commit hash, or 'HEAD'.
        limit: Maximum number of commits to return (guards against
               very long histories in interactive use).

    Returns:
        List of commit dicts, newest first. Empty list if no history.
    """
    start_hash = resolve_ref(ref)
    if start_hash is None:
        return []

    history = []
    current = start_hash
    seen    = set()  # cycle guard (shouldn't happen in a DAG, but be safe)

    while current and len(history) < limit:
        if current in seen:
            break
        seen.add(current)

        commit = load_commit(current)
        if commit is None:
            break

        history.append(commit)
        current = commit.get("parent")

    return history


def print_log(ref: str = "HEAD", limit: int = 20) -> None:
    """
    Pretty-print the commit history to stdout.

    Args:
        ref:   Starting ref (default HEAD).
        limit: Max commits to show.
    """
    history = log(ref, limit)
    if not history:
        print("No commits yet.")
        return

    branch = _current_branch()
    print(f"\n{'─' * 60}")
    print(f"  History from {ref}" + (f"  [branch: {branch}]" if branch else ""))
    print(f"{'─' * 60}")

    for i, commit in enumerate(history):
        h         = commit.get("hash", "?")[:8]
        ts        = commit.get("timestamp", "")[:19].replace("T", " ")
        msg       = commit.get("message", "")
        parent    = commit.get("parent", "")
        parent_s  = parent[:8] + "..." if parent else "none"
        meta      = commit.get("metadata", {})
        meta_str  = "  ".join(f"{k}={v}" for k, v in meta.items()) if meta else ""

        print(f"\n  {'●' if i == 0 else '○'}  {h}...  {ts}")
        print(f"     {msg}")
        if meta_str:
            print(f"     {meta_str}")
        print(f"     parent: {parent_s}")

    print(f"\n{'─' * 60}\n")


def get_ancestors(commit_hash: str) -> list[str]:
    """
    Return an ordered list of all ancestor hashes for a given commit,
    starting from the commit itself and walking to the root.

    Args:
        commit_hash: Starting commit.

    Returns:
        List of hashes [commit_hash, parent, grandparent, ...].
    """
    ancestors = []
    current   = commit_hash
    seen      = set()
    while current:
        if current in seen:
            break
        seen.add(current)
        ancestors.append(current)
        commit  = load_commit(current)
        current = commit.get("parent") if commit else None
    return ancestors


def common_ancestor(hash_a: str, hash_b: str) -> str | None:
    """
    Find the nearest common ancestor of two commits (the merge base).
    Used by diff() to understand how two branches diverged.

    Args:
        hash_a: First commit hash.
        hash_b: Second commit hash.

    Returns:
        Hash of the nearest common ancestor, or None if unrelated.
    """
    ancestors_a = set(get_ancestors(hash_a))
    current     = hash_b
    seen        = set()
    while current:
        if current in seen:
            break
        seen.add(current)
        if current in ancestors_a:
            return current
        commit  = load_commit(current)
        current = commit.get("parent") if commit else None
    return None


# ─────────────────────────────────────────────
#  Checkout
# ─────────────────────────────────────────────

def checkout(
    ref: str,
    destination: str | Path = "model_checkout.pt",
    detach: bool = False,
) -> dict | None:
    """
    Restore the model weights for a given commit to disk, and
    update HEAD to point at that commit.

    Args:
        ref:         Branch name, tag, commit hash (full or partial),
                     or 'HEAD'.
        destination: Where to write the restored weights file.
        detach:      If True, HEAD moves directly to the commit hash
                     (detached HEAD mode) rather than following a branch.
                     Set this when you want to inspect an old version
                     without affecting your branch pointer.

    Returns:
        The commit dict for the checked-out version, or None on failure.
    """
    commit_hash = resolve_ref(ref)
    if commit_hash is None:
        print(f"[checkout] Unknown ref: '{ref}'")
        return None

    commit = load_commit(commit_hash)
    if commit is None:
        print(f"[checkout] Could not load commit {commit_hash[:8]}...")
        return None

    weights_hash = commit.get("weights_hash")
    if not weights_hash:
        print(f"[checkout] Commit {commit_hash[:8]}... has no weights_hash.")
        return None

    # Restore weights to disk
    ok = retrieve_object(weights_hash, destination)
    if not ok:
        print(f"[checkout] Weights blob {weights_hash[:8]}... not found in object store.")
        return None

    # Update HEAD
    if detach:
        HEAD_FILE.write_text(commit_hash)
        print(f"[checkout] Detached HEAD at {commit_hash[:8]}...")
    else:
        # Try to find a branch pointing to this commit and switch to it,
        # otherwise stay on current branch (which now points at this commit)
        _advance_head(commit_hash)

    # Warn if environment differs from when this commit was made
    saved_env = commit.get("env", {})
    current_python = sys.version
    if saved_env.get("python_version") != current_python:
        print(
            f"[checkout] ⚠  Python version mismatch: "
            f"committed with {saved_env.get('python_version', '?')!r}, "
            f"current is {current_python!r}"
        )

    print(
        f"[checkout] {commit_hash[:8]}... → {destination}  "
        f"({commit.get('message', '')})"
    )
    return commit


# ─────────────────────────────────────────────
#  Diff
# ─────────────────────────────────────────────

def diff(ref_a: str, ref_b: str) -> dict:
    """
    Compare two commits and return a structured diff of their metadata.
    Does not compare raw weight bytes (that would require tensor-level
    comparison — build that on top if needed).

    Shows:
      - What metadata keys changed and by how much
      - Whether the weight blobs are identical
      - The nearest common ancestor

    Args:
        ref_a: First ref (e.g. a branch name or commit hash).
        ref_b: Second ref.

    Returns:
        Dict with keys: hash_a, hash_b, ancestor, weights_changed,
        metadata_diff, message_a, message_b.
    """
    hash_a = resolve_ref(ref_a)
    hash_b = resolve_ref(ref_b)

    if not hash_a:
        print(f"[diff] Cannot resolve ref: {ref_a}")
        return {}
    if not hash_b:
        print(f"[diff] Cannot resolve ref: {ref_b}")
        return {}

    commit_a = load_commit(hash_a)
    commit_b = load_commit(hash_b)

    if not commit_a or not commit_b:
        print("[diff] Could not load one or both commits.")
        return {}

    meta_a = commit_a.get("metadata", {})
    meta_b = commit_b.get("metadata", {})

    all_keys     = set(meta_a) | set(meta_b)
    metadata_diff = {}
    for key in sorted(all_keys):
        val_a = meta_a.get(key, "<absent>")
        val_b = meta_b.get(key, "<absent>")
        if val_a != val_b:
            metadata_diff[key] = {"a": val_a, "b": val_b}

    ancestor     = common_ancestor(hash_a, hash_b)
    weights_same = commit_a.get("weights_hash") == commit_b.get("weights_hash")

    result = {
        "hash_a":          hash_a,
        "hash_b":          hash_b,
        "message_a":       commit_a.get("message", ""),
        "message_b":       commit_b.get("message", ""),
        "timestamp_a":     commit_a.get("timestamp", ""),
        "timestamp_b":     commit_b.get("timestamp", ""),
        "weights_changed": not weights_same,
        "metadata_diff":   metadata_diff,
        "ancestor":        ancestor,
    }
    return result


def print_diff(ref_a: str, ref_b: str) -> None:
    """
    Pretty-print a diff between two refs.

    Args:
        ref_a: First ref.
        ref_b: Second ref.
    """
    d = diff(ref_a, ref_b)
    if not d:
        return

    print(f"\n{'─' * 60}")
    print(f"  diff  {d['hash_a'][:8]}  ↔  {d['hash_b'][:8]}")
    print(f"{'─' * 60}")
    print(f"  A: {d['message_a']}  ({d['timestamp_a'][:19]})")
    print(f"  B: {d['message_b']}  ({d['timestamp_b'][:19]})")
    print(f"  Ancestor: {d['ancestor'][:8] + '...' if d['ancestor'] else 'none'}")
    print(f"  Weights:  {'unchanged' if not d['weights_changed'] else 'CHANGED'}")
    if d["metadata_diff"]:
        print(f"\n  Metadata changes:")
        for key, vals in d["metadata_diff"].items():
            print(f"    {key}:  {vals['a']}  →  {vals['b']}")
    else:
        print(f"  Metadata: identical")
    print(f"{'─' * 60}\n")


# ─────────────────────────────────────────────
#  Branches
# ─────────────────────────────────────────────

def create_branch(branch_name: str, from_ref: str = "HEAD") -> bool:
    """
    Create a new branch pointing at the given ref.
    Does not switch HEAD to the new branch — use switch_branch() for that.

    Args:
        branch_name: Name for the new branch (e.g. 'experiment-lr').
        from_ref:    Starting point (default: current HEAD).

    Returns:
        True if created, False if the branch already exists.
    """
    branch_file = _read_branch_file(branch_name)
    if branch_file.exists():
        print(f"[branch] '{branch_name}' already exists.")
        return False

    commit_hash = resolve_ref(from_ref)
    if not commit_hash:
        print(f"[branch] Cannot resolve ref '{from_ref}' — is the repo empty?")
        return False

    branch_file.parent.mkdir(parents=True, exist_ok=True)
    branch_file.write_text(commit_hash)
    print(f"[branch] Created '{branch_name}' at {commit_hash[:8]}...")
    return True


def switch_branch(branch_name: str) -> bool:
    """
    Switch HEAD to point to an existing branch.
    Does not alter any files on disk — weights stay wherever they are.

    Args:
        branch_name: Name of the branch to switch to.

    Returns:
        True if switched, False if the branch does not exist.
    """
    branch_file = _read_branch_file(branch_name)
    if not branch_file.exists():
        print(f"[branch] '{branch_name}' does not exist. Create it first.")
        return False

    HEAD_FILE.write_text(f"ref: branches/{branch_name}")
    print(f"[branch] Switched to '{branch_name}'")
    return True


def list_branches() -> list[str]:
    """
    Return a list of all branch names in the repo.

    Returns:
        Sorted list of branch name strings.
    """
    if not BRANCHES_DIR.exists():
        return []
    return sorted(f.name for f in BRANCHES_DIR.iterdir() if f.is_file())


def print_branches() -> None:
    """Print all branches, marking the currently active one."""
    branches = list_branches()
    current  = _current_branch()
    if not branches:
        print("No branches yet.")
        return
    print("\nBranches:")
    for b in branches:
        marker = "* " if b == current else "  "
        tip    = _read_branch_file(b).read_text().strip()
        print(f"  {marker}{b}  ({tip[:8]}...)")
    print()


# ─────────────────────────────────────────────
#  Tags
# ─────────────────────────────────────────────

def tag(tag_name: str, ref: str = "HEAD", message: str = "") -> bool:
    """
    Create a lightweight tag pointing at a commit.
    Tags are immutable named pointers — use them to mark releases
    or production deployments.

    Args:
        tag_name: Name for the tag (e.g. 'v1.0', 'prod-2026-03').
        ref:      Commit to tag (default: current HEAD).
        message:  Optional annotation stored alongside the tag.

    Returns:
        True if created, False if the tag already exists.
    """
    tag_file = _read_tag_file(tag_name)
    if tag_file.exists():
        print(f"[tag] '{tag_name}' already exists.")
        return False

    commit_hash = resolve_ref(ref)
    if not commit_hash:
        print(f"[tag] Cannot resolve ref '{ref}'.")
        return False

    tag_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {"commit": commit_hash, "message": message}
    tag_file.write_text(json.dumps(payload))
    print(f"[tag] Created tag '{tag_name}' → {commit_hash[:8]}...")
    return True


def list_tags() -> list[str]:
    """Return all tag names."""
    if not TAGS_DIR.exists():
        return []
    return sorted(f.name for f in TAGS_DIR.iterdir() if f.is_file())


# ─────────────────────────────────────────────
#  Quick self-test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np
    import tempfile

    print("=" * 60)
    print("commit.py — self test")
    print("=" * 60)

    # Bootstrap
    init_repo()

    # Point HEAD at the default 'main' branch
    BRANCHES_DIR.mkdir(parents=True, exist_ok=True)
    if not HEAD_FILE.read_text().strip():
        HEAD_FILE.write_text("ref: branches/main")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        def make_weights(name: str, seed: int) -> Path:
            np.random.seed(seed)
            w = np.random.rand(50, 50).astype(np.float32)
            p = tmp / name
            np.save(p, w)
            return p

        # ── Commit 1: baseline ──────────────────────────────────────
        w1 = make_weights("weights_v1.npy", seed=1)
        h1 = commit_model(
            w1,
            message="baseline — ResNet50, ImageNet",
            metadata={"accuracy": 0.87, "epochs": 10, "lr": 0.01},
        )

        # ── Commit 2: better LR ─────────────────────────────────────
        w2 = make_weights("weights_v2.npy", seed=2)
        h2 = commit_model(
            w2,
            message="lr=0.001, accuracy improved",
            metadata={"accuracy": 0.91, "epochs": 10, "lr": 0.001},
        )

        # ── Branch: experiment ──────────────────────────────────────
        create_branch("experiment-aug", from_ref="HEAD")
        switch_branch("experiment-aug")

        w3 = make_weights("weights_v3.npy", seed=3)
        h3 = commit_model(
            w3,
            message="data augmentation, accuracy up",
            metadata={"accuracy": 0.93, "epochs": 15, "lr": 0.001},
        )

        # ── Switch back to main ─────────────────────────────────────
        switch_branch("main")

        # ── Tag the current main tip ────────────────────────────────
        tag("v1.0", message="First production release")

        # ── Log ─────────────────────────────────────────────────────
        print("\n── Log (main) ──")
        print_log("main")

        print("\n── Log (experiment-aug) ──")
        print_log("experiment-aug")

        # ── Branches ────────────────────────────────────────────────
        print_branches()

        # ── Diff ────────────────────────────────────────────────────
        print("\n── Diff main vs experiment-aug ──")
        print_diff("main", "experiment-aug")

        # ── Common ancestor ─────────────────────────────────────────
        anc = common_ancestor(h2, h3)
        print(f"Common ancestor of main and experiment: {anc[:8] if anc else 'none'}")

        # ── Checkout ────────────────────────────────────────────────
        restored = tmp / "restored.npy"
        commit   = checkout(h1, destination=restored, detach=True)
        print(f"Checked-out commit message: {commit['message']}")
        original = np.load(w1)
        loaded   = np.load(restored)
        print(f"Weights round-trip identical: {np.array_equal(original, loaded)}")

        # ── resolve_ref on a partial hash ───────────────────────────
        partial  = h2[:10]
        resolved = resolve_ref(partial)
        print(f"Partial hash '{partial}' resolved to: {resolved[:8]}...")

        # ── Tags ────────────────────────────────────────────────────
        print(f"Tags: {list_tags()}")

    print("\nAll tests passed.")