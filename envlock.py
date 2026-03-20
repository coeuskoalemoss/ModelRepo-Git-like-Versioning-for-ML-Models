"""
env_lock.py — Environment locking for the ML model versioning system.

Phase 6 of the build. Sits alongside commit.py and is called
automatically by an enhanced commit_model() wrapper defined here.

THE PROBLEM
───────────
Rolling back model weights is only half the story. A model's behaviour
depends on its entire software environment:

  - Python version          (3.10 vs 3.11 — different float precision bugs)
  - PyTorch / TensorFlow    (weight format changes between major versions)
  - NumPy                   (RNG behaviour changed in 1.17)
  - CUDA version            (affects reproducibility on GPU)
  - Any preprocessing lib   (tokenizer, albumentations, etc.)

If you check out weights from 6 months ago but run them with today's
libraries, you'll get subtly wrong results and spend days debugging.

THE SOLUTION
────────────
On every commit we capture a full environment snapshot:
  - requirements freeze (pip freeze output)
  - Python version + platform
  - key library versions (torch, numpy, sklearn, etc.)
  - a SHA-256 hash of the frozen requirements

The snapshot is stored as a blob in the object store (so it's
immutable and content-addressed, just like weights). Its hash is
recorded in the commit under "env_snapshot_hash".

On checkout we compare the current environment against the snapshot
and print a structured diff: what's missing, what's the wrong version,
what's new. This surfaces drift immediately without blocking you.

Optionally, env_restore() generates a restoration script and can
create a fresh virtual environment with the exact locked versions.

INTEGRATION
───────────
Use commit_with_env() instead of commit_model() for full locking:

  from env_lock import commit_with_env
  hash = commit_with_env("model.pt", "fine-tuned v2", metadata={...})

  # On checkout:
  from env_lock import check_env_on_checkout
  check_env_on_checkout(commit_hash)
"""

import hashlib
import json
import platform
import subprocess
import sys
import venv
from pathlib import Path
from typing import Any

from object_store import init_repo, store_bytes, retrieve_bytes
from commit import (
    REPO_DIR,
    _current_branch,
    _resolve_head,
    load_commit,
    resolve_ref,
    commit_model,
)

# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────

# Libraries we always try to record even if not installed
TRACKED_LIBS = [
    "torch", "tensorflow", "keras", "jax", "flax",
    "numpy", "scipy", "pandas", "scikit-learn",
    "transformers", "tokenizers", "datasets",
    "opencv-python", "Pillow", "albumentations",
    "onnx", "onnxruntime",
    "lightning", "pytorch-lightning",
    "safetensors", "accelerate", "peft",
    "xgboost", "lightgbm", "catboost",
]

ENV_SNAPSHOTS_DIR = REPO_DIR / "env_snapshots"


# ─────────────────────────────────────────────
#  Capture
# ─────────────────────────────────────────────

def _pip_freeze() -> str:
    """
    Run `pip freeze` and return its output as a string.
    Returns an empty string if pip is unavailable.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze", "--all"],
            capture_output=True, text=True, timeout=30,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def _get_lib_version(lib_name: str) -> str | None:
    """
    Return the installed version of a library, or None if not installed.
    Uses importlib.metadata which is stdlib in Python 3.8+.
    """
    try:
        from importlib.metadata import version, PackageNotFoundError
        return version(lib_name)
    except Exception:
        return None


def _cuda_info() -> dict:
    """
    Try to detect CUDA version and GPU info.
    Returns empty dict if CUDA is not available.
    """
    info = {}
    try:
        import torch
        if torch.cuda.is_available():
            info["cuda_available"]  = True
            info["cuda_version"]    = torch.version.cuda
            info["gpu_count"]       = torch.cuda.device_count()
            info["gpu_name"]        = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    if not info:
        try:
            result = subprocess.run(
                ["nvcc", "--version"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                info["cuda_available"] = True
                info["nvcc_output"]    = result.stdout.strip().splitlines()[-1]
        except Exception:
            info["cuda_available"] = False

    return info


def capture_env() -> dict:
    """
    Capture a complete snapshot of the current Python environment.

    Returns a dict containing:
      - python:     version string + implementation
      - platform:   OS, architecture
      - pip_freeze: full output of pip freeze
      - req_hash:   SHA-256 of the pip freeze output (for quick comparison)
      - libs:       dict of {lib_name: version} for tracked libraries
      - cuda:       CUDA / GPU info (if available)

    This is the canonical environment record stored with every commit.
    """
    freeze     = _pip_freeze()
    req_hash   = hashlib.sha256(freeze.encode()).hexdigest()

    libs = {}
    for lib in TRACKED_LIBS:
        v = _get_lib_version(lib)
        if v is not None:
            libs[lib] = v

    return {
        "python": {
            "version":        sys.version,
            "version_info":   list(sys.version_info[:3]),
            "implementation": platform.python_implementation(),
            "executable":     sys.executable,
        },
        "platform": {
            "system":    platform.system(),
            "release":   platform.release(),
            "machine":   platform.machine(),
            "processor": platform.processor(),
            "node":      platform.node(),
        },
        "pip_freeze": freeze,
        "req_hash":   req_hash,
        "libs":       libs,
        "cuda":       _cuda_info(),
    }


def store_env_snapshot(env: dict) -> str:
    """
    Serialise an environment snapshot and store it as a blob in the
    object store. Returns the blob hash.

    Args:
        env: Dict returned by capture_env().

    Returns:
        SHA-256 hash of the stored snapshot blob.
    """
    data = json.dumps(env, sort_keys=True, indent=2).encode("utf-8")
    return store_bytes(data)


def load_env_snapshot(snapshot_hash: str) -> dict | None:
    """
    Load an environment snapshot from the object store.

    Args:
        snapshot_hash: Hash returned by store_env_snapshot().

    Returns:
        The env dict, or None if not found.
    """
    raw = retrieve_bytes(snapshot_hash)
    if raw is None:
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


# ─────────────────────────────────────────────
#  Comparison
# ─────────────────────────────────────────────

def compare_envs(saved: dict, current: dict) -> dict:
    """
    Compare a saved environment snapshot against the current environment.
    Returns a structured diff — what changed, what's missing, what's new.

    Args:
        saved:   The environment dict stored at commit time.
        current: The environment dict captured right now.

    Returns:
        Dict with keys:
          clean          — True if environments are identical
          req_hash_match — True if pip freeze hash matches (fast check)
          python_match   — True if Python version matches
          python_diff    — dict describing version difference if any
          lib_diffs      — list of {lib, saved_ver, current_ver, status}
          missing_libs   — libs in saved but not in current
          new_libs       — libs in current but not in saved
          summary        — human-readable one-line summary
    """
    result: dict[str, Any] = {
        "clean":          True,
        "req_hash_match": False,
        "python_match":   False,
        "python_diff":    {},
        "lib_diffs":      [],
        "missing_libs":   [],
        "new_libs":       [],
        "summary":        "",
    }

    # ── Quick hash check ─────────────────────────────────────────
    saved_req_hash   = saved.get("req_hash", "")
    current_req_hash = current.get("req_hash", "")
    result["req_hash_match"] = (saved_req_hash == current_req_hash)
    if result["req_hash_match"]:
        result["clean"]   = True
        result["python_match"] = True
        result["summary"] = "Environment identical."
        return result

    result["clean"] = False

    # ── Python version ──────────────────────────────────────────
    saved_py   = saved.get("python", {}).get("version_info", [])
    current_py = current.get("python", {}).get("version_info", [])
    result["python_match"] = (saved_py == current_py)
    if not result["python_match"]:
        result["python_diff"] = {
            "saved":   ".".join(str(x) for x in saved_py),
            "current": ".".join(str(x) for x in current_py),
        }

    # ── Library versions ─────────────────────────────────────────
    saved_libs   = saved.get("libs", {})
    current_libs = current.get("libs", {})

    all_libs = set(saved_libs) | set(current_libs)
    diffs    = []

    for lib in sorted(all_libs):
        sv = saved_libs.get(lib)
        cv = current_libs.get(lib)

        if sv is None and cv is not None:
            result["new_libs"].append({"lib": lib, "version": cv})
        elif sv is not None and cv is None:
            result["missing_libs"].append({"lib": lib, "saved_version": sv})
            diffs.append({"lib": lib, "saved": sv, "current": None, "status": "MISSING"})
        elif sv != cv:
            diffs.append({"lib": lib, "saved": sv, "current": cv, "status": "VERSION_CHANGE"})

    result["lib_diffs"] = diffs

    # ── Summary ──────────────────────────────────────────────────
    parts = []
    if not result["python_match"]:
        d = result["python_diff"]
        parts.append(f"Python {d['saved']} → {d['current']}")
    if result["missing_libs"]:
        names = ", ".join(x["lib"] for x in result["missing_libs"])
        parts.append(f"missing: {names}")
    if diffs:
        changed = [d["lib"] for d in diffs if d["status"] == "VERSION_CHANGE"]
        if changed:
            parts.append(f"version changed: {', '.join(changed)}")

    result["summary"] = "; ".join(parts) if parts else "Minor differences in unlocked packages."
    return result


def print_env_diff(diff: dict) -> None:
    """
    Pretty-print an env diff dict to stdout.

    Args:
        diff: Dict returned by compare_envs().
    """
    if diff["clean"] or diff["req_hash_match"]:
        print("  Environment: identical ✓")
        return

    print(f"\n  {'─' * 56}")
    print(f"  Environment drift detected")
    print(f"  {'─' * 56}")

    if not diff["python_match"]:
        d = diff["python_diff"]
        print(f"  ⚠  Python:  saved={d['saved']}  current={d['current']}")

    for entry in diff["lib_diffs"]:
        if entry["status"] == "MISSING":
            print(f"  ✗  {entry['lib']:<30} saved={entry['saved']}  MISSING")
        else:
            print(
                f"  ~  {entry['lib']:<30} "
                f"saved={entry['saved']}  →  current={entry['current']}"
            )

    if diff["new_libs"]:
        for entry in diff["new_libs"][:5]:   # cap at 5 to avoid noise
            print(f"  +  {entry['lib']:<30} new={entry['version']}")
        extra = len(diff["new_libs"]) - 5
        if extra > 0:
            print(f"     ... and {extra} more new packages")

    print(f"\n  Summary: {diff['summary']}")
    print(f"  {'─' * 56}\n")


# ─────────────────────────────────────────────
#  Restore helpers
# ─────────────────────────────────────────────

def generate_requirements_txt(snapshot_hash: str) -> str | None:
    """
    Extract the pip freeze output from a stored snapshot and return it
    as a string ready to write to requirements.txt.

    Args:
        snapshot_hash: Hash of the stored env snapshot.

    Returns:
        requirements.txt content as a string, or None if not found.
    """
    env = load_env_snapshot(snapshot_hash)
    if not env:
        return None
    return env.get("pip_freeze", "")


def write_requirements_file(
    snapshot_hash: str,
    output_path: str | Path = "requirements_locked.txt",
) -> bool:
    """
    Write the exact pip freeze from a snapshot to a requirements file.
    You can then restore the environment with:
        pip install -r requirements_locked.txt

    Args:
        snapshot_hash: Hash of the stored env snapshot.
        output_path:   Where to write the file (default: requirements_locked.txt).

    Returns:
        True if written successfully, False if snapshot not found.
    """
    content = generate_requirements_txt(snapshot_hash)
    if content is None:
        print(f"[env] Snapshot {snapshot_hash[:8]}... not found.")
        return False

    output_path = Path(output_path)
    output_path.write_text(content)
    print(f"[env] Wrote {output_path} ({len(content.splitlines())} packages)")
    return True


def create_venv_script(
    snapshot_hash: str,
    venv_name: str = ".venv_restored",
    output_script: str | Path = "restore_env.sh",
) -> bool:
    """
    Generate a shell script that creates a fresh virtual environment
    and installs the exact locked requirements.

    Args:
        snapshot_hash:  Hash of the env snapshot to restore.
        venv_name:      Name of the virtual environment directory.
        output_script:  Where to write the shell script.

    Returns:
        True if the script was written, False if snapshot not found.
    """
    content = generate_requirements_txt(snapshot_hash)
    if content is None:
        print(f"[env] Snapshot {snapshot_hash[:8]}... not found.")
        return False

    env     = load_env_snapshot(snapshot_hash)
    py_ver  = env.get("python", {}).get("version_info", [3, 10, 0])
    py_str  = f"python{py_ver[0]}.{py_ver[1]}"

    req_file = f"{venv_name}_requirements.txt"
    script = f"""#!/bin/bash
# Auto-generated environment restoration script
# Snapshot: {snapshot_hash[:16]}...
# Python:   {'.'.join(str(x) for x in py_ver)}

set -e

echo "Creating virtual environment: {venv_name}"
{py_str} -m venv {venv_name} || python3 -m venv {venv_name}

echo "Installing locked requirements..."
{venv_name}/bin/pip install --upgrade pip
{venv_name}/bin/pip install -r {req_file}

echo "Done. Activate with: source {venv_name}/bin/activate"
"""

    # Write requirements file alongside the script
    req_path = Path(req_file)
    req_path.write_text(content)

    output_script = Path(output_script)
    output_script.write_text(script)
    output_script.chmod(0o755)

    print(f"[env] Wrote restore script: {output_script}")
    print(f"[env] Wrote requirements:   {req_path}")
    print(f"[env] Run: bash {output_script}")
    return True


# ─────────────────────────────────────────────
#  Checkout integration
# ─────────────────────────────────────────────

def check_env_on_checkout(
    commit_hash: str,
    warn_only: bool = True,
) -> dict | None:
    """
    After checking out a commit, compare the current environment against
    what was recorded at commit time. Prints a diff if drift is detected.

    Args:
        commit_hash: The commit that was just checked out.
        warn_only:   If True (default), print warnings but don't raise.
                     If False, raise RuntimeError on any drift.

    Returns:
        The diff dict (from compare_envs), or None if no snapshot stored.
    """
    commit = load_commit(commit_hash)
    if not commit:
        return None

    snapshot_hash = commit.get("env_snapshot_hash")
    if not snapshot_hash:
        # Older commit made without env locking — skip silently
        return None

    saved   = load_env_snapshot(snapshot_hash)
    if not saved:
        print(f"[env] Warning: env snapshot {snapshot_hash[:8]}... not found.")
        return None

    current = capture_env()
    diff    = compare_envs(saved, current)

    if not diff["clean"]:
        print(f"\n[env] ⚠  Environment drift detected for commit {commit_hash[:8]}...")
        print_env_diff(diff)
        print(
            f"[env] To restore the exact environment:\n"
            f"      python env_lock.py restore {commit_hash[:8]} --venv\n"
        )
        if not warn_only:
            raise RuntimeError(
                f"Environment mismatch for commit {commit_hash[:8]}. "
                f"See diff above. Use --force to override."
            )
    else:
        print(f"[env] Environment matches commit {commit_hash[:8]}... ✓")

    return diff


# ─────────────────────────────────────────────
#  Enhanced commit  (main entry point for Phase 6)
# ─────────────────────────────────────────────

def commit_with_env(
    weights_path: str | Path,
    message: str,
    metadata: dict[str, Any] | None = None,
    index: bool = True,
) -> str:
    """
    Full Phase 5 + 6 commit:
      1. Store weights using chunked deduplication (Phase 5)
      2. Capture the full environment snapshot (Phase 6)
      3. Create a commit with both weights_hash and env_snapshot_hash
      4. Index the commit in SQLite (Phase 3)

    This is the recommended commit function once both phases are active.
    Replaces commit_model() and commit_and_index() as the primary entry point.

    Args:
        weights_path: Path to the serialised model weights.
        message:      Commit message.
        metadata:     Optional metrics / hyperparameters dict.
        index:        Whether to also write to the SQLite index (default True).

    Returns:
        The commit hash.
    """
    import json
    from object_store import store_bytes as _store_bytes
    from commit import (
        _resolve_head, _advance_head, _current_branch,
        _serialise_commit, _now_iso
    )
    from dedup import store_chunked

    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    # 1. Store weights using chunked dedup
    print("[commit] Storing weights (chunked dedup)...")
    weights_hash = store_chunked(weights_path)

    # 2. Capture and store environment snapshot
    print("[commit] Capturing environment snapshot...")
    env          = capture_env()
    env_hash     = store_env_snapshot(env)
    print(f"[commit] Env snapshot: {env_hash[:8]}...  ({len(env.get('libs', {}))} tracked libs)")

    # 3. Build commit object (same structure as commit.py, extra field added)
    parent_hash = _resolve_head()
    commit = {
        "type":               "commit",
        "weights_hash":       weights_hash,
        "parent":             parent_hash,
        "message":            message,
        "timestamp":          _now_iso(),
        "weights_file":       weights_path.name,
        "metadata":           metadata or {},
        "env":                {
            "python_version": env["python"]["version"],
            "platform":       env["platform"]["system"],
        },
        "env_snapshot_hash":  env_hash,   # ← Phase 6 addition
        "storage":            "chunked",  # ← Phase 5 marker
    }

    commit_bytes = _serialise_commit(commit)
    commit_hash  = _store_bytes(commit_bytes)
    commit["hash"] = commit_hash
    _store_bytes(_serialise_commit(commit))
    _advance_head(commit_hash)

    branch       = _current_branch()
    branch_label = f" [{branch}]" if branch else " [detached]"
    print(f"[commit] {commit_hash[:8]}...{branch_label} — {message}")

    # 4. Index
    if index:
        try:
            from index import index_commit, index_branch
            index_commit(commit)
            if branch:
                index_branch(branch, commit_hash)
        except ImportError:
            pass   # index not available — not fatal

    return commit_hash


# ─────────────────────────────────────────────
#  CLI helpers (called from cli.py restore cmd)
# ─────────────────────────────────────────────

def restore_env_for_commit(
    ref: str,
    venv: bool = False,
    output_dir: str | Path = ".",
) -> bool:
    """
    Generate restoration artefacts for the environment locked in a commit.

    Args:
        ref:        Branch, tag, or commit hash.
        venv:       If True, also generate the venv creation script.
        output_dir: Where to write output files.

    Returns:
        True on success, False if the commit or snapshot is not found.
    """
    commit_hash = resolve_ref(ref)
    if not commit_hash:
        print(f"[env] Cannot resolve ref: '{ref}'")
        return False

    commit = load_commit(commit_hash)
    if not commit:
        print(f"[env] Commit not found: {commit_hash[:8]}...")
        return False

    snapshot_hash = commit.get("env_snapshot_hash")
    if not snapshot_hash:
        print(
            f"[env] Commit {commit_hash[:8]}... was made without env locking "
            f"(use commit_with_env() going forward)."
        )
        return False

    output_dir = Path(output_dir)
    req_path   = output_dir / f"requirements_{commit_hash[:8]}.txt"
    ok = write_requirements_file(snapshot_hash, req_path)

    if venv and ok:
        script_path = output_dir / f"restore_env_{commit_hash[:8]}.sh"
        create_venv_script(
            snapshot_hash,
            venv_name  = f".venv_{commit_hash[:8]}",
            output_script = script_path,
        )

    return ok


# ─────────────────────────────────────────────
#  Self-test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile
    import numpy as np
    from object_store import init_repo
    from commit import BRANCHES_DIR, HEAD_FILE

    print("=" * 60)
    print("env_lock.py — self test")
    print("=" * 60)

    init_repo()
    BRANCHES_DIR.mkdir(parents=True, exist_ok=True)
    if not HEAD_FILE.read_text().strip():
        HEAD_FILE.write_text("ref: branches/main")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # ── Capture current env ─────────────────────────────────────
        print("\n── capture_env() ──")
        env = capture_env()
        print(f"  Python:    {env['python']['version'][:30]}")
        print(f"  Platform:  {env['platform']['system']} {env['platform']['machine']}")
        print(f"  req_hash:  {env['req_hash'][:16]}...")
        print(f"  libs:      {list(env['libs'].keys())[:5]} ...")
        print(f"  cuda:      {env['cuda']}")

        # ── Store + load round-trip ─────────────────────────────────
        print("\n── store + load snapshot ──")
        h = store_env_snapshot(env)
        loaded = load_env_snapshot(h)
        assert loaded["req_hash"] == env["req_hash"], "Round-trip mismatch"
        print(f"  Stored:  {h[:16]}...")
        print(f"  Loaded:  req_hash matches ✓")

        # ── Compare identical env ───────────────────────────────────
        print("\n── compare_envs (identical) ──")
        diff = compare_envs(env, env)
        print(f"  clean: {diff['clean']}  summary: {diff['summary']}")
        assert diff["clean"]

        # ── Compare with a simulated drift ─────────────────────────
        print("\n── compare_envs (simulated drift) ──")
        import copy
        drifted = copy.deepcopy(env)
        drifted["python"]["version_info"] = [3, 9, 0]   # older Python
        drifted["libs"]["numpy"] = "1.21.0"              # older numpy
        if "torch" in drifted["libs"]:
            del drifted["libs"]["torch"]
        drifted["req_hash"] = "different"

        diff2 = compare_envs(env, drifted)
        print_env_diff(diff2)
        assert not diff2["clean"]

        # ── Full commit_with_env() ──────────────────────────────────
        print("\n── commit_with_env() ──")
        np.random.seed(1)
        w = np.random.rand(32, 32).astype(np.float32)
        weights_path = tmp / "model.npy"
        np.save(weights_path, w)

        try:
            from index import init_index
            init_index()
        except Exception:
            pass

        h_commit = commit_with_env(
            weights_path,
            "test commit with env lock",
            metadata={"accuracy": 0.88},
        )
        print(f"  Commit hash: {h_commit[:16]}...")

        # ── check_env_on_checkout ───────────────────────────────────
        print("\n── check_env_on_checkout() ──")
        diff3 = check_env_on_checkout(h_commit)
        # Should be clean since we just committed from this very env
        if diff3:
            print(f"  clean: {diff3['clean']}")

        # ── write_requirements_file ─────────────────────────────────
        print("\n── write_requirements_file() ──")
        commit_obj   = load_commit(h_commit)
        snap_hash    = commit_obj.get("env_snapshot_hash", "")
        req_out      = tmp / "requirements_locked.txt"
        ok = write_requirements_file(snap_hash, req_out)
        if ok:
            lines = req_out.read_text().splitlines()
            print(f"  Written {len(lines)} lines to {req_out.name}")

        # ── restore_env_for_commit ──────────────────────────────────
        print("\n── restore_env_for_commit() ──")
        ok = restore_env_for_commit(h_commit, venv=True, output_dir=tmp)
        if ok:
            print(f"  Restore artefacts written to {tmp}")

    print("\nAll tests passed.")