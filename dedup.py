"""
dedup.py — Chunk-level deduplication for large model weight files.

Phase 5 of the build. An optional replacement for store_object() /
retrieve_object() in object_store.py when storage efficiency matters.

THE PROBLEM WITH WHOLE-FILE STORAGE
────────────────────────────────────
Phase 1's store_object() hashes and stores the ENTIRE weights file as
one blob. This is simple and correct, but wasteful when you fine-tune:

  base_model.pt     → 7 GB   (stored once, good)
  fine_tuned_v1.pt  → 7 GB   (100% identical bytes except the last
  fine_tuned_v2.pt  → 7 GB    few layers) → 21 GB total on disk

With chunked dedup, those three files share 95%+ of their chunks. Only
the changed chunks are stored again. Total on disk: ~7.3 GB instead of 21.

HOW IT WORKS
────────────
1. Split the weights file into fixed-size chunks (default 64 MB).
2. Hash each chunk independently (SHA-256).
3. Store each unique chunk as a blob in the object store (same layout
   as before — objects/ab/cdef...).
4. Store a *manifest* file: a JSON list of [chunk_index, chunk_hash]
   pairs that describes how to reassemble the original file.
5. The manifest itself gets stored as a blob and its hash is what the
   commit points to — the manifest hash IS the weights_hash.

On retrieval, load the manifest, fetch each chunk in order, and
concatenate them back to disk. The original file is reconstructed
byte-for-byte.

WHAT STAYS THE SAME
───────────────────
The interface is identical to object_store.store_object() /
retrieve_object(). Callers (commit.py, index.py, cli.py) don't change
at all — just swap the import.

  # Before (whole-file):
  from object_store import store_object, retrieve_object

  # After (chunked dedup):
  from dedup import store_chunked as store_object
  from dedup import retrieve_chunked as retrieve_object

STORAGE LAYOUT
──────────────
  ~/.modelrepo/
    objects/
      ab/cdef...   ← individual 64 MB chunks (same as before)
      12/3456...   ← manifest JSON blobs (also in object store)

The manifest format:
  {
    "type":        "manifest",
    "file_name":   "model.pt",
    "file_size":   7516192768,
    "chunk_size":  67108864,
    "chunk_count": 112,
    "chunks": [
      {"index": 0, "hash": "a3f9b2..."},
      {"index": 1, "hash": "ff04ab..."},
      ...
    ]
  }
"""

import json
from pathlib import Path
from typing import Iterator

from object_store import (
    OBJECTS_DIR,
    REPO_DIR,
    _object_path,
    hash_bytes,
    store_bytes,
    retrieve_bytes,
    object_exists,
)

# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────

DEFAULT_CHUNK_SIZE = 64 * 1024 * 1024   # 64 MB — balances granularity vs overhead
MIN_CHUNK_SIZE     = 1  * 1024 * 1024   # 1 MB floor (smaller = more overhead)
MAX_CHUNK_SIZE     = 512 * 1024 * 1024  # 512 MB ceiling


# ─────────────────────────────────────────────
#  Chunking
# ─────────────────────────────────────────────

def _iter_chunks(
    file_path: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Iterator[tuple[int, bytes]]:
    """
    Read a file in fixed-size chunks and yield (index, chunk_bytes).
    The last chunk may be smaller than chunk_size.

    Args:
        file_path:  File to read.
        chunk_size: Bytes per chunk.

    Yields:
        (chunk_index, chunk_bytes) tuples.
    """
    with open(file_path, "rb") as f:
        idx = 0
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield idx, chunk
            idx += 1


def _store_chunk(chunk_data: bytes) -> str:
    """
    Store a single chunk blob and return its hash.
    Identical to store_bytes() — reuses the object store directly.

    Args:
        chunk_data: Raw bytes for this chunk.

    Returns:
        SHA-256 hash of the chunk.
    """
    return store_bytes(chunk_data)


# ─────────────────────────────────────────────
#  Manifest
# ─────────────────────────────────────────────

def _build_manifest(
    file_path: Path,
    chunk_hashes: list[str],
    chunk_size: int,
) -> dict:
    """
    Build the manifest dict describing a chunked file.

    Args:
        file_path:    Original file path (for name and size metadata).
        chunk_hashes: Ordered list of SHA-256 hashes, one per chunk.
        chunk_size:   The chunk size used when splitting.

    Returns:
        Manifest dict (not yet stored).
    """
    return {
        "type":        "manifest",
        "file_name":   file_path.name,
        "file_size":   file_path.stat().st_size,
        "chunk_size":  chunk_size,
        "chunk_count": len(chunk_hashes),
        "chunks": [
            {"index": i, "hash": h}
            for i, h in enumerate(chunk_hashes)
        ],
    }


def _store_manifest(manifest: dict) -> str:
    """
    Serialise a manifest to JSON and store it as a blob.

    Args:
        manifest: The manifest dict.

    Returns:
        Hash of the manifest blob — this becomes the weights_hash
        that a commit points to.
    """
    data = json.dumps(manifest, sort_keys=True, indent=2).encode("utf-8")
    return store_bytes(data)


def _load_manifest(manifest_hash: str) -> dict | None:
    """
    Load and deserialise a manifest from the object store.

    Args:
        manifest_hash: Hash returned by _store_manifest().

    Returns:
        Manifest dict, or None if not found.
    """
    raw = retrieve_bytes(manifest_hash)
    if raw is None:
        return None
    try:
        obj = json.loads(raw.decode("utf-8"))
        if obj.get("type") == "manifest":
            return obj
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass
    return None


# ─────────────────────────────────────────────
#  Core API  (drop-in for store_object / retrieve_object)
# ─────────────────────────────────────────────

def store_chunked(
    file_path: str | Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    verbose: bool = True,
) -> str:
    """
    Split a weights file into chunks, store each chunk once, then store
    a manifest describing the chunk sequence. Returns the manifest hash.

    This is a drop-in replacement for object_store.store_object().
    The returned hash is used exactly the same way in commits.

    Already-stored chunks are skipped automatically — this is where the
    deduplication happens. If you commit a fine-tuned model that shares
    95% of its weights with the base, only the changed chunks are written.

    Args:
        file_path:  Path to the weights file.
        chunk_size: Bytes per chunk (default 64 MB).
        verbose:    Print progress lines.

    Returns:
        Manifest hash (the weights_hash stored in the commit).

    Raises:
        FileNotFoundError: If file_path does not exist.
        ValueError:        If chunk_size is out of the allowed range.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not (MIN_CHUNK_SIZE <= chunk_size <= MAX_CHUNK_SIZE):
        raise ValueError(
            f"chunk_size must be between {MIN_CHUNK_SIZE} and {MAX_CHUNK_SIZE} bytes."
        )

    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    if verbose:
        print(f"[dedup] Chunking {file_path.name} ({file_size_mb:.1f} MB) "
              f"into {chunk_size // (1024*1024)} MB chunks...")

    chunk_hashes: list[str] = []
    new_chunks   = 0
    reused_chunks = 0

    for idx, chunk_data in _iter_chunks(file_path, chunk_size):
        chunk_hash = hash_bytes(chunk_data)

        if object_exists(chunk_hash):
            reused_chunks += 1
        else:
            _store_chunk(chunk_data)
            new_chunks += 1

        chunk_hashes.append(chunk_hash)

    # Build and store the manifest
    manifest      = _build_manifest(file_path, chunk_hashes, chunk_size)
    manifest_hash = _store_manifest(manifest)

    total = new_chunks + reused_chunks
    if verbose:
        print(
            f"[dedup] {total} chunks total — "
            f"{new_chunks} new, {reused_chunks} reused "
            f"({reused_chunks / max(total, 1) * 100:.0f}% dedup ratio)"
        )
        print(f"[dedup] Manifest hash: {manifest_hash[:8]}...")

    return manifest_hash


def retrieve_chunked(
    manifest_hash: str,
    destination_path: str | Path,
    verbose: bool = True,
) -> bool:
    """
    Reassemble a weights file from its chunks using the manifest.

    This is a drop-in replacement for object_store.retrieve_object().
    Pass the manifest hash (what was returned by store_chunked) and a
    destination path — the original file is reconstructed byte-for-byte.

    Args:
        manifest_hash:    Hash of the manifest (the commit's weights_hash).
        destination_path: Where to write the reassembled file.
        verbose:          Print progress lines.

    Returns:
        True if successful, False if manifest or any chunk is missing.
    """
    manifest = _load_manifest(manifest_hash)
    if manifest is None:
        # Fall back: maybe this hash points to a whole-file blob (Phase 1
        # commit). Try the legacy retrieve path.
        from object_store import retrieve_object
        return retrieve_object(manifest_hash, destination_path)

    destination_path = Path(destination_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    chunks      = sorted(manifest["chunks"], key=lambda c: c["index"])
    total       = len(chunks)
    expected_sz = manifest.get("file_size", 0)

    if verbose:
        print(
            f"[dedup] Reassembling {manifest['file_name']} "
            f"from {total} chunks → {destination_path}"
        )

    with open(destination_path, "wb") as out:
        for entry in chunks:
            chunk_hash = entry["hash"]
            chunk_data = retrieve_bytes(chunk_hash)
            if chunk_data is None:
                print(
                    f"[dedup] ERROR: chunk {entry['index']} "
                    f"({chunk_hash[:8]}...) missing from object store."
                )
                destination_path.unlink(missing_ok=True)
                return False
            out.write(chunk_data)

    actual_sz = destination_path.stat().st_size
    if expected_sz and actual_sz != expected_sz:
        print(
            f"[dedup] WARNING: size mismatch — "
            f"expected {expected_sz} bytes, got {actual_sz}."
        )
        return False

    if verbose:
        print(f"[dedup] Reassembled {actual_sz / (1024*1024):.1f} MB OK.")
    return True


def is_manifest(weights_hash: str) -> bool:
    """
    Return True if a weights_hash points to a chunked manifest rather
    than a whole-file blob. Useful for deciding which retrieve path to use.

    Args:
        weights_hash: The hash stored in a commit's weights_hash field.

    Returns:
        True if it's a manifest, False if it's a legacy whole-file blob.
    """
    return _load_manifest(weights_hash) is not None


# ─────────────────────────────────────────────
#  Dedup statistics
# ─────────────────────────────────────────────

def dedup_stats(manifest_hash: str) -> dict | None:
    """
    Return deduplication statistics for a single committed model.

    Shows how many of this model's chunks are shared with other
    models already in the store (i.e. were NOT newly stored when
    this commit was made — though we can only approximate this after
    the fact by checking which chunks exist multiple times).

    Args:
        manifest_hash: Hash of the manifest to analyse.

    Returns:
        Dict with keys: file_name, file_size_mb, chunk_count,
        unique_chunks, chunk_size_mb, or None if not a manifest.
    """
    manifest = _load_manifest(manifest_hash)
    if not manifest:
        return None

    chunk_hashes = [c["hash"] for c in manifest["chunks"]]
    unique       = len(set(chunk_hashes))   # dedup within the file itself

    return {
        "file_name":     manifest.get("file_name", ""),
        "file_size_mb":  round(manifest.get("file_size", 0) / (1024 * 1024), 2),
        "chunk_count":   manifest.get("chunk_count", 0),
        "unique_chunks": unique,
        "chunk_size_mb": round(manifest.get("chunk_size", 0) / (1024 * 1024), 1),
    }


def repo_dedup_ratio() -> dict:
    """
    Scan all manifests in the object store and compute the overall
    deduplication ratio across the entire repo.

    Compares:
      - logical_size: sum of all file sizes across all manifests
        (what you'd use if storing every file independently)
      - physical_size: actual bytes stored (unique chunks only)

    Returns:
        Dict with keys: manifests, logical_size_mb, physical_size_mb,
        saved_mb, dedup_ratio (1.0 = no saving, 3.0 = 3× compression).
    """
    from object_store import list_objects

    all_chunk_hashes: set[str] = set()
    logical_bytes = 0
    manifest_count = 0

    for obj_hash in list_objects():
        raw = retrieve_bytes(obj_hash)
        if not raw:
            continue
        try:
            obj = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

        if obj.get("type") != "manifest":
            continue

        manifest_count += 1
        logical_bytes  += obj.get("file_size", 0)
        for chunk in obj.get("chunks", []):
            all_chunk_hashes.add(chunk["hash"])

    # Physical size = sum of unique chunk blob sizes
    physical_bytes = sum(
        _object_path(h).stat().st_size
        for h in all_chunk_hashes
        if _object_path(h).exists()
    )

    ratio = logical_bytes / max(physical_bytes, 1)

    return {
        "manifests":        manifest_count,
        "logical_size_mb":  round(logical_bytes  / (1024 * 1024), 2),
        "physical_size_mb": round(physical_bytes / (1024 * 1024), 2),
        "saved_mb":         round((logical_bytes - physical_bytes) / (1024 * 1024), 2),
        "dedup_ratio":      round(ratio, 2),
    }


# ─────────────────────────────────────────────
#  Verify a chunked model
# ─────────────────────────────────────────────

def verify_chunked(manifest_hash: str) -> bool:
    """
    Verify that every chunk referenced by a manifest exists in the
    object store and that its content still matches its hash.

    Args:
        manifest_hash: The manifest hash to verify.

    Returns:
        True if all chunks are present and intact, False otherwise.
    """
    manifest = _load_manifest(manifest_hash)
    if manifest is None:
        print(f"[dedup] {manifest_hash[:8]}... is not a manifest.")
        return False

    all_ok   = True
    total    = len(manifest["chunks"])
    corrupt  = 0
    missing  = 0

    for entry in manifest["chunks"]:
        chunk_hash = entry["hash"]
        chunk_path = _object_path(chunk_hash)

        if not chunk_path.exists():
            print(f"[dedup] chunk {entry['index']} ({chunk_hash[:8]}...) MISSING")
            missing += 1
            all_ok   = False
            continue

        # Re-hash the stored chunk
        actual = hash_bytes(chunk_path.read_bytes())
        if actual != chunk_hash:
            print(f"[dedup] chunk {entry['index']} ({chunk_hash[:8]}...) CORRUPTED")
            corrupt += 1
            all_ok   = False

    status = "OK" if all_ok else f"FAILED ({missing} missing, {corrupt} corrupted)"
    print(
        f"[dedup] {manifest_hash[:8]}...  {total} chunks  {status}"
    )
    return all_ok


# ─────────────────────────────────────────────
#  Self-test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile
    import numpy as np
    from object_store import init_repo

    print("=" * 60)
    print("dedup.py — self test")
    print("=" * 60)

    init_repo()

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # ── Create two "models" that share most of their weights ────
        # Simulate a fine-tune: base has 10 chunks, fine-tune differs
        # only in the last chunk.
        CHUNK = 1 * 1024 * 1024  # 1 MB chunks for the test

        # 10 MB base model
        np.random.seed(42)
        base_data    = np.random.bytes(10 * CHUNK)
        base_path    = tmp / "base_model.bin"
        base_path.write_bytes(base_data)

        # Fine-tuned model: identical first 9 MB, different last 1 MB
        np.random.seed(99)
        ft_data      = base_data[:9 * CHUNK] + np.random.bytes(CHUNK)
        ft_path      = tmp / "fine_tuned.bin"
        ft_path.write_bytes(ft_data)

        print(f"\nBase model:       {len(base_data) / (1024*1024):.0f} MB")
        print(f"Fine-tuned model: {len(ft_data)  / (1024*1024):.0f} MB")
        print(f"Expected dedup:   9/10 chunks shared\n")

        # ── Store base model ────────────────────────────────────────
        print("── Storing base model ──")
        base_manifest_hash = store_chunked(base_path, chunk_size=CHUNK)

        # ── Store fine-tuned model ──────────────────────────────────
        print("\n── Storing fine-tuned model ──")
        ft_manifest_hash = store_chunked(ft_path, chunk_size=CHUNK)

        # ── Round-trip: retrieve base ───────────────────────────────
        print("\n── Retrieving base model ──")
        restored_base = tmp / "base_restored.bin"
        ok = retrieve_chunked(base_manifest_hash, restored_base)
        assert ok, "Base retrieval failed"
        assert restored_base.read_bytes() == base_data, "Base round-trip mismatch"
        print("Base round-trip: OK")

        # ── Round-trip: retrieve fine-tuned ────────────────────────
        print("\n── Retrieving fine-tuned model ──")
        restored_ft = tmp / "ft_restored.bin"
        ok = retrieve_chunked(ft_manifest_hash, restored_ft)
        assert ok, "Fine-tune retrieval failed"
        assert restored_ft.read_bytes() == ft_data, "Fine-tune round-trip mismatch"
        print("Fine-tune round-trip: OK")

        # ── Dedup stats per model ───────────────────────────────────
        print("\n── dedup_stats() ──")
        for label, mh in [("base", base_manifest_hash), ("fine-tune", ft_manifest_hash)]:
            s = dedup_stats(mh)
            print(f"  {label}: {s}")

        # ── Repo-wide ratio ─────────────────────────────────────────
        print("\n── repo_dedup_ratio() ──")
        ratio = repo_dedup_ratio()
        for k, v in ratio.items():
            print(f"  {k}: {v}")
        assert ratio["dedup_ratio"] > 1.0, "Expected dedup savings"
        print(f"  Dedup working: {ratio['dedup_ratio']}× ratio ✓")

        # ── Verify integrity ────────────────────────────────────────
        print("\n── verify_chunked() ──")
        assert verify_chunked(base_manifest_hash)
        assert verify_chunked(ft_manifest_hash)

        # ── is_manifest() ───────────────────────────────────────────
        assert is_manifest(base_manifest_hash), "Should be manifest"
        assert not is_manifest("0" * 64),       "Should not be manifest"
        print("is_manifest(): OK")

    print("\nAll tests passed.")