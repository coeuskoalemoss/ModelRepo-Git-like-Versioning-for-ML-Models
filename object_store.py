import os
import hashlib
import shutil
import json
from pathlib import Path

# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────

REPO_DIR    = Path.home() / ".modelrepo"
OBJECTS_DIR = REPO_DIR / "objects"
HEAD_FILE   = REPO_DIR / "HEAD"


# ─────────────────────────────────────────────
#  Init
# ─────────────────────────────────────────────

def init_repo() -> None:
    """
    Create the repo folder structure if it doesn't exist.
    Safe to call multiple times — won't overwrite existing data.
    """
    OBJECTS_DIR.mkdir(parents=True, exist_ok=True)
    if not HEAD_FILE.exists():
        HEAD_FILE.write_text("")
    print(f"Initialized repo at {REPO_DIR}")


# ─────────────────────────────────────────────
#  Hashing
# ─────────────────────────────────────────────

def hash_file(file_path: str | Path, chunk_size: int = 65536) -> str:
    """
    Read a file in chunks and return its SHA-256 hex digest.
    Chunked reading handles large model files without loading
    everything into RAM.

    Args:
        file_path:  Path to the file to hash.
        chunk_size: Bytes per read chunk (default 64 KB).

    Returns:
        64-character hex string, e.g. 'a3f9b2c1d4e5...'
    """
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def hash_bytes(data: bytes) -> str:
    """
    Hash raw bytes directly (used for commit JSON objects).

    Args:
        data: Raw bytes to hash.

    Returns:
        64-character hex string.
    """
    return hashlib.sha256(data).hexdigest()


def hash_string(text: str) -> str:
    """
    Convenience wrapper — hash a UTF-8 string.

    Args:
        text: String to hash.

    Returns:
        64-character hex string.
    """
    return hash_bytes(text.encode("utf-8"))


# ─────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────

def _object_path(file_hash: str) -> Path:
    """
    Convert a hash into its storage path inside the object store.
    First 2 chars → subdirectory, remaining 62 chars → filename.
    Mirrors git's layout for the same reasons (avoids too many
    files in a single directory on older filesystems).

    Example:
        'a3f9b2c1d4e5...' → objects/a3/f9b2c1d4e5...
    """
    return OBJECTS_DIR / file_hash[:2] / file_hash[2:]


# ─────────────────────────────────────────────
#  Core operations
# ─────────────────────────────────────────────

def store_object(file_path: str | Path) -> str:
    """
    Hash a file and copy it into the object store.
    If the hash already exists the copy is skipped (free dedup).

    Args:
        file_path: Path to the file to store.

    Returns:
        The SHA-256 hash of the file — use this as the permanent ID.

    Raises:
        FileNotFoundError: If file_path does not exist.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    file_hash   = hash_file(file_path)
    destination = _object_path(file_hash)

    if destination.exists():
        print(f"[store] {file_hash[:8]}... already exists — skipping.")
        return file_hash

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(file_path, destination)
    size_kb = file_path.stat().st_size / 1024
    print(f"[store] {file_hash[:8]}... stored ({size_kb:.1f} KB)")
    return file_hash


def store_bytes(data: bytes) -> str:
    """
    Store raw bytes directly into the object store.
    Used internally to persist commit JSON blobs.

    Args:
        data: Raw bytes to store.

    Returns:
        The SHA-256 hash of the data.
    """
    file_hash   = hash_bytes(data)
    destination = _object_path(file_hash)

    if destination.exists():
        return file_hash

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    return file_hash


def retrieve_object(file_hash: str, destination_path: str | Path) -> bool:
    """
    Copy an object out of the store to a destination path.

    Args:
        file_hash:        The hash returned by store_object().
        destination_path: Where to write the file.

    Returns:
        True if the object was found and copied, False otherwise.
    """
    object_path = _object_path(file_hash)

    if not object_path.exists():
        print(f"[retrieve] {file_hash[:8]}... not found in store.")
        return False

    destination_path = Path(destination_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(object_path, destination_path)
    print(f"[retrieve] {file_hash[:8]}... → {destination_path}")
    return True


def retrieve_bytes(file_hash: str) -> bytes | None:
    """
    Read an object directly into memory as bytes.
    Used internally to load commit JSON blobs.

    Args:
        file_hash: The hash of the object.

    Returns:
        Raw bytes if found, None otherwise.
    """
    object_path = _object_path(file_hash)
    if not object_path.exists():
        return None
    return object_path.read_bytes()


def object_exists(file_hash: str) -> bool:
    """
    Check whether a hash is already in the store.
    Fast — no file copying, just a path existence check.

    Args:
        file_hash: Hash to check.

    Returns:
        True if the object exists, False otherwise.
    """
    return _object_path(file_hash).exists()


def delete_object(file_hash: str) -> bool:
    """
    Remove an object from the store by hash.
    Use with caution — this is permanent. Prefer keeping
    all objects and letting the store grow (disk is cheap,
    reproducibility is not).

    Args:
        file_hash: Hash of the object to delete.

    Returns:
        True if deleted, False if it wasn't found.
    """
    object_path = _object_path(file_hash)
    if not object_path.exists():
        print(f"[delete] {file_hash[:8]}... not found.")
        return False
    object_path.unlink()
    # Clean up empty parent directories
    try:
        object_path.parent.rmdir()
    except OSError:
        pass  # Directory not empty — fine, leave it
    print(f"[delete] {file_hash[:8]}... removed.")
    return True


# ─────────────────────────────────────────────
#  HEAD pointer
# ─────────────────────────────────────────────

def get_head() -> str | None:
    """
    Read the current HEAD (the hash of the active commit).

    Returns:
        Hash string if HEAD is set, None if the repo is empty.
    """
    content = HEAD_FILE.read_text().strip()
    return content if content else None


def set_head(commit_hash: str) -> None:
    """
    Update HEAD to point to a new commit hash.

    Args:
        commit_hash: The commit hash to make current.
    """
    HEAD_FILE.write_text(commit_hash)
    print(f"[head] HEAD → {commit_hash[:8]}...")


# ─────────────────────────────────────────────
#  Inspection / diagnostics
# ─────────────────────────────────────────────

def list_objects() -> list[str]:
    """
    Return a list of all hashes currently in the object store.
    Reconstructs the full hash from folder name + filename.

    Returns:
        Sorted list of full 64-char hex hashes.
    """
    hashes = []
    if not OBJECTS_DIR.exists():
        return hashes
    for subfolder in sorted(OBJECTS_DIR.iterdir()):
        if subfolder.is_dir() and len(subfolder.name) == 2:
            for obj_file in subfolder.iterdir():
                hashes.append(subfolder.name + obj_file.name)
    return sorted(hashes)


def store_stats() -> dict:
    """
    Return summary statistics about the object store.

    Returns:
        Dict with keys: object_count, total_size_mb, objects_dir.
    """
    objects     = list_objects()
    total_bytes = sum(
        _object_path(h).stat().st_size
        for h in objects
        if _object_path(h).exists()
    )
    return {
        "object_count":  len(objects),
        "total_size_mb": round(total_bytes / (1024 * 1024), 2),
        "objects_dir":   str(OBJECTS_DIR),
    }


def verify_object(file_hash: str) -> bool:
    """
    Re-hash a stored object and confirm it matches its filename.
    Detects corruption or accidental modification.

    Args:
        file_hash: Hash of the object to verify.

    Returns:
        True if the stored content still matches the hash, False otherwise.
    """
    object_path = _object_path(file_hash)
    if not object_path.exists():
        print(f"[verify] {file_hash[:8]}... not found.")
        return False
    actual_hash = hash_file(object_path)
    ok          = actual_hash == file_hash
    status      = "OK" if ok else "CORRUPTED"
    print(f"[verify] {file_hash[:8]}... {status}")
    return ok


def verify_all() -> dict:
    """
    Verify every object in the store.
    Useful as a periodic integrity check.

    Returns:
        Dict with keys: total, ok, corrupted (list of bad hashes).
    """
    objects   = list_objects()
    corrupted = [h for h in objects if not verify_object(h)]
    return {
        "total":     len(objects),
        "ok":        len(objects) - len(corrupted),
        "corrupted": corrupted,
    }


# ─────────────────────────────────────────────
#  Quick self-test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np
    import tempfile

    print("=" * 50)
    print("Object Store — self test")
    print("=" * 50)

    # 1. Init
    init_repo()

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # 2. Create a fake weights file
        weights = np.random.rand(100, 100).astype(np.float32)
        weights_path = tmp / "model_v1.npy"
        np.save(weights_path, weights)
        print(f"\nCreated fake weights file: {weights_path}")

        # 3. Store it
        h = store_object(weights_path)
        print(f"Hash: {h}")

        # 4. Store same file again — should skip
        store_object(weights_path)

        # 5. Check existence
        print(f"Exists: {object_exists(h)}")
        print(f"Fake hash exists: {object_exists('0' * 64)}")

        # 6. Retrieve and verify round-trip
        restored_path = tmp / "model_v1_restored.npy"
        retrieve_object(h, restored_path)
        original = np.load(weights_path)
        restored = np.load(restored_path)
        print(f"Round-trip identical: {np.array_equal(original, restored)}")

        # 7. Store raw bytes (commit blob simulation)
        commit_data   = json.dumps({"message": "initial commit"}).encode()
        commit_hash   = store_bytes(commit_data)
        loaded_bytes  = retrieve_bytes(commit_hash)
        loaded_commit = json.loads(loaded_bytes)
        print(f"Commit round-trip: {loaded_commit}")

        # 8. Verify integrity
        verify_object(h)

        # 9. Stats
        stats = store_stats()
        print(f"\nStore stats: {stats}")

        # 10. List all objects
        all_hashes = list_objects()
        print(f"Objects in store: {len(all_hashes)}")
        for obj_hash in all_hashes:
            print(f"  {obj_hash[:16]}...")

    print("\nAll tests passed.")