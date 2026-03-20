#!/usr/bin/env python3
"""
cli.py — Command-line interface for the ML model versioning system.

Phase 4 of the build. Thin wrapper over object_store, commit, and index.
All business logic lives in those modules — the CLI just parses arguments
and calls the right functions.

USAGE
─────
  python cli.py init
  python cli.py commit  model.pt  -m "baseline ResNet50"  --accuracy 0.87
  python cli.py log
  python cli.py log     --branch experiment-aug  --limit 10
  python cli.py checkout a3f9b2c1
  python cli.py checkout a3f9b2c1  --out restored.pt  --detach
  python cli.py diff    HEAD  experiment-aug
  python cli.py branch  --list
  python cli.py branch  --create experiment-lr  --from HEAD
  python cli.py branch  --switch experiment-lr
  python cli.py tag     v1.0  --message "production release"
  python cli.py search  --min-accuracy 0.92  --dataset imagenet-v2
  python cli.py best    --metric accuracy  --top 5
  python cli.py stats
  python cli.py lineage a3f9b2c1
  python cli.py rebuild
  python cli.py verify

Install as a global command (optional):
  pip install -e .   # if you add a pyproject.toml with [scripts] entry
  # or just:
  alias model="python /path/to/cli.py"
"""

import argparse
import json
import sys
from pathlib import Path


# ─────────────────────────────────────────────
#  Lazy imports — only load heavy modules when
#  the relevant subcommand is actually invoked
# ─────────────────────────────────────────────

def _bootstrap():
    """Ensure the repo + index exist. Called at the top of every command."""
    from object_store import init_repo
    from index import init_index
    init_repo()
    init_index()


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def _die(message: str, code: int = 1) -> None:
    """Print an error message to stderr and exit."""
    print(f"error: {message}", file=sys.stderr)
    sys.exit(code)


def _ok(message: str) -> None:
    """Print a success message."""
    print(message)


def _parse_metadata_args(args: argparse.Namespace) -> dict:
    """
    Pull the well-known metric/hyperparam flags off the parsed args
    and return them as a metadata dict (omitting None values).
    Also merges any extra --meta KEY=VALUE pairs.
    """
    meta = {}
    for field in ("accuracy", "loss", "lr", "epochs", "dataset", "model_arch"):
        v = getattr(args, field, None)
        if v is not None:
            meta[field] = v

    # --meta KEY=VALUE  (zero or more)
    for kv in getattr(args, "meta", []) or []:
        if "=" not in kv:
            _die(f"--meta value must be KEY=VALUE, got: '{kv}'")
        k, v = kv.split("=", 1)
        # Try to coerce to int/float before storing as string
        try:
            v = int(v)
        except ValueError:
            try:
                v = float(v)
            except ValueError:
                pass
        meta[k] = v

    return meta


# ─────────────────────────────────────────────
#  Subcommand handlers
# ─────────────────────────────────────────────

def cmd_init(args: argparse.Namespace) -> None:
    """model init — initialise a new repo in ~/.modelrepo."""
    from object_store import init_repo, REPO_DIR
    from index import init_index
    from commit import BRANCHES_DIR, HEAD_FILE

    init_repo()
    init_index()

    # Point HEAD at 'main' branch if not already set
    BRANCHES_DIR.mkdir(parents=True, exist_ok=True)
    if not HEAD_FILE.read_text().strip():
        HEAD_FILE.write_text("ref: branches/main")

    _ok(f"Initialised model repo at {REPO_DIR}")
    _ok("Default branch: main")


def cmd_commit(args: argparse.Namespace) -> None:
    """model commit <weights_file> -m <message> [metric flags]"""
    _bootstrap()

    weights = Path(args.weights)
    if not weights.exists():
        _die(f"File not found: {weights}")

    if not args.message:
        _die("Commit message is required. Use -m 'your message'.")

    metadata = _parse_metadata_args(args)

    from index import commit_and_index
    commit_hash = commit_and_index(weights, args.message, metadata or None)
    _ok(f"\nCommit:  {commit_hash}")
    _ok(f"Message: {args.message}")
    if metadata:
        for k, v in metadata.items():
            _ok(f"         {k} = {v}")


def cmd_log(args: argparse.Namespace) -> None:
    """model log [--branch <name>] [--limit N] [--json]"""
    _bootstrap()

    ref   = args.branch or "HEAD"
    limit = args.limit  or 20

    if args.json:
        from index import fast_log
        rows = fast_log(ref, limit)
        print(json.dumps(rows, indent=2))
    else:
        from index import print_fast_log
        print_fast_log(ref, limit)


def cmd_checkout(args: argparse.Namespace) -> None:
    """model checkout <ref> [--out <path>] [--detach]"""
    _bootstrap()

    from commit import checkout
    destination = args.out or "model_checkout.pt"
    commit = checkout(args.ref, destination=destination, detach=args.detach)

    if commit is None:
        _die(f"Could not check out ref: '{args.ref}'")

    _ok(f"\nChecked out: {commit.get('hash', '')[:8]}...")
    _ok(f"Message:     {commit.get('message', '')}")
    _ok(f"Weights →    {destination}")
    meta = commit.get("metadata", {})
    if meta:
        _ok("Metadata:")
        for k, v in meta.items():
            _ok(f"  {k} = {v}")


def cmd_diff(args: argparse.Namespace) -> None:
    """model diff <ref_a> <ref_b> [--json]"""
    _bootstrap()

    from commit import diff, print_diff
    if args.json:
        d = diff(args.ref_a, args.ref_b)
        print(json.dumps(d, indent=2))
    else:
        print_diff(args.ref_a, args.ref_b)


def cmd_branch(args: argparse.Namespace) -> None:
    """
    model branch --list
    model branch --create <name> [--from <ref>]
    model branch --switch <name>
    model branch --delete <name>
    """
    _bootstrap()

    from commit import (
        create_branch,
        switch_branch,
        list_branches,
        print_branches,
        BRANCHES_DIR,
    )

    if args.list or (not args.create and not args.switch and not args.delete):
        print_branches()
        return

    if args.create:
        from_ref = args.from_ref or "HEAD"
        ok = create_branch(args.create, from_ref=from_ref)
        if ok:
            _ok(f"Created branch '{args.create}' from {from_ref}")
        return

    if args.switch:
        ok = switch_branch(args.switch)
        if not ok:
            _die(f"Branch '{args.switch}' does not exist.")
        return

    if args.delete:
        branch_file = BRANCHES_DIR / args.delete
        if not branch_file.exists():
            _die(f"Branch '{args.delete}' does not exist.")
        # Safety: refuse to delete the currently active branch
        from commit import _current_branch
        if _current_branch() == args.delete:
            _die(
                f"Cannot delete the currently active branch '{args.delete}'. "
                "Switch to another branch first."
            )
        branch_file.unlink()
        _ok(f"Deleted branch '{args.delete}'")


def cmd_tag(args: argparse.Namespace) -> None:
    """
    model tag <name> [--ref <ref>] [--message <msg>] [--list]
    """
    _bootstrap()

    if args.list:
        from index import get_tags
        tags = get_tags()
        if not tags:
            _ok("No tags yet.")
            return
        print(f"\n  {'NAME':<20} {'COMMIT':<12} MESSAGE")
        print(f"  {'─'*20} {'─'*12} {'─'*30}")
        for t in tags:
            print(
                f"  {t['name']:<20} "
                f"{t['commit_hash'][:10]:<12} "
                f"{t['message']}"
            )
        print()
        return

    if not args.name:
        _die("Provide a tag name, or use --list to list all tags.")

    from commit import tag as commit_tag
    from index import index_tag, resolve_ref

    ref     = args.ref or "HEAD"
    message = args.message or ""
    ok      = commit_tag(args.name, ref=ref, message=message)
    if ok:
        commit_hash = resolve_ref(ref)
        if commit_hash:
            index_tag(args.name, commit_hash, message)
        _ok(f"Tagged '{args.name}' → {(commit_hash or 'unknown')[:8]}...")


def cmd_search(args: argparse.Namespace) -> None:
    """model search [--min-accuracy N] [--max-loss N] [--dataset X] ..."""
    _bootstrap()

    from index import search, print_search_results

    results = search(
        min_accuracy     = args.min_accuracy,
        max_loss         = args.max_loss,
        lr               = args.lr,
        dataset          = args.dataset,
        model_arch       = args.model_arch,
        message_contains = args.message,
        since            = args.since,
        until            = args.until,
        limit            = args.limit or 50,
    )

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_search_results(results)


def cmd_best(args: argparse.Namespace) -> None:
    """model best [--metric accuracy|loss|lr|epochs] [--top N]"""
    _bootstrap()

    from index import best_model

    metric = args.metric or "accuracy"
    top_n  = args.top    or 5

    try:
        rows = best_model(metric, top_n=top_n)
    except ValueError as e:
        _die(str(e))

    if args.json:
        print(json.dumps(rows, indent=2))
        return

    order_label = "↑ higher is better" if metric != "loss" else "↓ lower is better"
    print(f"\n  Top {top_n} by {metric}  ({order_label})\n")
    print(f"  {'HASH':<12} {metric.upper():>10}  {'DATASET':<18} MESSAGE")
    print(f"  {'─'*12} {'─'*10}  {'─'*18} {'─'*30}")

    for row in rows:
        val  = row.get(metric)
        val_s = f"{val:.4f}" if isinstance(val, float) else str(val or "—")
        ds   = (row.get("dataset") or "—")[:18]
        msg  = (row.get("message") or "")[:40]
        h    = row["hash"][:10]
        print(f"  {h:<12} {val_s:>10}  {ds:<18} {msg}")
    print()


def cmd_stats(args: argparse.Namespace) -> None:
    """model stats — aggregate summary of the whole repo."""
    _bootstrap()

    from index import stats, compare_datasets, get_branch_tips, get_tags

    s = stats()

    print(f"\n{'─' * 50}")
    print(f"  Model Repo — Statistics")
    print(f"{'─' * 50}")
    print(f"  Total commits    : {s['total_commits']}")
    print(f"  Unique datasets  : {s['unique_datasets']}")
    print(f"  Best accuracy    : {s['best_accuracy']}")
    print(f"  Best loss        : {s['best_loss']}")
    print(f"  Avg accuracy     : {s['avg_accuracy']}")
    print(f"  First commit     : {s['first_commit']}")
    print(f"  Latest commit    : {s['latest_commit']}")

    datasets = compare_datasets()
    if datasets:
        print(f"\n  Per-dataset best accuracy:")
        for row in datasets:
            print(
                f"    {(row['dataset'] or '—'):<20} "
                f"best={row['best_accuracy']}  "
                f"({row['commit_count']} commits)"
            )

    branches = get_branch_tips()
    if branches:
        print(f"\n  Branches:")
        for b in branches:
            print(f"    {b['name']:<20} → {b['commit_hash'][:8]}...")

    tags = get_tags()
    if tags:
        print(f"\n  Tags:")
        for t in tags:
            print(f"    {t['name']:<20} → {t['commit_hash'][:8]}...")

    print(f"{'─' * 50}\n")


def cmd_lineage(args: argparse.Namespace) -> None:
    """model lineage <ref> — show full ancestor chain."""
    _bootstrap()

    from commit import resolve_ref
    from index import lineage

    commit_hash = resolve_ref(args.ref)
    if not commit_hash:
        _die(f"Cannot resolve ref: '{args.ref}'")

    rows = lineage(commit_hash)
    if not rows:
        _die(f"No lineage found for {args.ref}. Is the index up to date? Try: model rebuild")

    if args.json:
        print(json.dumps(rows, indent=2))
        return

    print(f"\n  Lineage of {args.ref} ({commit_hash[:8]}...)\n")
    print(f"  {'DEPTH':<6} {'HASH':<12} {'ACCURACY':>10}  MESSAGE")
    print(f"  {'─'*6} {'─'*12} {'─'*10}  {'─'*40}")
    for row in rows:
        acc = f"{row['accuracy']:.4f}" if row.get("accuracy") is not None else "—"
        print(
            f"  {row['depth']:<6} "
            f"{row['hash'][:10]:<12} "
            f"{acc:>10}  "
            f"{row.get('message', '')}"
        )
    print()


def cmd_rebuild(args: argparse.Namespace) -> None:
    """model rebuild — regenerate the SQLite index from the object store."""
    _bootstrap()

    from index import rebuild_index
    result = rebuild_index()
    _ok(
        f"\nRebuilt index:"
        f"\n  {result['commits_indexed']} commits"
        f"\n  {result['branches_indexed']} branches"
        f"\n  {result['tags_indexed']} tags"
    )


def cmd_verify(args: argparse.Namespace) -> None:
    """model verify — re-hash every object and report any corruption."""
    _bootstrap()

    from object_store import verify_all
    result = verify_all()

    print(f"\n  Object store verification")
    print(f"  Total   : {result['total']}")
    print(f"  OK      : {result['ok']}")
    print(f"  Corrupt : {len(result['corrupted'])}")

    if result["corrupted"]:
        print("\n  Corrupted objects:")
        for h in result["corrupted"]:
            print(f"    {h}")
        sys.exit(1)
    else:
        _ok("\n  All objects intact.")


def cmd_show(args: argparse.Namespace) -> None:
    """model show <ref> — print full details of a single commit."""
    _bootstrap()

    from commit import resolve_ref, load_commit

    commit_hash = resolve_ref(args.ref)
    if not commit_hash:
        _die(f"Cannot resolve ref: '{args.ref}'")

    commit = load_commit(commit_hash)
    if not commit:
        _die(f"Could not load commit {commit_hash[:8]}...")

    if args.json:
        print(json.dumps(commit, indent=2))
        return

    print(f"\n{'─' * 60}")
    print(f"  Commit   {commit_hash[:8]}...")
    print(f"{'─' * 60}")
    print(f"  Message  : {commit.get('message', '')}")
    print(f"  Time     : {(commit.get('timestamp') or '')[:19].replace('T', ' ')}")
    print(f"  Weights  : {commit.get('weights_hash', '')[:16]}...")
    parent = commit.get("parent")
    print(f"  Parent   : {parent[:8] + '...' if parent else 'none (root)'}")

    meta = commit.get("metadata", {})
    if meta:
        print(f"\n  Metadata:")
        for k, v in meta.items():
            print(f"    {k} = {v}")

    env = commit.get("env", {})
    if env:
        print(f"\n  Environment:")
        print(f"    python  : {(env.get('python_version') or '')[:40]}")
        print(f"    platform: {(env.get('platform') or '')[:40]}")

    print(f"{'─' * 60}\n")


# ─────────────────────────────────────────────
#  Argument parser
# ─────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="model",
        description="Git-like versioning for ML models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  model init
  model commit weights.pt -m "baseline" --accuracy 0.87 --lr 0.01 --dataset imagenet
  model log
  model log --branch experiment-aug --limit 5
  model checkout a3f9b2c1 --out restored.pt
  model diff HEAD experiment-aug
  model branch --create experiment-lr --from HEAD
  model branch --switch experiment-lr
  model tag v1.0 --message "production release"
  model search --min-accuracy 0.92 --dataset imagenet
  model best --metric accuracy --top 5
  model show HEAD
  model stats
  model lineage HEAD
  model rebuild
  model verify
        """,
    )

    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    # ── init ────────────────────────────────────────────────────────
    sub.add_parser("init", help="Initialise a new model repo")

    # ── commit ──────────────────────────────────────────────────────
    p_commit = sub.add_parser("commit", help="Commit a model weights file")
    p_commit.add_argument("weights",          help="Path to the weights file")
    p_commit.add_argument("-m", "--message",  required=True, help="Commit message")
    # Well-known metric / hyperparam shortcuts
    p_commit.add_argument("--accuracy",  type=float, help="Accuracy metric")
    p_commit.add_argument("--loss",      type=float, help="Loss metric")
    p_commit.add_argument("--lr",        type=float, help="Learning rate")
    p_commit.add_argument("--epochs",    type=int,   help="Number of epochs")
    p_commit.add_argument("--dataset",              help="Dataset identifier")
    p_commit.add_argument("--model-arch",dest="model_arch", help="Architecture name")
    # Arbitrary extra metadata
    p_commit.add_argument(
        "--meta", action="append", metavar="KEY=VALUE",
        help="Extra metadata (repeatable): --meta batch_size=32 --meta optimizer=adam",
    )

    # ── log ─────────────────────────────────────────────────────────
    p_log = sub.add_parser("log", help="Show commit history")
    p_log.add_argument("--branch", help="Branch or ref to start from (default: HEAD)")
    p_log.add_argument("--limit",  type=int, default=20, help="Max commits (default 20)")
    p_log.add_argument("--json",   action="store_true",  help="Output raw JSON")

    # ── checkout ────────────────────────────────────────────────────
    p_co = sub.add_parser("checkout", help="Restore weights for a commit")
    p_co.add_argument("ref",             help="Branch, tag, commit hash, or HEAD")
    p_co.add_argument("--out",           help="Destination path (default: model_checkout.pt)")
    p_co.add_argument("--detach", action="store_true",
                      help="Detach HEAD (inspect without affecting branch)")

    # ── diff ────────────────────────────────────────────────────────
    p_diff = sub.add_parser("diff", help="Compare two commits or branches")
    p_diff.add_argument("ref_a", help="First ref")
    p_diff.add_argument("ref_b", help="Second ref")
    p_diff.add_argument("--json", action="store_true", help="Output raw JSON")

    # ── branch ──────────────────────────────────────────────────────
    p_br = sub.add_parser("branch", help="Manage branches")
    p_br.add_argument("--list",   action="store_true", help="List all branches")
    p_br.add_argument("--create", metavar="NAME",      help="Create a new branch")
    p_br.add_argument("--from",   dest="from_ref",     help="Starting point for --create")
    p_br.add_argument("--switch", metavar="NAME",      help="Switch to a branch")
    p_br.add_argument("--delete", metavar="NAME",      help="Delete a branch")

    # ── tag ─────────────────────────────────────────────────────────
    p_tag = sub.add_parser("tag", help="Create or list tags")
    p_tag.add_argument("name",          nargs="?",           help="Tag name")
    p_tag.add_argument("--ref",                              help="Ref to tag (default HEAD)")
    p_tag.add_argument("--message", "-m",                   help="Tag annotation")
    p_tag.add_argument("--list",    action="store_true",    help="List all tags")

    # ── search ──────────────────────────────────────────────────────
    p_sr = sub.add_parser("search", help="Search commits by metadata")
    p_sr.add_argument("--min-accuracy", type=float, dest="min_accuracy")
    p_sr.add_argument("--max-loss",     type=float, dest="max_loss")
    p_sr.add_argument("--lr",           type=float)
    p_sr.add_argument("--dataset")
    p_sr.add_argument("--model-arch",   dest="model_arch")
    p_sr.add_argument("--message",      help="Substring match on commit message")
    p_sr.add_argument("--since",        help="ISO timestamp lower bound")
    p_sr.add_argument("--until",        help="ISO timestamp upper bound")
    p_sr.add_argument("--limit",        type=int, default=50)
    p_sr.add_argument("--json",         action="store_true")

    # ── best ────────────────────────────────────────────────────────
    p_best = sub.add_parser("best", help="Top N commits by a metric")
    p_best.add_argument("--metric", default="accuracy",
                        choices=["accuracy", "loss", "lr", "epochs"],
                        help="Metric to rank by (default: accuracy)")
    p_best.add_argument("--top",  type=int, default=5, help="How many results (default 5)")
    p_best.add_argument("--json", action="store_true")

    # ── show ────────────────────────────────────────────────────────
    p_show = sub.add_parser("show", help="Show full details of a single commit")
    p_show.add_argument("ref", help="Branch, tag, commit hash, or HEAD")
    p_show.add_argument("--json", action="store_true")

    # ── stats ───────────────────────────────────────────────────────
    sub.add_parser("stats", help="Aggregate statistics for the whole repo")

    # ── lineage ─────────────────────────────────────────────────────
    p_lin = sub.add_parser("lineage", help="Show full ancestor chain of a commit")
    p_lin.add_argument("ref", help="Branch, tag, commit hash, or HEAD")
    p_lin.add_argument("--json", action="store_true")

    # ── rebuild ─────────────────────────────────────────────────────
    sub.add_parser("rebuild", help="Rebuild the SQLite index from the object store")

    # ── verify ──────────────────────────────────────────────────────
    sub.add_parser("verify", help="Verify integrity of every stored object")

    return parser


# ─────────────────────────────────────────────
#  Dispatch table
# ─────────────────────────────────────────────

_COMMANDS = {
    "init":     cmd_init,
    "commit":   cmd_commit,
    "log":      cmd_log,
    "checkout": cmd_checkout,
    "diff":     cmd_diff,
    "branch":   cmd_branch,
    "tag":      cmd_tag,
    "search":   cmd_search,
    "best":     cmd_best,
    "show":     cmd_show,
    "stats":    cmd_stats,
    "lineage":  cmd_lineage,
    "rebuild":  cmd_rebuild,
    "verify":   cmd_verify,
}


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    handler = _COMMANDS.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    try:
        handler(args)
    except KeyboardInterrupt:
        print("\nAborted.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        if "--debug" in sys.argv:
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()