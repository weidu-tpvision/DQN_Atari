"""Isolated training output directories and their provenance."""

import csv
import json
import os
import subprocess
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from filelock import FileLock, Timeout


def checkpoint_step(checkpoint: dict) -> int | None:
    return checkpoint.get("env_steps", checkpoint.get("global_step"))


def checkpoint_stamp(path):
    stat = Path(path).stat()
    return stat.st_ino, stat.st_size, stat.st_mtime_ns


def validate_resume(checkpoint, trainer, environment, *, legacy_environment=None):
    """Use explicit identity when available and recognize older repo checkpoints."""
    config = checkpoint.get("config") or {}
    saved_trainer = checkpoint.get("trainer")
    saved_environment = checkpoint.get("environment") or config.get("env")
    if saved_trainer is None:
        if any(key in checkpoint for key in ("pool_state", "pool_snapshots", "matches_played", "pool_size")):
            saved_trainer, saved_environment = "selfplay", "PettingZoo/pong_v3"
        elif "conv_h.weight" in checkpoint.get("online_net", {}) or any(
            key in config for key in ("action_masking", "symmetry_augment", "reward")
        ):
            saved_trainer, saved_environment = "2048", "2048"
        elif saved_environment:
            saved_trainer = "atari"

    if saved_trainer is None and trainer == "atari" and legacy_environment:
        saved_trainer, saved_environment = "atari", legacy_environment
    if saved_trainer != trainer or saved_environment != environment:
        source = saved_environment or "unknown environment"
        hint = ""
        if trainer in ("atari", "selfplay"):
            hint = " Use --base-model for weights-only initialization."
        if saved_trainer is None and trainer == "atari":
            hint += " For an older checkpoint, --legacy-resume-env can declare its known game."
        raise ValueError(f"Cannot resume {source} training as {environment}.{hint}")


def _git_info():
    root = Path(__file__).resolve().parent.parent

    def git(*args):
        try:
            result = subprocess.run(["git", *args], cwd=root, capture_output=True, encoding="utf-8",
                                    errors="replace", timeout=5)
            return result.stdout.strip() if result.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            return None

    revision = git("rev-parse", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=normal") if revision else None
    return {"revision": revision, "dirty": None if status is None else bool(status)}


def _can_append_metrics(directory, step):
    for path in directory.glob("*_metrics.csv"):
        try:
            with path.open("rb") as raw:
                if path.stat().st_size:
                    raw.seek(-1, os.SEEK_END)
                    if raw.read(1) != b"\n":
                        return False
            with path.open(encoding="utf-8", newline="") as metrics:
                last_step = -1
                for row in csv.DictReader(metrics):
                    current = int(row["step"])
                    if not last_step <= current <= step or None in row or any(value is None for value in row.values()):
                        return False
                    last_step = current
        except (ValueError, TypeError, KeyError, csv.Error):
            return False
    return True


def _source(path, checkpoint, kind):
    if path is None:
        return None
    return {
        "kind": kind,
        "path": str(path),
        "run_id": checkpoint.get("run_id"),
        "step": checkpoint_step(checkpoint),
    }


@contextmanager
def open_run(
    args, *, default_dir, trainer, environment, agent, evaluation, replay_context,
    checkpoint, resume_path=None, resume_stamp=None, weights_path=None,
):
    """Claim a run directory while training and append its session provenance."""
    source_path = Path(resume_path).resolve() if resume_path else None
    parent = _source(source_path, checkpoint, "resume")
    if weights_path and parent is None:
        parent = _source(Path(weights_path).resolve(), checkpoint, "weights")
    requested = Path(args.save_dir or (source_path.parent if source_path else default_dir)).resolve()
    requested.mkdir(parents=True, exist_ok=True)
    git = _git_info()
    now = datetime.now(timezone.utc).isoformat()
    new_id = uuid4().hex
    directory = requested
    manifest = None

    lock = FileLock(str(directory / ".run.lock"), timeout=0)
    try:
        lock.acquire()
    except Timeout:
        directory = requested / f"run-{new_id}"
    else:
        try:
            manifest_path = directory / "run.json"
            existing = None
            if manifest_path.is_file() and source_path == directory / "latest.pt" and checkpoint.get("run_id"):
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                existing
                and existing.get("schema_version") == 1
                and existing["run_id"] == checkpoint["run_id"]
                and existing["trainer"] == trainer
                and existing["environment"] == environment
                and existing["sessions"][-1]["evaluation"] == evaluation
                and existing["sessions"][-1]["replay_context"] == replay_context
                and existing["sessions"][-1]["start_step"] <= agent.env_steps
                and not args.replay_path
                and checkpoint_stamp(source_path) == resume_stamp
                and _can_append_metrics(directory, agent.env_steps)
            ):
                manifest = existing
            elif any(path.name != ".run.lock" for path in directory.iterdir()):
                directory = requested / f"run-{new_id}"
        except BaseException:
            lock.release()
            raise
        if directory != requested:
            lock.release()

    if directory != requested:
        directory.mkdir()
        lock = FileLock(str(directory / ".run.lock"), timeout=0)
        lock.acquire()
    try:
        continuing = manifest is not None
        if manifest is None:
            manifest = {
                "schema_version": 1, "run_id": new_id, "trainer": trainer,
                "environment": environment, "created_at": now, "parent_checkpoint": parent,
                "sessions": [],
            }
        args.save_dir = str(directory)
        manifest["sessions"].append({
            "started_at": now, "start_step": agent.env_steps, "source_checkpoint": parent,
            "git": git, "evaluation": evaluation, "replay_context": replay_context,
            "config": dict(vars(args), eps_start=agent.eps_start, eps_end=agent.eps_end,
                           eps_decay=agent.eps_decay_steps),
        })
        temporary = directory / ".run.json.tmp"
        try:
            temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            os.replace(temporary, directory / "run.json")
        finally:
            temporary.unlink(missing_ok=True)
        print(f"Run {manifest['run_id']} | {directory}", flush=True)
        yield {key: manifest[key] for key in ("run_id", "trainer", "environment")}, continuing
    finally:
        lock.release()
