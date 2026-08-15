"""Cross-process authority for profile identity lifecycle changes."""

from __future__ import annotations

import contextlib
import hashlib
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Iterator

try:  # pragma: no cover - platform selected at import time
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None

try:  # pragma: no cover - platform selected at import time
    import msvcrt
except ImportError:  # pragma: no cover - POSIX
    msvcrt = None


_LOCK_TIMEOUT_SECONDS = 30.0
_holder = threading.local()


def _shared_root() -> Path:
    override = os.environ.get("HERMES_KANBAN_HOME", "").strip()
    if override:
        return Path(override).expanduser()
    from hermes_constants import get_default_hermes_root

    return get_default_hermes_root()


def _lock_path() -> Path:
    return _shared_root() / "profiles" / ".lifecycle.lock"


def _unavailable_path(profile_name: str) -> Path:
    """Return a path-safe persistent fence for a retired profile identity."""
    digest = hashlib.sha256(profile_name.encode("utf-8")).hexdigest()
    return (
        _shared_root()
        / "profiles"
        / ".lifecycle-unavailable"
        / f"{digest}.tombstone"
    )


def mark_profile_unavailable(profile_name: str) -> None:
    """Persist a fail-closed identity fence while holding the lifecycle lock."""
    path = _unavailable_path(profile_name)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        path.parent.chmod(0o700)
    except OSError:
        pass
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, f"{profile_name}\n".encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        path.chmod(0o600)
    except OSError:
        pass


def clear_profile_unavailable(profile_name: str) -> None:
    """Remove a fence when its associated lifecycle mutation rolls back."""
    _unavailable_path(profile_name).unlink(missing_ok=True)


def assert_profile_assignment_available(profile_name: str) -> None:
    """Reject assignment to a retired identity unless it has been recreated.

    Never-seen assignee strings remain valid for backwards compatibility. A
    tombstone becomes blocking only while the corresponding named profile
    directory is absent. Callers must hold :func:`profile_lifecycle_lock` so
    the directory and tombstone observation cannot race a delete or rename.
    """
    path = _unavailable_path(profile_name)
    try:
        path.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise RuntimeError(
            f"cannot verify lifecycle fence for profile '{profile_name}'"
        ) from exc

    # HERMES_KANBAN_HOME may relocate the shared lock/fence authority without
    # relocating profile directories, so resolve liveness through the profile
    # module rather than deriving a path from the Kanban root.
    from hermes_cli.profiles import profile_exists, validate_profile_name

    try:
        validate_profile_name(profile_name)
    except ValueError as exc:
        raise ValueError(f"assignee profile is unavailable: {profile_name}") from exc
    if profile_exists(profile_name):
        return
    raise ValueError(f"assignee profile is unavailable: {profile_name}")


@contextlib.contextmanager
def profile_lifecycle_lock(
    timeout_seconds: float = _LOCK_TIMEOUT_SECONDS,
) -> Iterator[None]:
    """Serialize graph validation with destructive profile mutations."""
    if getattr(_holder, "depth", 0) > 0:
        _holder.depth += 1
        try:
            yield
        finally:
            _holder.depth -= 1
        return

    path = _lock_path()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if msvcrt is not None and (not path.exists() or path.stat().st_size == 0):
        path.write_text(" ", encoding="utf-8")

    with path.open("r+" if msvcrt is not None else "a+", encoding="utf-8") as handle:
        deadline = time.monotonic() + max(1.0, timeout_seconds)
        while True:
            try:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                elif msvcrt is not None:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:  # pragma: no cover - unsupported platform fallback
                    raise RuntimeError("profile lifecycle lock backend unavailable")
                break
            except (BlockingIOError, OSError, PermissionError) as exc:
                if time.monotonic() >= deadline:
                    raise TimeoutError("profile lifecycle authority lock timed out") from exc
                time.sleep(0.05)

        _holder.depth = 1
        try:
            yield
        finally:
            _holder.depth = 0
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            elif msvcrt is not None:  # pragma: no cover - Windows
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def _board_db_paths() -> list[Path]:
    root = _shared_root()
    candidates: list[Path] = []
    override = os.environ.get("HERMES_KANBAN_DB", "").strip()
    if override:
        candidates.append(Path(override).expanduser())
    candidates.append(root / "kanban.db")
    boards = root / "kanban" / "boards"
    if boards.is_dir():
        candidates.extend(sorted(boards.glob("*/kanban.db")))

    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        canonical = str(path.resolve(strict=False))
        if canonical not in seen:
            seen.add(canonical)
            unique.append(path)
    return unique


def assert_profile_has_no_open_assignments(profile_name: str) -> None:
    """Fail closed when any discoverable board has live work for a profile."""
    references: list[str] = []
    for path in _board_db_paths():
        if not path.is_file():
            continue
        uri = path.resolve(strict=False).as_uri() + "?mode=ro"
        try:
            conn = sqlite3.connect(uri, uri=True, timeout=5.0)
            try:
                exists = conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='tasks'"
                ).fetchone()
                if not exists:
                    continue
                rows = conn.execute(
                    "SELECT id FROM tasks WHERE assignee=? "
                    "AND status NOT IN ('done', 'archived') LIMIT 5",
                    (profile_name,),
                ).fetchall()
                references.extend(f"{path}:{row[0]}" for row in rows)
            finally:
                conn.close()
        except sqlite3.Error as exc:
            raise RuntimeError(
                f"cannot verify Kanban assignments in {path}; profile mutation refused"
            ) from exc
    if references:
        raise RuntimeError(
            f"profile '{profile_name}' has open Kanban assignments: "
            + ", ".join(references)
        )
