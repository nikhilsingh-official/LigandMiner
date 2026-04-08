from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from rich.progress import Progress


ProgressAction = Literal["start", "update", "advance", "finish"]


@dataclass(frozen=True)
class ProgressEvent:
    action: ProgressAction
    key: str
    message: str | None = None
    total: int | None = None
    completed: int | None = None
    advance: int = 1


class ProgressSink(Protocol):
    @property
    def raw_progress(self) -> Progress | None: ...

    def emit(self, event: ProgressEvent) -> None: ...

    def start(self, key: str, message: str, total: int | None = None) -> None: ...

    def update(
        self,
        key: str,
        message: str,
        *,
        total: int | None = None,
        completed: int | None = None,
    ) -> None: ...

    def advance(self, key: str, step: int = 1, message: str | None = None) -> None: ...

    def finish(self, key: str, message: str | None = None) -> None: ...

    def child(self, prefix: str) -> "ProgressSink": ...


class NullProgressSink:
    @property
    def raw_progress(self) -> Progress | None:
        return None

    def emit(self, event: ProgressEvent) -> None:
        pass

    def start(self, key: str, message: str, total: int | None = None) -> None:
        pass

    def update(
        self,
        key: str,
        message: str,
        *,
        total: int | None = None,
        completed: int | None = None,
    ) -> None:
        pass

    def advance(self, key: str, step: int = 1, message: str | None = None) -> None:
        pass

    def finish(self, key: str, message: str | None = None) -> None:
        pass

    def child(self, prefix: str) -> "NullProgressSink":
        return self

    def __enter__(self) -> "NullProgressSink":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class RichProgressSink:
    def __init__(
        self,
        progress: Progress,
        *,
        prefix: tuple[str, ...] = (),
        depth: int = 0,
        tasks: dict[str, int] | None = None,
    ) -> None:
        self._progress = progress
        self._prefix = prefix
        self._depth = depth
        self._tasks = tasks if tasks is not None else {}

    def child(self, prefix: str) -> "RichProgressSink":
        return RichProgressSink(
            self._progress,
            prefix=self._prefix + (prefix,),
            depth=self._depth + 1,
            tasks=self._tasks,
        )

    @property
    def raw_progress(self) -> Progress | None:
        return self._progress

    def emit(self, event: ProgressEvent) -> None:
        full_key = self._full_key(event.key)

        if event.action == "start":
            self._start(full_key, event.message or "", event.total)
            return

        if event.action == "update":
            self._update(
                full_key,
                event.message or "",
                total=event.total,
                completed=event.completed,
            )
            return

        if event.action == "advance":
            self._advance(full_key, event.advance, event.message)
            return

        if event.action == "finish":
            self._finish(full_key, event.message)
            return

        raise ValueError(f"Unsupported progress action: {event.action}")

    def start(self, key: str, message: str, total: int | None = None) -> None:
        self.emit(ProgressEvent("start", key, message=message, total=total))

    def update(
        self,
        key: str,
        message: str,
        *,
        total: int | None = None,
        completed: int | None = None,
    ) -> None:
        self.emit(
            ProgressEvent(
                "update",
                key,
                message=message,
                total=total,
                completed=completed,
            )
        )

    def advance(self, key: str, step: int = 1, message: str | None = None) -> None:
        self.emit(ProgressEvent("advance", key, message=message, advance=step))

    def finish(self, key: str, message: str | None = None) -> None:
        self.emit(ProgressEvent("finish", key, message=message))

    def __enter__(self) -> "RichProgressSink":
        self._progress.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return self._progress.__exit__(exc_type, exc, tb)

    def _full_key(self, key: str) -> str:
        return ".".join((*self._prefix, key)) if self._prefix else key

    def _format_message(self, message: str) -> str:
        return f"{'   ' * self._depth}{message}"

    def _start(self, key: str, message: str, total: int | None) -> None:
        if key in self._tasks:
            self._update(key, message, total=total)
            return

        self._tasks[key] = self._progress.add_task(
            self._format_message(message),
            total=total,
        )

    def _update(
        self,
        key: str,
        message: str,
        *,
        total: int | None = None,
        completed: int | None = None,
    ) -> None:
        task_id = self._tasks.get(key)
        if task_id is None:
            self._start(key, message, total)
            task_id = self._tasks[key]

        kwargs: dict[str, object] = {"description": self._format_message(message)}
        if total is not None:
            kwargs["total"] = total
        if completed is not None:
            kwargs["completed"] = completed
        self._progress.update(task_id, **kwargs)

    def _advance(self, key: str, step: int, message: str | None) -> None:
        task_id = self._tasks.get(key)
        if task_id is None:
            self._start(key, message or key, None)
            task_id = self._tasks[key]

        if message is not None:
            self._progress.update(task_id, description=self._format_message(message))
        self._progress.advance(task_id, step)

    def _finish(self, key: str, message: str | None) -> None:
        task_id = self._tasks.get(key)
        if task_id is None:
            if message is not None:
                self._start(key, message, 1)
                task_id = self._tasks[key]
            else:
                return

        kwargs: dict[str, object] = {}
        if message is not None:
            kwargs["description"] = self._format_message(message)
        self._progress.update(task_id, **kwargs)
