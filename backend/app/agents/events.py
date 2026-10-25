"""Custom stream events from inside nodes (node_started, tool_called). Outside a graph run (unit
tests of a single function) there is no stream writer and emitting is a no-op.

A subgraph has its own stream, so events written inside it never reach the run's stream by
themselves. A node that runs a subgraph wraps it in `forward_events()`, which remembers the OUTER
writer; `emit` prefers it."""

import contextlib
from collections.abc import Callable, Iterator
from contextvars import ContextVar
from typing import Any

_outer: ContextVar[Callable[[Any], None] | None] = ContextVar("outer_stream_writer", default=None)


def _current_writer() -> Callable[[Any], None] | None:
    try:
        from langgraph.config import get_stream_writer

        return get_stream_writer()
    except (RuntimeError, ImportError):
        return None


@contextlib.contextmanager
def forward_events() -> Iterator[None]:
    """Use inside a node, around code that runs a subgraph."""
    token = _outer.set(_current_writer())
    try:
        yield
    finally:
        _outer.reset(token)


def emit(event: dict[str, Any]) -> None:
    writer = _outer.get() or _current_writer()
    if writer is not None:
        writer(event)
