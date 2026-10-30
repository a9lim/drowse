"""The exclusive-GPU section's call shape, shared by every collaborator.

A session holds one process-wide exclusive section around every model op:
PyTorch's MPS backend serializes device work through one global command
buffer that is not thread-safe, so a second thread committing it aborts the
process.  Collaborators that run model work (the read instruments, the token
replay, the completion scorer, the neutral baseline, the fit glue) take the
section as an explicit constructor dependency of this type.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any

#: ``exclusive(busy_msg, *, phase_msg=None, exc=ConcurrentExtractionError)``:
#: enter the exclusive-GPU section, raising ``exc(busy_msg)`` when another
#: model op holds it and ``exc(phase_msg)`` when ``phase_msg`` is given and a
#: generation is in flight.
ExclusiveGuard = Callable[..., AbstractContextManager[Any]]

__all__ = ["ExclusiveGuard"]
