"""Shared test infrastructure: real tiny models, real sessions, scripted output.

``tests/conftest.py`` holds the pytest fixtures; this package holds the plain
helpers they are built from, importable by any test module:

- :mod:`tests.support.tiny_models` — a byte-level tokenizer with a real chat
  template, a ``from_config`` causal LM per chat family, an on-disk checkpoint
  writer, and :class:`~tests.support.tiny_models.ScriptedReplies` (forces the
  model to spell scripted replies while keeping real hidden states).
- :mod:`tests.support.sessions` — :class:`~tests.support.sessions.TinySessionFactory`
  (real ``DrowseSession.from_pretrained`` on the tiny checkpoint, CPU, with the
  neutral baseline seeded so the whitener resolves from cache) and the
  ``drowse_home`` context manager.

Test modules must not import helpers from other test modules; shared fakes
belong here.
"""
