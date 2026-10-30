"""Shared test infrastructure: real tiny models, real sessions, fakes, rigs.

``tests/conftest.py`` holds the pytest fixtures; this package holds the plain
helpers, importable by any test module.  Test modules never import from other
test modules; a helper two modules share lives here.

- :mod:`~tests.support.tiny_models` — byte-level tokenizer with a real chat
  template, a ``from_config`` causal LM per chat family, the on-disk
  checkpoint writer, and ``ScriptedReplies``.
- :mod:`~tests.support.sessions` — ``TinySessionFactory`` (a real
  ``DrowseSession.from_pretrained`` on the tiny checkpoint, CPU, neutral
  baseline seeded) and the ``drowse_home`` context manager.
- :mod:`~tests.support.composer` — ``SteeringRig``: a real steering composer
  with its registry, neutral baseline and steering manager, no model.
- :mod:`~tests.support.read_plane` — ``ReadRig``: a real read plane over real
  (or substituted) instruments and a swappable ``CaptureSlot``.
- :mod:`~tests.support.engine` — ``engine_over``: a generation engine built
  directly over a host.
- :mod:`~tests.support.lens` / :mod:`~tests.support.sae` — direct
  construction of the lens and SAE instruments (``LensHarness``,
  ``lens_kwargs``, ``sae_instrument``, ``MockSaeBackend``).
- :mod:`~tests.support.replay` — ``ReplayRig`` and ``unwired_replay``.
- :mod:`~tests.support.capture` — the model-free fit-capture harness
  (``StubHandle``, ``install_capture``).
- :mod:`~tests.support.extraction` — an ``Extraction`` over recording fakes.
- :mod:`~tests.support.decode` — fake decode-loop models and tokenizers.
- :mod:`~tests.support.server` — the route tests' ``mock_session`` and
  ``StubSession``.
- :mod:`~tests.support.chat_templates`, :mod:`~tests.support.manifolds`,
  :mod:`~tests.support.artifacts`, :mod:`~tests.support.tokens` — shared
  fixtures data and builders.
"""
